# compare_sn_vs_kan_multioutput.py
import argparse, math, time, os, json
from dataclasses import dataclass
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# --- Use your SN code directly ---
from sn_core import get_dataset, train_network, CONFIG

# -------------------------------
# Minimal KAN-like implementation
# -------------------------------

class PiecewiseLinearSpline1D(nn.Module):
    """
    Simple, differentiable piecewise-linear spline on [in_min, in_max]
    with K knots and trainable values (coeffs) at the knots.
    """
    def __init__(self, K=60, in_min=0.0, in_max=1.0, init_identity=False):
        super().__init__()
        assert K >= 2
        self.K = K
        self.register_buffer('in_min', torch.tensor(float(in_min)))
        self.register_buffer('in_max', torch.tensor(float(in_max)))
        self.register_buffer('grid', torch.linspace(float(in_min), float(in_max), steps=K))
        # coeffs are function values at knots
        self.coeffs = nn.Parameter(torch.zeros(K))
        if init_identity:
            # Initialize approximately as identity on [in_min,in_max]
            with torch.no_grad():
                self.coeffs.copy_(self.grid.clone())

    def forward(self, x):
        # clamp to domain, then linear interpolation
        x = torch.clamp(x, min=self.in_min.item(), max=self.in_max.item())
        # map x to [0, K-1]
        t = (x - self.in_min) / (self.in_max - self.in_min) * (self.K - 1)
        i0 = torch.clamp(t.floor().long(), 0, self.K - 2)
        i1 = i0 + 1
        alpha = (t - i0.float()).unsqueeze(-1)  # broadcast if needed
        c0 = self.coeffs[i0]
        c1 = self.coeffs[i1]
        return (1 - alpha.squeeze(-1)) * c0 + alpha.squeeze(-1) * c1


class KANUnivariate(nn.Module):
    """
    A single learned 1D activation ϕ(x) = wb * silu(x) + ws * spline(x).
    """
    def __init__(self, K=60, in_min=0.0, in_max=1.0):
        super().__init__()
        self.spline = PiecewiseLinearSpline1D(K=K, in_min=in_min, in_max=in_max, init_identity=False)
        self.wb = nn.Parameter(torch.tensor(1.0))  # residual basis scale
        self.ws = nn.Parameter(torch.tensor(1.0))  # spline scale

    def forward(self, x):
        return self.wb * F.silu(x) + self.ws * self.spline(x)


class KANLayer(nn.Module):
    """
    KAN layer mapping d_in -> d_out:
    y_j = sum_i ϕ_{ij}(x_i), with one univariate function per edge (i,j).
    """
    def __init__(self, d_in, d_out, K=60, in_min=0.0, in_max=1.0):
        super().__init__()
        self.d_in = d_in
        self.d_out = d_out
        # ϕ_{ij}: ModuleList of length d_in * d_out
        self.phi = nn.ModuleList([KANUnivariate(K=K, in_min=in_min, in_max=in_max)
                                  for _ in range(d_in * d_out)])

    def forward(self, x):
        B = x.shape[0]
        out = x.new_zeros(B, self.d_out)
        idx = 0
        # sum per edge
        for j in range(self.d_out):
            s = 0.0
            for i in range(self.d_in):
                s = s + self.phi[idx](x[:, i])
                idx += 1
            out[:, j] = s
        return out


class KANNet(nn.Module):
    """
    Multi-layer KAN: interleave KANLayer blocks, then a final affine scale+bias.
    Widths in 'architecture' define hidden layer sizes.
    """
    def __init__(self, input_dim, architecture, final_dim, K=60):
        super().__init__()
        dims = [input_dim] + list(architecture) + [final_dim]
        layers = []
        for a, b in zip(dims[:-1], dims[1:]):
            layers.append(KANLayer(a, b, K=K, in_min=0.0, in_max=1.0))
        self.layers = nn.ModuleList(layers)
        self.out_scale = nn.Parameter(torch.tensor(1.0))
        self.out_bias  = nn.Parameter(torch.tensor(0.0))

    def forward(self, x):
        h = x
        for L in self.layers:
            h = L(h)
        return self.out_scale * h + self.out_bias

def count_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

@dataclass
class RunResult:
    model_name: str
    params: int
    train_mse: float
    test_rmse_mean: float
    test_rmse_per_head: list
    corr_Frob: float
    seconds: float


def rmse_per_head(y_true, y_pred):
    # y shape: [N, D]
    mse = torch.mean((y_true - y_pred)**2, dim=0)
    return torch.sqrt(mse).cpu().tolist(), float(torch.sqrt(torch.mean((y_true - y_pred)**2)).cpu())


def corr_frobenius(y_true, y_pred):
    # Corr matrices per output head on the same samples
    # Center:
    yt = y_true - y_true.mean(dim=0, keepdim=True)
    yp = y_pred - y_pred.mean(dim=0, keepdim=True)
    # Normalize
    yt = yt / (yt.std(dim=0, keepdim=True) + 1e-8)
    yp = yp / (yp.std(dim=0, keepdim=True) + 1e-8)
    Ct = yt.t().mm(yt) / (y_true.shape[0] - 1)
    Cp = yp.t().mm(yp) / (y_true.shape[0] - 1)
    return float(torch.norm(Ct - Cp, p='fro').cpu())


def train_kan(x_train, y_train, input_dim, final_dim, arch, K, device, epochs=4000, lr=1e-3, wd=1e-6, seed=0):
    torch.manual_seed(seed); np.random.seed(seed)
    model = KANNet(input_dim=input_dim, architecture=arch, final_dim=final_dim, K=K).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
    loss_fn = nn.MSELoss()
    x_train = x_train.to(device); y_train = y_train.to(device)
    t0 = time.time()
    for e in range(epochs):
        opt.zero_grad()
        yhat = model(x_train)
        loss = loss_fn(yhat, y_train)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
    secs = time.time() - t0
    with torch.no_grad():
        train_mse = float(loss_fn(model(x_train), y_train).cpu())
    return model, train_mse, secs


def evaluate(model, dataset, device, n_test=20000):
    with torch.no_grad():
        x_test, y_test = dataset.sample(n_test, device=device)
        y_pred = model(x_test)
        per_head, mean_rmse = rmse_per_head(y_test, y_pred)
        corrF = corr_frobenius(y_test, y_pred)
    return per_head, mean_rmse, corrF


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=str, default="auto", choices=["auto","cpu","cuda"])
    parser.add_argument("--seed", type=int, default=45)
    parser.add_argument("--epochs", type=int, default=4000)

    # Dataset
    parser.add_argument("--dataset", type=str, default="toy_4d_to_5d")

    # SN config
    parser.add_argument("--sn_arch", type=str, default="15,15")
    parser.add_argument("--sn_phi_knots", type=int, default=60)
    parser.add_argument("--sn_Phi_knots", type=int, default=60)
    parser.add_argument("--sn_norm", type=str, default=None, choices=[None,"none","batch","layer"])

    # KAN config
    parser.add_argument("--kan_arch", type=str, default="15,15")
    parser.add_argument("--kan_K", type=int, default=60)

    # Test set size
    parser.add_argument("--n_test", type=int, default=20000)

    args = parser.parse_args()
    device = ("cuda" if (args.device=="auto" and torch.cuda.is_available()) else
              ("cpu" if args.device=="auto" else args.device))

    # dataset
    dataset = get_dataset(args.dataset)
    input_dim = dataset.input_dim
    final_dim = dataset.output_dim

    # ---- Train SN using your trainer (this returns the actual x_train used) ----
    if args.sn_norm is not None:
        # Respect explicit norm setting; otherwise use CONFIG
        norm_type = args.sn_norm
        norm_pos  = CONFIG.get('norm_position', 'after')
        norm_skip_first = CONFIG.get('norm_skip_first', True)
    else:
        # From CONFIG defaults
        norm_type = CONFIG.get('norm_type', 'batch') if CONFIG.get('use_normalization', True) else 'none'
        norm_pos  = CONFIG.get('norm_position', 'after')
        norm_skip_first = CONFIG.get('norm_skip_first', True)

    sn_arch = [int(x) for x in args.sn_arch.split(",")] if args.sn_arch.strip() else []
    plotting_snapshot, _ = train_network(
        dataset=dataset,
        architecture=sn_arch,
        total_epochs=args.epochs,
        print_every=max(1, args.epochs // 10),
        device=device,
        phi_knots=args.sn_phi_knots,
        Phi_knots=args.sn_Phi_knots,
        seed=args.seed,
        norm_type=norm_type,
        norm_position=norm_pos,
        norm_skip_first=norm_skip_first,
        no_load_best=False,
        bn_recalc_on_load=False,
    )
    sn_model = plotting_snapshot["model"].to(device)
    x_train  = plotting_snapshot["x_train"].to(device)
    y_train  = plotting_snapshot["y_train"].to(device)

    # Param count (exact from the model)
    sn_params = count_params(sn_model)

    # Train performance (MSE)
    with torch.no_grad():
        sn_train_mse = float(torch.mean((sn_model(x_train) - y_train)**2).cpu())

    # Test evaluation
    sn_per_head, sn_rmse_mean, sn_corrF = evaluate(sn_model, dataset, device, n_test=args.n_test)

    sn_result = RunResult(
        model_name=f"SN(arch={sn_arch}, K={args.sn_phi_knots}/{args.sn_Phi_knots})",
        params=sn_params,
        train_mse=sn_train_mse,
        test_rmse_mean=sn_rmse_mean,
        test_rmse_per_head=sn_per_head,
        corr_Frob=sn_corrF,
        seconds=0.0 # training time already printed by SN trainer; skip here
    )

    # ---- Train KAN on the EXACT SAME (x_train,y_train) ----
    kan_arch = [int(x) for x in args.kan_arch.split(",")] if args.kan_arch.strip() else []
    kan_model, kan_train_mse, kan_secs = train_kan(
        x_train=x_train, y_train=y_train,
        input_dim=input_dim, final_dim=final_dim,
        arch=kan_arch, K=args.kan_K,
        device=device, epochs=args.epochs, seed=args.seed
    )
    kan_params = count_params(kan_model)
    kan_per_head, kan_rmse_mean, kan_corrF = evaluate(kan_model, dataset, device, n_test=args.n_test)
    kan_result = RunResult(
        model_name=f"KAN(arch={kan_arch}, K={args.kan_K})",
        params=kan_params,
        train_mse=kan_train_mse,
        test_rmse_mean=kan_rmse_mean,
        test_rmse_per_head=kan_per_head,
        corr_Frob=kan_corrF,
        seconds=kan_secs
    )

    # ---- Print & save results ----
    def pretty(r: RunResult):
        return {
            "model": r.model_name,
            "params": r.params,
            "train_mse": r.train_mse,
            "test_rmse_mean": r.test_rmse_mean,
            "test_rmse_per_head": [float(x) for x in r.test_rmse_per_head],
            "corr_Frob_error": r.corr_Frob,
            "train_seconds": r.seconds
        }

    results = [pretty(sn_result), pretty(kan_result)]
    print("\n=== Head-to-Head Results (Toy4Dto5D) ===")
    for r in results:
        print(json.dumps(r, indent=2))

    os.makedirs("results", exist_ok=True)
    with open("results/sn_vs_kan_toy4dto5d.json", "w") as f:
        json.dump(results, f, indent=2)
    print("\nSaved: results/sn_vs_kan_toy4dto5d.json")

if __name__ == "__main__":
    main()