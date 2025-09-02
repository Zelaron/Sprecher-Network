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
    Differentiable piecewise-linear spline on [in_min, in_max] with K knots and
    trainable values (coeffs) at the knots. IMPORTANT: we use linear
    extrapolation outside [in_min, in_max] instead of clamping, to avoid
    'flatlining' the activation when upstream values drift out of range.
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
            with torch.no_grad():
                self.coeffs.copy_(self.grid.clone())

    def forward(self, x):
        # map x to knot index t in [0, K-1] *without clamping*, then extrapolate linearly
        denom = (self.in_max - self.in_min).clamp_min(1e-12)
        t = (x - self.in_min) / denom * (self.K - 1)  # can be <0 or >K-1
        i0 = torch.floor(t).long()
        i0c = torch.clamp(i0, 0, self.K - 2)
        i1c = i0c + 1
        alpha = t - i0c.to(t.dtype)  # alpha can be <0 or >1 -> linear extrapolation
        c0 = self.coeffs[i0c]
        c1 = self.coeffs[i1c]
        return c0 + alpha * (c1 - c0)


class KANUnivariate(nn.Module):
    """
    A single learned 1D activation: ϕ(x) = wb * silu(x) + ws * spline(x).
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
        self.phi = nn.ModuleList(
            [KANUnivariate(K=K, in_min=in_min, in_max=in_max) for _ in range(d_in * d_out)]
        )

    def forward(self, x):
        B = x.shape[0]
        out = x.new_zeros(B, self.d_out)
        idx = 0
        for j in range(self.d_out):
            s = 0.0
            for i in range(self.d_in):
                s = s + self.phi[idx](x[:, i])
                idx += 1
            out[:, j] = s
        return out


class KANNet(nn.Module):
    """
    Multi-layer KAN: interleave KANLayer blocks, then a final per-head affine scale+bias.
    Widths in 'architecture' define hidden layer sizes.
    """
    def __init__(self, input_dim, architecture, final_dim, K=60):
        super().__init__()
        dims = [input_dim] + list(architecture) + [final_dim]
        layers = []
        for a, b in zip(dims[:-1], dims[1:]):
            layers.append(KANLayer(a, b, K=K, in_min=0.0, in_max=1.0))
        self.layers = nn.ModuleList(layers)
        # Per-output affine to avoid over-constraining all heads to share a global scale/bias
        self.out_scale = nn.Parameter(torch.ones(final_dim))
        self.out_bias  = nn.Parameter(torch.zeros(final_dim))

    def forward(self, x):
        h = x  # upstream code or dataset can normalize inputs if desired
        for L in self.layers:
            h = L(h)
        return h * self.out_scale + self.out_bias


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
    corr_heads_used: int
    seconds: float
    notes: str = ""


def rmse_per_head(y_true, y_pred):
    # y shape: [N, D]
    mse = torch.mean((y_true - y_pred) ** 2, dim=0)
    per_head = torch.sqrt(mse)
    mean_rmse = torch.sqrt(torch.mean((y_true - y_pred) ** 2))
    return per_head.cpu().tolist(), float(mean_rmse.cpu())


def corr_frobenius(y_true, y_pred, eps=1e-8):
    """
    Compare correlation structure across output heads.
    We guard against degenerate predictions by only using heads whose
    true AND predicted std are both > eps. Returns (fro_norm, heads_used).
    """
    # center
    yt = y_true - y_true.mean(dim=0, keepdim=True)
    yp = y_pred - y_pred.mean(dim=0, keepdim=True)
    # stds
    std_t = yt.std(dim=0, keepdim=True)
    std_p = yp.std(dim=0, keepdim=True)
    use = ((std_t > eps) & (std_p > eps)).squeeze(0)

    if use.sum() >= 2:
        yt = yt[:, use] / (std_t[:, use] + eps)
        yp = yp[:, use] / (std_p[:, use] + eps)
        Ct = yt.t().mm(yt) / (yt.shape[0] - 1)
        Cp = yp.t().mm(yp) / (yp.shape[0] - 1)
        fr = float(torch.norm(Ct - Cp, p='fro').cpu())
        heads_used = int(use.sum().item())
    else:
        # Not enough non-degenerate heads to define a correlation matrix difference
        fr = float('nan')
        heads_used = int(use.sum().item())

    return fr, heads_used


def train_kan(x_train, y_train, input_dim, final_dim, arch, K, device, epochs=4000, lr=1e-3, wd=1e-6, seed=0):
    torch.manual_seed(seed); np.random.seed(seed)
    model = KANNet(input_dim=input_dim, architecture=arch, final_dim=final_dim, K=K).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
    loss_fn = nn.MSELoss()
    x_train = x_train.to(device); y_train = y_train.to(device)
    t0 = time.time()
    for _ in range(epochs):
        opt.zero_grad(set_to_none=True)
        yhat = model(x_train)
        loss = loss_fn(yhat, y_train)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
    secs = time.time() - t0
    with torch.no_grad():
        train_mse = float(loss_fn(model(x_train), y_train).cpu())
    return model, train_mse, secs


def evaluate_on_fixed_test(model, x_test, y_test):
    with torch.no_grad():
        y_pred = model(x_test)
        per_head, mean_rmse = rmse_per_head(y_test, y_pred)
        corrF, used = corr_frobenius(y_test, y_pred)
    return per_head, mean_rmse, corrF, used


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--seed", type=int, default=45)
    parser.add_argument("--epochs", type=int, default=4000)

    # Dataset
    parser.add_argument("--dataset", type=str, default="toy_4d_to_5d")

    # SN config
    parser.add_argument("--sn_arch", type=str, default="15,15")
    parser.add_argument("--sn_phi_knots", type=int, default=60)
    parser.add_argument("--sn_Phi_knots", type=int, default=60)
    parser.add_argument("--sn_norm", type=str, default=None, choices=[None, "none", "batch", "layer"])

    # KAN config
    parser.add_argument("--kan_arch", type=str, default="15,15")
    parser.add_argument("--kan_K", type=int, default=60)

    # Test set size
    parser.add_argument("--n_test", type=int, default=20000)

    args = parser.parse_args()
    device = ("cuda" if (args.device == "auto" and torch.cuda.is_available()) else
              ("cpu" if args.device == "auto" else args.device))

    # dataset
    dataset = get_dataset(args.dataset)
    input_dim = dataset.input_dim
    final_dim = dataset.output_dim

    # ------------------------------
    # Sample a fixed test set ONCE
    # ------------------------------
    # Ensure reproducibility of the test split within this run and comparability across models
    torch.manual_seed(args.seed + 2025)
    with torch.no_grad():
        x_test, y_test = dataset.sample(args.n_test, device=device)

    # ---- Train SN using your trainer (this returns the actual x_train used) ----
    if args.sn_norm is not None:
        norm_type = args.sn_norm
        norm_pos = CONFIG.get('norm_position', 'after')
        norm_skip_first = CONFIG.get('norm_skip_first', True)
    else:
        norm_type = CONFIG.get('norm_type', 'batch') if CONFIG.get('use_normalization', True) else 'none'
        norm_pos = CONFIG.get('norm_position', 'after')
        norm_skip_first = CONFIG.get('norm_skip_first', True)

    sn_arch = [int(x) for x in args.sn_arch.split(",")] if args.sn_arch.strip() else []

    sn_t0 = time.time()
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
    sn_secs = time.time() - sn_t0

    sn_model = plotting_snapshot["model"].to(device)
    x_train  = plotting_snapshot["x_train"].to(device)
    y_train  = plotting_snapshot["y_train"].to(device)

    # Param count (exact from the model)
    sn_params = count_params(sn_model)

    # Train performance (MSE)
    with torch.no_grad():
        sn_train_mse = float(torch.mean((sn_model(x_train) - y_train) ** 2).cpu())

    # Test evaluation
    sn_per_head, sn_rmse_mean, sn_corrF, sn_corr_used = evaluate_on_fixed_test(sn_model, x_test, y_test)

    sn_result = RunResult(
        model_name=f"SN(arch={sn_arch}, K={args.sn_phi_knots}/{args.sn_Phi_knots})",
        params=sn_params,
        train_mse=sn_train_mse,
        test_rmse_mean=sn_rmse_mean,
        test_rmse_per_head=sn_per_head,
        corr_Frob=sn_corrF,
        corr_heads_used=sn_corr_used,
        seconds=sn_secs,
        notes="SN timing measured around train_network()."
    )

    # ---- Train KAN on the EXACT SAME (x_train,y_train) & evaluate on the SAME test set ----
    kan_arch = [int(x) for x in args.kan_arch.split(",")] if args.kan_arch.strip() else []
    kan_model, kan_train_mse, kan_secs = train_kan(
        x_train=x_train, y_train=y_train,
        input_dim=input_dim, final_dim=final_dim,
        arch=kan_arch, K=args.kan_K,
        device=device, epochs=args.epochs, seed=args.seed
    )
    kan_params = count_params(kan_model)
    kan_per_head, kan_rmse_mean, kan_corrF, kan_corr_used = evaluate_on_fixed_test(kan_model, x_test, y_test)
    # Helpful note if correlation is NaN or very few heads are usable
    note = ""
    if not math.isfinite(kan_corrF) or kan_corr_used < max(2, final_dim // 2):
        note = f"Warning: corr_Frob based on only {kan_corr_used}/{final_dim} heads; " \
               f"predicted std too small on the others."

    kan_result = RunResult(
        model_name=f"KAN(arch={kan_arch}, K={args.kan_K})",
        params=kan_params,
        train_mse=kan_train_mse,
        test_rmse_mean=kan_rmse_mean,
        test_rmse_per_head=kan_per_head,
        corr_Frob=kan_corrF,
        corr_heads_used=kan_corr_used,
        seconds=kan_secs,
        notes=note
    )

    # ---- Print & save results ----
    def pretty(r: RunResult):
        # JSON-friendly dict; keep original keys but add helpful fields
        out = {
            "model": r.model_name,
            "params": r.params,
            "train_mse": r.train_mse,
            "test_rmse_mean": r.test_rmse_mean,
            "test_rmse_per_head": [float(x) for x in r.test_rmse_per_head],
            "corr_Frob_error": (None if (r.corr_Frob is None or not math.isfinite(r.corr_Frob)) else r.corr_Frob),
            "corr_Frob_heads_used": r.corr_heads_used,
            "train_seconds": r.seconds
        }
        if r.notes:
            out["notes"] = r.notes
        return out

    results = [pretty(sn_result), pretty(kan_result)]
    print(f"\n=== Head-to-Head Results ({args.dataset}) ===")
    for r in results:
        print(json.dumps(r, indent=2))

    os.makedirs("results", exist_ok=True)
    out_path = f"results/sn_vs_kan_{args.dataset}.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()