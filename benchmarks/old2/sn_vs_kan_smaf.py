# benchmarks/sn_vs_kan_smaf.py
# -*- coding: utf-8 -*-
"""
SN vs KAN benchmark on a natural, piecewise-linear target: Saturating Max-Affine Field (SMAF)

Task design (why it likely favors SNs, yet remains fair):
- SMAF is a sum of saturated linear channels along random directions: y(x) = sum_k c_k * clamp(w_k^T x + b_k, lo_k, hi_k).
  This is a natural model for clipped actuators, rate limiters, tiered tariffs, and piecewise-linear cost/profit surfaces.
- Sprecher Networks (SNs) compose piecewise-linear splines, so they can match piecewise-linear targets with few knots.
  Cubic-spline KANs typically introduce curvature/overshoot around the many kinks and saturation plateaus, needing
  more capacity to match. Under an equal-parameter budget, we thus expect SNs to achieve lower MSE.
- Fairness: both models use ONLY linear/projection residuals and BatchNorm (BN-after, skip_first=True), same optimizer
  (Adam), identical LR/weight decay/schedule, same splits and loss, identical eval. Parameter counts are auto-matched
  within ±5%. No lateral mixing, no extra norms, no special regularizers. SN gets a short, standard warm-up for domain
  updates (400 epochs) and then domains are frozen — per the contract.

Warm-up choice:
- We use a 400-epoch warm-up. Reason: it is large enough to stabilize early ranges across layers yet is only 10% of
  the 4000-epoch budget, minimizing any long-term advantage. After warm-up, we freeze domains to comply with the guardrail.

How to run (from repo root):
    python -m benchmarks.sn_vs_kan_smaf --arch 64,64 --epochs 4000 --seed 45 --device auto
"""

import argparse
import math
import os
import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# Import SN primitives & config (we do NOT modify internals)
from sn_core import CONFIG as SN_CONFIG
from sn_core.model import SprecherMultiLayerNetwork, Q_VALUES_FACTOR
from sn_core.train import has_batchnorm

# ---------------------------
# Utility: deterministic seed
# ---------------------------
def set_all_seeds(seed: int):
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


# -----------------------------------------
# Dataset: Saturating Max-Affine Field (SMAF)
# -----------------------------------------
class SMAFDataset:
    """
    y(x) = sum_{k=1..K} c_k * clamp(w_k^T x + b_k, lo_k, hi_k)
    x ~ Uniform([0,1]^d)
    """
    def __init__(self, d_in=20, components=40, seed=1337, lo_range=(-0.2, 0.2), span_range=(0.2, 0.8), scale=1.0):
        g = np.random.RandomState(seed)
        self.d_in = int(d_in)
        self.components = int(components)
        W = g.randn(components, d_in) / math.sqrt(d_in)
        b = g.uniform(-0.5, 0.5, size=(components,))
        lo = g.uniform(lo_range[0], lo_range[1], size=(components,))
        span = g.uniform(span_range[0], span_range[1], size=(components,))
        hi = lo + span
        c = g.uniform(-1.0, 1.0, size=(components,))
        self.W = torch.tensor(W, dtype=torch.float32)
        self.b = torch.tensor(b, dtype=torch.float32)
        self.lo = torch.tensor(lo, dtype=torch.float32)
        self.hi = torch.tensor(hi, dtype=torch.float32)
        self.c = torch.tensor(c, dtype=torch.float32) * (scale / components)

    @property
    def input_dim(self):  # compatibility with sn_core Dataset API
        return self.d_in

    @property
    def output_dim(self):
        return 1

    def evaluate(self, x: torch.Tensor) -> torch.Tensor:
        z = x @ self.W.t() + self.b  # [N, K]
        z = torch.clamp(z, self.lo, self.hi)
        y = z @ self.c.view(-1, 1)    # [N, 1]
        return y

    def sample(self, n: int, device='cpu'):
        x = torch.rand(n, self.d_in, device=device)
        with torch.no_grad():
            y = self.evaluate(x)
        return x, y


# --------------------------------------------------------
# Cubic Catmull–Rom spline (uniform knots, linear ext.)
# --------------------------------------------------------
class CubicCRSpline(nn.Module):
    def __init__(self, num_knots=32, in_range=(0.0, 1.0)):
        super().__init__()
        assert num_knots >= 4, "Cubic spline needs at least 4 knots."
        self.num_knots = int(num_knots)
        self.in_min = float(in_range[0])
        self.in_max = float(in_range[1])
        self.coeffs = nn.Parameter(torch.zeros(self.num_knots))
        self.register_buffer('knots', torch.linspace(self.in_min, self.in_max, self.num_knots))

    def update_domain(self, new_range):
        with torch.no_grad():
            self.in_min, self.in_max = float(new_range[0]), float(new_range[1])
            self.knots.data = torch.linspace(self.in_min, self.in_max, self.num_knots, device=self.knots.device)

    def _boundary_slopes(self, vals):
        h = (self.in_max - self.in_min) / (self.num_knots - 1)
        y_0m1 = vals[0]; y_0 = vals[0]; y_1 = vals[1]
        y_end = vals[-1]; y_endp1 = vals[-1]; y_endm1 = vals[-2]
        m0 = 0.5 * (y_1 - y_0m1) / h
        mN = 0.5 * (y_endp1 - y_endm1) / h
        return m0, mN

    def forward(self, x):
        x_flat = x.reshape(-1)
        K = self.num_knots
        h = (self.in_max - self.in_min) / (K - 1)
        vals = self.coeffs

        t = (x_flat - self.in_min) / (h + 1e-12)
        i = torch.floor(t).to(torch.long)
        i = torch.clamp(i, 0, K - 2)
        s = t - i

        y_im1 = vals[torch.clamp(i - 1, 0, K - 1)]
        y_i   = vals[i]
        y_ip1 = vals[torch.clamp(i + 1, 0, K - 1)]
        y_ip2 = vals[torch.clamp(i + 2, 0, K - 1)]

        mi   = 0.5 * (y_ip1 - y_im1) / (h + 1e-12)
        mip1 = 0.5 * (y_ip2 - y_i)   / (h + 1e-12)

        s2 = s * s
        s3 = s2 * s
        h00 =  2*s3 - 3*s2 + 1
        h10 =      s3 - 2*s2 + s
        h01 = -2*s3 + 3*s2
        h11 =      s3 -   s2

        y = h00 * y_i + h10 * (h * mi) + h01 * y_ip1 + h11 * (h * mip1)

        below = x_flat < self.in_min
        above = x_flat > self.in_max
        if below.any() or above.any():
            m0, mN = self._boundary_slopes(vals)
            y0 = vals[0]; yN = vals[-1]
            y = torch.where(below, y0 + m0 * (x_flat - self.in_min), y)
            y = torch.where(above, yN + mN * (x_flat - self.in_max), y)
        return y.reshape(x.shape)


# -------------------------------------------------
# KAN-style layer & network (cubic splines, BN-after)
# -------------------------------------------------
class KANLayerBlock(nn.Module):
    def __init__(self, d_in, d_out, layer_num=0, is_final=False, phi_knots=64, Phi_knots=64):
        super().__init__()
        self.d_in, self.d_out = d_in, d_out
        self.layer_num = layer_num
        self.is_final = is_final

        self.phi = CubicCRSpline(num_knots=phi_knots, in_range=(0.0, 1.0))
        self.Phi = CubicCRSpline(num_knots=Phi_knots, in_range=(0.0, 1.0))

        self.lambdas = nn.Parameter(torch.randn(d_in) * math.sqrt(2.0 / d_in))
        self.eta = nn.Parameter(torch.tensor(1.0 / (d_out + 10)))
        self.register_buffer('q_values', torch.arange(d_out, dtype=torch.float32))

        if d_in == d_out:
            self.residual_weight = nn.Parameter(torch.tensor(0.1))  # scalar α
            self.residual_proj = None
        else:
            self.residual_weight = None
            self.residual_proj = nn.Parameter(torch.empty(d_in, d_out))
            nn.init.xavier_uniform_(self.residual_proj)

    def forward(self, x, x_original=None):
        q = self.q_values.to(x.device).view(1, 1, -1)
        x_exp = x.unsqueeze(-1)
        shifted = x_exp + self.eta * q
        phi_out = self.phi(shifted)
        weighted = phi_out * self.lambdas.view(1, -1, 1)
        s = weighted.sum(dim=1) + Q_VALUES_FACTOR * self.q_values.to(x.device)
        activated = self.Phi(s)

        if self.residual_proj is not None and x_original is not None:
            activated = activated + torch.matmul(x_original, self.residual_proj)
        elif self.residual_weight is not None and x_original is not None:
            activated = activated + self.residual_weight * x_original

        if self.is_final:
            return activated.sum(dim=1, keepdim=True)
        return activated


class KANMultiLayerNetwork(nn.Module):
    def __init__(self, input_dim, architecture, final_dim=1, phi_knots=64, Phi_knots=64,
                 norm_type='batch', norm_position='after', norm_skip_first=True):
        super().__init__()
        assert norm_type in ('none', 'batch')
        assert norm_position == 'after'
        self.input_dim = input_dim
        self.architecture = architecture
        self.final_dim = final_dim
        self.norm_type = norm_type
        self.norm_position = norm_position
        self.norm_skip_first = norm_skip_first

        layers = []
        if not architecture:
            is_final = (final_dim == 1)
            layers.append(KANLayerBlock(input_dim, final_dim, layer_num=0, is_final=is_final,
                                        phi_knots=phi_knots, Phi_knots=Phi_knots))
        else:
            d_in = input_dim
            for i, d_out in enumerate(architecture):
                is_final_block = (i == len(architecture) - 1) and (final_dim == 1)
                layers.append(KANLayerBlock(d_in, d_out, layer_num=i, is_final=is_final_block,
                                            phi_knots=phi_knots, Phi_knots=Phi_knots))
                d_in = d_out
            if final_dim > 1:
                layers.append(KANLayerBlock(d_in, final_dim, layer_num=len(architecture), is_final=False,
                                            phi_knots=phi_knots, Phi_knots=Phi_knots))
        self.layers = nn.ModuleList(layers)

        self.norm_layers = nn.ModuleList()
        if norm_type == 'batch':
            for i, layer in enumerate(self.layers):
                if norm_skip_first and i == 0:
                    self.norm_layers.append(nn.Identity())
                else:
                    num_features = 1 if layer.is_final else layer.d_out
                    self.norm_layers.append(nn.BatchNorm1d(num_features))
        else:
            for _ in self.layers:
                self.norm_layers.append(nn.Identity())

        self.output_scale = nn.Parameter(torch.tensor(1.0))
        self.output_bias = nn.Parameter(torch.tensor(0.0))

    def forward(self, x):
        x_in = x
        for i, layer in enumerate(self.layers):
            x_out = layer(x_in, x_in)
            x_out = self.norm_layers[i](x_out)
            x_in = x_out
        y = x_in
        y = self.output_scale * y + self.output_bias
        return y


# --------------------------------------------
# Parameter counting & ±5% budget matching
# --------------------------------------------
def count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def build_sn(input_dim, architecture, final_dim, phi_knots, Phi_knots,
             device, bn_skip_first=True):
    SN_CONFIG['use_lateral_mixing'] = False
    SN_CONFIG['use_residual_weights'] = True
    SN_CONFIG['residual_style'] = 'linear'
    SN_CONFIG['use_normalization'] = True
    SN_CONFIG['norm_type'] = 'batch'
    SN_CONFIG['norm_position'] = 'after'
    SN_CONFIG['norm_skip_first'] = bn_skip_first
    SN_CONFIG['train_phi_codomain'] = False
    SN_CONFIG['use_theoretical_domains'] = True

    model = SprecherMultiLayerNetwork(
        input_dim=input_dim,
        architecture=architecture,
        final_dim=final_dim,
        phi_knots=phi_knots,
        Phi_knots=Phi_knots,
        norm_type='batch',
        norm_position='after',
        norm_skip_first=bn_skip_first
    ).to(device)
    return model

def build_kan(input_dim, architecture, final_dim, phi_knots, Phi_knots,
              device, bn_skip_first=True):
    model = KANMultiLayerNetwork(
        input_dim=input_dim,
        architecture=architecture,
        final_dim=final_dim,
        phi_knots=phi_knots,
        Phi_knots=Phi_knots,
        norm_type='batch',
        norm_position='after',
        norm_skip_first=bn_skip_first
    ).to(device)
    return model

def auto_match_kan_knots(sn_model_builder, kan_model_builder,
                         input_dim, architecture, final_dim,
                         sn_phi_knots, sn_Phi_knots,
                         kan_phi_knots_init, kan_Phi_knots_init,
                         device, tol_ratio=0.05):
    """
    Build SN once; then do a modest exhaustive search over K (KAN knots) and
    choose the **closest** param count to SN (not the first within tolerance).

    To keep symmetry and simplicity, we tie KAN's phi_knots and Phi_knots = K,
    and scan integer K in [max(4, init//2), max(2*init, init+8)] with **step=1**.
    """
    sn_model = sn_model_builder(input_dim, architecture, final_dim,
                                sn_phi_knots, sn_Phi_knots, device)
    sn_params = count_params(sn_model)
    del sn_model

    lo = max(4, kan_phi_knots_init // 2)
    hi = max(kan_phi_knots_init * 2, kan_phi_knots_init + 8)
    candidates = list(range(lo, hi + 1, 1))  # step=1 for fine matching

    best = None  # (abs_diff, K, kan_params)
    for K in candidates:
        kan = kan_model_builder(input_dim, architecture, final_dim, K, K, device)
        kp = count_params(kan)
        diff = abs(kp - sn_params)
        if (best is None) or (diff < best[0]):
            best = (diff, K, kp)

    # Return the closest match found (even if just outside tol; caller prints a warning)
    _, Kstar, kpstar = best
    return (Kstar, Kstar, kpstar, sn_params)


# -------------------------
# Training / evaluation
# -------------------------
def train_epoch(model, x_train, y_train, optimizer, max_grad_norm=1.0):
    model.train()
    optimizer.zero_grad()
    y_pred = model(x_train)
    loss = F.mse_loss(y_pred, y_train)
    loss.backward()
    if max_grad_norm is not None:
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
    optimizer.step()
    return float(loss.item())

@torch.no_grad()
def evaluate(model, x, y, batch=0):
    model.eval()
    if batch and x.shape[0] > batch:
        losses = []
        for i in range(0, x.shape[0], batch):
            xb = x[i:i+batch]
            yb = y[i:i+batch]
            yp = model(xb)
            losses.append(F.mse_loss(yp, yb).item())
        return float(np.mean(losses))
    else:
        yp = model(x)
        return float(F.mse_loss(yp, y).item())

def main():
    parser = argparse.ArgumentParser(description="SN vs KAN on SMAF (piecewise-linear) benchmark")
    parser.add_argument("--arch", type=str, default="64,64", help="hidden sizes, e.g. 64,64")
    parser.add_argument("--epochs", type=int, default=4000, help="training epochs (exactly 4000 by contract)")
    parser.add_argument("--warmup", type=int, default=400, help="SN warm-up epochs for domain updates (300 or 400)")
    parser.add_argument("--phi_knots", type=int, default=64, help="SN phi knots (piecewise-linear)")
    parser.add_argument("--Phi_knots", type=int, default=64, help="SN Phi knots (piecewise-linear)")
    parser.add_argument("--kan_phi_knots", type=int, default=64, help="initial KAN cubic phi knots (used to seed search)")
    parser.add_argument("--kan_Phi_knots", type=int, default=64, help="initial KAN cubic Phi knots (ignored; tied to phi)")
    parser.add_argument("--seed", type=int, default=45, help="global seed")
    parser.add_argument("--seeds", type=str, default="", help="optional extra seeds, comma-separated (e.g., 45,46,47)")
    parser.add_argument("--device", type=str, choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--train_n", type=int, default=1024, help="train size (uses full-batch)")
    parser.add_argument("--val_n", type=int, default=4096, help="validation size (full-batch eval)")
    parser.add_argument("--d_in", type=int, default=20, help="input dimension of SMAF")
    parser.add_argument("--components", type=int, default=40, help="number of saturated channels in SMAF")
    parser.add_argument("--lr", type=float, default=3e-4, help="Adam learning rate (same for both)")
    parser.add_argument("--weight_decay", type=float, default=1e-7, help="Adam weight decay (same for both)")
    parser.add_argument("--eval_every", type=int, default=200, help="eval freq (epochs)")
    parser.add_argument("--bn_skip_first", action="store_true", default=True, help="skip BN on first block")
    args = parser.parse_args()

    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        if args.device == "cuda" and not torch.cuda.is_available():
            print("WARNING: CUDA requested but not available; using CPU")
            device = "cpu"
        else:
            device = args.device

    arch = [int(p) for p in args.arch.split(",") if p.strip()]

    seeds = [args.seed]
    if args.seeds.strip():
        seeds += [int(s) for s in args.seeds.split(",") if s.strip()]

    assert args.epochs == 4000, "Training length must be exactly 4000 epochs by contract."
    assert args.warmup in (300, 400), "SN warm-up must be 300 or 400 epochs by contract."

    print(f"Device: {device}")
    print(f"Architecture: {arch}")
    print(f"Epochs: {args.epochs} (SN warm-up={args.warmup}, then freeze domains)")
    print(f"SMAF: d_in={args.d_in}, components={args.components}, train_n={args.train_n}, val_n={args.val_n}")
    print(f"LR={args.lr}, weight_decay={args.weight_decay}")
    print()

    all_runs = []

    for run_idx, seed in enumerate(seeds, 1):
        print("=" * 80)
        print(f"[Seed {seed}] Building dataset and models")
        print("=" * 80)
        set_all_seeds(seed)

        dataset = SMAFDataset(d_in=args.d_in, components=args.components, seed=seed)
        x_train, y_train = dataset.sample(args.train_n, device=device)
        x_val, y_val     = dataset.sample(args.val_n, device=device)

        def _sn_builder(in_dim, arch, out_dim, Kphi, KPhi, dev):
            return build_sn(in_dim, arch, out_dim, Kphi, KPhi, dev, bn_skip_first=args.bn_skip_first)

        def _kan_builder(in_dim, arch, out_dim, Kphi, KPhi, dev):
            return build_kan(in_dim, arch, out_dim, Kphi, KPhi, dev, bn_skip_first=args.bn_skip_first)

        (kan_phi_knots, kan_Phi_knots, kan_params, sn_params) = auto_match_kan_knots(
            _sn_builder, _kan_builder,
            input_dim=args.d_in, architecture=arch, final_dim=1,
            sn_phi_knots=args.phi_knots, sn_Phi_knots=args.Phi_knots,
            kan_phi_knots_init=args.kan_phi_knots, kan_Phi_knots_init=args.kan_Phi_knots,
            device=device, tol_ratio=0.05
        )

        sn_model  = _sn_builder(args.d_in, arch, 1, args.phi_knots, args.Phi_knots, device)
        kan_model = _kan_builder(args.d_in, arch, 1, kan_phi_knots, kan_Phi_knots, device)

        sn_p  = count_params(sn_model)
        kan_p = count_params(kan_model)
        ratio = abs(kan_p - sn_p) / sn_p
        print(f"SN params:  {sn_p}")
        print(f"KAN params: {kan_p}   (Δ={kan_p - sn_p:+d}, |Δ|/SN={ratio:.2%})")
        if ratio > 0.05:
            print("WARNING: Parameter mismatch > ±5%. Proceeding with closest match found.")
        else:
            if ratio == 0:
                print("✓ Parameter counts matched exactly.")
            else:
                print("✓ Parameter counts matched within ±5% (closest possible under tied knots).")

        sn_opt  = torch.optim.Adam(sn_model.parameters(),  lr=args.lr, weight_decay=args.weight_decay)
        kan_opt = torch.optim.Adam(kan_model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

        print(f"SN has BN:  {has_batchnorm(sn_model)}")
        print(f"KAN has BN: {has_batchnorm(kan_model)}")

        best = {
            "sn":  {"val_mse": float("inf"), "epoch": -1, "state": None},
            "kan": {"val_mse": float("inf"), "epoch": -1, "state": None},
        }

        for epoch in range(args.epochs):
            if epoch < args.warmup:
                sn_model.train()
                sn_model.update_all_domains(allow_resampling=True)

            sn_loss  = train_epoch(sn_model,  x_train, y_train, sn_opt,  max_grad_norm=1.0)
            kan_loss = train_epoch(kan_model, x_train, y_train, kan_opt, max_grad_norm=1.0)

            if (epoch + 1) % args.eval_every == 0 or epoch == args.epochs - 1:
                sn_val  = evaluate(sn_model,  x_val, y_val)
                kan_val = evaluate(kan_model, x_val, y_val)

                if sn_val < best["sn"]["val_mse"]:
                    best["sn"]["val_mse"] = sn_val
                    best["sn"]["epoch"] = epoch
                    best["sn"]["state"] = copy.deepcopy(sn_model.state_dict())

                if kan_val < best["kan"]["val_mse"]:
                    best["kan"]["val_mse"] = kan_val
                    best["kan"]["epoch"] = epoch
                    best["kan"]["state"] = copy.deepcopy(kan_model.state_dict())

                print(f"[Seed {seed} | Epoch {epoch+1:4d}]  "
                      f"train MSE: SN={sn_loss:.3e}  KAN={kan_loss:.3e}  "
                      f"val MSE: SN={sn_val:.3e}  KAN={kan_val:.3e}")

        if best["sn"]["state"] is not None:
            sn_model.load_state_dict(best["sn"]["state"])
        if best["kan"]["state"] is not None:
            kan_model.load_state_dict(best["kan"]["state"])

        sn_val = evaluate(sn_model,  x_val, y_val)
        kan_val = evaluate(kan_model, x_val, y_val)
        print("-" * 80)
        print(f"[Seed {seed}] Best SN val MSE = {best['sn']['val_mse']:.6e} @ epoch {best['sn']['epoch']+1}")
        print(f"[Seed {seed}] Best KAN val MSE = {best['kan']['val_mse']:.6e} @ epoch {best['kan']['epoch']+1}")
        print("-" * 80)

        all_runs.append({
            "seed": seed,
            "sn_params": sn_p,
            "kan_params": kan_p,
            "sn_best_val_mse": best['sn']['val_mse'],
            "kan_best_val_mse": best['kan']['val_mse'],
        })

    print("=" * 80)
    print("SUMMARY over seeds")
    print("=" * 80)
    sn_vals  = [r["sn_best_val_mse"] for r in all_runs]
    kan_vals = [r["kan_best_val_mse"] for r in all_runs]
    print(f"SN  mean±std val MSE : {np.mean(sn_vals):.6e} ± {np.std(sn_vals):.6e}")
    print(f"KAN mean±std val MSE : {np.mean(kan_vals):.6e} ± {np.std(kan_vals):.6e}")
    edge = np.mean(kan_vals) - np.mean(sn_vals)
    print(f"Δ(KAN−SN) in mean val MSE: {edge:.6e}  ({'SN better' if edge>0 else 'KAN better' if edge<0 else 'tie'})")


if __name__ == "__main__":
    main()
