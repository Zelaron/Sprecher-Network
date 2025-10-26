# benchmarks/trimix_sn_vs_kan.py
# -*- coding: utf-8 -*-
"""
TriWaveMix benchmark: Sprecher Networks (SN, piecewise-linear splines with warm-up domain updates)
vs. KAN-style blocks (cubic Catmull–Rom splines) under a tightly controlled fairness contract.

Rationale (kept succinct on purpose):
- Task: A 10D "TriWaveMix" regression f: [0,1]^10 -> R built from triangular waves (PWL) per-dimension plus
  a sparse linear-mix triangular term. This is *natural* in signal/spectral synthesis and commonly used in
  control/EE (triangle carriers, PWM), yet rich in kinks. It's not a trivial tweak of any existing repo dataset.
- Why SNs should win (without cheating):
  * SN uses piecewise-linear splines; PWL targets can be represented exactly with few knots.
  * KAN's cubic splines must bend around kinks, which increases local curvature demand and tends to overshoot;
    with a matched parameter budget and identical training schedule, cubic-spline KANs usually converge slower
    and fit kinks less crisply.
  * SN "domain updates" (only for the first 400 epochs here) align φ/Φ domains to internal ranges; then domains
    are *frozen*, keeping the playing field fair while letting SN leverage its intended mechanism during warm-up.

Fairness contract implemented here:
- Training length: EXACTLY 4000 epochs for both.
- Normalization: ONLY BatchNorm (position='after', skip_first=True) in both models.
- Residuals: ONLY linear/projection residuals in both models (no node-centric residuals).
- No lateral mixing, no attention/lateral tricks, no bespoke regularizers, no special losses.
- Same optimizer (Adam), same base LR, weight decay, batch size, and schedule (no advanced scheduler).
- SN domain updates: ENABLED ONLY for a 400-epoch warm-up (argument `--warmup {300|400}`, default 400),
  then domains are frozen (no further updates).
- Parameter counts matched within ±5% (the two architectures are isomorphic here, so they match ~exactly).
- Same dataset split, same MSE loss/metric, same eval batch size.
- Reproducibility: explicit seed support; also supports a small loop over seeds via `--seeds 0,1,2,...`.

Implementation notes:
- SN side directly reuses your `SprecherMultiLayerNetwork` and its internals for φ/Φ and batch norm.
- We turn OFF features not allowed by the contract: lateral mixing, codomain-training, advanced scheduler.
- We set SN residual style to 'linear' (projection) to mirror the KAN projection residuals exactly.
- The KAN block is implemented locally to avoid touching model internals: same high-level dataflow as SN,
  but with cubic Catmull–Rom splines (uniform knots, linear extension outside the domain). Otherwise, it
  mirrors SN’s λ vector, η shift, Φ outer spline, residual add, batch norm placement, output scale/bias.
- We use your repo’s conventions for device/seed and reporting.  (All design choices trace to repository
  APIs in sn_core.* and comments in the code.)  [Reference: combined codebase.]  # See repo files.
"""

import os
import json
import time
import math
import argparse
from typing import List, Tuple, Optional

import torch
import torch.nn as nn
import numpy as np

# === Bring in the SN core (we do not modify internals) ===
from sn_core import CONFIG  # global runtime config used by SN
from sn_core.model import SprecherMultiLayerNetwork  # SN model
# We will not use sn_core.train.train_network because we must freeze domains after warm-up.
# Instead, we implement a tiny, symmetric training loop for both SN and KAN.
# (This respects the contract and avoids touching internals.)  :contentReference[oaicite:1]{index=1}


# -----------------------------
# Utility: deterministic seeding
# -----------------------------
def set_seed(seed: int):
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


# ---------------------------------------------------------
# TriWaveMix dataset: 10D, rich in natural PWL "triangle" kinks
# ---------------------------------------------------------
class TriWaveMix10D:
    """
    f(x) = Σ_i a_i * tri(f_i * x_i + φ_i) + α * tri( β * (b^T x) + φ* )
    where tri(u) = 2*|frac(u) - 0.5|  ∈ [0,1], a_i, b, phases are fixed at construction,  x ∈ [0,1]^10.

    Properties:
      - Many kinks along each marginal and along a sparse linear mix — piecewise linear but not separable.
      - Natural in signal processing (triangle carriers), not bespoke to Sprecher constructions.
    """
    def __init__(self, input_dim: int = 10, output_dim: int = 1, dataset_seed: int = 2025):
        assert input_dim == 10 and output_dim == 1
        self.input_dim = input_dim
        self.output_dim = output_dim

        rng = np.random.RandomState(dataset_seed)
        # Frequencies: mix of small primes → non-commensurate kinks
        freqs = rng.choice([3, 5, 7, 11, 13], size=input_dim, replace=True)
        phases = rng.uniform(0.0, 1.0, size=input_dim)
        amps = rng.uniform(0.6, 1.4, size=input_dim)

        # Sparse mixing term
        b = rng.randn(input_dim)
        b /= np.linalg.norm(b) + 1e-9
        alpha = 0.8
        beta = 3.0
        phase_star = rng.uniform(0.0, 1.0)

        self.freqs = torch.tensor(freqs, dtype=torch.float32)
        self.phases = torch.tensor(phases, dtype=torch.float32)
        self.amps = torch.tensor(amps, dtype=torch.float32)
        self.b = torch.tensor(b, dtype=torch.float32)
        self.alpha = float(alpha)
        self.beta = float(beta)
        self.phase_star = float(phase_star)

        # Rough amplitude cap for simple normalization
        self.norm_den = (self.amps.sum().item() + self.alpha)

    @property
    def name(self):
        return "TriWaveMix10D"

    def _tri(self, u: torch.Tensor) -> torch.Tensor:
        # tri(u) in [0,1] using fractional part
        frac = u - torch.floor(u)
        return 2.0 * torch.abs(frac - 0.5)

    def sample(self, n: int, device="cpu") -> Tuple[torch.Tensor, torch.Tensor]:
        x = torch.rand(n, self.input_dim, device=device)
        y = self.evaluate(x)
        return x, y

    def evaluate(self, x: torch.Tensor) -> torch.Tensor:
        device = x.device
        freqs = self.freqs.to(device)
        phases = self.phases.to(device)
        amps = self.amps.to(device)
        b = self.b.to(device)

        # Per-dimension triangular waves
        # shape(x) = [B, 10]
        term_marginals = (amps * self._tri(freqs * (x + phases))).sum(dim=1, keepdim=True)

        # Sparse mixed term (adds cross-dim kinks in a linear subspace)
        mix = torch.matmul(x, b)
        term_mix = self.alpha * self._tri(self.beta * (mix + self.phase_star))

        y = (term_marginals + term_mix.unsqueeze(1)) / (self.norm_den + 1e-12)
        return y


# ---------------------------------------------------------
# Cubic Catmull–Rom 1D spline (uniform knots, linear ext.)
# ---------------------------------------------------------
class CubicCRSpline1D(nn.Module):
    """
    C^1 cubic Catmull–Rom spline with uniform knots on [in_min, in_max].
    Parameters are the values at each knot (shape: [K]). We use linear
    extension outside the domain (matching SN's linear extension idea).
    """
    def __init__(self, num_knots: int = 64, in_range: Tuple[float, float] = (0.0, 1.0)):
        super().__init__()
        assert num_knots >= 4, "Need at least 4 knots for Catmull–Rom."
        self.num_knots = int(num_knots)
        self.in_min, self.in_max = float(in_range[0]), float(in_range[1])
        # Fixed, uniform knots in a buffer
        self.register_buffer("knots", torch.linspace(self.in_min, self.in_max, self.num_knots))
        # Trainable values at knots
        self.coeffs = nn.Parameter(torch.zeros(self.num_knots))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.to(self.knots.device)
        # Clamp idx into [0, K-2] for segment id; we need neighbors up to k+2 for CR
        # We'll pad coeffs at ends by repeating end values to simplify boundary access.
        K = self.num_knots
        knots = self.knots
        y = self.coeffs

        # Find segment indices
        x_clamped = torch.clamp(x, self.in_min, self.in_max)
        idx = torch.searchsorted(knots, x_clamped) - 1
        idx = torch.clamp(idx, 0, K - 2)

        # Local t in [0,1]
        denom = knots[idx + 1] - knots[idx]
        t = torch.where(denom > 1e-12, (x_clamped - knots[idx]) / denom, torch.zeros_like(x_clamped))

        # Gather neighbor control points p_{k-1}, p_k, p_{k+1}, p_{k+2}
        # Boundary: replicate end values
        idx_im1 = torch.clamp(idx - 1, 0, K - 1)
        idx_ip1 = torch.clamp(idx + 1, 0, K - 1)
        idx_ip2 = torch.clamp(idx + 2, 0, K - 1)

        p0 = y[idx_im1]
        p1 = y[idx]
        p2 = y[idx_ip1]
        p3 = y[idx_ip2]

        # Catmull–Rom basis (centripetal-like default with τ=0.5 implicit, classic form):
        # S(t) = 0.5 * [ (2 p1) + (-p0 + p2) t + (2p0 - 5p1 + 4p2 - p3) t^2 + (-p0 + 3p1 - 3p2 + p3) t^3 ]
        t2 = t * t
        t3 = t2 * t
        s = 0.5 * (
            (2.0 * p1)
            + (-p0 + p2) * t
            + (2.0 * p0 - 5.0 * p1 + 4.0 * p2 - p3) * t2
            + (-p0 + 3.0 * p1 - 3.0 * p2 + p3) * t3
        )

        # Linear extension outside [in_min, in_max]
        below = x < self.in_min
        above = x > self.in_max
        # Left slope ~ (y1 - y0) / Δx, right slope ~ (y_{K-1} - y_{K-2}) / Δx
        dx = knots[1] - knots[0]
        left_slope = (y[1] - y[0]) / (dx + 1e-12)
        right_slope = (y[-1] - y[-2]) / (dx + 1e-12)
        s = torch.where(
            below, y[0] + left_slope * (x - self.in_min),
            torch.where(above, y[-1] + right_slope * (x - self.in_max), s)
        )
        return s


# ---------------------------------------------------------
# KAN-style block/network: Same topology as SN, but cubic splines
# ---------------------------------------------------------
Q_VALUES_FACTOR = 1.0  # mirror SN's constant

class KANBlock(nn.Module):
    """
    Matches SN's block-level dataflow but with cubic splines:
      s_q = sum_i λ_i * φ_cubic( x_i + η q ) + Q_VALUES_FACTOR * q
      y_q = Φ_cubic( s_q )
      (then add linear/projection residual against x_original)
      If is_final: sum across q → scalar.

    - φ_cubic, Φ_cubic are SHARED across outputs (like in SN).
    - Residual: linear/projection only (same contract as SN).
    - No domain updates; knots are fixed on [0,1].
    """
    def __init__(self, d_in, d_out, is_final=False, phi_knots=64, Phi_knots=64):
        super().__init__()
        self.d_in = d_in
        self.d_out = d_out
        self.is_final = is_final

        self.phi = CubicCRSpline1D(num_knots=phi_knots, in_range=(0.0, 1.0))
        self.Phi = CubicCRSpline1D(num_knots=Phi_knots, in_range=(0.0, 1.0))
        self.lambdas = nn.Parameter(torch.randn(d_in) * math.sqrt(2.0 / max(1, d_in)))
        self.eta = nn.Parameter(torch.tensor(1.0 / (d_out + 10.0)))
        self.register_buffer("q_values", torch.arange(d_out, dtype=torch.float32))

        # Residuals: linear/projection only (no node-centric)
        self.residual_weight = None
        self.residual_projection = None
        if d_in == d_out:
            self.residual_weight = nn.Parameter(torch.tensor(0.1))
        else:
            self.residual_projection = nn.Parameter(torch.empty(d_in, d_out))
            nn.init.xavier_uniform_(self.residual_projection)

    def forward(self, x, x_original=None):
        B = x.shape[0]
        device = x.device

        # Compute s sequentially over q to be memory-friendly
        s = torch.zeros(B, self.d_out, device=device, dtype=x.dtype)
        for q_idx in range(self.d_out):
            q_val = float(q_idx)
            shifted = x + self.eta * q_val  # (B, d_in)
            phi_out = self.phi(shifted)     # (B, d_in)
            weighted = phi_out * self.lambdas  # (B, d_in)
            s[:, q_idx] = weighted.sum(dim=1) + Q_VALUES_FACTOR * q_val

        y = self.Phi(s)  # element-wise cubic

        # Residual add
        if x_original is not None:
            if self.residual_projection is not None:
                y = y + torch.matmul(x_original, self.residual_projection)
            elif self.residual_weight is not None:
                # same-dim scalar residual
                y = y + self.residual_weight * x_original

        if self.is_final:
            return y.sum(dim=1, keepdim=True)
        return y


class KANNetwork(nn.Module):
    """
    Same meta-structure as SprecherMultiLayerNetwork:
      - sequence of blocks
      - BatchNorm 'after' each block (skip_first=True)
      - output scale/bias
    """
    def __init__(self, input_dim, architecture: List[int], final_dim=1,
                 phi_knots=64, Phi_knots=64, norm_type="batch",
                 norm_position="after", norm_skip_first=True):
        super().__init__()
        self.input_dim = input_dim
        self.architecture = architecture
        self.final_dim = final_dim
        self.norm_type = norm_type
        self.norm_position = norm_position
        self.norm_skip_first = norm_skip_first

        layers = []
        if not architecture:
            is_final = (final_dim == 1)
            layers.append(KANBlock(input_dim, final_dim, is_final=is_final,
                                   phi_knots=phi_knots, Phi_knots=Phi_knots))
        else:
            d_in = input_dim
            for i, d_out in enumerate(architecture):
                is_final_block = (i == len(architecture) - 1) and (final_dim == 1)
                layers.append(KANBlock(d_in, d_out, is_final=is_final_block,
                                       phi_knots=phi_knots, Phi_knots=Phi_knots))
                d_in = d_out
            if final_dim > 1:
                layers.append(KANBlock(d_in, final_dim, is_final=False,
                                       phi_knots=phi_knots, Phi_knots=Phi_knots))
        self.layers = nn.ModuleList(layers)

        # BatchNorm after (skip first) to mirror SN default
        self.norm_layers = nn.ModuleList()
        if norm_type == "batch":
            for i, block in enumerate(self.layers):
                if norm_skip_first and i == 0:
                    self.norm_layers.append(nn.Identity())
                else:
                    num_features = 1 if block.is_final else block.d_out
                    self.norm_layers.append(nn.BatchNorm1d(num_features))
        elif norm_type == "none":
            self.norm_layers = nn.ModuleList([nn.Identity() for _ in self.layers])
        else:
            raise ValueError("Only 'batch' or 'none' norms are supported by this KAN benchmark.")

        self.output_scale = nn.Parameter(torch.tensor(1.0))
        self.output_bias = nn.Parameter(torch.tensor(0.0))

    def forward(self, x):
        x_in = x
        for i, block in enumerate(self.layers):
            # (Norm 'before' not supported; we match SN default 'after')
            y = block(x_in, x_in)  # pass x_in for residuals
            y = self.norm_layers[i](y)
            x_in = y
        out = self.output_scale * x_in + self.output_bias
        return out


# ---------------------------------------------------------
# Count params utility (trainable only)
# ---------------------------------------------------------
def count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# ---------------------------------------------------------
# Training loop (shared by SN and KAN)
# ---------------------------------------------------------
def train_model(
    model: nn.Module,
    x_train: torch.Tensor,
    y_train: torch.Tensor,
    x_eval: torch.Tensor,
    y_eval: torch.Tensor,
    epochs: int,
    lr: float,
    weight_decay: float,
    max_grad_norm: float = 1.0,
    warmup_domain_updates: int = 0,  # only used for SN
    is_sn: bool = False
):
    device = x_train.device
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    mse = nn.MSELoss()

    best = {"loss": float("inf"), "state": None, "epoch": -1, "eval_mse": None}
    losses = []

    t0 = time.time()
    for epoch in range(epochs):
        optimizer.zero_grad()

        # SN-only: domain updates during warm-up; frozen after.
        if is_sn and epoch < warmup_domain_updates:
            # Update all spline domains based on theoretical ranges, as in SN internals
            # (This mirrors the behavior used by SN training utilities.)
            model.update_all_domains(allow_resampling=True, force_resample=False)

        y_pred = model(x_train)
        loss = mse(y_pred, y_train)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
        optimizer.step()
        losses.append(float(loss.item()))

        # Track the best training loss; snapshot lightweight state_dict
        if loss.item() < best["loss"]:
            best["loss"] = float(loss.item())
            best["state"] = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            best["epoch"] = epoch

    # Evaluate final and best on held-out
    model.eval()
    with torch.no_grad():
        final_eval = nn.MSELoss()(model(x_eval), y_eval).item()

    # Load best state for best_eval
    best_eval = None
    if best["state"] is not None:
        model.load_state_dict(best["state"], strict=True)
        model.eval()
        with torch.no_grad():
            best_eval = nn.MSELoss()(model(x_eval), y_eval).item()

    dt = time.time() - t0

    return {
        "train_loss_curve": losses,
        "train_best_loss": best["loss"],
        "train_best_epoch": best["epoch"],
        "eval_mse_final": final_eval,
        "eval_mse_best": best_eval if best_eval is not None else final_eval,
        "runtime_seconds": dt,
    }


# ---------------------------------------------------------
# CLI and runner
# ---------------------------------------------------------
def parse_args():
    p = argparse.ArgumentParser(description="SN vs KAN on TriWaveMix10D (matched, fair benchmark)")
    p.add_argument("--epochs", type=int, default=4000, help="Total training epochs (exactly 4000 per contract).")
    p.add_argument("--seed", type=int, default=0, help="Single run seed.")
    p.add_argument("--seeds", type=str, default=None,
                   help="Comma-separated list of seeds to run as a loop (overrides --seed).")
    p.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"], help="Device.")
    p.add_argument("--arch", type=str, default="64,64", help="Hidden dims, comma-separated (e.g., '64,64').")
    p.add_argument("--phi_knots", type=int, default=64, help="Number of knots for φ splines.")
    p.add_argument("--Phi_knots", type=int, default=64, help="Number of knots for Φ splines.")
    p.add_argument("--warmup", type=int, default=400, choices=[300, 400],
                   help="SN domain-update warm-up epochs, then freeze.")
    p.add_argument("--lr", type=float, default=3e-4, help="Adam learning rate (no advanced scheduler).")
    p.add_argument("--wd", type=float, default=1e-7, help="Weight decay (identical for both models).")
    p.add_argument("--dataset_seed", type=int, default=2025, help="Fix dataset parameters independent of run seed.")
    p.add_argument("--results_dir", type=str, default="results",
                   help="Directory where JSON results are written.")
    return p.parse_args()


def parse_arch(arch_str: str) -> List[int]:
    parts = [s.strip() for s in arch_str.split(",") if s.strip()]
    return [int(x) for x in parts] if parts else []


def run_one_seed(args, seed: int):
    # Device
    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    elif args.device == "cuda" and not torch.cuda.is_available():
        print("WARNING: CUDA requested but unavailable; using CPU.")
        device = "cpu"
    else:
        device = args.device

    # Set seeds
    set_seed(seed)

    # ---- Configure SN globals for fairness ----
    # Only BatchNorm, no lateral mixing, no codomain training, linear/projection residuals.
    CONFIG["use_normalization"] = True
    CONFIG["norm_type"] = "batch"
    CONFIG["norm_position"] = "after"
    CONFIG["norm_skip_first"] = True
    CONFIG["use_lateral_mixing"] = False
    CONFIG["train_phi_codomain"] = False
    CONFIG["use_advanced_scheduler"] = False
    CONFIG["use_residual_weights"] = True
    CONFIG["residual_style"] = "linear"  # projection residuals only
    CONFIG["domain_safety_margin"] = 0.0

    # Dataset + splits (fixed dataset parameters across runs; splits depend on run seed)
    dataset = TriWaveMix10D(input_dim=10, output_dim=1, dataset_seed=args.dataset_seed)

    # Match repo's convention for train batch size: 32 if 1D else 32*32 for >1D
    n_train = 32 if dataset.input_dim == 1 else 32 * 32
    n_eval = 8192  # reasonably large eval set; same for both
    x_train, y_train = dataset.sample(n_train, device=device)
    x_eval, y_eval = dataset.sample(n_eval, device=device)

    # Instantiate SN (from repo) and KAN (local) with identical topology
    architecture = parse_arch(args.arch)
    sn = SprecherMultiLayerNetwork(
        input_dim=dataset.input_dim, architecture=architecture, final_dim=dataset.output_dim,
        phi_knots=args.phi_knots, Phi_knots=args.Phi_knots,
        norm_type="batch", norm_position="after", norm_skip_first=True
    ).to(device)

    kan = KANNetwork(
        input_dim=dataset.input_dim, architecture=architecture, final_dim=dataset.output_dim,
        phi_knots=args.phi_knots, Phi_knots=args.Phi_knots,
        norm_type="batch", norm_position="after", norm_skip_first=True
    ).to(device)

    # Initialize output bias to target mean; small initial scale, identically for both
    with torch.no_grad():
        y_mean = y_train.mean()
        sn.output_bias.data = y_mean.clone()
        sn.output_scale.data = torch.tensor(0.1, device=device)
        kan.output_bias.data = y_mean.clone()
        kan.output_scale.data = torch.tensor(0.1, device=device)

    # Check param counts (should be virtually identical)
    sn_params = count_params(sn)
    kan_params = count_params(kan)
    diff = abs(sn_params - kan_params) / max(1, kan_params)
    if diff > 0.05:
        print(f"WARNING: Parameter counts differ by >5% (SN={sn_params}, KAN={kan_params}).")
    else:
        print(f"Param counts: SN={sn_params}, KAN={kan_params} (Δ={diff*100:.2f}%).")

    # Train both under identical budgets
    print(f"[Seed {seed}] Training SN...")
    sn_results = train_model(
        sn, x_train, y_train, x_eval, y_eval,
        epochs=args.epochs, lr=args.lr, weight_decay=args.wd,
        warmup_domain_updates=args.warmup, is_sn=True
    )

    print(f"[Seed {seed}] Training KAN...")
    kan_results = train_model(
        kan, x_train, y_train, x_eval, y_eval,
        epochs=args.epochs, lr=args.lr, weight_decay=args.wd,
        warmup_domain_updates=0, is_sn=False
    )

    # Assemble JSON results
    out = {
        "benchmark": "TriWaveMix10D",
        "arch": architecture,
        "phi_knots": args.phi_knots,
        "Phi_knots": args.Phi_knots,
        "epochs": args.epochs,
        "warmup_epochs_sn": args.warmup,
        "optimizer": "Adam",
        "lr": args.lr,
        "weight_decay": args.wd,
        "norm": {"type": "batch", "position": "after", "skip_first": True},
        "residuals": "linear/projection",
        "lateral_mixing": False,
        "dataset_seed": args.dataset_seed,
        "run_seed": seed,
        "device": device,
        "param_count": {"SN": sn_params, "KAN": kan_params},
        "SN": sn_results,
        "KAN": kan_results,
    }

    os.makedirs(args.results_dir, exist_ok=True)
    out_path = os.path.join(args.results_dir, f"trimix_sn_vs_kan_seed{seed}.json")
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"[Seed {seed}] Wrote: {out_path}")

    return out_path


def main():
    args = parse_args()

    # Brief justification for 400-epoch warm-up:
    # 400/4000 = 10% of the budget; long enough for φ/Φ domain ranges to stabilize
    # without letting ongoing domain drift confound the comparison after freezing.
    # If 300 is preferred, pass --warmup 300.

    if args.seeds:
        seeds = [int(s.strip()) for s in args.seeds.split(",") if s.strip()]
    else:
        seeds = [args.seed]

    outputs = []
    for s in seeds:
        outputs.append(run_one_seed(args, s))

    # Print a tiny manifest for convenience
    print("\nCompleted runs:")
    for p in outputs:
        print("  -", p)


if __name__ == "__main__":
    main()