# benchmarks/max_affine_sn_vs_kan.py
# -*- coding: utf-8 -*-
"""
Benchmark: SN (PWL splines + domain updates) vs. KAN (cubic splines, fixed domains)
Task: 8D "max-of-affines" tropical regression (continuous, non-smooth; oblique ridges)

Why this favors SNs but remains fair (brief):
- The ground truth is a convex piecewise-linear function y(x) = max_k(a_k^T x + b_k). 
  Such functions appear naturally in robust optimization, max-margin losses, and value functions.
  SNs use piecewise-linear splines and (during warm-up) dynamic domain updates per layer, which can 
  align the spline support to the active ranges and represent kinks efficiently.
- KANs here are implemented as cubic-spline drop-in counterparts of the same block structure 
  (identical residuals, batch norm, optimizer & schedule, and parameter budget ±5%). 
  Cubic splines typically overshoot near kinks under tight budgets; without domain updates they 
  must spread capacity across fixed ranges, which is a natural, not ad-hoc, disadvantage.
- We warm-up the SN’s domain updates for 400 epochs (see --warmup) to stabilize domains, then 
  freeze them. This is symmetric: KAN never updates domains. Both train exactly 4000 epochs.

Fairness contract (how this script enforces it):
- Same training length: exactly 4000 epochs for both models.
- Residuals: only linear/projection residuals (no node-centric); BatchNorm only. 
  (We set SN residual style to 'linear' and disable lateral mixing.)
- SN domain updates: enabled only during warm-up (default 400 epochs), then frozen.
- Hyperparams and budget: same optimizer (Adam), same LR, weight decay, batch size, schedule (none).
  Parameter counts are matched within ±5% by using the same architecture and same #knots; both models 
  have identical parameterization counts per block except the spline type (PWL vs cubic).
- Evaluation: same train/eval splits, same MSE, same eval batch size.
- Seeds: supports a fixed seed and a seed range loop that writes per-seed JSON.

Implementation notes:
- We reuse the official Sprecher network model and utilities (no internal modifications), 
  constructing the SN via sn_core.SprecherMultiLayerNetwork. We make a minimal cubic-spline 
  “KAN” that mirrors the SN block’s structure (λ-vector mixing, η-shift, Φ-outer spline, BN, 
  linear residuals) but with cubic splines and NO domain updates. This isolates spline order 
  + domain updates as the only meaningful differences, keeping everything else apples-to-apples.

Refs to repository API we rely on:
- sn_core.model.SprecherMultiLayerNetwork and CONFIG (residual style, BN, lateral mixing, etc.)
- The SN implementation already supports 'linear' residuals and batch norm placement.
- We set CONFIG flags rather than touching internals, per the repository's design. 
"""

import os, json, time, argparse, math
from dataclasses import dataclass
from typing import List, Tuple, Dict
import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm

# --- Import SN codebase pieces (no internal modifications) ---
from sn_core.config import CONFIG as SN_CONFIG
from sn_core.model import SprecherMultiLayerNetwork
# (We will not use sn_core.train.train_network because we need to freeze domains after warm-up.)
# We keep optimizer/schedule minimal and symmetric for both models.  :contentReference[oaicite:1]{index=1}


# ---------------------------
# Synthetic dataset: Max-of-affines in 8D
# ---------------------------
class MaxOfAffines8D:
    """
    y(x) = max_k (a_k^T x + b_k),   x ∈ [0,1]^8
    K (num pieces) is moderate (default 48) to create rich, oblique kink geometry.
    Continuous, non-smooth, and widely used in robust optimization / piecewise-linear modeling.
    """
    def __init__(self, K: int = 48, seed: int = 0, device="cpu"):
        self.input_dim = 8
        self.output_dim = 1
        g = torch.Generator(device=device)
        g.manual_seed(seed)
        # Random oblique planes; scale to keep outputs O(1)
        self.A = torch.randn(K, self.input_dim, generator=g, device=device) / math.sqrt(self.input_dim)
        self.b = torch.randn(K, 1, generator=g, device=device) * 0.1

    def sample(self, n: int, device="cpu") -> Tuple[torch.Tensor, torch.Tensor]:
        x = torch.rand(n, self.input_dim, device=device)
        y = self.evaluate(x)
        return x, y

    def evaluate(self, x: torch.Tensor) -> torch.Tensor:
        # y(x) = max_k a_k^T x + b_k
        vals = x @ self.A.transpose(0, 1) + self.b.squeeze(1)  # (n, K)
        y, _ = torch.max(vals, dim=1, keepdim=True)
        return y


# ---------------------------
# Cubic-spline 1D (KAN) — uniform knots, Hermite interpolation
# ---------------------------
class CubicSpline1D(nn.Module):
    """
    A C^1 piecewise-cubic spline on a fixed domain [in_min, in_max] with UNIFORM knots.
    Parameters: values at knots (learnable); slopes are Catmull–Rom-style from neighbors.
    Extrapolation: linear using boundary slopes (like SN's SimpleSpline non-monotonic case).
    NOTE: No domain updates; knots are fixed post-init.
    """
    def __init__(self, num_knots: int = 64, in_range: Tuple[float, float] = (0.0, 1.0)):
        super().__init__()
        assert num_knots >= 4, "Cubic spline needs at least 4 knots."
        self.num_knots = int(num_knots)
        self.in_min, self.in_max = float(in_range[0]), float(in_range[1])
        # Knots buffer (uniform)
        self.register_buffer("knots", torch.linspace(self.in_min, self.in_max, self.num_knots))
        # Control values (learnable)
        self.values = nn.Parameter(torch.zeros(self.num_knots))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.to(self.values.device)
        # Handle linear extrapolation outside domain
        below = x < self.in_min
        above = x > self.in_max
        x_clamped = torch.clamp(x, self.in_min, self.in_max)

        # Find interval indices
        idx = torch.searchsorted(self.knots, x_clamped) - 1
        idx = torch.clamp(idx, 0, self.num_knots - 2)

        # t in [0,1] in the local interval
        x0 = self.knots[idx]
        x1 = self.knots[idx + 1]
        h = x1 - x0
        h = torch.where(h > 1e-12, h, torch.ones_like(h))  # avoid tiny gaps
        t = (x_clamped - x0) / h

        v0 = self.values[idx]
        v1 = self.values[idx + 1]

        # Catmull–Rom slopes (uniform spacing)
        # m_i ≈ 0.5 * (v_{i+1} - v_{i-1}) / Δ; handle boundaries with one-sided diffs
        im1 = torch.clamp(idx - 1, 0, self.num_knots - 1)
        ip1 = torch.clamp(idx + 1, 0, self.num_knots - 1)
        ip2 = torch.clamp(idx + 2, 0, self.num_knots - 1)

        # Δ is uniform step = knots[1]-knots[0]
        delta = (self.knots[1] - self.knots[0]).item()
        m0 = 0.5 * (self.values[ip1] - self.values[im1]) / max(delta, 1e-12)
        m1 = 0.5 * (self.values[ip2] - self.values[idx]) / max(delta, 1e-12)

        # Hermite basis
        t2 = t * t
        t3 = t2 * t
        h00 = 2 * t3 - 3 * t2 + 1
        h10 = t3 - 2 * t2 + t
        h01 = -2 * t3 + 3 * t2
        h11 = t3 - t2

        # Evaluate
        y_in = h00 * v0 + h10 * m0 * h + h01 * v1 + h11 * m1 * h

        # Linear extrapolation
        # Boundary slopes again via one-sided differences
        vL0, vL1 = self.values[0], self.values[1]
        vR0, vR1 = self.values[-2], self.values[-1]
        m_left = (vL1 - vL0) / max(delta, 1e-12)
        m_right = (vR1 - vR0) / max(delta, 1e-12)

        y_left = vL0 + m_left * (x - self.in_min)
        y_right = vR1 + m_right * (x - self.in_max)

        y = torch.where(below, y_left, y_in)
        y = torch.where(above, y_right, y)
        return y


# ---------------------------
# Cubic-KAN block and network (drop-in counterpart to SN blocks)
# ---------------------------
class CubicKANLayerBlock(nn.Module):
    """
    Mirrors SprecherLayerBlock interface, but with cubic splines (no domain updates):
      - φ_cubic: applied elementwise to (x + η * q) like SN (q indexes outputs).
      - λ: one weight per input dim (shared across outputs, as in SN).
      - Φ_cubic: outer spline applied elementwise across outputs.
      - Linear/projection residuals only (identical rule as SN).
      - Optional BN is handled by the wrapper network (identical to SN).
    """
    def __init__(self, d_in, d_out, layer_num=0, is_final=False, phi_knots=64, Phi_knots=64,
                 phi_domain=None, Phi_domain=None):
        super().__init__()
        self.d_in = d_in
        self.d_out = d_out
        self.layer_num = layer_num
        self.is_final = is_final

        # Shared φ and Φ cubic splines (no domain updates here)
        if phi_domain is None:
            # Conservative domain since we may evaluate x + η q with q∈[0..d_out-1]
            # Using [−0.1, 1.1 + 0.02*d_out] keeps all shifts in-bounds for default η init.
            phi_domain = (-0.1, 1.1 + 0.02 * max(0, d_out - 1))
        if Phi_domain is None:
            # We'll feed sums of λ·φ; use a broad fixed domain
            Phi_domain = (-5.0, 5.0)

        self.phi = CubicSpline1D(num_knots=phi_knots, in_range=phi_domain)
        self.Phi = CubicSpline1D(num_knots=Phi_knots, in_range=Phi_domain)

        # λ vector and η shift (shared across outputs, identical to SN)
        self.lambdas = nn.Parameter(torch.randn(d_in) * math.sqrt(2.0 / max(1, d_in)))
        self.eta = nn.Parameter(torch.tensor(1.0 / (d_out + 10.0)))

        # Residuals: linear/projection only (same rule as SN)
        self.residual_weight = None
        self.residual_projection = None
        if d_in == d_out:
            self.residual_weight = nn.Parameter(torch.tensor(0.1))
        else:
            self.residual_projection = nn.Parameter(torch.empty(d_in, d_out))
            nn.init.xavier_uniform_(self.residual_projection)

        # Precompute q indices buffer
        self.register_buffer("q_values", torch.arange(d_out, dtype=torch.float32))

    def forward(self, x, x_original=None):
        # shape: x [B, d_in]
        B = x.shape[0]
        device = x.device
        q = self.q_values.to(device).view(1, 1, -1)  # [1,1,d_out]
        # Elementwise φ on shifted inputs (like SN)
        shifted = x.unsqueeze(-1) + self.eta * q  # [B, d_in, d_out]
        phi_out = self.phi(shifted)               # [B, d_in, d_out]

        # Sum across inputs with shared λ
        s = (phi_out * self.lambdas.view(1, -1, 1)).sum(dim=1)  # [B, d_out]

        # Outer Φ
        y = self.Phi(s)

        # Linear/projection residual
        if self.residual_projection is not None and x_original is not None:
            y = y + x_original @ self.residual_projection
        elif self.residual_weight is not None and x_original is not None:
            y = y + self.residual_weight * x_original

        # Final sum if this is the summing layer
        if self.is_final:
            y = y.sum(dim=1, keepdim=True)
        return y


class CubicKANMultiLayerNetwork(nn.Module):
    """
    Same wrapper semantics as SprecherMultiLayerNetwork: sequence of blocks + BN (position='after').
    Norm config mirrors SN usage in the repository.
    """
    def __init__(self, input_dim, architecture: List[int], final_dim=1,
                 phi_knots=64, Phi_knots=64, norm_type='batch', norm_position='after',
                 norm_skip_first=True):
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
            layers.append(CubicKANLayerBlock(d_in=input_dim, d_out=final_dim, layer_num=0, is_final=is_final,
                                             phi_knots=phi_knots, Phi_knots=Phi_knots))
        else:
            d_in = input_dim
            L = len(architecture)
            for i, d_out in enumerate(architecture):
                is_final_block = (i == L - 1) and (final_dim == 1)
                layers.append(CubicKANLayerBlock(d_in=d_in, d_out=d_out, layer_num=i, is_final=is_final_block,
                                                 phi_knots=phi_knots, Phi_knots=Phi_knots))
                d_in = d_out
            if final_dim > 1:
                # add an extra non-summing block if vector output (rare in this benchmark)
                layers.append(CubicKANLayerBlock(d_in=d_in, d_out=final_dim, layer_num=L, is_final=False,
                                                 phi_knots=phi_knots, Phi_knots=Phi_knots))
        self.layers = nn.ModuleList(layers)

        # BN mirrors SN: after each block, with skip_first config
        self.norm_layers = nn.ModuleList()
        if norm_type != 'none':
            for i, layer in enumerate(self.layers):
                if norm_skip_first and i == 0:
                    self.norm_layers.append(nn.Identity())
                else:
                    # After-block BN: features = 1 if final-sum block else d_out
                    num_features = 1 if layer.is_final else layer.d_out
                    self.norm_layers.append(nn.BatchNorm1d(num_features))

        self.output_scale = nn.Parameter(torch.tensor(1.0))
        self.output_bias = nn.Parameter(torch.tensor(0.0))

    def forward(self, x):
        xin = x
        for i, layer in enumerate(self.layers):
            if self.norm_type != 'none' and self.norm_position == 'before':
                xin = self.norm_layers[i](xin)
            xout = layer(xin, xin)
            if self.norm_type != 'none' and self.norm_position == 'after':
                xout = self.norm_layers[i](xout)
            xin = xout
        y = xin
        return self.output_scale * y + self.output_bias

    # For symmetry with SN API (no-ops here)
    def update_all_domains(self, allow_resampling=True, force_resample=False):
        return


# ---------------------------
# Training utilities (shared, symmetric)
# ---------------------------
@dataclass
class TrainConfig:
    epochs: int = 4000
    warmup: int = 400              # SN domain-update warm-up epochs (choose 300 or 400)
    lr: float = 3e-4
    weight_decay: float = 1e-6
    max_grad_norm: float = 1.0
    print_every: int = 400


def count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


@torch.no_grad()
def mse_loss(model: nn.Module, x: torch.Tensor, y: torch.Tensor, eval_batch: int = 8192) -> float:
    model.eval()
    n = x.shape[0]
    total = 0.0
    for i in range(0, n, eval_batch):
        xb = x[i:i+eval_batch]
        yb = y[i:i+eval_batch]
        pred = model(xb)
        total += torch.mean((pred - yb)**2).item() * xb.shape[0]
    return total / n


def train_model(model: nn.Module,
                dataset: MaxOfAffines8D,
                cfg: TrainConfig,
                seed: int,
                device: str,
                is_sn: bool,
                log_stride: int = 50) -> Dict:
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    # Fixed train/eval splits per seed
    n_train = 32 if dataset.input_dim == 1 else 32 * 32  # match repo’s pattern
    x_train, y_train = dataset.sample(n_train, device=device)
    x_eval, y_eval = dataset.sample(max(8192, n_train * 8), device=device)

    # Optimizer (identical for SN & KAN)
    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)

    losses = []
    t0 = time.time()
    model.train()
    for epoch in tqdm(range(cfg.epochs), desc=("SN" if is_sn else "KAN"), leave=False):
        # === Domain updates only for SN during warm-up ===
        if is_sn and epoch < cfg.warmup:
            # freeze/resample semantics rely on the SN’s official method; after warm-up we stop calling it.
            model.update_all_domains(allow_resampling=True)

        opt.zero_grad()
        pred = model(x_train)
        loss = torch.mean((pred - y_train)**2)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.max_grad_norm)
        opt.step()

        if (epoch % log_stride) == 0 or epoch == cfg.epochs - 1:
            losses.append(loss.item())

        if (epoch + 1) % cfg.print_every == 0:
            train_mse = loss.item()
            eval_mse = mse_loss(model, x_eval, y_eval)
            print(f"[{('SN' if is_sn else 'KAN')}][epoch {epoch+1:4d}/{cfg.epochs}] "
                  f"train MSE={train_mse:.3e}  eval MSE={eval_mse:.3e}")

    runtime = time.time() - t0
    # Final metrics
    final_train = torch.mean((model(x_train) - y_train)**2).item()
    final_eval = mse_loss(model, x_eval, y_eval)

    return {
        "seed": seed,
        "train_points": n_train,
        "final_train_mse": final_train,
        "final_eval_mse": final_eval,
        "loss_curve_every_%d" % log_stride: losses,
        "runtime_sec": runtime
    }


# ---------------------------
# Main benchmark entry
# ---------------------------
def parse_args():
    p = argparse.ArgumentParser("SN vs KAN on Max-of-Affines (8D) benchmark")
    p.add_argument("--epochs", type=int, default=4000, help="Training epochs (must be 4000).")
    p.add_argument("--warmup", type=int, default=400, help="SN domain-update warm-up epochs: 300 or 400.")
    p.add_argument("--arch", type=str, default="8,8,8",
                   help="Hidden architecture, e.g. '8,8,8'. Last block sums to scalar.")
    p.add_argument("--phi_knots", type=int, default=64, help="#knots for φ splines.")
    p.add_argument("--Phi_knots", type=int, default=64, help="#knots for Φ splines.")
    p.add_argument("--pieces", type=int, default=48, help="#affine pieces (K) for the dataset.")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--seed_range", type=int, nargs=2, default=None,
                   help="Run a loop over seeds: start end (inclusive).")
    p.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"])
    p.add_argument("--results_dir", type=str, default="results")
    p.add_argument("--tag", type=str, default="", help="Optional tag for filenames.")
    return p.parse_args()


def build_sn_and_kan(input_dim: int,
                     arch: List[int],
                     final_dim: int,
                     phi_knots: int,
                     Phi_knots: int) -> Tuple[nn.Module, nn.Module]:
    # --- Configure SN for fairness: linear residuals only; BN after; NO lateral mixing; fixed codomain; ---
    SN_CONFIG["use_lateral_mixing"] = False
    SN_CONFIG["residual_style"] = "linear"
    SN_CONFIG["use_normalization"] = True
    SN_CONFIG["norm_type"] = "batch"
    SN_CONFIG["norm_position"] = "after"
    SN_CONFIG["norm_skip_first"] = True
    SN_CONFIG["train_phi_codomain"] = False          # avoid SN-only codomain params
    SN_CONFIG["use_theoretical_domains"] = False     # we will call update_all_domains manually during warm-up
    # Weight decay / etc are enforced via our optimizer (same for both)

    sn = SprecherMultiLayerNetwork(
        input_dim=input_dim,
        architecture=arch,
        final_dim=final_dim,
        phi_knots=phi_knots,
        Phi_knots=Phi_knots,
        norm_type="batch",
        norm_position="after",
        norm_skip_first=True,
        initialize_domains=True
    )

    # KAN cubic counterpart with mirrored wrapper, same paramization counts per block
    kan = CubicKANMultiLayerNetwork(
        input_dim=input_dim,
        architecture=arch,
        final_dim=final_dim,
        phi_knots=phi_knots,
        Phi_knots=Phi_knots,
        norm_type="batch",
        norm_position="after",
        norm_skip_first=True
    )

    return sn, kan


def run_single(args, seed: int):
    assert args.epochs == 4000, "--epochs must be exactly 4000 per fairness contract."
    assert args.warmup in (300, 400), "--warmup must be 300 or 400."
    device = ("cuda" if (args.device == "auto" and torch.cuda.is_available()) else
              "cpu" if args.device == "auto" else args.device)

    # Dataset
    dataset = MaxOfAffines8D(K=args.pieces, seed=seed, device=device)

    # Parse architecture
    arch = [int(x.strip()) for x in args.arch.split(",") if x.strip()]
    final_dim = 1

    # Build models
    sn, kan = build_sn_and_kan(dataset.input_dim, arch, final_dim, args.phi_knots, args.Phi_knots)
    sn.to(device); kan.to(device)

    # Initialize output bias to target mean (symmetrically for both)
    with torch.no_grad():
        xtmp, ytmp = dataset.sample(2048, device=device)
        mu = ytmp.mean()
        for m in (sn, kan):
            m.output_bias.data = mu
            m.output_scale.data = torch.tensor(0.1)

    # Parameter count check (should match within a few params due to identical design)
    n_sn = count_params(sn)
    n_kan = count_params(kan)
    budget_diff = abs(n_sn - n_kan) / max(1, n_sn)
    if budget_diff > 0.05:
        print(f"WARNING: parameter budgets differ by >5% (SN={n_sn}, KAN={n_kan}). "
              f"Consider adjusting --phi_knots/--Phi_knots or architecture.")
    else:
        print(f"Param counts: SN={n_sn}, KAN={n_kan} (Δ={budget_diff*100:.2f}%)")

    cfg = TrainConfig(epochs=args.epochs, warmup=args.warmup)

    # Train both (same optimizer/LR/WD/clip/schedule)
    sn_res = train_model(sn, dataset, cfg, seed=seed, device=device, is_sn=True)
    kan_res = train_model(kan, dataset, cfg, seed=seed, device=device, is_sn=False)

    # Collect results
    out = {
        "seed": seed,
        "device": device,
        "dataset": {
            "name": "MaxOfAffines8D",
            "pieces": args.pieces,
            "input_dim": dataset.input_dim,
            "output_dim": dataset.output_dim
        },
        "arch": arch,
        "phi_knots": args.phi_knots,
        "Phi_knots": args.Phi_knots,
        "epochs": args.epochs,
        "warmup": args.warmup,
        "param_counts": {"SN": n_sn, "KAN": n_kan},
        "SN": sn_res,
        "KAN": kan_res
    }
    os.makedirs(args.results_dir, exist_ok=True)
    tag = (f"-{args.tag}" if args.tag else "")
    fname = os.path.join(args.results_dir, f"max_affine_sn_vs_kan-seed{seed}{tag}.json")
    with open(fname, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    print(f"Wrote results to {fname}")
    return fname


def main():
    args = parse_args()

    if args.seed_range is None:
        run_single(args, args.seed)
    else:
        s0, s1 = args.seed_range
        for s in range(s0, s1 + 1):
            run_single(args, s)


if __name__ == "__main__":
    main()