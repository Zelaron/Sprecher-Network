# benchmarks/sn_vs_kan_saturating_hinge.py
# -*- coding: utf-8 -*-
"""
SN vs KAN benchmark on a new, realistic piecewise-linear-with-saturation task.

Why this task (short rationale):
- Many real systems (sensor pipelines, audio dynamics compressors, telemetry with clamping, ad bid modifiers,
  piecewise tariff/risk schedules) are well-modeled by **additive hinge + saturation** maps: sums of
  (x_i - τ_i)_+ with group-wise clipping or caps. These are **piecewise-linear with frequent kinks** and
  sparse, axis-aligned discontinuities—natural, not hand-crafted to “cheat”.
- Under a matched parameter budget, **SN’s piecewise-linear (PWL) splines with domain updates** can represent
  such kinks **exactly** with few knots and avoid cubic overshoot. **KAN’s cubic splines** are smooth and
  typically need more local degrees of freedom to fit sharp hinges without ringing; when budgets are matched,
  we expect KAN to approximate more slowly and underfit corner regions.
- Fairness: both models share the **same macro-architecture** (layers, widths, residual style), **same optimizer,
  schedule, batch size, weight decay**, **BatchNorm only**, **identical splits/seed**, and **parameter counts within ±5%**.
  The only architectural difference is the spline family (PWL vs cubic) and the **standard SN warm-up** (400 epochs)
  for data-driven domain updates; domains are then frozen to ensure a stationary target during the bulk of training.

Contract highlights implemented here:
- 4000 epochs for both models, MSE loss, identical train/val/test splits and eval batch.
- Only linear/projection residuals and BatchNorm for both; no lateral mixing, attention, or other extras.
- SN gets a **400-epoch domain-update warm-up**, then domains are frozen (no updates afterwards).
- Parameter counts automatically checked and kept within ±5%; the KAN uses the **same skeleton** as SN
  but replaces PWL with **cubic Hermite splines** (no domain updates).
"""

import os
import argparse
import math
from dataclasses import dataclass
from typing import List, Tuple

import torch
import torch.nn as nn
import numpy as np

# Use the repo's SN core utilities (model + config, not the training loop; we keep training symmetric).
from sn_core.config import CONFIG, Q_VALUES_FACTOR        # repo config (used carefully; we do not modify internals)
from sn_core.model import SprecherMultiLayerNetwork       # SN model (piecewise-linear with domain updates)
from sn_core.data import Dataset                          # base typing only (we provide a new dataset here)

# ---------------------------
# Utilities: seeds & parsing
# ---------------------------

def set_seed(seed: int):
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def parse_arch(arch_str: str) -> List[int]:
    parts = [p.strip() for p in arch_str.split(",") if p.strip()]
    return [int(p) for p in parts]


def select_device(device_arg: str) -> torch.device:
    if device_arg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_arg)


def count_trainable_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# ---------------------------------------------------------
# New dataset: Saturating Hinge Aggregation (SHA) generator
# ---------------------------------------------------------

class SaturatingHingeAggregation(Dataset):
    """
    Natural synthetic regression: grouped hinge-sums with piecewise-linear saturation.
    x ∈ [0,1]^d. Partition features into G groups. For each group g:

       z_g = ∑_{i in group g} w_i * ReLU(x_i - τ_i)
       h_g(z) = clip(a_g * z + b_g, L_g, U_g)

    Output y = ∑_g γ_g * h_g(z_g) + ε, with small Gaussian noise.

    This mimics common pipelines: thresholded sensors feeding into capped linear aggregators
    (e.g., telemetry counters with caps, compressor-like audio units, policy “tiers”).
    It is piecewise-linear with many kinks and saturations—favorable terrain for PWL splines.
    """
    def __init__(self, d=16, groups=4, seed=123, noise_std=0.01):
        self._d = int(d)
        self._groups = int(groups)
        self._noise_std = float(noise_std)
        assert self._d >= self._groups and self._groups >= 1
        rng = np.random.default_rng(seed)

        # Group assignment: contiguous blocks for reproducibility
        sizes = [self._d // self._groups] * self._groups
        for k in range(self._d % self._groups):
            sizes[k] += 1
        idx = 0
        self.group_indices = []
        for s in sizes:
            self.group_indices.append(np.arange(idx, idx + s, dtype=int))
            idx += s

        # Per-feature thresholds and weights
        self.tau = rng.uniform(0.2, 0.8, size=self._d).astype(np.float32)
        # Mix of positive/negative weights to avoid trivial monotonicity
        signs = rng.choice([-1.0, 1.0], size=self._d, replace=True).astype(np.float32)
        self.w = (signs * rng.uniform(0.5, 1.5, size=self._d)).astype(np.float32)

        # Group-wise linear + saturation parameters
        self.a = rng.uniform(0.7, 1.3, size=self._groups).astype(np.float32)
        self.b = rng.uniform(-0.25, 0.25, size=self._groups).astype(np.float32)
        # Saturation bounds per group; heterogeneous caps to create realistic corner cases
        self.L = rng.uniform(-0.75, -0.25, size=self._groups).astype(np.float32)
        self.U = rng.uniform( 0.25,  0.75, size=self._groups).astype(np.float32)

        # Group combination weights
        self.gamma = rng.uniform(0.7, 1.3, size=self._groups).astype(np.float32)

    @property
    def input_dim(self) -> int:
        return self._d

    @property
    def output_dim(self) -> int:
        return 1

    def evaluate(self, x: torch.Tensor) -> torch.Tensor:
        # x: [N, d] in [0,1]
        N, d = x.shape
        device = x.device
        tau = torch.as_tensor(self.tau, device=device)   # [d]
        w   = torch.as_tensor(self.w, device=device)     # [d]
        a   = torch.as_tensor(self.a, device=device)     # [G]
        b   = torch.as_tensor(self.b, device=device)     # [G]
        L   = torch.as_tensor(self.L, device=device)
        U   = torch.as_tensor(self.U, device=device)
        gamma = torch.as_tensor(self.gamma, device=device)

        # Hinge per feature
        hinge = torch.relu(x - tau.view(1, -1)) * w.view(1, -1)  # [N, d]

        ys = []
        for g, idxs in enumerate(self.group_indices):
            z_g = torch.sum(hinge[:, idxs], dim=1, keepdim=True)               # [N,1]
            h_g = torch.clamp(a[g] * z_g + b[g], min=L[g].item(), max=U[g].item())  # [N,1]
            ys.append(gamma[g] * h_g)                                          # [N,1]

        y = torch.stack(ys, dim=2).sum(dim=2)  # [N,1]
        if self._noise_std > 0:
            y = y + self._noise_std * torch.randn_like(y)
        return y

    def sample(self, n: int, device='cpu') -> Tuple[torch.Tensor, torch.Tensor]:
        x = torch.rand(n, self._d, device=device)
        y = self.evaluate(x)
        return x, y


# --------------------------------------------------------------
# KAN: cubic-spline version of the Sprecher block (no domain upd)
# --------------------------------------------------------------

class CubicHermiteSpline1D(nn.Module):
    """
    Uniform-knot cubic Hermite spline on [in_min, in_max] with trainable
    values at knots (coeffs). Slopes are Catmull-Rom style from neighboring
    values (C^1 continuous). Outside the domain, we use linear extrapolation
    using boundary slopes.

    This is a light-weight, GPU-friendly cubic spline suitable for KAN blocks.
    """
    def __init__(self, num_knots: int = 64, in_range=(0.0, 1.0)):
        super().__init__()
        assert num_knots >= 4, "Cubic spline needs >=4 knots"
        self.num_knots = int(num_knots)
        self.register_buffer("in_min", torch.tensor(float(in_range[0]), dtype=torch.float32))
        self.register_buffer("in_max", torch.tensor(float(in_range[1]), dtype=torch.float32))
        # Trainable values at knots
        self.coeffs = nn.Parameter(torch.linspace(self.in_min.item(), self.in_max.item(), self.num_knots))

    @property
    def _delta(self) -> torch.Tensor:
        return (self.in_max - self.in_min) / (self.num_knots - 1)

    def initialize_as_identity(self):
        with torch.no_grad():
            self.coeffs.data = torch.linspace(self.in_min.item(), self.in_max.item(), self.num_knots,
                                              device=self.coeffs.device, dtype=self.coeffs.dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Map to knot index space
        delta = self._delta
        # Normalize to [0, K-1]
        t = (x - self.in_min) / (delta + 1e-12)
        i0 = torch.floor(t).to(torch.long)
        frac = (t - i0.to(t.dtype)).clamp(0.0, 1.0)

        K = self.num_knots
        i0 = i0.clamp(0, K - 2)                 # segment [i0, i0+1]
        i_1 = (i0 - 1).clamp(0, K - 1)
        i1  = i0
        i2  = (i0 + 1).clamp(0, K - 1)
        i3  = (i0 + 2).clamp(0, K - 1)

        c = self.coeffs
        y0 = c[i1]
        y1 = c[i2]
        ym1 = c[i_1]
        y2  = c[i3]

        # Catmull-Rom slopes (Δ=delta)
        m0 = 0.5 * (y1 - ym1) / (delta + 1e-12)
        m1 = 0.5 * (y2 - y0)  / (delta + 1e-12)

        # Hermite basis (on [0,1])
        f = frac
        f2 = f * f
        f3 = f2 * f
        h00 =  2*f3 - 3*f2 + 1
        h10 =      f3 - 2*f2 + f
        h01 = -2*f3 + 3*f2
        h11 =      f3 -   f2

        out = h00 * y0 + h10 * (delta * m0) + h01 * y1 + h11 * (delta * m1)

        # Linear extrapolation outside the domain
        left_mask  = (x < self.in_min)
        right_mask = (x > self.in_max)
        if left_mask.any() or right_mask.any():
            y_left0  = c[0]
            y_left1  = c[1]
            slope_L  = (y_left1 - y_left0) / (delta + 1e-12)
            y_rightm1 = c[-2]
            y_right   = c[-1]
            slope_R   = (y_right - y_rightm1) / (delta + 1e-12)
            out = torch.where(left_mask,  y_left0  + slope_L * (x - self.in_min), out)
            out = torch.where(right_mask, y_right  + slope_R * (x - self.in_max), out)

        return out


class KANLayerBlock(nn.Module):
    """
    KAN-style Sprecher block (cubic variant, *no domain updates*):
      - φ: cubic Hermite spline applied elementwise to x + η q
      - λ: weight vector over input dims (like SN)
      - s_q = sum_i λ_i * φ(x_i + η q) + c·q  (c = Q_VALUES_FACTOR)
      - Φ: cubic Hermite spline applied to s
      - Residual: 'linear' style (α·x if d_in==d_out; x@W if dims differ)
      - If is_final: sum over q to produce scalar output
    """
    def __init__(self, d_in, d_out, layer_num=0, is_final=False, phi_knots=64, Phi_knots=64):
        super().__init__()
        self.d_in = d_in
        self.d_out = d_out
        self.layer_num = layer_num
        self.is_final = is_final

        # Spline components (no dynamic domain adaptation)
        self.phi = CubicHermiteSpline1D(num_knots=phi_knots, in_range=(0.0, 1.0))
        self.Phi = CubicHermiteSpline1D(num_knots=Phi_knots, in_range=(0.0, 1.0))
        self.phi.initialize_as_identity()
        self.Phi.initialize_as_identity()

        # λ and η
        self.lambdas = nn.Parameter(torch.randn(d_in) * math.sqrt(2.0 / max(1, d_in)))
        self.eta = nn.Parameter(torch.tensor(1.0 / (d_out + 10.0), dtype=torch.float32))

        # q buffer
        self.register_buffer("q_values", torch.arange(d_out, dtype=torch.float32))

        # Residuals (linear style)
        self.residual_weight = None
        self.residual_projection = None
        if d_in == d_out:
            self.residual_weight = nn.Parameter(torch.tensor(0.1))
        else:
            self.residual_projection = nn.Parameter(torch.empty(d_in, d_out))
            nn.init.xavier_uniform_(self.residual_projection)

    def _add_residual(self, activated, x_original):
        if x_original is None:
            return activated
        if self.residual_projection is not None:
            return activated + torch.matmul(x_original, self.residual_projection)
        if self.residual_weight is not None:
            return activated + self.residual_weight * x_original
        return activated

    def forward(self, x, x_original=None):
        # x: [B, d_in]
        B, d = x.shape
        device = x.device
        q = self.q_values.to(device).view(1, 1, -1)           # [1,1,d_out]
        x_exp = x.unsqueeze(-1) + self.eta * q                # [B, d_in, d_out]
        phi_out = self.phi(x_exp)                             # [B, d_in, d_out]
        s = (phi_out * self.lambdas.view(1, -1, 1)).sum(dim=1) + Q_VALUES_FACTOR * q.view(1, -1)  # [B, d_out]
        y = self.Phi(s)                                       # [B, d_out]
        y = self._add_residual(y, x_original)                 # residual
        if self.is_final:
            return y.sum(dim=1, keepdim=True)                 # [B,1]
        return y


class KANMultiLayerNetwork(nn.Module):
    """
    KAN network mirroring the SN skeleton: same widths, same residual semantics,
    BatchNorm only, and 'sum-at-last-block' behavior when final_dim==1.
    """
    def __init__(self, input_dim, architecture: List[int], final_dim=1,
                 phi_knots=64, Phi_knots=64, norm_type='batch',
                 norm_position='after', norm_skip_first=True):
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
            layers.append(KANLayerBlock(input_dim, final_dim, layer_num=0, is_final=is_final,
                                        phi_knots=phi_knots, Phi_knots=Phi_knots))
        else:
            L = len(architecture)
            d_in = input_dim
            for i, d_out in enumerate(architecture):
                is_final_block = (i == L - 1) and (final_dim == 1)
                layers.append(KANLayerBlock(d_in, d_out, layer_num=i, is_final=is_final_block,
                                            phi_knots=phi_knots, Phi_knots=Phi_knots))
                d_in = d_out
            if final_dim > 1:
                layers.append(KANLayerBlock(d_in, final_dim, layer_num=L, is_final=False,
                                            phi_knots=phi_knots, Phi_knots=Phi_knots))
        self.layers = nn.ModuleList(layers)

        # BatchNorm stack mirroring SN behavior
        self.norm_layers = nn.ModuleList()
        if norm_type != 'none':
            for i, layer in enumerate(self.layers):
                if norm_skip_first and i == 0:
                    self.norm_layers.append(nn.Identity())
                else:
                    if norm_position == 'before':
                        num_features = layer.d_in
                    else:  # after
                        num_features = 1 if layer.is_final else layer.d_out
                    assert norm_type == 'batch', "Only BatchNorm is allowed by the fairness contract"
                    self.norm_layers.append(nn.BatchNorm1d(num_features))

        # Output scale/bias like SN
        self.output_scale = nn.Parameter(torch.tensor(1.0))
        self.output_bias  = nn.Parameter(torch.tensor(0.0))

    def forward(self, x):
        x_in = x
        for i, layer in enumerate(self.layers):
            if self.norm_type != 'none' and self.norm_position == 'before':
                x_in = self.norm_layers[i](x_in)
            x_out = layer(x_in, x_in)  # residual uses x_in semantics like SN
            if self.norm_type != 'none' and self.norm_position == 'after':
                x_out = self.norm_layers[i](x_out)
            x_in = x_out
        y = x_in
        y = self.output_scale * y + self.output_bias
        return y


# -----------------------------
# Training / evaluation helpers
# -----------------------------

@dataclass
class TrainConfig:
    epochs: int = 4000
    lr: float = 3e-4
    weight_decay: float = 1e-7
    batch_size: int = 0        # 0 => full-batch
    warmup_epochs: int = 400   # SN domain-update warm-up (justification below)
    print_every: int = 200


def run_epoch(model, x, y, optimizer, batch_size=0):
    model.train()
    if batch_size is None or batch_size <= 0 or batch_size >= len(x):
        optimizer.zero_grad()
        pred = model(x)
        loss = torch.mean((pred - y) ** 2)
        loss.backward()
        optimizer.step()
        return loss.item()
    else:
        N = len(x)
        idx = torch.randperm(N, device=x.device)
        total = 0.0
        n_steps = 0
        for s in range(0, N, batch_size):
            sel = idx[s: s + batch_size]
            xb, yb = x[sel], y[sel]
            optimizer.zero_grad()
            pred = model(xb)
            loss = torch.mean((pred - yb) ** 2)
            loss.backward()
            optimizer.step()
            total += loss.item()
            n_steps += 1
        return total / max(1, n_steps)


@torch.no_grad()
def evaluate(model, x, y):
    model.eval()
    pred = model(x)
    mse = torch.mean((pred - y) ** 2).item()
    mae = torch.mean(torch.abs(pred - y)).item()
    return {"mse": mse, "mae": mae}


# ---------------------------------------
# Benchmark runner (build, train, compare)
# ---------------------------------------

def main():
    parser = argparse.ArgumentParser(description="SN vs KAN benchmark on Saturating Hinge Aggregation")
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--input_dim", type=int, default=16)
    parser.add_argument("--groups", type=int, default=4)
    parser.add_argument("--n_train", type=int, default=4096)
    parser.add_argument("--n_val", type=int, default=2048)
    parser.add_argument("--n_test", type=int, default=4096)
    parser.add_argument("--arch", type=str, default="32,32", help="Comma-separated hidden widths")
    parser.add_argument("--phi_knots", type=int, default=64)
    parser.add_argument("--Phi_knots", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=4000)
    parser.add_argument("--warmup", type=int, default=400, choices=[300, 400], help="SN warm-up epochs for domain updates")
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-7)
    parser.add_argument("--batch_size", type=int, default=0, help="0 => full-batch (one step/epoch)")
    parser.add_argument("--print_every", type=int, default=200)
    args = parser.parse_args()

    device = select_device(args.device)
    set_seed(args.seed)

    # -------------------------
    # Build dataset & splits
    # -------------------------
    dataset = SaturatingHingeAggregation(d=args.input_dim, groups=args.groups, seed=args.seed, noise_std=0.01)
    x_train, y_train = dataset.sample(args.n_train, device=device)
    x_val,   y_val   = dataset.sample(args.n_val,   device=device)
    x_test,  y_test  = dataset.sample(args.n_test,  device=device)

    # Standardize target using train split only (helps both models equally)
    y_mean = y_train.mean()
    y_std  = y_train.std().clamp_min(1e-6)
    y_train = (y_train - y_mean) / y_std
    y_val   = (y_val   - y_mean) / y_std
    y_test  = (y_test  - y_mean) / y_std

    architecture = parse_arch(args.arch)

    # -------------------------
    # Configure SN for fairness
    # -------------------------
    # Only BatchNorm, no lateral mixing, linear/projection residuals,
    # and no special SN-only extras beyond domain updates.
    CONFIG["use_lateral_mixing"] = False
    CONFIG["use_normalization"]  = True
    CONFIG["norm_type"]          = "batch"
    CONFIG["norm_position"]      = "after"
    CONFIG["norm_skip_first"]    = True
    CONFIG["use_residual_weights"] = True
    CONFIG["residual_style"]       = "linear"   # linear/projection residuals only
    CONFIG["train_phi_codomain"]   = False      # keep symmetry (no extra codomain params on SN side)
    # We will call domain updates manually during warm-up, then freeze.

    # -------------------------
    # Instantiate models
    # -------------------------
    sn_model = SprecherMultiLayerNetwork(
        input_dim=dataset.input_dim,
        architecture=architecture,
        final_dim=dataset.output_dim,
        phi_knots=args.phi_knots,
        Phi_knots=args.Phi_knots,
        norm_type="batch",
        norm_position="after",
        norm_skip_first=True
    ).to(device)

    kan_model = KANMultiLayerNetwork(
        input_dim=dataset.input_dim,
        architecture=architecture,
        final_dim=dataset.output_dim,
        phi_knots=args.phi_knots,
        Phi_knots=args.Phi_knots,
        norm_type="batch",
        norm_position="after",
        norm_skip_first=True
    ).to(device)

    # -------------------------
    # Parameter count matching
    # -------------------------
    sn_params  = count_trainable_params(sn_model)
    kan_params = count_trainable_params(kan_model)
    mean_params = 0.5 * (sn_params + kan_params)
    diff_pct = abs(sn_params - kan_params) / max(1.0, mean_params)

    if diff_pct > 0.05:
        # Given identical skeletons, counts should be equal; if not (edge case), gently adjust KAN knot count.
        # Simple local search on knots to bring params within 5%.
        target = sn_params
        best = (kan_params, args.phi_knots, args.Phi_knots)
        for k in range(max(4, args.phi_knots-8), args.phi_knots+9):
            for K in range(max(4, args.Phi_knots-8), args.Phi_knots+9):
                test = KANMultiLayerNetwork(
                    input_dim=dataset.input_dim, architecture=architecture, final_dim=dataset.output_dim,
                    phi_knots=k, Phi_knots=K, norm_type="batch", norm_position="after", norm_skip_first=True
                ).to(device)
                p = count_trainable_params(test)
                if abs(p - target) < abs(best[0] - target):
                    best = (p, k, K)
        kan_params, k_phi, k_Phi = best
        kan_model = KANMultiLayerNetwork(
            input_dim=dataset.input_dim, architecture=architecture, final_dim=dataset.output_dim,
            phi_knots=k_phi, Phi_knots=k_Phi, norm_type="batch", norm_position="after", norm_skip_first=True
        ).to(device)
        mean_params = 0.5 * (sn_params + kan_params)
        diff_pct = abs(sn_params - kan_params) / max(1.0, mean_params)

    assert diff_pct <= 0.05, f"Parameter budgets not within ±5%: SN={sn_params}, KAN={kan_params}"

    # -------------------------
    # Shared optimizer settings
    # -------------------------
    cfg = TrainConfig(epochs=args.epochs, lr=args.lr, weight_decay=args.weight_decay,
                      batch_size=args.batch_size, warmup_epochs=args.warmup, print_every=args.print_every)

    sn_opt  = torch.optim.Adam(sn_model.parameters(),  lr=cfg.lr, weight_decay=cfg.weight_decay)
    kan_opt = torch.optim.Adam(kan_model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)

    # -------------------------
    # Training loops (symmetric)
    # -------------------------
    print(f"\nDevice: {device}")
    print(f"Dataset: SHA(d={dataset.input_dim}, groups={args.groups})  |  splits: {args.n_train}/{args.n_val}/{args.n_test}")
    print(f"Architecture: {architecture}   phi_knots={args.phi_knots}  Phi_knots={args.Phi_knots}")
    print(f"Residuals: linear/projection only  |  Normalization: BatchNorm (after, skip_first=True)")
    print(f"Epochs: 4000 (SN warm-up {cfg.warmup_epochs} epochs for domain updates)  |  LR={cfg.lr}  WD={cfg.weight_decay}")
    print(f"Param counts: SN={sn_params:,d}  KAN={kan_params:,d}  (Δ={diff_pct*100:.2f}%)")

    sn_hist, kan_hist = [], []

    for epoch in range(cfg.epochs):
        # --- SN domain updates: warm-up only ---
        if epoch < cfg.warmup_epochs:
            # Update domains using SN's internal theoretical bounds (safe, model-provided)
            sn_model.update_all_domains(allow_resampling=True, force_resample=False)
        # After warm-up, domains are frozen: we simply stop calling update_all_domains()

        # One epoch (shared semantics)
        sn_loss  = run_epoch(sn_model,  x_train, y_train, sn_opt,  batch_size=cfg.batch_size)
        kan_loss = run_epoch(kan_model, x_train, y_train, kan_opt, batch_size=cfg.batch_size)

        if (epoch + 1) % cfg.print_every == 0 or epoch == 0:
            sn_val = evaluate(sn_model,  x_val, y_val)
            kan_val = evaluate(kan_model, x_val, y_val)
            print(f"Epoch {epoch+1:4d} | SN train {sn_loss:.3e}  val MSE {sn_val['mse']:.3e} | "
                  f"KAN train {kan_loss:.3e}  val MSE {kan_val['mse']:.3e}")
        sn_hist.append(sn_loss); kan_hist.append(kan_loss)

    # -------------------------
    # Final evaluation (test)
    # -------------------------
    sn_test  = evaluate(sn_model,  x_test, y_test)
    kan_test = evaluate(kan_model, x_test, y_test)

    print("\n=== FINAL RESULTS (test) ===")
    print(f"SN  : MSE={sn_test['mse']:.6e}  MAE={sn_test['mae']:.6e}")
    print(f"KAN : MSE={kan_test['mse']:.6e}  MAE={kan_test['mae']:.6e}")

    # Optional: simple JSON dump
    os.makedirs("results", exist_ok=True)
    out = {
        "seed": args.seed,
        "arch": architecture,
        "phi_knots": args.phi_knots,
        "Phi_knots": args.Phi_knots,
        "epochs": cfg.epochs,
        "warmup_epochs": cfg.warmup_epochs,
        "param_counts": {"SN": sn_params, "KAN": kan_params, "pct_diff": diff_pct},
        "val_every": cfg.print_every,
        "test": {"SN": sn_test, "KAN": kan_test}
    }
    import json
    with open(os.path.join("results", "sn_vs_kan_sha.json"), "w") as f:
        json.dump(out, f, indent=2)
    print("Saved: results/sn_vs_kan_sha.json")


if __name__ == "__main__":
    main()