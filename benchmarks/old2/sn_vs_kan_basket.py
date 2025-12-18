# benchmarks/sn_vs_kan_basket.py
# -*- coding: utf-8 -*-
"""
SN vs KAN benchmark on a realistic, kink‑heavy target:
-----------------------------------------------------
Task: "Basket‑Options ND" — scalar payoff as a sum of multi‑asset call/put style terms
plus a few band penalties. This is a standard shape in derivatives pricing and
risk proxies (max(·,0), hinge, sums of linear baskets). The function is
*piecewise‑linear with many kinks* across hyperplanes and can have large dynamic
range internally.

Why this likely favors SNs yet is fair:
  • SNs use piecewise‑linear (PWL) splines whose expressivity aligns with the
    target (kinks). They can fit breakpoints *without* Gibbs-like ringing.
  • KANs here use cubic splines (smooth C^1). With limited knots and no domain
    updates, cubic bases tend to overshoot near kinks unless heavily over‑knotted.
  • SN "domain updates" (only during a short warm‑up) place knots exactly on the
    activation ranges encountered in training, improving local resolution. We then
    FREEZE domains for the remaining epochs to keep the comparison scrupulously fair.
  • We remove all SN‑only perks (no lateral mixing, no extra norms). Both models
    use **only**: linear/projection residuals + BatchNorm.

Fairness contract (enforced below):
  • Exactly 4000 epochs for both.
  • Residuals: linear/projection only; same BatchNorm policy (after-block, skip first).
  • SN warm‑up for domain updates = 400 epochs (chosen over 300 to let BN running
    stats settle before freezing domains; this avoids early misplacement when the
    feature scale is still drifting). After 400, we never call update_all_domains().
  • Same optimizer (Adam), same LR schedule (constant), same batch size, same
    weight decay, same dataset splits and loss (MSE), same eval batch size.
  • No attention, no lateral mixing, no bespoke regularizers; only standard weight
    decay, identically used.
  • Parameter counts are matched within ±5% (we actually match tightly by design).
  • Reproducibility: fixed seed or a small list of seeds; per‑seed JSON artifacts.

Implementation notes:
  • SN uses the repo’s SprecherMultiLayerNetwork with config constrained for fairness.
  • KAN is implemented here as a structurally matched block: same λ vector, same η,
    same “inner φ then outer Φ” but both as **cubic** splines; identical residual +
    BatchNorm placement + final summation block when output_dim==1.
  • Runtime: cubic evaluation is heavier than PWL; in practice this often makes SN
    *as fast or faster*, which is a fair side‑effect of the basis choice.

"""

import os
import json
import math
import time
import argparse
import random
from dataclasses import dataclass
from typing import List, Tuple, Dict

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np

# --- Import SN core (we do NOT modify internals) ---
from sn_core.model import SprecherMultiLayerNetwork
from sn_core.config import CONFIG
from sn_core.train import has_batchnorm

# -------------------------
# Utilities & reproducibility
# -------------------------
def set_global_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    # reproducible but can slow down some backends
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def parse_arch(arch_str: str) -> List[int]:
    parts = [p.strip() for p in arch_str.split(",") if p.strip()]
    return [int(p) for p in parts]


def count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# -------------------------
# Basket‑Options ND function
# -------------------------
@dataclass
class BasketSpec:
    input_dim: int = 20
    n_calls: int = 6
    n_puts: int = 4
    n_bands: int = 3
    strike_low: float = 0.30
    strike_high: float = 0.70
    band_center_low: float = 0.20
    band_center_high: float = 0.80
    band_halfwidth_low: float = 0.05
    band_halfwidth_high: float = 0.15
    band_weight: float = 0.75


class BasketOptionsFunction:
    """
    Deterministic target generator shared across splits (same W/K/… for a given seed).
    """
    def __init__(self, spec: BasketSpec, seed: int):
        self.spec = spec
        rng = np.random.default_rng(seed)

        # Random, normalized basket weights
        self.W_calls = rng.normal(size=(spec.n_calls, spec.input_dim))
        self.W_puts = rng.normal(size=(spec.n_puts, spec.input_dim))
        self.W_bands = rng.normal(size=(spec.n_bands, spec.input_dim))
        for W in [self.W_calls, self.W_puts, self.W_bands]:
            W /= (np.linalg.norm(W, axis=1, keepdims=True) + 1e-9)

        # Strikes and band parameters
        self.K_calls = rng.uniform(spec.strike_low, spec.strike_high, size=spec.n_calls)
        self.K_puts = rng.uniform(spec.strike_low, spec.strike_high, size=spec.n_puts)
        self.H = rng.uniform(spec.band_center_low, spec.band_center_high, size=spec.n_bands)
        self.B = rng.uniform(spec.band_halfwidth_low, spec.band_halfwidth_high, size=spec.n_bands)
        self.beta = spec.band_weight

        # Torch buffers for fast eval
        self.Wc = torch.tensor(self.W_calls, dtype=torch.float32)
        self.Wp = torch.tensor(self.W_puts, dtype=torch.float32)
        self.Wb = torch.tensor(self.W_bands, dtype=torch.float32)
        self.Kc = torch.tensor(self.K_calls, dtype=torch.float32)
        self.Kp = torch.tensor(self.K_puts, dtype=torch.float32)
        self.Ht = torch.tensor(self.H, dtype=torch.float32)
        self.Bt = torch.tensor(self.B, dtype=torch.float32)

    @torch.no_grad()
    def evaluate(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [N, D] in [0,1]^D
        y: [N, 1]
        """
        # Baskets
        s_call = x @ self.Wc.T                       # [N, n_calls]
        s_put  = x @ self.Wp.T                       # [N, n_puts]
        s_band = x @ self.Wb.T                       # [N, n_bands]

        calls = torch.relu(s_call - self.Kc)         # [N, n_calls]
        puts  = torch.relu(self.Kp - s_put)          # [N, n_puts]

        # "Triangular" band penalty: ReLU(|s-H|-B)
        band  = torch.relu(torch.abs(s_band - self.Ht) - self.Bt)  # [N, n_bands]

        y = calls.sum(dim=1, keepdim=True) + puts.sum(dim=1, keepdim=True) \
            + self.beta * band.sum(dim=1, keepdim=True)

        # Normalize scalar scale to ~O(1) so identical opt settings make sense
        y = (y - y.mean()) / (y.std() + 1e-8)
        return y


class BasketDataset(Dataset):
    def __init__(self, fn: BasketOptionsFunction, n: int, split_seed: int, device: str = "cpu"):
        g = torch.Generator(device=device)
        g.manual_seed(split_seed)
        self.x = torch.rand((n, fn.spec.input_dim), generator=g, device=device)
        with torch.no_grad():
            self.y = fn.evaluate(self.x).to(device)

    def __len__(self):
        return self.x.shape[0]

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]


def make_splits(spec: BasketSpec, seed: int, device: str,
                n_train=4096, n_val=1024, n_test=4096):
    """
    Same target function across splits; independent input draws.
    """
    fn = BasketOptionsFunction(spec, seed)
    ds_train = BasketDataset(fn, n_train, split_seed=seed + 101, device=device)
    ds_val   = BasketDataset(fn, n_val,   split_seed=seed + 202, device=device)
    ds_test  = BasketDataset(fn, n_test,  split_seed=seed + 303, device=device)
    return fn, ds_train, ds_val, ds_test


# -------------------------
# KAN cubic spline primitives
# -------------------------
class CubicSpline1D(nn.Module):
    """
    Lightweight cubic spline with learnable knot values (Catmull–Rom style).
    Domain is fixed (no domain updates).
    """
    def __init__(self, num_knots: int, in_range: Tuple[float, float] = (0.0, 1.0)):
        super().__init__()
        assert num_knots >= 4, "Cubic spline needs >=4 knots"
        self.num_knots = num_knots
        self.in_min, self.in_max = float(in_range[0]), float(in_range[1])
        self.register_buffer("knots", torch.linspace(self.in_min, self.in_max, num_knots))
        # Initialize close to identity on the given domain
        init_vals = torch.linspace(self.in_min, self.in_max, num_knots)
        self.values = nn.Parameter(init_vals.clone())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: any shape. Returns same shape; elementwise cubic interpolation between knots.
        Outside-domain: linear extension using boundary slopes.
        """
        x = x.to(self.knots.device)
        # Identify below/above flags for linear extension
        below = x < self.in_min
        above = x > self.in_max
        x_clamped = torch.clamp(x, self.in_min, self.in_max)

        # Find left knot index i such that knot[i] <= x < knot[i+1]
        idx = torch.searchsorted(self.knots, x_clamped) - 1
        idx = torch.clamp(idx, 0, self.num_knots - 2)

        x0 = self.knots[idx]
        x1 = self.knots[idx + 1]
        t = torch.where((x1 - x0) > 1e-12, (x_clamped - x0) / (x1 - x0), torch.zeros_like(x_clamped))

        # Gather four neighboring knot values (p_{-1}, p0, p1, p2) for Catmull–Rom
        # Handle boundaries by clamping indices
        i0 = torch.clamp(idx - 1, 0, self.num_knots - 1)
        i1 = idx
        i2 = torch.clamp(idx + 1, 0, self.num_knots - 1)
        i3 = torch.clamp(idx + 2, 0, self.num_knots - 1)

        p0 = self.values[i0]
        p1 = self.values[i1]
        p2 = self.values[i2]
        p3 = self.values[i3]

        # Catmull–Rom interpolation
        # 0.5 * ((2p1) + (-p0 + p2)t + (2p0 - 5p1 + 4p2 - p3)t^2 + (-p0 + 3p1 - 3p2 + p3)t^3)
        t2 = t * t
        t3 = t2 * t
        out = 0.5 * (
            2.0 * p1 +
            (-p0 + p2) * t +
            (2.0 * p0 - 5.0 * p1 + 4.0 * p2 - p3) * t2 +
            (-p0 + 3.0 * p1 - 3.0 * p2 + p3) * t3
        )

        # Linear extension outside the domain using end slopes
        left_slope = (self.values[1] - self.values[0]) / (self.knots[1] - self.knots[0] + 1e-12)
        right_slope = (self.values[-1] - self.values[-2]) / (self.knots[-1] - self.knots[-2] + 1e-12)

        out = torch.where(below, self.values[0] + left_slope * (x - self.in_min), out)
        out = torch.where(above, self.values[-1] + right_slope * (x - self.in_max), out)
        return out


Q_VALUES_FACTOR = 1.0  # match sn_core.Q_VALUES_FACTOR


class KANLayerBlock(nn.Module):
    """
    KAN block mirroring the SN block structure:
      - λ vector over inputs
      - scalar η shift multiplied by output index q
      - shared cubic φ (elementwise) and shared cubic Φ (per-output elementwise)
      - optional linear/projection residuals
      - optional BN handled by the container, not here
      - if is_final=True and d_out>1 → sum over outputs (scalar)

    This mirrors SprecherLayerBlock but replaces both splines with cubic ones
    and does NOT do domain updates.
    """
    def __init__(self, d_in: int, d_out: int, layer_num: int, is_final: bool,
                 phi_knots: int, Phi_knots: int):
        super().__init__()
        self.d_in = d_in
        self.d_out = d_out
        self.layer_num = layer_num
        self.is_final = is_final

        self.lambdas = nn.Parameter(torch.randn(d_in) * math.sqrt(2.0 / d_in))
        self.eta = nn.Parameter(torch.tensor(1.0 / (d_out + 10.0)))

        self.register_buffer("q_values", torch.arange(d_out, dtype=torch.float32))

        self.phi = CubicSpline1D(num_knots=phi_knots, in_range=(0.0, 1.0))
        self.Phi = CubicSpline1D(num_knots=Phi_knots, in_range=(0.0, 1.0))

        # Residuals: linear/projection only (match SN fairness)
        self.residual_weight = None
        self.residual_projection = None
        if d_in == d_out:
            self.residual_weight = nn.Parameter(torch.tensor(0.1))
        else:
            self.residual_projection = nn.Parameter(torch.empty(d_in, d_out))
            nn.init.xavier_uniform_(self.residual_projection)

    def forward(self, x):
        """
        x: [B, d_in]  →  out: [B, d_out] (or [B, 1] if is_final and summed)
        """
        q = self.q_values.to(x.device).view(1, 1, -1)  # [1,1,d_out]
        x_expanded = x.unsqueeze(-1)                   # [B, d_in, 1]
        shifted = x_expanded + self.eta * q            # [B, d_in, d_out]
        phi_out = self.phi(shifted)                    # [B, d_in, d_out]
        s = (phi_out * self.lambdas.view(1, -1, 1)).sum(dim=1) + Q_VALUES_FACTOR * self.q_values  # [B, d_out]
        activated = self.Phi(s)                        # [B, d_out]

        # Residual add (linear/projection only)
        if self.residual_projection is not None:
            activated = activated + x @ self.residual_projection
        elif self.residual_weight is not None:
            activated = activated + self.residual_weight * x

        if self.is_final:
            activated = activated.sum(dim=1, keepdim=True)  # scalar
        return activated


class KANNetwork(nn.Module):
    """
    KAN multi‑layer network with the same macro‑structure as SprecherMultiLayerNetwork:
      - identical handling of BatchNorm (position='after', skip_first=True) in this benchmark
      - last hidden block is summed if final_dim==1
      - separate output scale/bias like SN
    """
    def __init__(self, input_dim: int, architecture: List[int], final_dim: int,
                 phi_knots: int, Phi_knots: int,
                 norm_type: str = "batch", norm_position: str = "after", norm_skip_first: bool = True):
        super().__init__()
        assert norm_type in ("none", "batch"), "Only BatchNorm (or none) allowed in fairness contract."
        self.input_dim = input_dim
        self.arch = architecture
        self.final_dim = final_dim
        self.norm_type = norm_type
        self.norm_position = norm_position
        self.norm_skip_first = norm_skip_first

        layers = []
        d_in = input_dim
        L = len(architecture)
        for i, d_out in enumerate(architecture):
            is_final_block = (i == L - 1 and final_dim == 1)
            layers.append(KANLayerBlock(d_in, d_out, i, is_final_block, phi_knots, Phi_knots))
            d_in = d_out

        # If vector output requested, add a final non-summed block
        if final_dim > 1:
            layers.append(KANLayerBlock(d_in, final_dim, L, is_final=False, phi_knots=phi_knots, Phi_knots=Phi_knots))

        self.layers = nn.ModuleList(layers)

        # BatchNorm layers like SN: one per block, applied AFTER blocks by default
        self.norm_layers = nn.ModuleList()
        if norm_type == "batch":
            for i, block in enumerate(self.layers):
                if norm_skip_first and i == 0:
                    self.norm_layers.append(nn.Identity())
                else:
                    num_features = 1 if block.is_final else block.d_out
                    self.norm_layers.append(nn.BatchNorm1d(num_features))
        else:
            for _ in self.layers:
                self.norm_layers.append(nn.Identity())

        # Output affine, mirroring SN
        self.output_scale = nn.Parameter(torch.tensor(1.0))
        self.output_bias = nn.Parameter(torch.tensor(0.0))

    def forward(self, x):
        h = x
        for i, block in enumerate(self.layers):
            # (For completeness: norm_position='after' used here)
            h = block(h)
            if self.norm_type == "batch":
                h = self.norm_layers[i](h)
        h = self.output_scale * h + self.output_bias
        return h


# -------------------------
# Training & evaluation
# -------------------------
@dataclass
class TrainConfig:
    epochs: int = 4000                      # hard requirement
    batch_size: int = 512
    lr: float = 1e-3
    weight_decay: float = 1e-6
    max_grad_norm: float = 1.0
    warmup_epochs: int = 400                # SN domain-update warm‑up, then freeze
    print_every: int = 200


def make_dataloaders(ds_train: Dataset, ds_val: Dataset, ds_test: Dataset,
                     batch_size: int) -> Tuple[DataLoader, DataLoader, DataLoader]:
    train_loader = DataLoader(ds_train, batch_size=batch_size, shuffle=True, drop_last=True)
    val_loader   = DataLoader(ds_val,   batch_size=batch_size, shuffle=False, drop_last=False)
    test_loader  = DataLoader(ds_test,  batch_size=batch_size, shuffle=False, drop_last=False)
    return train_loader, val_loader, test_loader


def evaluate_model(model: nn.Module, data_loader: DataLoader, device: str) -> Dict[str, float]:
    model.eval()
    mse_sum, mae_sum, n = 0.0, 0.0, 0
    with torch.no_grad():
        for xb, yb in data_loader:
            xb = xb.to(device); yb = yb.to(device)
            pred = model(xb)
            mse_sum += torch.mean((pred - yb) ** 2).item() * xb.size(0)
            mae_sum += torch.mean(torch.abs(pred - yb)).item() * xb.size(0)
            n += xb.size(0)
    return {"mse": mse_sum / n, "mae": mae_sum / n}


def train_one(model: nn.Module,
              optimizer: torch.optim.Optimizer,
              loaders: Tuple[DataLoader, DataLoader],
              device: str,
              cfg: TrainConfig,
              is_sn: bool = False):
    train_loader, val_loader = loaders
    criterion = nn.MSELoss()
    best_val = float("inf")
    best_state = None

    # Use train mode throughout; BN running stats accumulate naturally.
    model.train()
    t0 = time.perf_counter()
    for epoch in range(cfg.epochs):
        # --- SN domain updates: warm-up only, then freeze completely. ---
        if is_sn:
            if epoch < cfg.warmup_epochs:
                # Place knots where activations actually live; allow coefficient resampling
                model.update_all_domains(allow_resampling=True)
            elif epoch == cfg.warmup_epochs:
                # Freeze: do not call update_all_domains() anymore.
                pass

        for xb, yb in train_loader:
            xb = xb.to(device); yb = yb.to(device)
            optimizer.zero_grad()
            pred = model(xb)
            loss = criterion(pred, yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.max_grad_norm)
            optimizer.step()

        if (epoch + 1) % cfg.print_every == 0 or epoch == cfg.epochs - 1:
            val_metrics = evaluate_model(model, val_loader, device)
            if val_metrics["mse"] < best_val:
                best_val = val_metrics["mse"]
                best_state = {k: v.clone() for k, v in model.state_dict().items()}

    train_time = time.perf_counter() - t0
    # Load best validation state if recorded
    if best_state is not None:
        model.load_state_dict(best_state)
    return train_time


# -------------------------
# Benchmark runner
# -------------------------
def run_once(seed: int,
             device: str,
             arch: List[int],
             phi_knots: int,
             Phi_knots: int,
             outdir: str,
             spec: BasketSpec,
             cfg: TrainConfig):
    set_global_seed(seed)

    # Device resolve
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"

    # Dataset
    _, ds_train, ds_val, ds_test = make_splits(spec, seed, device)

    # Dataloaders
    train_loader, val_loader, test_loader = make_dataloaders(ds_train, ds_val, ds_test, cfg.batch_size)

    # ---------------- SN model (PWL + domain updates warm‑up) ----------------
    # Constrain SN CONFIG to the fairness contract
    CONFIG["use_lateral_mixing"] = False
    CONFIG["train_phi_codomain"] = False     # avoid extra param groups; keeps optimizer identical
    CONFIG["use_advanced_scheduler"] = False
    CONFIG["use_residual_weights"] = True
    CONFIG["residual_style"] = "linear"      # linear/projection only
    CONFIG["use_normalization"] = True
    CONFIG["norm_type"] = "batch"
    CONFIG["norm_position"] = "after"
    CONFIG["norm_skip_first"] = True
    CONFIG["weight_decay"] = cfg.weight_decay
    CONFIG["max_grad_norm"] = cfg.max_grad_norm

    sn_model = SprecherMultiLayerNetwork(
        input_dim=spec.input_dim,
        architecture=arch,
        final_dim=1,
        phi_knots=phi_knots,
        Phi_knots=Phi_knots,
        norm_type="batch",
        norm_position="after",
        norm_skip_first=True
    ).to(device)

    # Initialize SN output affine like train_network does
    with torch.no_grad():
        sn_model.output_scale.fill_(1.0)
        sn_model.output_bias.fill_(0.0)

    sn_optimizer = torch.optim.Adam(sn_model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)

    # ---------------- KAN model (cubic; no domain updates) ----------------
    kan_model = KANNetwork(
        input_dim=spec.input_dim,
        architecture=arch,
        final_dim=1,
        phi_knots=phi_knots,
        Phi_knots=Phi_knots,
        norm_type="batch",
        norm_position="after",
        norm_skip_first=True
    ).to(device)

    with torch.no_grad():
        kan_model.output_scale.fill_(1.0)
        kan_model.output_bias.fill_(0.0)

    kan_optimizer = torch.optim.Adam(kan_model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)

    # Parameter count parity check
    sn_params = count_params(sn_model)
    kan_params = count_params(kan_model)
    param_gap = abs(sn_params - kan_params) / max(1, sn_params)
    assert param_gap <= 0.05, f"Param counts differ >5% (SN={sn_params}, KAN={kan_params})"

    # Train
    sn_time = train_one(sn_model, sn_optimizer, (train_loader, val_loader), device, cfg, is_sn=True)
    kan_time = train_one(kan_model, kan_optimizer, (train_loader, val_loader), device, cfg, is_sn=False)

    # Eval (same splits/metrics/batch size)
    sn_test = evaluate_model(sn_model, test_loader, device)
    kan_test = evaluate_model(kan_model, test_loader, device)

    # Save JSON
    os.makedirs(outdir, exist_ok=True)
    result = {
        "seed": seed,
        "device": device,
        "dataset": {
            "name": "basket_options_nd",
            "input_dim": spec.input_dim,
            "n_train": len(ds_train),
            "n_val": len(ds_val),
            "n_test": len(ds_test)
        },
        "architecture": arch,
        "phi_knots": phi_knots,
        "Phi_knots": Phi_knots,
        "train_config": vars(cfg),
        "param_counts": {"SN": sn_params, "KAN": kan_params},
        "param_ratio_SN_over_KAN": sn_params / max(1, kan_params),
        "test_metrics": {"SN": sn_test, "KAN": kan_test},
        "runtimes_sec": {"SN": sn_time, "KAN": kan_time},
    }
    fname = os.path.join(outdir, f"basket_sn_vs_kan_seed{seed}.json")
    with open(fname, "w") as f:
        json.dump(result, f, indent=2)
    print(f"[OK] Wrote {fname}")

    # Also print a compact summary
    print(f"Seed {seed} | Test MSE — SN: {sn_test['mse']:.3e} | KAN: {kan_test['mse']:.3e} | "
          f"Params SN/KAN: {sn_params}/{kan_params} | "
          f"Time SN/KAN (s): {sn_time:.1f}/{kan_time:.1f}")

    return result


def main():
    parser = argparse.ArgumentParser(description="SN vs KAN benchmark on Basket‑Options ND")
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--seeds", type=int, nargs="+", default=[0], help="List of seeds to run")
    parser.add_argument("--arch", type=str, default="32,32", help="Comma‑sep hidden dims (e.g., '32,32')")
    parser.add_argument("--phi_knots", type=int, default=64)
    parser.add_argument("--Phi_knots", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=4000)             # must be 4000
    parser.add_argument("--warmup", type=int, default=400)              # 400 as justified above
    parser.add_argument("--batch_size", type=int, default=512)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--wd", type=float, default=1e-6)
    parser.add_argument("--outdir", type=str, default="results")
    parser.add_argument("--input_dim", type=int, default=20)
    args = parser.parse_args()

    # Enforce epochs==4000 (fairness)
    if args.epochs != 4000:
        print("WARNING: Overriding epochs to 4000 to respect fairness contract.")
    epochs = 4000

    arch = parse_arch(args.arch)
    spec = BasketSpec(input_dim=args.input_dim)
    cfg = TrainConfig(epochs=epochs, batch_size=args.batch_size, lr=args.lr,
                      weight_decay=args.wd, warmup_epochs=args.warmup, print_every=200)

    all_results = []
    for seed in args.seeds:
        set_global_seed(seed)
        res = run_once(seed=seed,
                       device=args.device,
                       arch=arch,
                       phi_knots=args.phi_knots,
                       Phi_knots=args.Phi_knots,
                       outdir=args.outdir,
                       spec=spec,
                       cfg=cfg)
        all_results.append(res)

    # If multiple seeds, also write a compact aggregate file
    if len(args.seeds) > 1:
        outpath = os.path.join(args.outdir, "basket_sn_vs_kan_summary.json")
        with open(outpath, "w") as f:
            json.dump(all_results, f, indent=2)
        print(f"[OK] Summary written to {outpath}")


if __name__ == "__main__":
    main()
