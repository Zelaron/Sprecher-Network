#!/usr/bin/env python3
"""
benchmarks/benchmark_shared_warped_ridge.py

Barebones benchmark: Sprecher Network (SN) vs KAN.

What is being compared?
-----------------------
SN (Sprecher Network):
  - φ: piecewise-linear (PWL), monotone
  - Φ: piecewise-linear (PWL), general
  - NO residuals, NO lateral mixing, NO normalization, NO codomain training
  - Domain updates (theoretical) ONLY for the first 10% of epochs (default: 400 / 4000)

KAN (Kolmogorov-Arnold Network):
  - Edge splines: cubic Hermite splines with PCHIP slopes (shape-preserving cubic)
  - NO SiLU residual, NO grid updates, NO normalization
  - Fixed knot locations (no grid adaptation)

Fairness:
---------
  - Same train/test data, optimizer, LR schedule, dtype, epochs.
  - Parameters are matched: by default, KAN's knot count is automatically chosen so that
    the total parameter count matches the SN's parameter count as closely as possible.

Target function (chosen to highlight an inherent SN advantage):
--------------------------------------------------------------
High-dimensional *shared-warp ridge* function on x ∈ [0,1]^d:

  h(t) = 0.6 t + 0.2 σ(30(t-0.30)) + 0.2 σ(30(t-0.70))      (monotone, shared across dims)
  z    = mean_i h(x_i)
  y    = sin(12π z^2) * exp(-3(z-0.5)^2) + 0.15 sin(2π z) + 0.10 (z - 0.5)

This is intentionally symmetric across coordinates and depends on a *shared* monotone warp h.
SN shares a single monotone φ per layer across all input dimensions, so it can learn h once.
A barebones KAN must learn separate edge functions; under a fixed parameter budget this
forces very few knots per edge, which is precisely where SN's linear-in-d parameterization
should shine.

How to run
----------
Single run:
  python -m benchmarks.benchmark_shared_warped_ridge --seed 0

Seed sweep:
  for s in 0 1 2 3; do
    python -m benchmarks.benchmark_shared_warped_ridge --seed $s --phi-knots 350 --Phi-knots 350
  done
"""

from __future__ import annotations

import argparse
import math
import random
import time
from typing import Dict, Tuple

import numpy as np
import torch
import torch.nn as nn

from sn_core.config import CONFIG
from sn_core.model import SprecherMultiLayerNetwork


# -----------------------------
# Reproducibility
# -----------------------------

def set_global_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# -----------------------------
# Target function / dataset
# -----------------------------

@torch.no_grad()
def target_function(x: torch.Tensor) -> torch.Tensor:
    """
    x: [N, d] in [0,1]
    returns y: [N, 1]
    """
    # Shared monotone warp: multiple "shoulders" but still monotone.
    t = x
    h = 0.6 * t + 0.2 * torch.sigmoid(30.0 * (t - 0.30)) + 0.2 * torch.sigmoid(30.0 * (t - 0.70))
    z = h.mean(dim=1, keepdim=True)  # [N,1], roughly in [0,1]

    y = torch.sin(12.0 * math.pi * z * z) * torch.exp(-3.0 * (z - 0.5) ** 2)
    y = y + 0.15 * torch.sin(2.0 * math.pi * z) + 0.10 * (z - 0.5)
    return y


def make_train_test(
    *,
    input_dim: int,
    n_train: int,
    n_test: int,
    seed: int,
    device: torch.device,
    dtype: torch.dtype,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Returns: (x_train, y_train, x_test, y_test)
    """
    # Generate on CPU with a seeded generator for determinism across devices
    g = torch.Generator(device="cpu")
    g.manual_seed(seed)

    x_train = torch.rand((n_train, input_dim), generator=g, dtype=dtype)
    x_test = torch.rand((n_test, input_dim), generator=g, dtype=dtype)

    # Move to device
    x_train = x_train.to(device=device, dtype=dtype)
    x_test = x_test.to(device=device, dtype=dtype)

    y_train = target_function(x_train).to(device=device, dtype=dtype)
    y_test = target_function(x_test).to(device=device, dtype=dtype)
    return x_train, y_train, x_test, y_test


# -----------------------------
# KAN: cubic PCHIP splines (vectorized)
# -----------------------------

def _pchip_limit_endpoint(di: torch.Tensor, deltai: torch.Tensor) -> torch.Tensor:
    """
    Endpoint slope limiter from PCHIP.
    di, deltai: broadcastable tensors
    """
    # If slope has wrong sign, set to 0
    di = torch.where((di * deltai) <= 0, torch.zeros_like(di), di)
    # Limit magnitude to 3*delta
    di = torch.where(torch.abs(di) > 3.0 * torch.abs(deltai), 3.0 * deltai, di)
    return di


def pchip_slopes(y: torch.Tensor, knots: torch.Tensor) -> torch.Tensor:
    """
    Vectorized PCHIP slopes.
    y: [..., K] knot values
    knots: [K] strictly increasing
    returns d: [..., K]
    """
    K = y.shape[-1]
    if K == 1:
        return torch.zeros_like(y)

    h = knots[1:] - knots[:-1]  # [K-1]
    delta = (y[..., 1:] - y[..., :-1]) / (h + 1e-12)  # [..., K-1]

    d = torch.zeros_like(y)

    if K > 2:
        hkm1 = h[:-1]  # [K-2]
        hk = h[1:]     # [K-2]
        w1 = 2.0 * hk + hkm1
        w2 = hk + 2.0 * hkm1

        same_sign = (delta[..., :-1] * delta[..., 1:]) > 0
        d_interior = (w1 + w2) / (w1 / (delta[..., :-1] + 1e-12) + w2 / (delta[..., 1:] + 1e-12))
        d[..., 1:-1] = torch.where(same_sign, d_interior, torch.zeros_like(d_interior))

        d0 = ((2.0 * h[0] + h[1]) * delta[..., 0] - h[0] * delta[..., 1]) / (h[0] + h[1] + 1e-12)
        dN = ((2.0 * h[-1] + h[-2]) * delta[..., -1] - h[-1] * delta[..., -2]) / (h[-1] + h[-2] + 1e-12)
    else:
        # K == 2
        d0 = delta[..., 0]
        dN = delta[..., -1]

    d0 = _pchip_limit_endpoint(d0, delta[..., 0])
    dN = _pchip_limit_endpoint(dN, delta[..., -1])

    d[..., 0] = d0
    d[..., -1] = dN
    return d


def hermite_eval_many(x: torch.Tensor, knots: torch.Tensor, y: torch.Tensor, d: torch.Tensor) -> torch.Tensor:
    """
    Evaluate many cubic Hermite splines sharing the same knots at points x.

    x: [B]
    knots: [K]
    y: [M, K]   (M splines)
    d: [M, K]   (PCHIP slopes)
    returns: [B, M]
    """
    assert x.ndim == 1, "x must be [B]"
    K = knots.shape[0]
    M = y.shape[0]

    # Clamp for interpolation
    x_clamped = torch.clamp(x, knots[0], knots[-1])

    # Segment indices
    idx = torch.searchsorted(knots, x_clamped) - 1
    idx = torch.clamp(idx, 0, K - 2)  # [B]

    xk = knots[idx]       # [B]
    xk1 = knots[idx + 1]  # [B]
    h = xk1 - xk          # [B]

    safe_h = torch.where(h.abs() < 1e-12, torch.ones_like(h), h)
    t = torch.where(h.abs() < 1e-12, torch.zeros_like(x_clamped), (x_clamped - xk) / safe_h)

    t2 = t * t
    t3 = t2 * t

    h00 = 2.0 * t3 - 3.0 * t2 + 1.0
    h10 = t3 - 2.0 * t2 + t
    h01 = -2.0 * t3 + 3.0 * t2
    h11 = t3 - t2

    # Gather y/d at idx and idx+1
    idx0 = idx.unsqueeze(0).expand(M, -1)  # [M,B]
    idx1 = (idx + 1).unsqueeze(0).expand(M, -1)

    yk = y.gather(1, idx0)     # [M,B]
    yk1 = y.gather(1, idx1)    # [M,B]
    dk = d.gather(1, idx0)     # [M,B]
    dk1 = d.gather(1, idx1)    # [M,B]

    hM = h.unsqueeze(0).expand(M, -1)  # [M,B]

    out = (
        h00.unsqueeze(0) * yk +
        h10.unsqueeze(0) * hM * dk +
        h01.unsqueeze(0) * yk1 +
        h11.unsqueeze(0) * hM * dk1
    )

    # Linear extrapolation outside domain using endpoint slopes
    below = x < knots[0]
    above = x > knots[-1]

    left_val = y[:, 0].unsqueeze(1)    # [M,1]
    left_slope = d[:, 0].unsqueeze(1)  # [M,1]
    out = torch.where(
        below.unsqueeze(0),
        left_val + left_slope * (x - knots[0]).unsqueeze(0),
        out,
    )

    right_val = y[:, -1].unsqueeze(1)
    right_slope = d[:, -1].unsqueeze(1)
    out = torch.where(
        above.unsqueeze(0),
        right_val + right_slope * (x - knots[-1]).unsqueeze(0),
        out,
    )

    return out.transpose(0, 1)  # [B,M]


class KANLayerPCHIP(nn.Module):
    """
    Barebones KAN layer with cubic PCHIP splines on edges.
    y_out[j] = sum_i spline[j,i](x_in[i]) + bias[j]
    """

    def __init__(
        self,
        d_in: int,
        d_out: int,
        num_knots: int,
        x_min: float = 0.0,
        x_max: float = 1.0,
        init_scale: float = 0.05,
        dtype: torch.dtype = torch.float32,
    ) -> None:
        super().__init__()
        self.d_in = int(d_in)
        self.d_out = int(d_out)
        self.num_knots = int(num_knots)

        self.register_buffer("knots", torch.linspace(float(x_min), float(x_max), self.num_knots, dtype=dtype))
        # coeffs[o, i, k]
        self.coeffs = nn.Parameter(torch.zeros(self.d_out, self.d_in, self.num_knots, dtype=dtype))
        self.bias = nn.Parameter(torch.zeros(self.d_out, dtype=dtype))

        # Small random init helps break symmetry
        with torch.no_grad():
            self.coeffs.normal_(mean=0.0, std=init_scale)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, d_in]
        B = x.shape[0]

        # Buffers/params are already moved by model.to(device, dtype)
        knots = self.knots
        y = self.coeffs  # [d_out, d_in, K]
        d = pchip_slopes(y, knots)

        out = torch.zeros((B, self.d_out), device=x.device, dtype=x.dtype)
        for i in range(self.d_in):
            out = out + hermite_eval_many(x[:, i], knots, y[:, i, :], d[:, i, :])
        out = out + self.bias
        return out


class KANNetPCHIP(nn.Module):
    """
    Barebones stacked KAN with two hidden layers.
    dims: [input_dim, width, width, 1]
    """

    def __init__(
        self,
        input_dim: int,
        width: int,
        num_knots: int,
        x_min: float = 0.0,
        x_max: float = 1.0,
        init_scale: float = 0.05,
        dtype: torch.dtype = torch.float32,
    ) -> None:
        super().__init__()
        dims = [int(input_dim), int(width), int(width), 1]
        layers = []
        for li in range(len(dims) - 1):
            layers.append(
                KANLayerPCHIP(
                    d_in=dims[li],
                    d_out=dims[li + 1],
                    num_knots=num_knots,
                    x_min=x_min,
                    x_max=x_max,
                    init_scale=init_scale,
                    dtype=dtype,
                )
            )
        self.layers = nn.ModuleList(layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x)
        return x


# -----------------------------
# Parameter counting / matching
# -----------------------------

def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


def kan_param_count_formula(input_dim: int, width: int, num_knots: int) -> int:
    dims = [int(input_dim), int(width), int(width), 1]
    edges = sum(dims[i] * dims[i + 1] for i in range(len(dims) - 1))
    biases = sum(dims[i + 1] for i in range(len(dims) - 1))
    return int(edges * int(num_knots) + biases)


def choose_kan_knots_to_match(sn_params: int, input_dim: int, width: int, *, min_knots: int = 3) -> int:
    dims = [int(input_dim), int(width), int(width), 1]
    edges = sum(dims[i] * dims[i + 1] for i in range(len(dims) - 1))
    biases = sum(dims[i + 1] for i in range(len(dims) - 1))

    # Solve edges*k + biases ≈ sn_params
    k0 = int(round((sn_params - biases) / max(edges, 1)))
    k0 = max(min_knots, k0)

    # Search small neighborhood for closest match
    best_k = k0
    best_diff = abs(kan_param_count_formula(input_dim, width, best_k) - sn_params)
    for k in range(max(min_knots, k0 - 5), k0 + 6):
        diff = abs(kan_param_count_formula(input_dim, width, k) - sn_params)
        if diff < best_diff:
            best_k, best_diff = k, diff
    return best_k


# -----------------------------
# Training / evaluation
# -----------------------------

@torch.no_grad()
def rmse(model: nn.Module, x: torch.Tensor, y: torch.Tensor) -> float:
    model.eval()
    pred = model(x)
    return torch.sqrt(torch.mean((pred - y) ** 2)).item()


def train_regression(
    *,
    model: nn.Module,
    x_train: torch.Tensor,
    y_train: torch.Tensor,
    x_test: torch.Tensor,
    y_test: torch.Tensor,
    epochs: int,
    lr: float,
    weight_decay: float,
    max_grad_norm: float,
    log_every: int,
    label: str,
    # SN-only
    domain_update_epochs: int = 0,
    sn_domain_update_fn=None,
) -> Dict[str, float]:
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=lr * 0.05)

    best_test = float("inf")
    best_epoch = -1

    t0 = time.time()
    for epoch in range(epochs):
        model.train()

        # SN domain warmup (first 10% epochs)
        if sn_domain_update_fn is not None and epoch < domain_update_epochs:
            sn_domain_update_fn(allow_resampling=True)

        optimizer.zero_grad(set_to_none=True)
        pred = model(x_train)
        loss = torch.mean((pred - y_train) ** 2)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
        optimizer.step()

        # Cheap refresh after step (mirrors sn_core.train behavior)
        if sn_domain_update_fn is not None and epoch < domain_update_epochs:
            sn_domain_update_fn(allow_resampling=False)

        scheduler.step()

        if (epoch == 0) or ((epoch + 1) % log_every == 0) or (epoch + 1 == epochs):
            tr = rmse(model, x_train, y_train)
            te = rmse(model, x_test, y_test)
            if te < best_test:
                best_test, best_epoch = te, epoch + 1
            lr_now = scheduler.get_last_lr()[0]
            print(
                f"[{label}] epoch {epoch+1:4d}/{epochs} "
                f"lr={lr_now:.2e} "
                f"train_RMSE={tr:.4e} test_RMSE={te:.4e} "
                f"(best={best_test:.4e} @ {best_epoch})"
            )

    elapsed = time.time() - t0
    final_train = rmse(model, x_train, y_train)
    final_test = rmse(model, x_test, y_test)

    return {
        "final_train_rmse": final_train,
        "final_test_rmse": final_test,
        "best_test_rmse": best_test,
        "best_epoch": float(best_epoch),
        "time_s": elapsed,
    }


# -----------------------------
# Main
# -----------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Barebones SN (PWL+warmup) vs KAN (cubic PCHIP) benchmark.")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", type=str, default=None, help="cpu | cuda | cuda:0 ... (default: auto)")
    p.add_argument("--dtype", type=str, default="float32", choices=["float32", "float64"])

    # Data
    p.add_argument("--input-dim", type=int, default=16, help="Input dimension d (in [0,1]^d).")
    p.add_argument("--n-train", type=int, default=1024)
    p.add_argument("--n-test", type=int, default=8192)

    # Architecture (two hidden layers, width <= 20 recommended)
    p.add_argument("--width", type=int, default=12, help="Hidden width for both SN and KAN (<= 20 recommended).")

    # SN spline knobs
    p.add_argument("--phi-knots", type=int, default=350, help="Number of knots for SN φ splines (PWL).")
    p.add_argument("--Phi-knots", type=int, default=350, help="Number of knots for SN Φ splines (PWL).")

    # KAN knot knob (default: auto matched)
    p.add_argument("--kan-knots", type=int, default=None, help="KAN knots per edge spline (default: auto-match params).")
    p.add_argument("--kan-xmin", type=float, default=0.0, help="KAN spline domain min.")
    p.add_argument("--kan-xmax", type=float, default=1.0, help="KAN spline domain max.")
    p.add_argument("--kan-init-scale", type=float, default=0.05, help="Stddev for KAN spline coeff init.")

    # Training
    p.add_argument("--epochs", type=int, default=4000)
    p.add_argument("--warmup-epochs", type=int, default=400, help="SN domain-update epochs (first 10%).")
    p.add_argument("--lr", type=float, default=2e-3)
    p.add_argument("--weight-decay", type=float, default=1e-6)
    p.add_argument("--max-grad-norm", type=float, default=10.0)
    p.add_argument("--log-every", type=int, default=400)

    return p.parse_args()


def main() -> None:
    args = parse_args()

    if args.width > 20:
        print(f"[warning] width={args.width} > 20 (you said to generally keep widths <= 20).")

    dtype = torch.float32 if args.dtype == "float32" else torch.float64

    if args.device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    set_global_seed(args.seed)

    # -------------------------
    # Make data (shared)
    # -------------------------
    x_train, y_train, x_test, y_test = make_train_test(
        input_dim=args.input_dim,
        n_train=args.n_train,
        n_test=args.n_test,
        seed=args.seed,
        device=device,
        dtype=dtype,
    )

    # -------------------------
    # Configure SN to be barebones
    # -------------------------
    CONFIG["seed"] = int(args.seed)
    CONFIG["use_theoretical_domains"] = True
    CONFIG["use_lateral_mixing"] = False
    CONFIG["use_residual_weights"] = False
    CONFIG["train_phi_codomain"] = False
    CONFIG["track_domain_violations"] = False
    CONFIG["verbose_domain_violations"] = False
    CONFIG["domain_safety_margin"] = 0.0

    # -------------------------
    # Build SN (PWL)
    # -------------------------
    sn_arch = [int(args.width), int(args.width), int(args.width)]  # 2 hidden blocks + final block (sum-to-scalar)
    sn_model = SprecherMultiLayerNetwork(
        input_dim=int(args.input_dim),
        architecture=sn_arch,
        final_dim=1,
        phi_knots=int(args.phi_knots),
        Phi_knots=int(args.Phi_knots),
        norm_type="none",
        norm_position="after",
        norm_skip_first=True,
        initialize_domains=True,
        domain_ranges=None,
        phi_spline_type="linear",
        Phi_spline_type="linear",
    ).to(device=device, dtype=dtype)

    sn_params = count_parameters(sn_model)

    # -------------------------
    # Build KAN (cubic PCHIP) with param matching
    # -------------------------
    if args.kan_knots is None:
        kan_knots = choose_kan_knots_to_match(sn_params, args.input_dim, args.width, min_knots=3)
    else:
        kan_knots = int(args.kan_knots)

    kan_model = KANNetPCHIP(
        input_dim=int(args.input_dim),
        width=int(args.width),
        num_knots=int(kan_knots),
        x_min=float(args.kan_xmin),
        x_max=float(args.kan_xmax),
        init_scale=float(args.kan_init_scale),
        dtype=dtype,
    ).to(device=device, dtype=dtype)

    kan_params = count_parameters(kan_model)

    # -------------------------
    # Report setup
    # -------------------------
    print("\n" + "=" * 72)
    print("Benchmark: SN (PWL + 10% domain warmup) vs KAN (cubic PCHIP)")
    print(f"Seed: {args.seed}")
    print(f"Device: {device}")
    print(f"Dtype: {args.dtype}")
    print(f"Data: n_train={args.n_train}, n_test={args.n_test}, input_dim={args.input_dim}")
    print(f"Epochs: {args.epochs} (SN warmup: {args.warmup_epochs})")
    print("-" * 72)
    print("SN config (barebones):")
    print(f"  architecture: {sn_arch}  (two hidden layers + final sum)")
    print(f"  φ knots: {args.phi_knots} (PWL), Φ knots: {args.Phi_knots} (PWL)")
    print(f"  params: {sn_params}")
    print("-" * 72)
    print("KAN config (barebones):")
    print(f"  dims: [{args.input_dim}, {args.width}, {args.width}, 1]  (two hidden layers)")
    print(f"  knots per edge spline: {kan_knots} (cubic PCHIP)")
    print(f"  spline domain: [{args.kan_xmin}, {args.kan_xmax}]")
    print(f"  params: {kan_params}")
    print("-" * 72)
    print(f"Param match: Δ = {abs(sn_params - kan_params)}  (SN - KAN = {sn_params - kan_params})")
    if sn_params < 2000 or kan_params < 2000:
        print("[warning] One of the models has < 2000 parameters. Consider increasing knot counts/width.")
    print("=" * 72 + "\n")

    # -------------------------
    # Train SN
    # -------------------------
    print("[SN] Training...")
    sn_stats = train_regression(
        model=sn_model,
        x_train=x_train,
        y_train=y_train,
        x_test=x_test,
        y_test=y_test,
        epochs=int(args.epochs),
        lr=float(args.lr),
        weight_decay=float(args.weight_decay),
        max_grad_norm=float(args.max_grad_norm),
        log_every=int(args.log_every),
        label="SN",
        domain_update_epochs=int(args.warmup_epochs),
        sn_domain_update_fn=sn_model.update_all_domains,
    )

    # -------------------------
    # Train KAN
    # -------------------------
    print("\n[KAN] Training...")
    kan_stats = train_regression(
        model=kan_model,
        x_train=x_train,
        y_train=y_train,
        x_test=x_test,
        y_test=y_test,
        epochs=int(args.epochs),
        lr=float(args.lr),
        weight_decay=float(args.weight_decay),
        max_grad_norm=float(args.max_grad_norm),
        log_every=int(args.log_every),
        label="KAN",
        domain_update_epochs=0,
        sn_domain_update_fn=None,
    )

    # -------------------------
    # Final summary (test RMSE is primary score)
    # -------------------------
    print("\n" + "=" * 72)
    print("FINAL RESULTS (test RMSE is the main score)")
    print(f"SN : final test RMSE = {sn_stats['final_test_rmse']:.6e}  (best = {sn_stats['best_test_rmse']:.6e} @ {int(sn_stats['best_epoch'])})")
    print(f"KAN: final test RMSE = {kan_stats['final_test_rmse']:.6e}  (best = {kan_stats['best_test_rmse']:.6e} @ {int(kan_stats['best_epoch'])})")
    print("-" * 72)
    print(f"Times: SN {sn_stats['time_s']:.1f}s, KAN {kan_stats['time_s']:.1f}s")
    print(f"Params: SN {sn_params}, KAN {kan_params} (Δ={abs(sn_params - kan_params)})")
    print("=" * 72 + "\n")


if __name__ == "__main__":
    main()