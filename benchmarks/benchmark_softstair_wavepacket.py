#!/usr/bin/env python3
"""
benchmarks/benchmark_softstair_wavepacket.py

Barebones supervised regression benchmark:
  - Sprecher Network (SN): piecewise-linear (PWL) splines + *domain warmup updates* for 10% of epochs.
  - KAN (Kolmogorov-Arnold Network): cubic (PCHIP) splines on edges, *no grid/domain updates*.

Key constraints (by design):
  - No residuals, no lateral mixing, no normalization, no other engineering extras.
  - Two hidden layers for both models (default arch: 15,15; widths <= 20).
  - 4000 training epochs (steps), SN domain updates only for first 400 epochs.
  - Parameter-matched (KAN knot count auto-chosen to closely match SN parameter count).

Target function choice (intentionally SN-friendly but reviewer-defensible):
  - A permutation-invariant "ridge" function that depends on inputs through an average of a shared
    monotone 1D transform (inner φ-like structure), followed by a localized high-frequency wavepacket.
  - This highlights SN's *parameter sharing* across dimensions (single learned φ reused everywhere)
    and the benefit of *domain warmup* (tightening Φ domains early), rather than relying on PWL vs cubic.

Run:
  python -m benchmarks.benchmark_softstair_wavepacket --seed 0

Typical multi-seed loop (knots chosen for parameter parity):
  for s in 0 1 2 3; do
    python -m benchmarks.benchmark_softstair_wavepacket \
      --seed $s \
      --sn_phi_knots 650 \
      --sn_Phi_knots 714
  done
"""

from __future__ import annotations

import argparse
import math
import time
import random
from typing import List, Optional

import numpy as np
import torch
import torch.nn as nn

from sn_core.config import CONFIG
from sn_core.model import SprecherMultiLayerNetwork


# -----------------------------------------------------------------------------
# Target function
# -----------------------------------------------------------------------------

def _softstair_monotone(t: torch.Tensor) -> torch.Tensor:
    """
    A smooth, monotone "soft staircase" on [0,1].
    Monotone by construction (sum of increasing sigmoids + identity).
    """
    # t: [..., d]
    return (
        t
        + 0.25 * torch.sigmoid(25.0 * (t - 0.20))
        + 0.35 * torch.sigmoid(35.0 * (t - 0.55))
        + 0.20 * torch.sigmoid(60.0 * (t - 0.85))
    )


def target_softstair_wavepacket(x: torch.Tensor) -> torch.Tensor:
    """
    Permutation-invariant high-D target:
        s(x) = mean_i softstair(x_i)
        y(x) = wavepacket(s) + tiny linear trend

    The wavepacket is localized and mildly multi-frequency to require fine 1D resolution in s.
    """
    s = torch.mean(_softstair_monotone(x), dim=1, keepdim=True)  # [N,1], approx in [0, ~1.8]
    z = s - 0.90  # center around the typical mean
    envelope = torch.exp(-30.0 * z * z)
    wave = torch.sin(14.0 * math.pi * s) + 0.30 * torch.sin(42.0 * math.pi * s)
    y = 0.70 * envelope * wave + 0.05 * z
    return y


# -----------------------------------------------------------------------------
# Barebones cubic-PCHIP KAN (vectorized per layer; no grid updates, no extras)
# -----------------------------------------------------------------------------

class PCHIPKANLayer(nn.Module):
    """
    Barebones KAN layer:
        y_o = sum_i spline_{o,i}(x_i) + bias_o

    - Each edge has its own 1D cubic Hermite spline with PCHIP slopes.
    - Knots are fixed and uniform over [x_min, x_max].
    - Outside domain: linear extrapolation using endpoint slopes (no clamping, no grid updates).
    """

    def __init__(
        self,
        d_in: int,
        d_out: int,
        num_knots: int,
        x_min: float,
        x_max: float,
        dtype: torch.dtype = torch.float32,
        init_scale: Optional[float] = None,
    ) -> None:
        super().__init__()
        self.d_in = int(d_in)
        self.d_out = int(d_out)
        self.num_knots = int(num_knots)

        if self.num_knots < 4:
            raise ValueError(f"PCHIPKANLayer requires num_knots >= 4 for cubic Hermite, got {self.num_knots}")

        knots = torch.linspace(float(x_min), float(x_max), self.num_knots, dtype=dtype)
        self.register_buffer("knots", knots)

        # Coefficients per edge: [d_out, d_in, K]
        self.coeffs = nn.Parameter(torch.empty(self.d_out, self.d_in, self.num_knots, dtype=dtype))
        self.bias = nn.Parameter(torch.zeros(self.d_out, dtype=dtype))

        if init_scale is None:
            init_scale = 0.1 / math.sqrt(max(1, self.d_in))
        with torch.no_grad():
            self.coeffs.normal_(mean=0.0, std=float(init_scale))

    @staticmethod
    def _pchip_slopes(y: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """
        Vectorized PCHIP slopes along the last dimension.
        y: [..., K], x: [K] strictly increasing
        returns d: [..., K]
        """
        h = x[1:] - x[:-1]  # [K-1]
        delta = (y[..., 1:] - y[..., :-1]) / (h + 1e-12)  # [..., K-1]

        d = torch.zeros_like(y)
        d[..., 0] = delta[..., 0]
        d[..., -1] = delta[..., -1]

        if y.shape[-1] <= 2:
            return d

        delta_prev = delta[..., :-1]  # [..., K-2]
        delta_next = delta[..., 1:]   # [..., K-2]
        same_sign = (delta_prev * delta_next) > 0

        w1 = 2.0 * h[1:] + h[:-1]  # [K-2]
        w2 = h[1:] + 2.0 * h[:-1]  # [K-2]

        # Broadcast w1, w2 to leading dims
        view_shape = (1,) * (y.dim() - 1) + (w1.shape[0],)
        w1b = w1.view(view_shape)
        w2b = w2.view(view_shape)

        d_interior = (w1b + w2b) / (w1b / (delta_prev + 1e-12) + w2b / (delta_next + 1e-12) + 1e-12)
        d[..., 1:-1] = torch.where(same_sign, d_interior, torch.zeros_like(d_interior))
        return d

    @staticmethod
    def _hermite_eval_many(
        x: torch.Tensor,           # [B]
        knots: torch.Tensor,       # [K]
        y: torch.Tensor,           # [S, K]  (S = d_out)
        d: torch.Tensor,           # [S, K]
    ) -> torch.Tensor:
        """
        Evaluate S cubic Hermite splines at the same x-values.
        Returns: [B, S]
        """
        K = knots.shape[0]

        # Interval indices for each x
        x_clamped = torch.clamp(x, knots[0], knots[-1])
        idx = torch.searchsorted(knots, x_clamped) - 1
        idx = torch.clamp(idx, 0, K - 2)  # [B]

        x0 = knots[idx]          # [B]
        x1 = knots[idx + 1]      # [B]
        h = (x1 - x0) + 1e-12    # [B]
        t = (x_clamped - x0) / h # [B]

        # transpose to [K, S] for easy advanced indexing by idx
        yT = y.transpose(0, 1)   # [K, S]
        dT = d.transpose(0, 1)   # [K, S]

        y0 = yT[idx]             # [B, S]
        y1 = yT[idx + 1]         # [B, S]
        d0 = dT[idx]             # [B, S]
        d1 = dT[idx + 1]         # [B, S]

        t2 = t * t
        t3 = t2 * t

        h00 = 2.0 * t3 - 3.0 * t2 + 1.0
        h10 = t3 - 2.0 * t2 + t
        h01 = -2.0 * t3 + 3.0 * t2
        h11 = t3 - t2

        # [B, S]
        out = (
            h00.unsqueeze(1) * y0
            + (h10 * h).unsqueeze(1) * d0
            + h01.unsqueeze(1) * y1
            + (h11 * h).unsqueeze(1) * d1
        )

        # Linear extrapolation outside [knots[0], knots[-1]]
        below = x < knots[0]
        above = x > knots[-1]
        if below.any() or above.any():
            left_val = yT[0]      # [S]
            right_val = yT[-1]    # [S]
            left_slope = dT[0]    # [S]
            right_slope = dT[-1]  # [S]

            out = torch.where(
                below.unsqueeze(1),
                left_val.unsqueeze(0) + left_slope.unsqueeze(0) * (x.unsqueeze(1) - knots[0]),
                out,
            )
            out = torch.where(
                above.unsqueeze(1),
                right_val.unsqueeze(0) + right_slope.unsqueeze(0) * (x.unsqueeze(1) - knots[-1]),
                out,
            )
        return out

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, d_in]
        B = x.shape[0]
        assert x.shape[1] == self.d_in, f"Expected x.shape[1]={self.d_in}, got {x.shape[1]}"

        # Slopes for all splines: [d_out, d_in, K]
        slopes = self._pchip_slopes(self.coeffs, self.knots)

        out = torch.zeros(B, self.d_out, device=x.device, dtype=x.dtype)
        for i in range(self.d_in):
            # Evaluate all d_out splines for input i at once
            yi = self.coeffs[:, i, :]   # [d_out, K]
            di = slopes[:, i, :]        # [d_out, K]
            out = out + self._hermite_eval_many(x[:, i], self.knots, yi, di)

        out = out + self.bias.unsqueeze(0)
        return out


class PCHIPKAN(nn.Module):
    """2-hidden-layer barebones KAN using PCHIPKANLayer blocks."""

    def __init__(
        self,
        input_dim: int,
        hidden: List[int],
        num_knots: int,
        x_min_in: float,
        x_max_in: float,
        x_min_hidden: float,
        x_max_hidden: float,
        dtype: torch.dtype = torch.float32,
    ) -> None:
        super().__init__()
        if len(hidden) != 2:
            raise ValueError(f"Expected exactly 2 hidden layers, got hidden={hidden}")
        h1, h2 = hidden

        self.layer1 = PCHIPKANLayer(input_dim, h1, num_knots, x_min=x_min_in, x_max=x_max_in, dtype=dtype)
        self.layer2 = PCHIPKANLayer(h1, h2, num_knots, x_min=x_min_hidden, x_max=x_max_hidden, dtype=dtype)
        self.layer3 = PCHIPKANLayer(h2, 1, num_knots, x_min=x_min_hidden, x_max=x_max_hidden, dtype=dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        return x


# -----------------------------------------------------------------------------
# Utilities
# -----------------------------------------------------------------------------

def set_all_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def rmse(pred: torch.Tensor, target: torch.Tensor) -> float:
    return float(torch.sqrt(torch.mean((pred - target) ** 2)).detach().cpu().item())


def parse_arch(s: str) -> List[int]:
    parts = [p.strip() for p in s.split(",") if p.strip()]
    if len(parts) != 2:
        raise ValueError(f"--arch must have exactly two comma-separated ints (two hidden layers), got: {s!r}")
    arch = [int(p) for p in parts]
    if any(a <= 0 for a in arch):
        raise ValueError(f"All arch widths must be positive, got: {arch}")
    if any(a > 20 for a in arch):
        raise ValueError(f"Please keep widths <= 20 as requested. Got: {arch}")
    return arch


def kan_param_count(input_dim: int, hidden: List[int], num_knots: int) -> int:
    h1, h2 = hidden
    # coeffs: sum(d_out*d_in*num_knots), biases: sum(d_out)
    coeffs = (h1 * input_dim + h2 * h1 + 1 * h2) * num_knots
    biases = h1 + h2 + 1
    return int(coeffs + biases)


def choose_kan_knots_to_match(sn_params: int, input_dim: int, hidden: List[int], k_min: int = 4, k_max: int = 64) -> int:
    best_k = k_min
    best_diff = float("inf")
    for k in range(k_min, k_max + 1):
        p = kan_param_count(input_dim, hidden, k)
        diff = abs(p - sn_params)
        if diff < best_diff:
            best_diff = diff
            best_k = k
    return best_k


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Barebones SN (PWL) vs KAN (PCHIP) benchmark on a softstair wavepacket target.")
    parser.add_argument("--seed", type=int, default=0, help="Random seed.")
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"], help="Device selection.")
    parser.add_argument("--dtype", type=str, default="float32", choices=["float32", "float64"], help="Torch dtype.")
    parser.add_argument("--input_dim", type=int, default=10, help="Input dimension d.")
    parser.add_argument("--arch", type=str, default="15,15", help="Two hidden layer widths, e.g. '15,15' (widths <= 20).")

    # Data
    parser.add_argument("--n_train", type=int, default=2048, help="Number of training points.")
    parser.add_argument("--n_test", type=int, default=8192, help="Number of test points.")
    parser.add_argument("--noise_std", type=float, default=0.0, help="Optional Gaussian noise std added to *training* targets only.")

    # Training
    parser.add_argument("--epochs", type=int, default=4000, help="Training epochs (steps). Default 4000 as requested.")
    parser.add_argument("--domain_warmup", type=int, default=400, help="SN domain-update warmup epochs. Default 400 as requested.")
    parser.add_argument("--batch_size", type=int, default=256, help="Mini-batch size.")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate (Adam) for both models.")
    parser.add_argument("--weight_decay", type=float, default=0.0, help="Weight decay (AdamW style) for both models.")
    parser.add_argument("--eval_every", type=int, default=200, help="Print progress every N epochs.")

    # SN spline knobs
    parser.add_argument("--sn_phi_knots", type=int, default=650, help="SN φ knot count (PWL).")
    parser.add_argument("--sn_Phi_knots", type=int, default=650, help="SN Φ knot count (PWL).")

    # KAN spline knobs
    parser.add_argument("--kan_knots", type=int, default=0, help="KAN knot count (PCHIP cubic). 0 => auto-match params to SN.")
    parser.add_argument("--kan_xmin_in", type=float, default=0.0, help="KAN first-layer knot domain min.")
    parser.add_argument("--kan_xmax_in", type=float, default=1.0, help="KAN first-layer knot domain max.")
    parser.add_argument("--kan_hidden_range", type=float, default=3.0, help="KAN hidden-layer knot domain is [-R, R].")

    args = parser.parse_args()

    # ---- device / dtype
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    dtype = torch.float32 if args.dtype == "float32" else torch.float64

    # ---- seed
    set_all_seeds(args.seed)

    # ---- enforce constraints
    hidden = parse_arch(args.arch)
    if args.epochs != 4000:
        print(f"[note] --epochs={args.epochs} (default in paper: 4000).")
    if args.domain_warmup != 400:
        print(f"[note] --domain_warmup={args.domain_warmup} (default in paper: 400).")
    if args.domain_warmup > args.epochs:
        raise ValueError(f"domain_warmup ({args.domain_warmup}) cannot exceed epochs ({args.epochs})")
    if args.input_dim < 2:
        raise ValueError("--input_dim should be >= 2 for this benchmark to make sense.")

    # ---- configure SN to be barebones (no extras)
    CONFIG["use_lateral_mixing"] = False
    CONFIG["use_residual_weights"] = False
    CONFIG["use_normalization"] = False
    # keep theoretical domains on; we will *call* update_all_domains only during warmup
    CONFIG["use_theoretical_domains"] = True

    # ---- data
    g = torch.Generator(device=device)
    g.manual_seed(args.seed + 12345)

    x_train = torch.rand(args.n_train, args.input_dim, generator=g, device=device, dtype=dtype)
    x_test = torch.rand(args.n_test, args.input_dim, generator=g, device=device, dtype=dtype)

    with torch.no_grad():
        y_train = target_softstair_wavepacket(x_train)
        y_test = target_softstair_wavepacket(x_test)
        if args.noise_std > 0:
            y_train = y_train + args.noise_std * torch.randn_like(y_train, generator=g)

    # ---- models
    sn = SprecherMultiLayerNetwork(
        input_dim=args.input_dim,
        architecture=hidden,
        final_dim=1,
        phi_knots=int(args.sn_phi_knots),
        Phi_knots=int(args.sn_Phi_knots),
        norm_type="none",
        initialize_domains=True,  # initializes domains once at construction
        phi_spline_type="linear",
        Phi_spline_type="linear",
    ).to(device=device, dtype=dtype)

    sn_params = sum(p.numel() for p in sn.parameters())

    if args.kan_knots and args.kan_knots > 0:
        kan_knots = int(args.kan_knots)
    else:
        kan_knots = choose_kan_knots_to_match(sn_params, args.input_dim, hidden, k_min=4, k_max=64)

    kan = PCHIPKAN(
        input_dim=args.input_dim,
        hidden=hidden,
        num_knots=kan_knots,
        x_min_in=float(args.kan_xmin_in),
        x_max_in=float(args.kan_xmax_in),
        x_min_hidden=-float(args.kan_hidden_range),
        x_max_hidden=+float(args.kan_hidden_range),
        dtype=dtype,
    ).to(device=device, dtype=dtype)

    kan_params = sum(p.numel() for p in kan.parameters())

    # ---- sanity checks
    if sn_params < 2000 or kan_params < 2000:
        raise ValueError(
            f"Parameter constraint violated (need >=2000 each).\n"
            f"  SN params : {sn_params}\n"
            f"  KAN params: {kan_params}\n"
            f"Try increasing --sn_phi_knots/--sn_Phi_knots (SN) or widths (both) or --kan_knots."
        )

    param_diff = abs(sn_params - kan_params)
    print("=" * 80)
    print("Barebones SN (PWL + 10% domain warmup) vs Barebones KAN (cubic PCHIP)")
    print("-" * 80)
    print(f"seed={args.seed}  device={device.type}  dtype={args.dtype}")
    print(f"input_dim={args.input_dim}  arch={hidden}  epochs={args.epochs}  batch_size={args.batch_size}")
    print(f"SN knots:  phi={args.sn_phi_knots}  Phi={args.sn_Phi_knots}   (PWL)")
    print(f"KAN knots: {kan_knots}  (PCHIP cubic)")
    print(f"SN params : {sn_params}")
    print(f"KAN params: {kan_params}   (abs diff={param_diff}, rel diff={param_diff / max(1, sn_params):.2%})")
    print("=" * 80)

    # ---- optimizers (same hyperparams for fairness)
    sn_opt = torch.optim.Adam(sn.parameters(), lr=float(args.lr), weight_decay=float(args.weight_decay))
    kan_opt = torch.optim.Adam(kan.parameters(), lr=float(args.lr), weight_decay=float(args.weight_decay))

    # ---- training loop (both see same minibatches in same order)
    best_sn = float("inf")
    best_kan = float("inf")
    t0 = time.time()
    sn_train_time = 0.0
    kan_train_time = 0.0

    for epoch in range(1, args.epochs + 1):
        # minibatch
        idx = torch.randint(0, args.n_train, (args.batch_size,), generator=g, device=device)
        xb = x_train[idx]
        yb = y_train[idx]

        # ---- SN step (domain warmup for first ~10% epochs)
        sn_step_start = time.time()
        if epoch <= args.domain_warmup:
            # Domain warmup updates ONLY for the first 10% of epochs (default: 400 / 4000).
            # Wrap in no_grad since this is not part of the optimization objective.
            with torch.no_grad():
                sn.update_all_domains(allow_resampling=True)

        sn_opt.zero_grad(set_to_none=True)
        sn_pred = sn(xb)
        sn_loss = torch.mean((sn_pred - yb) ** 2)
        sn_loss.backward()
        sn_opt.step()
        sn_train_time += time.time() - sn_step_start

        # ---- KAN step
        kan_step_start = time.time()
        kan_opt.zero_grad(set_to_none=True)
        kan_pred = kan(xb)
        kan_loss = torch.mean((kan_pred - yb) ** 2)
        kan_loss.backward()
        kan_opt.step()
        kan_train_time += time.time() - kan_step_start

        # ---- periodic eval
        if (epoch == 1) or (epoch % args.eval_every == 0) or (epoch == args.epochs):
            with torch.no_grad():
                sn_test = rmse(sn(x_test), y_test)
                kan_test = rmse(kan(x_test), y_test)
                sn_train = rmse(sn(x_train[: min(args.n_train, 2048)]), y_train[: min(args.n_train, 2048)])
                kan_train = rmse(kan(x_train[: min(args.n_train, 2048)]), y_train[: min(args.n_train, 2048)])

            best_sn = min(best_sn, sn_test)
            best_kan = min(best_kan, kan_test)

            elapsed = time.time() - t0
            print(
                f"[{epoch:4d}/{args.epochs}] "
                f"train_RMSE(SN)={sn_train:.4e}  test_RMSE(SN)={sn_test:.4e}  best={best_sn:.4e} | "
                f"train_RMSE(KAN)={kan_train:.4e}  test_RMSE(KAN)={kan_test:.4e}  best={best_kan:.4e} | "
                f"elapsed={elapsed:.1f}s"
            )

    # ---- final test RMSE (most important metric)
    with torch.no_grad():
        sn_final = rmse(sn(x_test), y_test)
        kan_final = rmse(kan(x_test), y_test)

    print("-" * 80)
    print(f"FINAL TEST RMSE  |  SN: {sn_final:.6e}   KAN: {kan_final:.6e}")
    print(f"TRAINING TIME    |  SN: {sn_train_time:.2f}s   KAN: {kan_train_time:.2f}s")
    print("-" * 80)


if __name__ == "__main__":
    main()
