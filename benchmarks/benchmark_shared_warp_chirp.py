#!/usr/bin/env python3
"""benchmarks/benchmark_shared_warp_chirp.py

Barebones benchmark: **piecewise-linear Sprecher Network (SN)** vs **cubic PCHIP KAN**.

Design goals (per request):
  - SN: PWL splines (φ and Φ are piecewise-linear) + **domain warmup** for ~10% epochs.
  - KAN: **cubic Hermite splines with PCHIP slopes** on every edge (no grid updates).
  - No extras: no residuals, no lateral mixing, no normalization, no grid updates.
  - Fairness: parameter-matched (within ~1% by default; often exact).

Target function rationale (why SN can win fairly):
  The target is *permutation-symmetric* and has a *shared monotone per-coordinate warp*:

      s(x) = mean_i sigmoid(alpha * (x_i - 0.5))
      y(x) = "chirpy"(s(x))

  This structure is friendly to SN's inductive bias: a single shared monotone φ can
  capture the warp once, instead of relearning it separately on every edge.
  A parameter-matched KAN must spread its parameters across many edge-splines, so
  it often has far fewer knots per edge (even with cubic splines), which can hurt
  sample-efficiency and test RMSE on this family.

How to run (example):

  # run 4 seeds
  for s in 0 1 2 3; do
    python -m benchmarks.benchmark_shared_warp_chirp \
      --seed $s \
      --phi_knots 400 --Phi_knots 400 \
      --width 16 --input_dim 10
  done

Notes:
  - Default training uses 4000 epochs, with SN domain updates for the first 400 epochs.
  - The printed "TEST RMSE" is the main score.
"""

from __future__ import annotations

import argparse
import math
import time
from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn

# Project-local: Sprecher Network core
from sn_core import CONFIG, SprecherMultiLayerNetwork


# -----------------------------
# Repro / small utilities
# -----------------------------


def set_seed(seed: int) -> None:
    seed = int(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def count_trainable_params(model: nn.Module) -> int:
    return int(sum(p.numel() for p in model.parameters() if p.requires_grad))


def rmse(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return torch.sqrt(torch.mean((pred - target) ** 2))


# -----------------------------
# Target function
# -----------------------------


def shared_warp_chirp(x: torch.Tensor, *, alpha: float = 18.0) -> torch.Tensor:
    """Permutation-symmetric regression target on [0,1]^d.

    x: (B,d) in [0,1]
    returns: (B,1)
    """
    # Shared monotone warp per coordinate
    z = torch.sigmoid(float(alpha) * (x - 0.5))  # (B,d) in (0,1)
    s = z.mean(dim=1, keepdim=True)  # (B,1) in (0,1)

    # Smooth, non-stationary 1D "chirp" of s
    two_pi = 2.0 * math.pi
    y = (
        torch.sin(two_pi * (6.0 * s + 5.0 * s**2))
        + 0.35 * torch.cos(two_pi * (3.0 * s))
        + 0.10 * (s - 0.5)
    )

    # Keep targets in a modest range to stabilize optimization
    return y


@dataclass
class DatasetSplit:
    x_train: torch.Tensor
    y_train: torch.Tensor
    x_test: torch.Tensor
    y_test: torch.Tensor


def make_dataset(
    *,
    seed: int,
    input_dim: int,
    n_train: int,
    n_test: int,
    device: torch.device,
    dtype: torch.dtype,
) -> DatasetSplit:
    """Generate a fixed train/test split for the benchmark."""
    g = torch.Generator(device="cpu")
    g.manual_seed(int(seed))

    x_train = torch.rand(int(n_train), int(input_dim), generator=g, dtype=dtype)
    x_test = torch.rand(int(n_test), int(input_dim), generator=g, dtype=dtype)

    # Move to device after generation (generator is CPU-only for full determinism)
    x_train = x_train.to(device)
    x_test = x_test.to(device)

    with torch.no_grad():
        y_train = shared_warp_chirp(x_train).to(device=device, dtype=dtype)
        y_test = shared_warp_chirp(x_test).to(device=device, dtype=dtype)

    return DatasetSplit(x_train=x_train, y_train=y_train, x_test=x_test, y_test=y_test)


# -----------------------------
# KAN: cubic PCHIP splines (vectorized)
# -----------------------------


class PCHIPCubicSplineBank(nn.Module):
    """Vectorized bank of 1D cubic Hermite splines with PCHIP slopes.

    This module represents MANY independent splines that all share the same
    knot x-positions (uniform over [x_min, x_max]), but each has its own
    learnable y-values at knots.

    Shapes:
      - knots: (K)
      - coeffs: (n_splines, K)
    """

    def __init__(self, n_splines: int, num_knots: int, *, x_min: float, x_max: float, dtype: torch.dtype):
        super().__init__()
        self.n_splines = int(n_splines)
        self.num_knots = int(num_knots)
        self.register_buffer("knots", torch.linspace(float(x_min), float(x_max), self.num_knots, dtype=dtype))

        # Learnable y-values at knots
        self.coeffs = nn.Parameter(torch.zeros(self.n_splines, self.num_knots, dtype=dtype))

    @staticmethod
    def _pchip_slopes(y: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """Vectorized PCHIP slopes (Fritsch–Carlson style).

        This is the shape-preserving derivative choice used by PCHIP.

        y: (..., K)
        x: (K)
        returns: (..., K)
        """

        K = int(y.shape[-1])
        if K < 2:
            return torch.zeros_like(y)

        h = x[1:] - x[:-1]  # (K-1)
        delta = (y[..., 1:] - y[..., :-1]) / (h + 1e-12)  # (..., K-1)

        d = torch.zeros_like(y)

        if K == 2:
            d[..., 0] = delta[..., 0]
            d[..., 1] = delta[..., 0]
            return d

        # ---- interior slopes ----
        # For k=1..K-2
        delta_prev = delta[..., :-1]  # (..., K-2)
        delta_next = delta[..., 1:]   # (..., K-2)
        same_sign = (delta_prev * delta_next) > 0

        h_prev = h[:-1]  # (K-2)
        h_next = h[1:]   # (K-2)
        w1 = 2.0 * h_next + h_prev
        w2 = h_next + 2.0 * h_prev
        denom = (w1 / (delta_prev + 1e-12)) + (w2 / (delta_next + 1e-12))
        d_int = (w1 + w2) / (denom + 1e-12)
        d[..., 1:-1] = torch.where(same_sign, d_int, torch.zeros_like(d_int))

        # ---- endpoint slopes ----
        h0 = h[0]
        h1 = h[1]
        delta0 = delta[..., 0]
        delta1 = delta[..., 1]

        d0 = ((2.0 * h0 + h1) * delta0 - h0 * delta1) / (h0 + h1 + 1e-12)
        # Enforce shape preservation
        d0 = torch.where(torch.sign(d0) != torch.sign(delta0), torch.zeros_like(d0), d0)
        d0 = torch.where(
            (torch.sign(delta0) != torch.sign(delta1)) & (torch.abs(d0) > 3.0 * torch.abs(delta0)),
            3.0 * delta0,
            d0,
        )
        d[..., 0] = d0

        hn1 = h[-1]
        hn2 = h[-2]
        deltan1 = delta[..., -1]
        deltan2 = delta[..., -2]

        dn = ((2.0 * hn1 + hn2) * deltan1 - hn1 * deltan2) / (hn1 + hn2 + 1e-12)
        dn = torch.where(torch.sign(dn) != torch.sign(deltan1), torch.zeros_like(dn), dn)
        dn = torch.where(
            (torch.sign(deltan1) != torch.sign(deltan2)) & (torch.abs(dn) > 3.0 * torch.abs(deltan1)),
            3.0 * deltan1,
            dn,
        )
        d[..., -1] = dn

        return d

    def forward(self, xq: torch.Tensor) -> torch.Tensor:
        """Evaluate all splines at query points.

        xq: (B, n_splines)
        returns: (B, n_splines)
        """
        knots = self.knots
        y = self.coeffs
        d = self._pchip_slopes(y, knots)

        x0_dom = knots[0]
        x1_dom = knots[-1]

        below = xq < x0_dom
        above = xq > x1_dom

        xq_clamped = torch.clamp(xq, x0_dom, x1_dom)

        # Interval indices
        idx = torch.searchsorted(knots, xq_clamped) - 1
        idx = torch.clamp(idx, 0, knots.numel() - 2)

        # Gather knot endpoints for each query
        x0 = knots[idx]
        x1 = knots[idx + 1]
        h = (x1 - x0) + 1e-12
        t = (xq_clamped - x0) / h

        h00 = (2 * t**3) - (3 * t**2) + 1
        h10 = (t**3) - (2 * t**2) + t
        h01 = (-2 * t**3) + (3 * t**2)
        h11 = (t**3) - (t**2)

        # Gather y0,y1,d0,d1 per spline
        # y: (n_splines,K). We need y0: (B,n_splines)
        gather_idx = idx.transpose(0, 1)  # (n_splines,B)
        y0 = y.gather(dim=1, index=gather_idx).transpose(0, 1)
        y1 = y.gather(dim=1, index=(gather_idx + 1)).transpose(0, 1)
        d0 = d.gather(dim=1, index=gather_idx).transpose(0, 1)
        d1 = d.gather(dim=1, index=(gather_idx + 1)).transpose(0, 1)

        interp = h00 * y0 + h10 * h * d0 + h01 * y1 + h11 * h * d1

        # Linear extrapolation
        left_val = y[:, 0].unsqueeze(0)
        right_val = y[:, -1].unsqueeze(0)
        left_slope = d[:, 0].unsqueeze(0)
        right_slope = d[:, -1].unsqueeze(0)

        out = interp
        out = torch.where(below, left_val + left_slope * (xq - x0_dom), out)
        out = torch.where(above, right_val + right_slope * (xq - x1_dom), out)
        return out


class PCHIPKANLayer(nn.Module):
    """Barebones KAN layer with cubic PCHIP splines on edges.

    Each output node o computes:
        y_o = sum_i spline_{o,i}(x_i) + bias_o

    No residual/base linear term, no grid updates.
    """

    def __init__(
        self,
        d_in: int,
        d_out: int,
        num_knots: int,
        *,
        x_min: float,
        x_max: float,
        dtype: torch.dtype,
        init_scale: Optional[float] = None,
    ) -> None:
        super().__init__()
        self.d_in = int(d_in)
        self.d_out = int(d_out)
        self.num_knots = int(num_knots)

        if self.num_knots < 4:
            raise ValueError("PCHIP cubic splines need at least 4 knots for a meaningful cubic shape.")

        if init_scale is None:
            init_scale = 0.10 / math.sqrt(max(1, self.d_in))

        # One spline per edge => d_out * d_in splines
        self._bank = PCHIPCubicSplineBank(
            n_splines=self.d_out * self.d_in,
            num_knots=self.num_knots,
            x_min=float(x_min),
            x_max=float(x_max),
            dtype=dtype,
        )

        # Initialize coeffs
        with torch.no_grad():
            self._bank.coeffs.normal_(0.0, float(init_scale))

        self.bias = nn.Parameter(torch.zeros(self.d_out, dtype=dtype))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B,d_in)
        B = x.shape[0]
        # Expand x to per-edge queries: (B, d_out*d_in)
        # For each output o, we repeat the full input vector.
        x_rep = x.unsqueeze(1).expand(B, self.d_out, self.d_in).reshape(B, self.d_out * self.d_in)

        y_edge = self._bank(x_rep)  # (B, d_out*d_in)
        y_edge = y_edge.view(B, self.d_out, self.d_in)
        y = y_edge.sum(dim=2) + self.bias.view(1, -1)
        return y


class PCHIPKANNet(nn.Module):
    """Stacked barebones KAN with per-layer knot counts."""

    def __init__(
        self,
        dims: List[int],
        knots_per_layer: List[int],
        *,
        x_min: float,
        x_max: float,
        dtype: torch.dtype,
    ) -> None:
        super().__init__()
        if len(dims) < 2:
            raise ValueError("dims must have at least input and output")
        if len(knots_per_layer) != (len(dims) - 1):
            raise ValueError("knots_per_layer must have len(dims)-1")

        layers: List[nn.Module] = []
        for li in range(len(dims) - 1):
            layers.append(
                PCHIPKANLayer(
                    dims[li],
                    dims[li + 1],
                    num_knots=int(knots_per_layer[li]),
                    x_min=float(x_min),
                    x_max=float(x_max),
                    dtype=dtype,
                )
            )
        self.layers = nn.ModuleList(layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x)
        return x


# -----------------------------
# SN builder
# -----------------------------


def build_barebones_sn(
    *,
    input_dim: int,
    width: int,
    phi_knots: int,
    Phi_knots: int,
    device: torch.device,
    dtype: torch.dtype,
) -> SprecherMultiLayerNetwork:
    """Two hidden layers + final block (sums to scalar)."""

    # Hard-disable all SN-only extras.
    CONFIG["use_residual_weights"] = False
    CONFIG["use_lateral_mixing"] = False
    CONFIG["use_normalization"] = False
    CONFIG["train_phi_codomain"] = False
    CONFIG["use_advanced_scheduler"] = False
    CONFIG["low_memory_mode"] = False
    CONFIG["track_domain_violations"] = False
    CONFIG["debug_domains"] = False

    # For fairness, keep theoretical domains on (but we control when updates occur).
    CONFIG["use_theoretical_domains"] = True
    CONFIG["domain_safety_margin"] = 0.0

    arch = [int(width), int(width), int(width)]  # 2 hidden blocks + final block width
    model = SprecherMultiLayerNetwork(
        input_dim=int(input_dim),
        architecture=arch,
        final_dim=1,
        phi_knots=int(phi_knots),
        Phi_knots=int(Phi_knots),
        norm_type="none",
        norm_position="after",
        norm_skip_first=True,
        initialize_domains=False,
        domain_ranges=None,
        phi_spline_type="linear",  # PWL
        Phi_spline_type="linear",  # PWL
        phi_spline_order=None,
        Phi_spline_order=None,
    ).to(device=device, dtype=dtype)
    return model


# -----------------------------
# Parameter matching
# -----------------------------


def kan_param_count_for(dims: List[int], knots_per_layer: List[int]) -> int:
    """Closed-form param count for our barebones KAN."""
    edges = 0
    biases = 0
    for li in range(len(dims) - 1):
        edges += int(dims[li]) * int(dims[li + 1])
        biases += int(dims[li + 1])
    # Each edge spline has `knots` parameters, but knots can differ per layer.
    spline_params = 0
    for li in range(len(dims) - 1):
        spline_params += int(knots_per_layer[li]) * (int(dims[li]) * int(dims[li + 1]))
    return int(spline_params + biases)


def choose_kan_knots_to_match(
    *,
    target_params: int,
    dims: List[int],
    min_knots: int = 4,
    max_knots: int = 16,
    prefer_small_variation: bool = True,
) -> List[int]:
    """Search small integer knot-counts per layer to match target_params."""

    L = len(dims) - 1
    if L <= 0:
        raise ValueError("dims must have at least 2 entries")

    # Brute force is fine: L=3 (input->h1->h2->out) and knots range is small.
    best: Optional[Tuple[int, List[int]]] = None  # (abs_diff, knots)
    best_score: Optional[Tuple[int, int]] = None  # (abs_diff, variation_penalty)

    ranges = [range(int(min_knots), int(max_knots) + 1) for _ in range(L)]
    for k0 in ranges[0]:
        for k1 in ranges[1] if L >= 2 else [k0]:
            for k2 in ranges[2] if L >= 3 else [k1]:
                ks = [k0]
                if L >= 2:
                    ks.append(k1)
                if L >= 3:
                    ks.append(k2)
                if L > 3:
                    # Not expected in this benchmark, but handle generically:
                    ks.extend([k2] * (L - 3))

                p = kan_param_count_for(dims, ks)
                abs_diff = abs(int(p) - int(target_params))

                if prefer_small_variation:
                    var_pen = (max(ks) - min(ks))
                else:
                    var_pen = 0

                score = (abs_diff, var_pen)
                if best is None or score < best_score:  # type: ignore[operator]
                    best = (abs_diff, ks)
                    best_score = score

    assert best is not None
    return best[1]


# -----------------------------
# Training
# -----------------------------


@torch.no_grad()
def eval_rmse(model: nn.Module, x: torch.Tensor, y: torch.Tensor) -> float:
    model.eval()
    pred = model(x)
    return float(rmse(pred, y).cpu())


def train_fullbatch_regression(
    *,
    model: nn.Module,
    x_train: torch.Tensor,
    y_train: torch.Tensor,
    epochs: int,
    lr: float,
    weight_decay: float,
    grad_clip: float,
    domain_warmup_epochs: int = 0,
    is_sn: bool = False,
    log_every: int = 500,
) -> None:
    opt = torch.optim.Adam(model.parameters(), lr=float(lr), weight_decay=float(weight_decay))
    mse = torch.nn.MSELoss()

    model.train()
    t0 = time.time()
    for ep in range(1, int(epochs) + 1):
        if is_sn and ep <= int(domain_warmup_epochs):
            # Tighten domains + (for Φ) allow resampling during warmup.
            # This is the ONLY SN-specific "extra" we allow.
            model.update_all_domains(allow_resampling=True)

        opt.zero_grad(set_to_none=True)
        pred = model(x_train)
        loss = mse(pred, y_train)
        loss.backward()

        if float(grad_clip) and float(grad_clip) > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=float(grad_clip))

        opt.step()

        if is_sn and ep <= int(domain_warmup_epochs):
            # Cheaper refresh (no resampling) after the update.
            model.update_all_domains(allow_resampling=False)

        if ep == 1 or ep % int(log_every) == 0 or ep == int(epochs):
            dt = time.time() - t0
            print(f"  ep {ep:4d}/{epochs} | train_mse={float(loss.detach().cpu()):.4e} | elapsed={dt:7.1f}s")


# -----------------------------
# CLI
# -----------------------------


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Barebones SN (PWL+domain warmup) vs KAN (cubic PCHIP) benchmark.")

    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda", "mps"])

    # Data
    p.add_argument("--input_dim", type=int, default=10)
    p.add_argument("--train_size", type=int, default=1024)
    p.add_argument("--test_size", type=int, default=8192)

    # Training
    p.add_argument("--epochs", type=int, default=4000)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight_decay", type=float, default=0.0)
    p.add_argument("--grad_clip", type=float, default=1.0)
    p.add_argument("--log_every", type=int, default=500)

    # SN (PWL)
    p.add_argument("--width", type=int, default=16)
    p.add_argument("--phi_knots", type=int, default=400)
    p.add_argument("--Phi_knots", type=int, default=400)
    p.add_argument("--sn_domain_warmup", type=int, default=400, help="#epochs with SN domain updates (default 400)")

    # KAN (cubic PCHIP)
    p.add_argument(
        "--kan_knots",
        type=str,
        default=None,
        help=(
            "KAN knots per layer. Either a single int (e.g. '6') or a comma list for "
            "[in->h1, h1->h2, h2->out] (e.g. '5,6,5'). If omitted, we auto-match params to SN."
        ),
    )
    p.add_argument("--kan_xmin", type=float, default=-2.0)
    p.add_argument("--kan_xmax", type=float, default=2.0)
    p.add_argument("--kan_kmin", type=int, default=4)
    p.add_argument("--kan_kmax", type=int, default=16)

    return p.parse_args()


def _resolve_device(device_flag: str) -> torch.device:
    if device_flag == "cpu":
        return torch.device("cpu")
    if device_flag == "cuda":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device_flag == "mps":
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available() and torch.backends.mps.is_built():
            return torch.device("mps")
        return torch.device("cpu")
    # auto
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available() and torch.backends.mps.is_built():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def _parse_kan_knots(s: Optional[str], num_layers: int) -> Optional[List[int]]:
    if s is None:
        return None
    s = str(s).strip()
    if "," not in s:
        k = int(s)
        return [k] * num_layers
    parts = [p.strip() for p in s.split(",") if p.strip()]
    ks = [int(p) for p in parts]
    if len(ks) != num_layers:
        raise ValueError(f"--kan_knots expects {num_layers} values, got {len(ks)}")
    return ks


def main() -> None:
    args = parse_args()

    if int(args.epochs) != 4000:
        print(f"[note] epochs is {args.epochs}; request asked for 4000.")
    if int(args.sn_domain_warmup) != 400:
        print(f"[note] sn_domain_warmup is {args.sn_domain_warmup}; request asked for 400.")

    if int(args.width) > 20:
        raise ValueError("Please keep --width <= 20 for this benchmark.")

    set_seed(args.seed)
    device = _resolve_device(args.device)
    dtype = torch.float32

    print("\n=== Benchmark: SN (PWL + domain warmup) vs KAN (cubic PCHIP) ===")
    print(f"seed={args.seed} | device={device} | dtype={dtype}")
    print(f"data: input_dim={args.input_dim} | train={args.train_size} | test={args.test_size}")
    print(f"train: epochs={args.epochs} | lr={args.lr:.2e} | wd={args.weight_decay:.2e} | clip={args.grad_clip}")
    print(f"SN:  width={args.width} | phi_knots={args.phi_knots} | Phi_knots={args.Phi_knots} | domain_warmup={args.sn_domain_warmup}")

    # Data
    data = make_dataset(
        seed=args.seed,
        input_dim=int(args.input_dim),
        n_train=int(args.train_size),
        n_test=int(args.test_size),
        device=device,
        dtype=dtype,
    )

    # ---------------- Build SN ----------------
    sn = build_barebones_sn(
        input_dim=int(args.input_dim),
        width=int(args.width),
        phi_knots=int(args.phi_knots),
        Phi_knots=int(args.Phi_knots),
        device=device,
        dtype=dtype,
    )
    sn_params = count_trainable_params(sn)

    # ---------------- Build KAN (param-matched) ----------------
    kan_dims = [int(args.input_dim), int(args.width), int(args.width), 1]
    num_layers = len(kan_dims) - 1
    kan_knots_user = _parse_kan_knots(args.kan_knots, num_layers)
    if kan_knots_user is None:
        kan_knots = choose_kan_knots_to_match(
            target_params=sn_params,
            dims=kan_dims,
            min_knots=int(args.kan_kmin),
            max_knots=int(args.kan_kmax),
            prefer_small_variation=True,
        )
        auto_str = "(auto-matched)"
    else:
        kan_knots = kan_knots_user
        auto_str = "(user)"

    kan = PCHIPKANNet(
        dims=kan_dims,
        knots_per_layer=kan_knots,
        x_min=float(args.kan_xmin),
        x_max=float(args.kan_xmax),
        dtype=dtype,
    ).to(device=device, dtype=dtype)
    kan_params = count_trainable_params(kan)

    # Checks for the requested constraints
    if sn_params < 2000 or kan_params < 2000:
        raise RuntimeError(
            f"Parameter floor violated: sn_params={sn_params}, kan_params={kan_params}. "
            "Increase --phi_knots/--Phi_knots and/or --width."
        )

    diff = abs(sn_params - kan_params)
    rel = diff / max(1, min(sn_params, kan_params))

    print(f"KAN: width={args.width} | knots_per_layer={kan_knots} {auto_str} | x_domain=[{args.kan_xmin},{args.kan_xmax}]")
    print(f"params: SN={sn_params} | KAN={kan_params} | abs_diff={diff} | rel_diff={100*rel:.2f}%")

    # ---------------- Train & Eval ----------------
    print("\n--- Training SN (PWL, domain warmup) ---")
    t0 = time.time()
    train_fullbatch_regression(
        model=sn,
        x_train=data.x_train,
        y_train=data.y_train,
        epochs=int(args.epochs),
        lr=float(args.lr),
        weight_decay=float(args.weight_decay),
        grad_clip=float(args.grad_clip),
        domain_warmup_epochs=int(args.sn_domain_warmup),
        is_sn=True,
        log_every=int(args.log_every),
    )
    sn_time = time.time() - t0

    print("\n--- Training KAN (cubic PCHIP) ---")
    t0 = time.time()
    train_fullbatch_regression(
        model=kan,
        x_train=data.x_train,
        y_train=data.y_train,
        epochs=int(args.epochs),
        lr=float(args.lr),
        weight_decay=float(args.weight_decay),
        grad_clip=float(args.grad_clip),
        domain_warmup_epochs=0,
        is_sn=False,
        log_every=int(args.log_every),
    )
    kan_time = time.time() - t0

    sn_test = eval_rmse(sn, data.x_test, data.y_test)
    kan_test = eval_rmse(kan, data.x_test, data.y_test)

    sn_train = eval_rmse(sn, data.x_train, data.y_train)
    kan_train = eval_rmse(kan, data.x_train, data.y_train)

    print("\n=== RESULTS ===")
    print(f"SN  train RMSE: {sn_train:.6f} | test RMSE: {sn_test:.6f} | time: {sn_time:.1f}s")
    print(f"KAN train RMSE: {kan_train:.6f} | test RMSE: {kan_test:.6f} | time: {kan_time:.1f}s")
    print(f"(lower is better)  TEST RMSE ratio KAN/SN: {kan_test / max(1e-12, sn_test):.3f}")


if __name__ == "__main__":
    main()