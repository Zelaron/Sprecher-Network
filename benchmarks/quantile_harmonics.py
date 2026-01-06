"""
Benchmark: Quantile Harmonics (barebones PWL-SN vs barebones PCHIP-KAN)

Run (4 seeds):
for s in 0 1 2 3; do
  python -m benchmarks.quantile_harmonics --seed $s --phi_knots 500 --Phi_knots 500
done

What this compares:
- Sprecher Network (SN): piecewise-linear (PWL) splines, *with* domain updates for the first 10% of epochs.
- KAN: cubic PCHIP splines on each edge, *no* grid/domain updates.

Engineering extras intentionally OFF for both:
- no residuals
- no lateral mixing
- no normalization layers
- no codomain learning for Φ
"""

from __future__ import annotations

import argparse
import math
import random
import time
from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset


# ----------------------------
# Utilities
# ----------------------------
def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # no-op if no cuda


def pick_device(name: str) -> torch.device:
    name = (name or "auto").lower()
    if name == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        # MPS can exist on macOS builds
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    if name == "cuda":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if name == "mps":
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device("cpu")


def count_trainable_params(model: nn.Module) -> int:
    return int(sum(p.numel() for p in model.parameters() if p.requires_grad))


def rmse(pred: torch.Tensor, y: torch.Tensor) -> float:
    pred = pred.view(-1)
    y = y.view(-1)
    return float(torch.sqrt(torch.mean((pred - y) ** 2)).item())


def parse_arch(arch_str: str) -> List[int]:
    # e.g. "20,20"
    parts = [p.strip() for p in arch_str.split(",") if p.strip()]
    if not parts:
        raise ValueError("--arch must be a comma-separated list like '20,20'")
    arch = [int(p) for p in parts]
    if any(a <= 0 for a in arch):
        raise ValueError("--arch must contain positive integers")
    return arch


# ----------------------------
# Target function + dataset
# ----------------------------
def quantile_harmonics_target(xy: torch.Tensor) -> torch.Tensor:
    """
    x,y in [0,1]. Output is a smooth but "regime-switching" harmonic mixture.

    Intuition: along x we have multiple smoothly-gated regimes (soft quantile bins),
    and inside each regime we use a different harmonic mixture whose amplitude is
    modulated by y. This creates structured nonlinearity where weight sharing across
    shifted channels can be advantageous.
    """
    x = xy[:, 0]
    y = xy[:, 1]

    # Soft quantile/bin gates along x
    # (centers roughly at 0.1, 0.3, 0.55, 0.8)
    centers = torch.tensor([0.10, 0.30, 0.55, 0.80], device=xy.device, dtype=xy.dtype)
    sigma = 0.055
    gates_unnorm = torch.exp(-0.5 * ((x[:, None] - centers[None, :]) / sigma) ** 2)
    gates = gates_unnorm / (gates_unnorm.sum(dim=1, keepdim=True) + 1e-12)  # [N,4]

    # Regime-specific harmonic mixtures (vary frequency & phase across regimes)
    freqs = torch.tensor([5.0, 9.0, 13.0, 17.0], device=xy.device, dtype=xy.dtype)
    phases = torch.tensor([0.2, -0.7, 1.1, -1.5], device=xy.device, dtype=xy.dtype)

    # Amplitude modulation by y (keeps the task genuinely 2D)
    amp = 0.65 + 0.35 * torch.cos(
        2.0 * math.pi * (2.0 * y + 0.15 * torch.sin(2.0 * math.pi * x))
    )

    # Harmonics: sin(2π f_k x + phase_k) + small secondary cosine
    h1 = torch.sin(2.0 * math.pi * (freqs[None, :] * x[:, None]) + phases[None, :])
    h2 = 0.35 * torch.cos(
        2.0 * math.pi * ((freqs + 2.0)[None, :] * x[:, None]) - 0.5 * phases[None, :]
    )
    mix = (h1 + h2) * amp[:, None]  # [N,4]

    # Combine regimes
    out = (gates * mix).sum(dim=1)

    # Add a small smooth interaction term to prevent pure "mixture-of-1D"
    # (still easy for SN-like compositional models)
    out = out + 0.15 * torch.sin(2.0 * math.pi * (x + y)) * torch.exp(-6.0 * (y - 0.5) ** 2)

    return out[:, None]  # [N,1]


@dataclass
class DatasetSplits:
    x_train: torch.Tensor
    y_train: torch.Tensor
    x_val: torch.Tensor
    y_val: torch.Tensor
    x_test: torch.Tensor
    y_test: torch.Tensor


def make_dataset(
    seed: int,
    n_train: int,
    n_val: int,
    n_test: int,
    device: torch.device,
    dtype: torch.dtype = torch.float32,
) -> DatasetSplits:
    gen = torch.Generator(device="cpu").manual_seed(seed)

    def _sample(n: int) -> torch.Tensor:
        # uniform in [0,1]^2
        return torch.rand((n, 2), generator=gen, dtype=dtype)

    x_train = _sample(n_train)
    x_val = _sample(n_val)
    x_test = _sample(n_test)

    with torch.no_grad():
        y_train = quantile_harmonics_target(x_train)
        y_val = quantile_harmonics_target(x_val)
        y_test = quantile_harmonics_target(x_test)

    return DatasetSplits(
        x_train=x_train.to(device),
        y_train=y_train.to(device),
        x_val=x_val.to(device),
        y_val=y_val.to(device),
        x_test=x_test.to(device),
        y_test=y_test.to(device),
    )


# ----------------------------
# Barebones PCHIP cubic spline (vectorized)
# ----------------------------
def _pchip_slopes(y: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    """
    Vectorized PCHIP slopes for y[..., K] with knots x[K] (strictly increasing).
    Returns d[..., K].

    This is a differentiable torch implementation (no SciPy dependency).
    """
    K = y.shape[-1]
    if K < 2:
        return torch.zeros_like(y)

    h = x[1:] - x[:-1]  # [K-1]
    # Avoid divide-by-zero
    h_safe = h + 1e-12

    delta = (y[..., 1:] - y[..., :-1]) / h_safe  # [..., K-1]

    d = torch.zeros_like(y)

    if K == 2:
        d[..., 0] = delta[..., 0]
        d[..., 1] = delta[..., 0]
        return d

    # Interior slopes
    h0 = h[:-1]  # [K-2]
    h1 = h[1:]  # [K-2]

    w1 = 2.0 * h1 + h0
    w2 = h1 + 2.0 * h0

    delta0 = delta[..., :-1]  # [..., K-2]
    delta1 = delta[..., 1:]  # [..., K-2]
    same_sign = (delta0 * delta1) > 0

    # Weighted harmonic mean
    denom = (w1 / (delta0 + 1e-12)) + (w2 / (delta1 + 1e-12))
    d_int = (w1 + w2) / (denom + 1e-12)

    d[..., 1:-1] = torch.where(same_sign, d_int, torch.zeros_like(d_int))

    # Endpoint slopes
    d0 = ((2.0 * h[0] + h[1]) * delta[..., 0] - h[0] * delta[..., 1]) / (h[0] + h[1] + 1e-12)
    dN = ((2.0 * h[-1] + h[-2]) * delta[..., -1] - h[-1] * delta[..., -2]) / (
        h[-1] + h[-2] + 1e-12
    )

    def _limit_endpoint(di: torch.Tensor, deltai: torch.Tensor) -> torch.Tensor:
        # If slope has opposite sign, set to 0
        di = torch.where((di * deltai) <= 0, torch.zeros_like(di), di)
        # If too large, clamp to 3*delta
        di = torch.where(di.abs() > 3.0 * deltai.abs(), 3.0 * deltai, di)
        return di

    d[..., 0] = _limit_endpoint(d0, delta[..., 0])
    d[..., -1] = _limit_endpoint(dN, delta[..., -1])

    return d


class KANLayer(nn.Module):
    """
    Barebones KAN layer:
      y_j = sum_i spline_{j,i}(x_i) + b_j
    where each spline_{j,i} is a cubic PCHIP spline with shared knot x-grid.
    """

    def __init__(self, d_in: int, d_out: int, num_knots: int, x_min: float = 0.0, x_max: float = 1.0):
        super().__init__()
        self.d_in = int(d_in)
        self.d_out = int(d_out)
        self.num_knots = int(num_knots)
        self.x_min = float(x_min)
        self.x_max = float(x_max)

        # Shared knots across all edges for this layer
        knots = torch.linspace(self.x_min, self.x_max, self.num_knots)
        self.register_buffer("knots", knots)

        # Edge coefficients: [d_out, d_in, K]
        self.coeffs = nn.Parameter(torch.zeros(self.d_out, self.d_in, self.num_knots))
        nn.init.normal_(self.coeffs, mean=0.0, std=0.2)

        # Bias: [d_out]
        self.bias = nn.Parameter(torch.zeros(self.d_out))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, d_in]
        B = x.shape[0]
        device = x.device
        dtype = x.dtype

        # Compute slopes for all splines in this layer: [d_out, d_in, K]
        slopes = _pchip_slopes(self.coeffs, self.knots)  # broadcasting over leading dims

        h = (self.x_max - self.x_min) / (self.num_knots - 1)
        h_t = torch.tensor(h, device=device, dtype=dtype)

        out = torch.zeros((B, self.d_out), device=device, dtype=dtype)

        # Loop over input dims only (<=20), vectorize over d_out
        for i in range(self.d_in):
            xi = x[:, i]
            xi_clamped = torch.clamp(xi, self.x_min, self.x_max)

            idx = torch.clamp(((xi_clamped - self.x_min) / h_t).floor().to(torch.long), 0, self.num_knots - 2)
            x0 = self.x_min + idx.to(dtype) * h_t
            t = (xi_clamped - x0) / h_t  # [B]

            # Basis
            t2 = t * t
            t3 = t2 * t
            h00 = 2 * t3 - 3 * t2 + 1
            h10 = t3 - 2 * t2 + t
            h01 = -2 * t3 + 3 * t2
            h11 = t3 - t2

            # Gather y0,y1,d0,d1 for all outputs at once
            # y: [d_out, K]
            y_i = self.coeffs[:, i, :]
            d_i = slopes[:, i, :]

            idx_expand = idx.unsqueeze(0).expand(self.d_out, B)
            y0 = torch.gather(y_i, 1, idx_expand)
            y1 = torch.gather(y_i, 1, idx_expand + 1)
            d0 = torch.gather(d_i, 1, idx_expand)
            d1 = torch.gather(d_i, 1, idx_expand + 1)

            # Evaluate: [d_out, B]
            yi_eval = (
                h00.unsqueeze(0) * y0
                + h10.unsqueeze(0) * h_t * d0
                + h01.unsqueeze(0) * y1
                + h11.unsqueeze(0) * h_t * d1
            )

            # Extrapolation (rare here because inputs in [0,1])
            left = xi < self.x_min
            right = xi > self.x_max
            if left.any():
                y_left = y_i[:, 0:1] + d_i[:, 0:1] * (xi[left].unsqueeze(0) - self.x_min)
                yi_eval[:, left] = y_left
            if right.any():
                y_right = y_i[:, -1:] + d_i[:, -1:] * (xi[right].unsqueeze(0) - self.x_max)
                yi_eval[:, right] = y_right

            out = out + yi_eval.transpose(0, 1)  # [B, d_out]

        out = out + self.bias.unsqueeze(0)
        return out


class BarebonesPchipKAN(nn.Module):
    """
    2-hidden-layer KAN with cubic PCHIP splines on edges.
    Architecture: 2 -> width -> width -> 1
    """

    def __init__(self, width: int, num_knots: int, x_min: float = 0.0, x_max: float = 1.0):
        super().__init__()
        self.width = int(width)
        self.num_knots = int(num_knots)
        self.layer1 = KANLayer(2, self.width, self.num_knots, x_min=x_min, x_max=x_max)
        self.layer2 = KANLayer(self.width, self.width, self.num_knots, x_min=x_min, x_max=x_max)
        self.layer3 = KANLayer(self.width, 1, self.num_knots, x_min=x_min, x_max=x_max)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        return x


# ----------------------------
# Parameter matching
# ----------------------------
@dataclass
class KanMatch:
    width: int
    knots: int
    params: int
    rel_diff: float
    score: float


def kan_param_count_formula(input_dim: int, width: int, num_knots: int) -> int:
    """
    Barebones 2-hidden-layer KAN:
      input_dim -> width -> width -> 1
    Each edge has num_knots parameters, each layer has biases.
    """
    w = int(width)
    K = int(num_knots)
    # layer1: input_dim->w
    p1 = input_dim * w * K + w
    # layer2: w->w
    p2 = w * w * K + w
    # layer3: w->1
    p3 = w * 1 * K + 1
    return int(p1 + p2 + p3)


def choose_kan_to_match(
    target_params: int,
    min_params: int,
    tol: float,
    width_max: int = 20,
    knots_min: int = 8,
    knots_max: int = 300,
    preferred_width: int = 12,
    preferred_knots: int = 16,
) -> KanMatch:
    """
    Search over (width, knots) to match KAN params to target_params.

    Tie-breaking is designed to be KAN-friendly:
      - stay close in params
      - prefer moderate width and moderate knot count (avoid tiny K or tiny width)
    """
    best: Optional[KanMatch] = None
    for w in range(4, width_max + 1):
        for K in range(knots_min, knots_max + 1):
            p = kan_param_count_formula(2, w, K)
            if p < min_params:
                continue
            rel = abs(p - target_params) / max(1, target_params)
            if rel > tol:
                continue

            # Secondary score: prefer (w,K) near a moderate sweet spot
            score = rel
            score += 0.0005 * abs(w - preferred_width)
            score += 0.0003 * abs(K - preferred_knots)

            cand = KanMatch(width=w, knots=K, params=p, rel_diff=rel, score=score)
            if best is None or cand.score < best.score:
                best = cand

    if best is None:
        # Diagnostic: find closest feasible above min_params
        closest: Optional[KanMatch] = None
        for w in range(4, width_max + 1):
            for K in range(knots_min, knots_max + 1):
                p = kan_param_count_formula(2, w, K)
                if p < min_params:
                    continue
                rel = abs(p - target_params) / max(1, target_params)
                score = rel + 0.0005 * abs(w - preferred_width) + 0.0003 * abs(K - preferred_knots)
                cand = KanMatch(width=w, knots=K, params=p, rel_diff=rel, score=score)
                if closest is None or cand.score < closest.score:
                    closest = cand

        msg = (
            f"Could not match KAN params to SN within tol={tol:.3f} while keeping KAN>=min_params.\n"
            f"  target_params={target_params}\n"
        )
        if closest is not None:
            msg += (
                f"Closest feasible KAN: width={closest.width}, knots={closest.knots}, "
                f"params={closest.params} (rel_diff={closest.rel_diff:.3%}).\n"
            )
        msg += "Try increasing --phi_knots/--Phi_knots (raises SN params) or increasing --tol.\n"
        raise ValueError(msg)

    return best


def min_phiPhi_for_sn_min_params(
    sn_arch: List[int],
    input_dim: int,
    min_params: int,
    train_phi_codomain: bool,
) -> int:
    """
    Compute the minimum (phi_knots + Phi_knots) needed so the SN hits min_params,
    using the exact SN param formula for SprecherMultiLayerNetwork with final_dim=1
    and no residual/lateral/codomain.

    SN param count (trainable) approximately:
      lambdas: input_dim + arch[0] + arch[1] + ... + arch[L-2]
      splines: L * (phi_knots + Phi_knots)
      etas:    L
      output_scale/output_bias: 2
      codomain params for Phi: 2*L if enabled
    """
    L = len(sn_arch) if len(sn_arch) > 0 else 1
    sum_din = input_dim + sum(sn_arch[:-1]) if len(sn_arch) >= 2 else input_dim
    eta_params = L
    codomain_params = (2 * L) if train_phi_codomain else 0
    base = sum_din + eta_params + codomain_params + 2  # +2 for output_scale/bias
    need = max(0, min_params - base)
    return int(math.ceil(need / max(1, L)))


# ----------------------------
# Training
# ----------------------------
@dataclass
class TrainResult:
    best_val_rmse: float
    test_rmse_at_best: float
    final_test_rmse: float
    training_time: float


def train_regressor(
    model: nn.Module,
    *,
    train_loader: DataLoader,
    x_val: torch.Tensor,
    y_val: torch.Tensor,
    x_test: torch.Tensor,
    y_test: torch.Tensor,
    epochs: int,
    lr: float,
    weight_decay: float,
    print_every: int,
    domain_warmup_epochs: int = 0,
    sn_core_for_domains: Optional[nn.Module] = None,
) -> TrainResult:
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    best_state: Optional[Dict[str, torch.Tensor]] = None
    best_val = float("inf")
    best_test = float("inf")

    start_time = time.time()

    for epoch in range(1, epochs + 1):
        model.train()

        if sn_core_for_domains is not None and epoch <= domain_warmup_epochs:
            # Domain update at start of epoch
            try:
                sn_core_for_domains.update_all_domains(allow_resampling=True, force_resample=False)
            except TypeError:
                # Some versions may not accept these args
                sn_core_for_domains.update_all_domains()

        for xb, yb in train_loader:
            optimizer.zero_grad(set_to_none=True)
            pred = model(xb)
            loss = F.mse_loss(pred, yb)
            loss.backward()
            optimizer.step()

        if (epoch % print_every) == 0 or epoch == 1 or epoch == epochs:
            model.eval()
            with torch.no_grad():
                val_pred = model(x_val)
                test_pred = model(x_test)
                val_rmse = rmse(val_pred, y_val)
                test_rmse = rmse(test_pred, y_test)

            if val_rmse < best_val:
                best_val = val_rmse
                best_test = test_rmse
                best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}

            print(
                f"  epoch {epoch:4d}/{epochs} | val_RMSE={val_rmse:.6f} "
                f"| test_RMSE={test_rmse:.6f} | best_val={best_val:.6f}"
            )

    training_time = time.time() - start_time

    # Final and best
    model.eval()
    with torch.no_grad():
        final_test = rmse(model(x_test), y_test)

    if best_state is not None:
        model.load_state_dict(best_state)
        model.eval()
        with torch.no_grad():
            best_test_eval = rmse(model(x_test), y_test)
    else:
        best_test_eval = final_test

    return TrainResult(best_val_rmse=best_val, test_rmse_at_best=best_test_eval, final_test_rmse=final_test, training_time=training_time)


# ----------------------------
# Main
# ----------------------------
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Quantile Harmonics: barebones PWL-SN vs PCHIP-KAN")

    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda", "mps"])

    # Data
    p.add_argument("--n_train", type=int, default=4096)
    p.add_argument("--n_val", type=int, default=2048)
    p.add_argument("--n_test", type=int, default=2048)

    # Training
    p.add_argument("--epochs", type=int, default=4000)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight_decay", type=float, default=0.0)
    p.add_argument("--batch_size", type=int, default=1024)
    p.add_argument("--print_every", type=int, default=200)

    # SN config
    p.add_argument("--arch", type=str, default="20,20", help="SN hidden architecture, e.g. '20,20'")
    p.add_argument("--phi_knots", type=int, default=500)
    p.add_argument("--Phi_knots", type=int, default=500)
    p.add_argument("--domain_warmup_frac", type=float, default=0.10)

    # Fairness / param constraints
    p.add_argument("--min_params", type=int, default=2000, help="Require >= this many trainable params per model")
    p.add_argument("--tol", type=float, default=0.02, help="Relative param mismatch tolerance for KAN vs SN")
    p.add_argument("--kan_width", type=int, default=None, help="Optional: override auto-matched KAN width")
    p.add_argument("--kan_knots", type=int, default=None, help="Optional: override auto-matched KAN knots")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    device = pick_device(args.device)
    set_seed(args.seed)

    sn_arch = parse_arch(args.arch)

    # ------------------------
    # Disable SN "extras" globally
    # ------------------------
    from sn_core.config import CONFIG

    CONFIG["seed"] = int(args.seed)
    CONFIG["use_residual_weights"] = False
    CONFIG["use_lateral_mixing"] = False
    CONFIG["use_normalization"] = False
    CONFIG["train_phi_codomain"] = False  # no Φ codomain learning (barebones)
    # Keep theoretical domains ON (domain warmup uses this)
    CONFIG["use_theoretical_domains"] = True

    # ------------------------
    # Dataset
    # ------------------------
    data = make_dataset(
        seed=args.seed,
        n_train=args.n_train,
        n_val=args.n_val,
        n_test=args.n_test,
        device=device,
        dtype=torch.float32,
    )

    train_ds = TensorDataset(data.x_train, data.y_train)
    # Use identical (deterministic) shuffles for SN and KAN for extra fairness.
    g_sn = torch.Generator(device="cpu").manual_seed(int(args.seed) + 12345)
    g_kan = torch.Generator(device="cpu").manual_seed(int(args.seed) + 12345)
    train_loader_sn = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, generator=g_sn)
    train_loader_kan = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, generator=g_kan)

    # ------------------------
    # Build SN (PWL) and count params
    # ------------------------
    from sn_core.model import SprecherMultiLayerNetwork

    sn = SprecherMultiLayerNetwork(
        input_dim=2,
        architecture=sn_arch,
        final_dim=1,
        phi_knots=int(args.phi_knots),
        Phi_knots=int(args.Phi_knots),
        norm_type="none",
        phi_spline_type="linear",
        Phi_spline_type="linear",
        initialize_domains=True,
    ).to(device)

    sn_params = count_trainable_params(sn)

    # Enforce min params (per original benchmark spec)
    if sn_params < args.min_params:
        min_sum = min_phiPhi_for_sn_min_params(
            sn_arch=sn_arch, input_dim=2, min_params=args.min_params, train_phi_codomain=False
        )
        suggested = int(math.ceil(min_sum / 2))
        raise ValueError(
            "SN parameter budget is below --min_params.\n"
            f"  SN params: {sn_params}\n"
            f"  min_params: {args.min_params}\n"
            f"  With arch={sn_arch} and final_dim=1, you need (phi_knots + Phi_knots) >= {min_sum}.\n"
            f"  Suggested: --phi_knots {suggested} --Phi_knots {suggested}\n"
        )

    # Re-seed before constructing KAN so its init is independent of SN internals.
    set_seed(int(args.seed))

    # ------------------------
    # Choose / build KAN to match params
    # ------------------------
    if args.kan_width is not None and args.kan_knots is not None:
        kan_w = int(args.kan_width)
        kan_K = int(args.kan_knots)
        kan_params_formula = kan_param_count_formula(2, kan_w, kan_K)
        rel = abs(kan_params_formula - sn_params) / sn_params
        if kan_params_formula < args.min_params or rel > args.tol:
            raise ValueError(
                "Manual KAN override violates constraints.\n"
                f"  SN params: {sn_params}\n"
                f"  KAN params (formula): {kan_params_formula}\n"
                f"  min_params: {args.min_params}\n"
                f"  rel_diff: {rel:.3%} (tol={args.tol:.3%})\n"
            )
        match = KanMatch(width=kan_w, knots=kan_K, params=kan_params_formula, rel_diff=rel, score=rel)
    else:
        match = choose_kan_to_match(
            target_params=sn_params,
            min_params=args.min_params,
            tol=float(args.tol),
            width_max=20,
            knots_min=8,
            knots_max=300,
            preferred_width=12,
            preferred_knots=16,
        )

    kan = BarebonesPchipKAN(width=match.width, num_knots=match.knots, x_min=0.0, x_max=1.0).to(device)
    kan_params = count_trainable_params(kan)

    # Final sanity checks (use actual param counts)
    rel_actual = abs(kan_params - sn_params) / sn_params
    if kan_params < args.min_params or rel_actual > args.tol:
        raise ValueError(
            "Param-count constraint violated (actual counts).\n"
            f"  SN={sn_params}, KAN={kan_params}\n"
            f"  min_params={args.min_params}, tol={args.tol}\n"
            "Try increasing --phi_knots/--Phi_knots or loosening --tol.\n"
        )

    # ------------------------
    # Print run header
    # ------------------------
    warmup_epochs = int(round(args.epochs * float(args.domain_warmup_frac)))
    print("\n========== Quantile Harmonics Benchmark ==========")
    print(f"seed: {args.seed} | device: {device}")
    print(f"epochs: {args.epochs} | batch_size: {args.batch_size} | lr: {args.lr}")
    print(f"dataset: n_train={args.n_train}, n_val={args.n_val}, n_test={args.n_test}")
    print("SN (PWL):")
    print(f"  arch={sn_arch} | phi_knots={args.phi_knots} | Phi_knots={args.Phi_knots}")
    print(f"  domain warmup epochs: {warmup_epochs} (={args.domain_warmup_frac:.0%})")
    print(f"  params: {sn_params}")
    print("KAN (PCHIP cubic):")
    print(f"  width={match.width} | knots={match.knots}")
    print(f"  params: {kan_params} (rel_diff vs SN: {rel_actual:.3%})")
    print("Extras OFF:")
    print("  SN residuals: OFF | SN lateral mixing: OFF | SN norm: OFF | SN Φ codomain: OFF")
    print("  KAN: no grid/domain updates; PCHIP cubic per edge")
    print("==================================================\n")

    # ------------------------
    # Train both
    # ------------------------
    print("Training SN ...")
    sn_res = train_regressor(
        sn,
        train_loader=train_loader_sn,
        x_val=data.x_val,
        y_val=data.y_val,
        x_test=data.x_test,
        y_test=data.y_test,
        epochs=args.epochs,
        lr=args.lr,
        weight_decay=args.weight_decay,
        print_every=max(1, args.print_every),
        domain_warmup_epochs=warmup_epochs,
        sn_core_for_domains=sn,
    )

    print("\nTraining KAN ...")
    kan_res = train_regressor(
        kan,
        train_loader=train_loader_kan,
        x_val=data.x_val,
        y_val=data.y_val,
        x_test=data.x_test,
        y_test=data.y_test,
        epochs=args.epochs,
        lr=args.lr,
        weight_decay=args.weight_decay,
        print_every=max(1, args.print_every),
        domain_warmup_epochs=0,
        sn_core_for_domains=None,
    )

    # ------------------------
    # Final report
    # ------------------------
    print("\n==================== RESULTS ====================")
    print(f"SN  best_val_RMSE: {sn_res.best_val_rmse:.6f} | test_RMSE@best: {sn_res.test_rmse_at_best:.6f}")
    print(f"KAN best_val_RMSE: {kan_res.best_val_rmse:.6f} | test_RMSE@best: {kan_res.test_rmse_at_best:.6f}")
    print("------------------------------------------------")
    print(f"SN  final_test_RMSE: {sn_res.final_test_rmse:.6f}")
    print(f"KAN final_test_RMSE: {kan_res.final_test_rmse:.6f}")
    print("------------------------------------------------")
    print(f"SN  training_time: {sn_res.training_time:.2f}s")
    print(f"KAN training_time: {kan_res.training_time:.2f}s")
    print("=================================================\n")


if __name__ == "__main__":
    main()
