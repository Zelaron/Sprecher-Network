#!/usr/bin/env python3
"""
benchmarks/kan_sn_oscillatory_headshift_bench.py

Fair(ish) baseline benchmark: Sprecher Network (SN) vs KAN on an oscillatory,
head-shifted synthetic regression task.

Rules enforced in this script:
  - SN: domain warmup only (domains can resample for ~10% of epochs), no other extras.
  - KAN: fixed cubic splines (PCHIP-style cubic Hermite), no grid updates / residuals / norms.
  - Parameter matching (SN vs KAN) via --equalize_params and --prefer_leq.

The script prints a copy-friendly RESULT line including test RMSE.

Run (example, 4 seeds):
for s in 0 1 2 3; do
  python -m benchmarks.kan_sn_oscillatory_headshift_bench \
    --dims 12 --heads 64 --train_size 1024 --test_size 50000 \
    --epochs 4000 --device cpu --seed $s \
    --sn_arch 8,8 --kan_arch 8,8 \
    --kan_num_knots 4 \
    --equalize_params --prefer_leq \
    --sn_freeze_domains_after 400 --sn_domain_margin 0.01
done
"""
from __future__ import annotations

import argparse
import math
import time
from dataclasses import dataclass
from typing import List, Tuple, Optional

import numpy as np
import torch
import torch.nn as nn

# IMPORTANT: This benchmark must not import other scripts from benchmarks/.
from sn_core.config import CONFIG
from sn_core.model import SprecherMultiLayerNetwork


# -----------------------------
# Utilities
# -----------------------------
def set_global_seeds(seed: int) -> None:
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def parse_arch(s: str) -> List[int]:
    s = (s or "").strip()
    if not s:
        return []
    return [int(x.strip()) for x in s.split(",") if x.strip()]


def count_trainable_params(model: nn.Module) -> int:
    return sum(int(p.numel()) for p in model.parameters() if p.requires_grad)


def rmse_all(yhat: torch.Tensor, y: torch.Tensor) -> float:
    # RMSE over all (sample, head) entries
    mse = torch.mean((yhat - y) ** 2)
    return float(torch.sqrt(mse).item())


# -----------------------------
# Dataset
# -----------------------------
@dataclass
class OscillatoryHeadShiftTask:
    """
    Synthetic mapping x -> y (vector of length heads).

    Inputs:
        x in [0, 1]^dims
    Outputs:
        bounded (sum of sin/cos), with head-dependent phase shifts.

    This task is intentionally *not* the same as DenseHeadShift or MonoIndex.
    The key structure is: a shared oscillatory "template" plus a head-dependent
    phase shift; SN's built-in q-shift inductive bias should help here.
    """
    dims: int
    heads: int
    alpha: float = 0.35            # head-phase shift scale
    omega: float = 6.0 * math.pi   # base angular frequency
    seed: int = 20270809           # fixed seed so the target function is the same across runs

    def __post_init__(self) -> None:
        g = torch.Generator()
        g.manual_seed(self.seed)

        # Two random but fixed projection directions (unit vectors)
        w1 = torch.randn(self.dims, generator=g)
        w2 = torch.randn(self.dims, generator=g)
        w1 = w1 / (w1.norm() + 1e-12)
        w2 = w2 / (w2.norm() + 1e-12)

        # A small random mixing matrix for extra structure
        M = torch.randn(self.dims, 3, generator=g) / math.sqrt(self.dims)

        self._w1 = w1
        self._w2 = w2
        self._M = M

    @property
    def input_dim(self) -> int:
        return self.dims

    @property
    def output_dim(self) -> int:
        return self.heads

    def evaluate(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (N, dims) in [0,1]
        returns y: (N, heads)
        """
        # Center inputs for nicer projections
        xc = x - 0.5  # roughly in [-0.5, 0.5]

        # Two scalar projections + a 3D mix for additional non-monoindex structure
        u = (xc * self._w1.to(x.device, x.dtype)).sum(dim=1)               # (N,)
        v = (xc * self._w2.to(x.device, x.dtype)).sum(dim=1)               # (N,)
        z = xc @ self._M.to(x.device, x.dtype)                              # (N,3)

        # Head index normalized to [-0.5, 0.5]
        q = torch.arange(self.heads, device=x.device, dtype=x.dtype)
        if self.heads > 1:
            q0 = (q - 0.5 * (self.heads - 1)) / (self.heads - 1)           # [-0.5,0.5]
        else:
            q0 = q * 0.0
        shift = self.alpha * q0                                            # (heads,)

        # Oscillatory components with head-dependent phase shifts
        # Shape details: u[:,None] is (N,1), shift[None,:] is (1,heads)
        phase1 = self.omega * (u[:, None] + shift[None, :])
        phase2 = 0.5 * self.omega * (v[:, None] - 0.7 * shift[None, :])

        # Mild amplitude modulation from z to avoid being a pure 1D monoindex
        amp = 0.75 + 0.25 * torch.tanh(2.5 * z[:, 0:1])                     # (N,1)

        y = (
            amp * torch.sin(phase1)
            + 0.35 * torch.cos(2.0 * phase2)
            + 0.15 * torch.sin(3.0 * phase1 + 0.25 * z[:, 1:2])
            + 0.10 * torch.cos(phase2 + 0.35 * z[:, 2:3])
        )

        # y is already bounded; no tanh needed.
        return y

    def sample(self, n: int, device: str) -> Tuple[torch.Tensor, torch.Tensor]:
        x = torch.rand(n, self.dims, device=device)
        y = self.evaluate(x)
        return x, y


# -----------------------------
# KAN (cubic splines, no extras)
# -----------------------------
class CubicPchipKANLayer(nn.Module):
    """
    A single KAN layer:
        out_j = sum_i spline_{i,j}(x_i) + bias_j

    Each spline_{i,j} is a *fixed-knot* cubic Hermite spline. Slopes are
    computed from knot values via a PCHIP-style rule to reduce overshoot.

    NOTE: We intentionally avoid torch.searchsorted on column slices (non-contiguous),
    which can trigger warnings and extra copies. With uniform knots we can index
    segments using arithmetic.
    """
    def __init__(
        self,
        d_in: int,
        d_out: int,
        num_knots: int,
        domain_min: float = -2.0,
        domain_max: float = 2.0,
        init_scale: float = 0.01,
    ) -> None:
        super().__init__()
        if num_knots < 4:
            raise ValueError("For this benchmark, require num_knots >= 4 for cubic splines.")
        if not (domain_max > domain_min):
            raise ValueError("domain_max must be > domain_min")

        self.d_in = int(d_in)
        self.d_out = int(d_out)
        self.num_knots = int(num_knots)
        self.domain_min = float(domain_min)
        self.domain_max = float(domain_max)

        # Learnable knot values per edge: (d_in, d_out, K)
        self.y = nn.Parameter(init_scale * torch.randn(self.d_in, self.d_out, self.num_knots))
        self.bias = nn.Parameter(torch.zeros(self.d_out))

        # Uniform knot spacing
        h = (self.domain_max - self.domain_min) / (self.num_knots - 1)
        self.register_buffer("_h", torch.tensor(h, dtype=torch.float32), persistent=False)

    @property
    def h(self) -> torch.Tensor:
        return self._h

    def _pchip_slopes(self, y: torch.Tensor) -> torch.Tensor:
        """
        Compute PCHIP-style slopes for uniform knots.
        y: (..., K)
        returns m: (..., K) slopes dy/dx at knots
        """
        K = y.shape[-1]
        h = self.h.to(y.device, y.dtype)
        delta = (y[..., 1:] - y[..., :-1]) / h  # (..., K-1)
        m = torch.zeros_like(y)

        # Interior slopes: harmonic mean when neighboring slopes have same sign
        d0 = delta[..., :-1]  # (..., K-2)
        d1 = delta[..., 1:]   # (..., K-2)
        same = (d0 * d1) > 0

        denom = d0 + d1
        # Avoid 0/0; if denom is ~0, set hm to 0.
        hm = 2.0 * d0 * d1 / torch.where(denom.abs() < 1e-12, torch.ones_like(denom), denom)
        hm = torch.where(denom.abs() < 1e-12, torch.zeros_like(hm), hm)

        m[..., 1:-1] = torch.where(same, hm, torch.zeros_like(hm))

        # Endpoint slopes (standard PCHIP end conditions for uniform spacing)
        m0 = (3.0 * delta[..., 0] - delta[..., 1]) / 2.0
        mN = (3.0 * delta[..., -1] - delta[..., -2]) / 2.0

        # If endpoint slope has wrong sign, set to 0
        m0 = torch.where(m0 * delta[..., 0] <= 0, torch.zeros_like(m0), m0)
        mN = torch.where(mN * delta[..., -1] <= 0, torch.zeros_like(mN), mN)

        # If slopes change sign across first/last interval, limit magnitude
        cond0 = (delta[..., 0] * delta[..., 1] < 0) & (m0.abs() > (3.0 * delta[..., 0]).abs())
        m0 = torch.where(cond0, 3.0 * delta[..., 0], m0)

        condN = (delta[..., -1] * delta[..., -2] < 0) & (mN.abs() > (3.0 * delta[..., -1]).abs())
        mN = torch.where(condN, 3.0 * delta[..., -1], mN)

        m[..., 0] = m0
        m[..., -1] = mN
        return m

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, d_in)
        returns: (B, d_out)
        """
        if x.ndim != 2 or x.shape[1] != self.d_in:
            raise ValueError(f"Expected x shape (B,{self.d_in}), got {tuple(x.shape)}")

        B = x.shape[0]
        dtype = x.dtype
        device = x.device

        # Clamp to spline domain
        x = torch.clamp(x, self.domain_min, self.domain_max)

        # Uniform segment indexing (no searchsorted, no contiguity warnings)
        h = self.h.to(device=device, dtype=dtype)
        t = (x - self.domain_min) / h                                 # (B, d_in)
        idx = torch.floor(t).to(torch.long)                           # (B, d_in)
        idx = torch.clamp(idx, 0, self.num_knots - 2)
        u = (t - idx.to(dtype)).unsqueeze(2)                          # (B, d_in, 1), in [0,1]

        # Compute slopes from knot values
        y = self.y.to(device=device, dtype=dtype)                     # (d_in, d_out, K)
        m = self._pchip_slopes(y)                                     # (d_in, d_out, K)

        # Gather y0,y1,m0,m1 for each (B, d_in) segment
        # Expand params with a batch dim so gather works
        y_exp = y.unsqueeze(0).expand(B, -1, -1, -1)                  # (B, d_in, d_out, K)
        m_exp = m.unsqueeze(0).expand(B, -1, -1, -1)                  # (B, d_in, d_out, K)

        idx0 = idx.unsqueeze(2).unsqueeze(3).expand(-1, -1, self.d_out, 1)
        idx1 = (idx + 1).unsqueeze(2).unsqueeze(3).expand(-1, -1, self.d_out, 1)

        y0 = torch.gather(y_exp, 3, idx0).squeeze(3)                  # (B, d_in, d_out)
        y1 = torch.gather(y_exp, 3, idx1).squeeze(3)                  # (B, d_in, d_out)
        m0 = torch.gather(m_exp, 3, idx0).squeeze(3)                  # (B, d_in, d_out)
        m1 = torch.gather(m_exp, 3, idx1).squeeze(3)                  # (B, d_in, d_out)

        # Cubic Hermite basis
        u2 = u * u
        u3 = u2 * u
        h00 = 2.0 * u3 - 3.0 * u2 + 1.0
        h10 = u3 - 2.0 * u2 + u
        h01 = -2.0 * u3 + 3.0 * u2
        h11 = u3 - u2

        # Interpolate per edge, then sum over d_in
        # m are dy/dx, so multiply by h to get the tangent contribution in y units.
        out_edges = h00 * y0 + h10 * (h * m0) + h01 * y1 + h11 * (h * m1)   # (B, d_in, d_out)
        out = out_edges.sum(dim=1) + self.bias.to(device=device, dtype=dtype)
        return out


class CubicPchipKAN(nn.Module):
    def __init__(
        self,
        input_dim: int,
        architecture: List[int],
        output_dim: int,
        num_knots: int,
        domain_min: float = -2.0,
        domain_max: float = 2.0,
        init_scale: float = 0.01,
    ) -> None:
        super().__init__()
        sizes = [int(input_dim)] + [int(x) for x in architecture] + [int(output_dim)]
        layers: List[nn.Module] = []
        for d_in, d_out in zip(sizes[:-1], sizes[1:]):
            layers.append(
                CubicPchipKANLayer(
                    d_in=d_in,
                    d_out=d_out,
                    num_knots=num_knots,
                    domain_min=domain_min,
                    domain_max=domain_max,
                    init_scale=init_scale,
                )
            )
        self.layers = nn.ModuleList(layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x)
        return x


def kan_param_count(dims: int, heads: int, arch: List[int], num_knots: int) -> int:
    sizes = [dims] + arch + [heads]
    total = 0
    for d_in, d_out in zip(sizes[:-1], sizes[1:]):
        total += d_in * d_out * num_knots  # spline knot values per edge
        total += d_out                     # bias per output unit
    return total


# -----------------------------
# Training loops (full-batch)
# -----------------------------
@torch.no_grad()
def _maybe_update_sn_domains(sn: SprecherMultiLayerNetwork, allow_resampling: bool) -> None:
    # The SN class has update_all_domains(allow_resampling, force_resample)
    sn.update_all_domains(allow_resampling=allow_resampling, force_resample=False)


def train_fullbatch(
    model: nn.Module,
    x_train: torch.Tensor,
    y_train: torch.Tensor,
    epochs: int,
    lr: float,
    weight_decay: float,
    grad_clip: float,
    print_every: int,
    *,
    sn_domain_warmup_epochs: int = 0,
) -> None:
    loss_fn = nn.MSELoss()
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    model.train()
    t0 = time.time()
    for ep in range(1, epochs + 1):
        # Domain warmup hook (only meaningful for SN)
        if sn_domain_warmup_epochs > 0 and hasattr(model, "update_all_domains"):
            if ep <= sn_domain_warmup_epochs:
                _maybe_update_sn_domains(model, allow_resampling=True)

        opt.zero_grad(set_to_none=True)
        yhat = model(x_train)
        loss = loss_fn(yhat, y_train)
        loss.backward()
        if grad_clip and grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        opt.step()

        if ep == 1 or ep % print_every == 0 or ep == epochs:
            dt = time.time() - t0
            train_rmse = math.sqrt(float(loss.item()))
            print(
                f"  epoch {ep:5d}/{epochs}  loss={loss.item():.6e}  "
                f"train_RMSE(mean_heads)={train_rmse:.6e}  ({dt:.1f}s)"
            )
            t0 = time.time()


# -----------------------------
# Param matching helper
# -----------------------------
def choose_sn_knots_to_match_params(
    target_params: int,
    dims: int,
    heads: int,
    sn_arch: List[int],
    *,
    prefer_leq: bool,
    device: str,
) -> Tuple[int, int]:
    """
    Choose (phi_knots, Phi_knots) with phi_knots==Phi_knots that makes SN params
    close to target_params. If prefer_leq=True, enforce SN <= target.
    """
    # We'll estimate linear relationship by probing k and k+1.
    def sn_params_for_k(k: int) -> int:
        sn = SprecherMultiLayerNetwork(
            input_dim=dims,
            architecture=sn_arch,
            final_dim=heads,
            phi_knots=k,
            Phi_knots=k,
            norm_type="none",
            norm_position="after",
            norm_skip_first=True,
            initialize_domains=True,
            domain_ranges=None,
            phi_spline_type="linear",
            Phi_spline_type="linear",
        ).to(device)
        return count_trainable_params(sn)

    k_probe = 10
    p10 = sn_params_for_k(k_probe)
    p11 = sn_params_for_k(k_probe + 1)
    slope = p11 - p10
    if slope <= 0:
        # Fallback: brute search (shouldn't happen)
        best_k = k_probe
        best_p = p10
        for k in range(4, 5000):
            p = sn_params_for_k(k)
            if prefer_leq and p > target_params:
                break
            if abs(p - target_params) < abs(best_p - target_params):
                best_k, best_p = k, p
        return best_k, best_k

    base = p10 - slope * k_probe
    k_real = (target_params - base) / float(slope)
    k0 = max(4, int(math.floor(k_real)))
    k1 = max(4, int(math.ceil(k_real)))

    # Check a small neighborhood
    candidates = sorted(set([k0 - 2, k0 - 1, k0, k1, k1 + 1, k1 + 2]))
    candidates = [k for k in candidates if k >= 4]

    best_k: Optional[int] = None
    best_diff = float("inf")

    for k in candidates:
        p = sn_params_for_k(k)
        if prefer_leq and p > target_params:
            continue
        diff = abs(p - target_params)
        if diff < best_diff:
            best_k = k
            best_diff = diff

    if best_k is None:
        # If all candidates violate prefer_leq, step down until we satisfy it.
        k = k0
        while k >= 4:
            p = sn_params_for_k(k)
            if p <= target_params:
                best_k = k
                break
            k -= 1
        if best_k is None:
            best_k = 4

    return int(best_k), int(best_k)


# -----------------------------
# Main
# -----------------------------
def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--dims", type=int, default=12)
    p.add_argument("--heads", type=int, default=64)
    p.add_argument("--train_size", type=int, default=1024)
    p.add_argument("--test_size", type=int, default=50000)
    p.add_argument("--epochs", type=int, default=4000)
    p.add_argument("--device", type=str, default="cpu")
    p.add_argument("--seed", type=int, default=0)

    # Task knobs (kept simple; you can ignore these in the command)
    p.add_argument("--alpha", type=float, default=0.35)
    p.add_argument("--omega", type=float, default=6.0 * math.pi)

    # SN knobs
    p.add_argument("--sn_arch", type=str, default="8,8")
    p.add_argument("--sn_phi_knots", type=int, default=60)
    p.add_argument("--sn_Phi_knots", type=int, default=60)
    p.add_argument("--sn_lr", type=float, default=1e-3)
    p.add_argument("--sn_weight_decay", type=float, default=0.0)
    p.add_argument("--sn_grad_clip", type=float, default=1.0)
    p.add_argument("--sn_freeze_domains_after", type=int, default=400)
    p.add_argument("--sn_domain_margin", type=float, default=0.01)

    # KAN knobs
    p.add_argument("--kan_arch", type=str, default="8,8")
    p.add_argument("--kan_num_knots", type=int, default=4)
    p.add_argument("--kan_lr", type=float, default=1e-3)
    p.add_argument("--kan_weight_decay", type=float, default=0.0)
    p.add_argument("--kan_grad_clip", type=float, default=1.0)
    p.add_argument("--kan_domain_min", type=float, default=-2.0)
    p.add_argument("--kan_domain_max", type=float, default=2.0)

    # Fairness knobs
    p.add_argument("--equalize_params", action="store_true")
    p.add_argument("--prefer_leq", action="store_true")

    args = p.parse_args()

    device = torch.device(args.device)

    # -------------------------
    # Global config: "no extras"
    # -------------------------
    # Turn off engineering extras for SN. Do this BEFORE instantiating the model.
    CONFIG["use_residual_weights"] = False
    CONFIG["use_lateral_mixing"] = False
    CONFIG["use_normalization"] = False
    CONFIG["train_phi_codomain"] = False
    CONFIG["domain_safety_margin"] = float(args.sn_domain_margin)

    # Reproducibility:
    # - Use seed for model init (reset for each model so they don't depend on construction order)
    # - Use seed+999 for data sampling (fixed per run)
    set_global_seeds(args.seed)

    dims = int(args.dims)
    heads = int(args.heads)
    sn_arch = parse_arch(args.sn_arch)
    kan_arch = parse_arch(args.kan_arch)

    # Task (fixed target function across all runs)
    task = OscillatoryHeadShiftTask(
        dims=dims,
        heads=heads,
        alpha=float(args.alpha),
        omega=float(args.omega),
    )

    # Data (fixed per run seed)
    set_global_seeds(args.seed + 999)
    x_train, y_train = task.sample(args.train_size, device=str(device))
    x_test, y_test = task.sample(args.test_size, device=str(device))

    # KAN params (for equalization)
    kan_params_target = kan_param_count(dims, heads, kan_arch, args.kan_num_knots)

    # Choose SN knots if requested
    if args.equalize_params:
        phi_knots, Phi_knots = choose_sn_knots_to_match_params(
            target_params=kan_params_target,
            dims=dims,
            heads=heads,
            sn_arch=sn_arch,
            prefer_leq=bool(args.prefer_leq),
            device=str(device),
        )
    else:
        phi_knots, Phi_knots = int(args.sn_phi_knots), int(args.sn_Phi_knots)

    # -------------------------
    # Build models (reset seed before each for cleaner reproducibility)
    # -------------------------
    set_global_seeds(args.seed)
    sn = SprecherMultiLayerNetwork(
        input_dim=dims,
        architecture=sn_arch,
        final_dim=heads,
        phi_knots=phi_knots,
        Phi_knots=Phi_knots,
        norm_type="none",
        norm_position="after",
        norm_skip_first=True,
        initialize_domains=True,
        domain_ranges=None,
        phi_spline_type="linear",
        Phi_spline_type="linear",
    ).to(device)

    set_global_seeds(args.seed)
    kan = CubicPchipKAN(
        input_dim=dims,
        architecture=kan_arch,
        output_dim=heads,
        num_knots=int(args.kan_num_knots),
        domain_min=float(args.kan_domain_min),
        domain_max=float(args.kan_domain_max),
        init_scale=0.01,
    ).to(device)

    # -------------------------
    # Print setup
    # -------------------------
    print("\n=== SN (no extras; domain warmup) ===")
    print(f"  dims={dims} heads={heads} sn_arch={sn_arch} phi_knots={phi_knots} Phi_knots={Phi_knots}")
    warmup = max(0, min(int(args.sn_freeze_domains_after), int(args.epochs)))
    print(f"  domain warmup epochs: {warmup}/{args.epochs}  (margin={args.sn_domain_margin})")

    print("\n=== KAN (cubic splines; no extras) ===")
    print(f"  dims={dims} heads={heads} kan_arch={kan_arch} num_knots={args.kan_num_knots}")
    print(f"  spline domain: [{args.kan_domain_min:.2f}, {args.kan_domain_max:.2f}]")

    sn_params = count_trainable_params(sn)
    kan_params = count_trainable_params(kan)

    print("\n=== Parameter counts (trainable) ===")
    print(f"  SN : {sn_params} params")
    print(f"  KAN: {kan_params} params")
    diff = sn_params - kan_params
    msg = f"  (SN - KAN) = {diff:+d}"
    if args.equalize_params and args.prefer_leq:
        msg += "  (<=0 as requested)"
    print(msg)

    # -------------------------
    # Train
    # -------------------------
    print_every = max(1, args.epochs // 8)

    print("\nTraining SN...")
    train_fullbatch(
        sn,
        x_train,
        y_train,
        epochs=int(args.epochs),
        lr=float(args.sn_lr),
        weight_decay=float(args.sn_weight_decay),
        grad_clip=float(args.sn_grad_clip),
        print_every=print_every,
        sn_domain_warmup_epochs=warmup,
    )

    print("\nTraining KAN...")
    train_fullbatch(
        kan,
        x_train,
        y_train,
        epochs=int(args.epochs),
        lr=float(args.kan_lr),
        weight_decay=float(args.kan_weight_decay),
        grad_clip=float(args.kan_grad_clip),
        print_every=print_every,
        sn_domain_warmup_epochs=0,
    )

    # -------------------------
    # Evaluate
    # -------------------------
    sn.eval()
    kan.eval()
    with torch.no_grad():
        sn_rmse = rmse_all(sn(x_test), y_test)
        kan_rmse = rmse_all(kan(x_test), y_test)

    print("\n=== Test RMSE (mean across heads) ===")
    print(f"  SN : {sn_rmse:.6e}")
    print(f"  KAN: {kan_rmse:.6e}")

    print("\n=== RESULT (copy-friendly) ===")
    print(
        "RESULT "
        f"seed={args.seed} "
        f"sn_rmse={sn_rmse:.6e} "
        f"kan_rmse={kan_rmse:.6e} "
        f"sn_params={sn_params} "
        f"kan_params={kan_params} "
        f"phi_knots={phi_knots} "
        f"Phi_knots={Phi_knots} "
        f"kan_num_knots={args.kan_num_knots}"
    )


if __name__ == "__main__":
    main()