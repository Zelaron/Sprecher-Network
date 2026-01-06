"""
benchmarks/kan_sn_inputshift_bump_bench.py

Fair SN vs KAN benchmark on a synthetic *vector-valued* regression task designed to
stress the "shifted inputs per output head" inductive bias.

Methodology (extras OFF):
- SN: domain warmup for ~10% of epochs (updates spline domains; no residuals, no mixing).
- KAN: cubic (PCHIP) splines on edges; no grid updates, no base activations, no residuals.

Run:
  for s in 0 1 2 3; do
    python -m benchmarks.kan_sn_inputshift_bump_bench \
      --seed $s \
      --sn_phi_knots 254 --sn_Phi_knots 260 \
      --sn_arch 20,20 \
      --kan_arch 10,10 \
      --equalize_params \
      --epochs 4000 \
      --test_size 50000
  done
"""
from __future__ import annotations

import argparse
import math
import random
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# --------------------------
# Repro / utils
# --------------------------

def set_all_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def parse_arch(s: str) -> List[int]:
    s = (s or "").strip()
    if not s:
        return []
    parts = [p.strip() for p in s.split(",") if p.strip()]
    return [int(p) for p in parts]


def count_params(module: nn.Module) -> int:
    return sum(int(p.numel()) for p in module.parameters())


def rmse(yhat: torch.Tensor, y: torch.Tensor) -> float:
    return float(torch.sqrt(torch.mean((yhat - y) ** 2)).detach().cpu().item())


def mse(yhat: torch.Tensor, y: torch.Tensor) -> float:
    return float(torch.mean((yhat - y) ** 2).detach().cpu().item())


def pick_device(device_str: str) -> torch.device:
    device_str = (device_str or "auto").lower()
    if device_str == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        # Apple Silicon
        if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(device_str)


# --------------------------
# Dataset: InputShiftBump
# --------------------------

@dataclass
class InputShiftBumpConfig:
    dims: int = 12
    heads: int = 17
    # teacher parameters
    eta_true: float = 0.045
    phi_slope: float = 8.0
    bump_center: float = 8.0
    bump_sigma: float = 2.6
    sine_amp: float = 0.25
    sine_freq: float = 0.85
    noise_std: float = 0.25


class InputShiftBump:
    """
    Vector-valued regression target:
      y_q = Φ( q + sum_i w_i * φ(x_i + η q) ) + ε

    where φ is monotone (sigmoid) and Φ is a "bump + sine" nonlinearity.
    The same η and the same inner φ are reused across all heads q, matching the
    structural bias of SNs (shifted inputs per head).
    """

    def __init__(
        self,
        cfg: InputShiftBumpConfig,
        seed: int,
        train_size: int,
        test_size: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> None:
        self.cfg = cfg
        self.seed = seed
        self.train_size = train_size
        self.test_size = test_size
        self.device = device
        self.dtype = dtype

        g = torch.Generator(device="cpu")
        g.manual_seed(seed + 12345)

        # Fixed teacher mixing weights (dims,)
        w = torch.randn(cfg.dims, generator=g)
        w = w / (w.norm() + 1e-8)
        # Make it slightly "structured" (alternating signs) to avoid trivial sum
        signs = torch.tensor([1.0 if (i % 2 == 0) else -1.0 for i in range(cfg.dims)])
        self.w = (0.9 * w + 0.1 * signs).to(dtype=torch.float32)

        # Sample train/test inputs
        x_train = torch.rand(train_size, cfg.dims, generator=g)
        x_test = torch.rand(test_size, cfg.dims, generator=g)

        y_train = self._target(x_train, g)
        y_test = self._target(x_test, g)

        self.x_train = x_train.to(device=device, dtype=dtype)
        self.y_train = y_train.to(device=device, dtype=dtype)
        self.x_test = x_test.to(device=device, dtype=dtype)
        self.y_test = y_test.to(device=device, dtype=dtype)

    def _phi_true(self, t: torch.Tensor) -> torch.Tensor:
        # monotone in t
        return torch.sigmoid(self.cfg.phi_slope * (t - 0.5))

    def _Phi_true(self, s: torch.Tensor) -> torch.Tensor:
        # bump + sine
        bump = torch.exp(-0.5 * ((s - self.cfg.bump_center) / self.cfg.bump_sigma) ** 2)
        return bump + self.cfg.sine_amp * torch.sin(self.cfg.sine_freq * s)

    def _target(self, x: torch.Tensor, g: torch.Generator) -> torch.Tensor:
        cfg = self.cfg
        # x: (N,dims)
        # heads index q: (heads,)
        q = torch.arange(cfg.heads, dtype=x.dtype)[None, :]  # (1,heads)
        # shift inputs per head
        # We'll compute s_q = q + sum_i w_i * phi(x_i + eta*q)
        # Do this with broadcasting: x -> (N, dims, 1), q -> (1,1,heads)
        x3 = x[:, :, None]  # (N, dims, 1)
        q3 = q[:, None, :]  # (1, 1, heads)
        shifted = x3 + cfg.eta_true * q3  # (N, dims, heads)
        phi = self._phi_true(shifted)  # (N, dims, heads)
        w = self.w.to(phi.device, phi.dtype)[:, None]  # (dims,1)
        # weighted sum over dims -> (N,heads)
        s = (phi * w[None, :, :]).sum(dim=1) + q  # (N,heads)
        y = self._Phi_true(s)
        if cfg.noise_std > 0:
            y = y + cfg.noise_std * torch.randn(y.shape, generator=g)
        return y


# --------------------------
# Splines: piecewise linear (SN) + cubic Hermite PCHIP (KAN)
# --------------------------

class PiecewiseLinearMonotoneSpline(nn.Module):
    """
    Monotone increasing spline with fixed codomain [0,1], implemented as
    piecewise-linear interpolation of knot values.

    Parameters: raw_increments (G,) -> positive increments via softplus.
    We normalize cumulative sums to hit 1 at the last knot, and we shift so ky[0]=0.
    """

    def __init__(self, n_knots: int, x_min: float = -0.5, x_max: float = 1.5) -> None:
        super().__init__()
        if n_knots < 2:
            raise ValueError("Need at least 2 knots")
        self.n_knots = int(n_knots)
        self.raw_increments = nn.Parameter(torch.zeros(self.n_knots))
        nn.init.constant_(self.raw_increments, 0.0)  # near-uniform after softplus
        self.register_buffer("x_min", torch.tensor(float(x_min)))
        self.register_buffer("x_max", torch.tensor(float(x_max)))

    @property
    def knots(self) -> torch.Tensor:
        # uniform knot locations (G,)
        return torch.linspace(
            float(self.x_min.item()),
            float(self.x_max.item()),
            self.n_knots,
            device=self.x_min.device,
            dtype=self.x_min.dtype,
        )

    def knot_values(self) -> torch.Tensor:
        # strictly increasing values with fixed endpoints: ky[0]=0, ky[-1]=1
        inc = F.softplus(self.raw_increments) + 1e-6
        cumsum = torch.cumsum(inc, dim=0)
        c0 = cumsum[0]
        denom = cumsum[-1] - c0
        vals = (cumsum - c0) / (denom + 1e-12)
        return vals

    def set_domain_(self, new_min: float, new_max: float) -> None:
        # only moves knot x-positions; values remain parameterized via increments
        if not (math.isfinite(new_min) and math.isfinite(new_max)):
            return
        if new_max <= new_min:
            return
        self.x_min.fill_(float(new_min))
        self.x_max.fill_(float(new_max))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: arbitrary shape
        ky = self.knot_values()  # (G,)
        x0 = self.x_min
        xN = self.x_max
        dx = (xN - x0) / (self.n_knots - 1)

        below = x <= x0
        above = x >= xN

        # uniform-grid index
        u = (x - x0) / (dx + 1e-12)
        idx = torch.floor(u).to(torch.long).clamp(0, self.n_knots - 2)
        t = u - idx.to(u.dtype)

        yL = ky[idx]
        yR = ky[idx + 1]
        y = yL + t * (yR - yL)

        y = torch.where(below, torch.zeros_like(y), y)
        y = torch.where(above, torch.ones_like(y), y)
        return y


class PiecewiseLinearSpline(nn.Module):
    """
    Unconstrained piecewise-linear spline with linear extrapolation outside domain.

    Parameters: y_values (G,).
    Domain: [x_min, x_max] stored as buffers, can be resampled.
    """

    def __init__(self, n_knots: int, x_min: float = -10.0, x_max: float = 10.0) -> None:
        super().__init__()
        if n_knots < 2:
            raise ValueError("Need at least 2 knots")
        self.n_knots = int(n_knots)
        self.y = nn.Parameter(torch.zeros(self.n_knots))
        # initialize to ~0 so early outputs are well-scaled
        with torch.no_grad():
            self.y.uniform_(-0.01, 0.01)
        self.register_buffer("x_min", torch.tensor(float(x_min)))
        self.register_buffer("x_max", torch.tensor(float(x_max)))

    @property
    def knots(self) -> torch.Tensor:
        return torch.linspace(
            float(self.x_min.item()),
            float(self.x_max.item()),
            self.n_knots,
            device=self.x_min.device,
            dtype=self.x_min.dtype,
        )

    def _eval_no_resample(self, x: torch.Tensor, kx: torch.Tensor, ky: torch.Tensor) -> torch.Tensor:
        # linear interpolation on a uniform grid + linear extrapolation
        x0 = kx[0]
        xN = kx[-1]
        dx = (xN - x0) / (self.n_knots - 1)

        # inside-domain interpolation
        u = (x - x0) / (dx + 1e-12)
        idx = torch.floor(u).to(torch.long).clamp(0, self.n_knots - 2)
        t = u - idx.to(u.dtype)

        yL = ky[idx]
        yR = ky[idx + 1]
        y = yL + t * (yR - yL)

        # left extrapolation slope
        m0 = (ky[1] - ky[0]) / (dx + 1e-12)
        y_left = ky[0] + m0 * (x - x0)
        # right extrapolation slope
        mN = (ky[-1] - ky[-2]) / (dx + 1e-12)
        y_right = ky[-1] + mN * (x - xN)

        y = torch.where(x < x0, y_left, y)
        y = torch.where(x > xN, y_right, y)
        return y

    def resample_domain_(self, new_min: float, new_max: float) -> None:
        if not (math.isfinite(new_min) and math.isfinite(new_max)):
            return
        if new_max <= new_min:
            return
        old_kx = self.knots.detach().clone()
        old_ky = self.y.detach().clone()
        # new knots
        new_kx = torch.linspace(
            float(new_min),
            float(new_max),
            self.n_knots,
            device=old_kx.device,
            dtype=old_kx.dtype,
        )
        # evaluate old spline at new knots (with extrapolation) to preserve function
        with torch.no_grad():
            new_ky = self._eval_no_resample(new_kx, old_kx, old_ky)
            self.y.copy_(new_ky)
            self.x_min.fill_(float(new_min))
            self.x_max.fill_(float(new_max))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        kx = self.knots
        ky = self.y
        return self._eval_no_resample(x, kx, ky)


# --------------------------
# SN: Sprecher blocks + network
# --------------------------

class SprecherBlock(nn.Module):
    def __init__(
        self,
        d_in: int,
        d_out: int,
        phi_knots: int,
        Phi_knots: int,
        alpha: float = 1.0,
    ) -> None:
        super().__init__()
        self.d_in = int(d_in)
        self.d_out = int(d_out)
        self.alpha = float(alpha)

        # shared splines
        self.phi = PiecewiseLinearMonotoneSpline(phi_knots, x_min=-0.5, x_max=1.5)
        self.Phi = PiecewiseLinearSpline(Phi_knots, x_min=-10.0, x_max=10.0)

        # Sprecher parameters
        self.lam = nn.Parameter(torch.randn(self.d_in) * 0.05)
        self.eta = nn.Parameter(torch.tensor(0.05))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B,d_in)
        outs = []
        for q in range(self.d_out):
            xq = x + self.eta * float(q)
            phi_xq = self.phi(xq)  # (B,d_in)
            s_q = (phi_xq * self.lam).sum(dim=-1) + self.alpha * float(q)  # (B,)
            out_q = self.Phi(s_q)  # (B,)
            outs.append(out_q)
        return torch.stack(outs, dim=-1)  # (B,d_out)

    @torch.no_grad()
    def capture_ranges(self, x: torch.Tensor) -> Dict[str, float]:
        # compute empirical min/max of phi inputs and Phi inputs on a batch
        phi_min = float("inf")
        phi_max = float("-inf")
        s_min = float("inf")
        s_max = float("-inf")
        for q in range(self.d_out):
            xq = x + self.eta * float(q)
            phi_min = min(phi_min, float(xq.min().cpu().item()))
            phi_max = max(phi_max, float(xq.max().cpu().item()))
            phi_xq = self.phi(xq)
            s_q = (phi_xq * self.lam).sum(dim=-1) + self.alpha * float(q)
            s_min = min(s_min, float(s_q.min().cpu().item()))
            s_max = max(s_max, float(s_q.max().cpu().item()))
        return {"phi_min": phi_min, "phi_max": phi_max, "Phi_min": s_min, "Phi_max": s_max}

    @torch.no_grad()
    def warmup_update_domains_(self, x: torch.Tensor, margin: float = 0.05) -> None:
        ranges = self.capture_ranges(x)
        a = ranges["phi_min"]
        b = ranges["phi_max"]
        c = ranges["Phi_min"]
        d = ranges["Phi_max"]
        # add small margin
        span = b - a
        if span <= 1e-9:
            span = 1.0
        a2 = a - margin * span
        b2 = b + margin * span
        self.phi.set_domain_(a2, b2)

        span2 = d - c
        if span2 <= 1e-9:
            span2 = 1.0
        c2 = c - margin * span2
        d2 = d + margin * span2
        self.Phi.resample_domain_(c2, d2)


class SprecherNetwork(nn.Module):
    def __init__(
        self,
        d_in: int,
        hidden: List[int],
        d_out: int,
        phi_knots: int,
        Phi_knots: int,
        alpha: float = 1.0,
    ) -> None:
        super().__init__()
        dims = [int(d_in)] + [int(h) for h in hidden] + [int(d_out)]
        self.blocks = nn.ModuleList(
            [
                SprecherBlock(
                    dims[i],
                    dims[i + 1],
                    phi_knots=phi_knots,
                    Phi_knots=Phi_knots,
                    alpha=alpha,
                )
                for i in range(len(dims) - 1)
            ]
        )
        # output affine head (2 params total)
        self.out_scale = nn.Parameter(torch.tensor(1.0))
        self.out_bias = nn.Parameter(torch.tensor(0.0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = x
        for blk in self.blocks:
            h = blk(h)
        return self.out_scale * h + self.out_bias

    @torch.no_grad()
    def warmup_update_domains_(self, x: torch.Tensor, margin: float = 0.05) -> None:
        h = x
        for blk in self.blocks:
            blk.warmup_update_domains_(h, margin=margin)
            h = blk(h)


# --------------------------
# KAN: cubic Hermite PCHIP edge splines
# --------------------------

def _pchip_slopes(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """
    Vectorized PCHIP slope computation.

    x: (K,)
    y: (..., K)
    returns d: (..., K)
    """
    K = x.numel()
    if K < 2:
        raise ValueError("Need at least 2 knots for cubic spline.")
    h = x[1:] - x[:-1]  # (K-1,)
    delta = (y[..., 1:] - y[..., :-1]) / (h + 1e-12)  # (..., K-1)

    if K == 2:
        return torch.stack([delta[..., 0], delta[..., 0]], dim=-1)

    d = torch.zeros_like(y)

    # interior slopes via weighted harmonic mean where delta keeps sign
    h0 = h[:-1]  # (K-2,)
    h1 = h[1:]  # (K-2,)
    w1 = 2 * h1 + h0
    w2 = h1 + 2 * h0
    del0 = delta[..., :-1]
    del1 = delta[..., 1:]
    same_sign = (del0 * del1) > 0
    denom = (w1 / (del0 + 1e-12)) + (w2 / (del1 + 1e-12))
    d_int = (w1 + w2) / (denom + 1e-12)
    d[..., 1:-1] = torch.where(same_sign, d_int, torch.zeros_like(d_int))

    # endpoints (Fritsch-Carlson)
    d0 = ((2 * h[0] + h[1]) * delta[..., 0] - h[0] * delta[..., 1]) / (h[0] + h[1] + 1e-12)
    d0 = torch.where(d0 * delta[..., 0] <= 0, torch.zeros_like(d0), d0)
    d0 = torch.where((delta[..., 0] * delta[..., 1] < 0) & (d0.abs() > 3 * delta[..., 0].abs()), 3 * delta[..., 0], d0)
    d[..., 0] = d0

    dn = ((2 * h[-1] + h[-2]) * delta[..., -1] - h[-1] * delta[..., -2]) / (h[-1] + h[-2] + 1e-12)
    dn = torch.where(dn * delta[..., -1] <= 0, torch.zeros_like(dn), dn)
    dn = torch.where((delta[..., -1] * delta[..., -2] < 0) & (dn.abs() > 3 * delta[..., -1].abs()), 3 * delta[..., -1], dn)
    d[..., -1] = dn

    return d


def _hermite_eval(
    x_in: torch.Tensor,
    knots: torch.Tensor,
    coeffs: torch.Tensor,
    slopes: torch.Tensor,
) -> torch.Tensor:
    """
    Evaluate cubic Hermite spline (PCHIP slopes) for every edge in a KAN layer.

    x_in: (B, d_in)
    knots: (K,) shared x-locations for all edges
    coeffs: (d_out, d_in, K) y-values at knots
    slopes: (d_out, d_in, K) dy/dx at knots

    returns: (B, d_out, d_in)
    """
    B, d_in = x_in.shape
    d_out, d_in2, K = coeffs.shape
    assert d_in2 == d_in
    assert slopes.shape == coeffs.shape

    # Expand x to (B, d_out, d_in) so each output has its own set of edge splines
    x = x_in[:, None, :].expand(B, d_out, d_in)

    x0 = knots[0]
    xN = knots[-1]
    mask_left = x < x0
    mask_right = x > xN
    mask_mid = (~mask_left) & (~mask_right)

    # segment index in [0, K-2] (uniform grid => no bucketize)
    x_min = knots[0]
    x_max = knots[-1]
    dx = (x_max - x_min) / (K - 1)

    u = (x - x_min) / (dx + 1e-12)
    idx = torch.floor(u).to(torch.long).clamp(0, K - 2)
    idxp1 = idx + 1

    # normalized coordinate in segment
    t = u - idx.to(u.dtype)
    h = dx

    # FIX: expand coeffs/slopes along batch dim before gather
    coeffs_b = coeffs.unsqueeze(0).expand(B, -1, -1, -1)  # (B, d_out, d_in, K)
    slopes_b = slopes.unsqueeze(0).expand(B, -1, -1, -1)  # (B, d_out, d_in, K)

    idxe = idx.unsqueeze(-1)
    idxp1e = idxp1.unsqueeze(-1)

    yk = torch.gather(coeffs_b, dim=-1, index=idxe).squeeze(-1)
    yk1 = torch.gather(coeffs_b, dim=-1, index=idxp1e).squeeze(-1)
    mk = torch.gather(slopes_b, dim=-1, index=idxe).squeeze(-1)
    mk1 = torch.gather(slopes_b, dim=-1, index=idxp1e).squeeze(-1)

    t2 = t * t
    t3 = t2 * t
    h00 = 2 * t3 - 3 * t2 + 1
    h10 = t3 - 2 * t2 + t
    h01 = -2 * t3 + 3 * t2
    h11 = t3 - t2
    y_mid = h00 * yk + h10 * h * mk + h01 * yk1 + h11 * h * mk1

    # Linear extrapolation using endpoint derivatives
    yL = coeffs[..., 0].unsqueeze(0).expand(B, -1, -1)
    mL = slopes[..., 0].unsqueeze(0).expand(B, -1, -1)
    y_left = yL + mL * (x - x0)

    yR = coeffs[..., -1].unsqueeze(0).expand(B, -1, -1)
    mR = slopes[..., -1].unsqueeze(0).expand(B, -1, -1)
    y_right = yR + mR * (x - xN)

    y = torch.where(mask_mid, y_mid, torch.zeros_like(y_mid))
    y = torch.where(mask_left, y_left, y)
    y = torch.where(mask_right, y_right, y)
    return y


class KANCubicLayer(nn.Module):
    """
    KAN layer: output dims d_out, input dims d_in.
    Each edge has its own cubic spline; nodes sum over inputs + bias.
    """

    def __init__(self, d_in: int, d_out: int, K: int, x_min: float = 0.0, x_max: float = 1.0) -> None:
        super().__init__()
        self.d_in = int(d_in)
        self.d_out = int(d_out)
        self.K = int(K)
        if self.K < 2:
            raise ValueError("KAN K must be >= 2")
        knots = torch.linspace(float(x_min), float(x_max), self.K)
        self.register_buffer("knots", knots)
        # coeffs per edge: (d_out, d_in, K)
        self.coeffs = nn.Parameter(torch.zeros(self.d_out, self.d_in, self.K))
        # initialize small to keep activations in a reasonable range (no extras / no normalization)
        with torch.no_grad():
            std = 0.02 / math.sqrt(max(1, self.d_in))
            self.coeffs.normal_(mean=0.0, std=std)
        self.bias = nn.Parameter(torch.zeros(self.d_out))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, d_in)
        d = _pchip_slopes(self.knots, self.coeffs)
        vals = _hermite_eval(x, self.knots, self.coeffs, d)  # (B, d_out, d_in)
        out = vals.sum(dim=-1) + self.bias  # (B, d_out)
        return out


class KANNetwork(nn.Module):
    def __init__(self, d_in: int, hidden: List[int], d_out: int, K: int) -> None:
        super().__init__()
        dims = [int(d_in)] + [int(h) for h in hidden] + [int(d_out)]
        self.layers = nn.ModuleList(
            [KANCubicLayer(dims[i], dims[i + 1], K=K, x_min=0.0, x_max=1.0) for i in range(len(dims) - 1)]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = x
        for layer in self.layers:
            h = layer(h)
        return h


# --------------------------
# Param matching helpers
# --------------------------

def sn_param_count(d_in: int, hidden: List[int], d_out: int, phi_knots: int, Phi_knots: int) -> int:
    dims = [d_in] + hidden + [d_out]
    total = 0
    for i in range(len(dims) - 1):
        din = dims[i]
        total += phi_knots + Phi_knots  # spline params
        total += din  # lambda vector
        total += 1  # eta
    total += 2  # output affine head
    return total


def kan_param_count(d_in: int, hidden: List[int], d_out: int, K: int) -> int:
    dims = [d_in] + hidden + [d_out]
    total = 0
    for i in range(len(dims) - 1):
        din = dims[i]
        dout = dims[i + 1]
        total += dout * din * K  # spline coeffs
        total += dout  # bias
    return total


def choose_kan_K_for_budget(
    d_in: int,
    hidden: List[int],
    d_out: int,
    target_params: int,
    K_min: int = 2,
    K_max: int = 16,
) -> int:
    best_K = K_min
    best_diff = float("inf")
    for K in range(K_min, K_max + 1):
        p = kan_param_count(d_in, hidden, d_out, K)
        diff = abs(p - target_params)
        if diff < best_diff:
            best_diff = diff
            best_K = K
    return best_K


# --------------------------
# Training
# --------------------------

def train_model(
    name: str,
    model: nn.Module,
    x_train: torch.Tensor,
    y_train: torch.Tensor,
    epochs: int,
    lr: float,
    warmup_domains: bool = False,
    warmup_frac: float = 0.10,
    log_every: Optional[int] = None,
) -> List[float]:
    model.train()
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    losses: List[float] = []
    best = float("inf")

    warmup_epochs = int(round(warmup_frac * epochs)) if warmup_domains else 0
    warmup_epochs = max(0, min(warmup_epochs, epochs))

    if log_every is None:
        log_every = max(1, epochs // 10)

    for ep in range(1, epochs + 1):
        opt.zero_grad(set_to_none=True)
        yhat = model(x_train)
        loss = torch.mean((yhat - y_train) ** 2)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()

        # domain warmup updates (SN only)
        if warmup_domains and ep <= warmup_epochs:
            with torch.no_grad():
                if hasattr(model, "warmup_update_domains_"):
                    model.warmup_update_domains_(x_train)

        loss_val = float(loss.detach().cpu().item())
        losses.append(loss_val)
        best = min(best, loss_val)

        if ep == 1 or ep % log_every == 0 or ep == epochs:
            print(f"[{name}] epoch {ep:5d}/{epochs}  loss={loss_val:8.3e}  best={best:8.3e}")

    return losses


@torch.no_grad()
def eval_model(model: nn.Module, x: torch.Tensor, y: torch.Tensor) -> Tuple[float, float]:
    model.eval()
    yhat = model(x)
    return rmse(yhat, y), mse(yhat, y)


# --------------------------
# Main
# --------------------------

def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--dims", type=int, default=12)
    p.add_argument("--heads", type=int, default=17)
    p.add_argument("--train_size", type=int, default=1024)
    p.add_argument("--test_size", type=int, default=50000)
    p.add_argument("--epochs", type=int, default=4000)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--device", type=str, default="auto")
    p.add_argument("--dtype", type=str, default="float32")

    # SN
    p.add_argument("--sn_arch", type=str, default="20,20")
    p.add_argument("--sn_phi_knots", type=int, default=260)
    p.add_argument("--sn_Phi_knots", type=int, default=260)
    p.add_argument("--sn_warmup_frac", type=float, default=0.10)

    # KAN
    p.add_argument(
        "--kan_arch",
        type=str,
        default="10,10",
    )
    p.add_argument(
        "--kan_K",
        type=int,
        default=None,
        help="Number of cubic knots per edge spline (if omitted and --equalize_params is set, we pick K to match SN params).",
    )
    p.add_argument("--equalize_params", action="store_true")

    args = p.parse_args()

    dtype = torch.float32 if args.dtype.lower() == "float32" else torch.float64
    device = pick_device(args.device)

    set_all_seeds(args.seed)

    # Data
    cfg = InputShiftBumpConfig(dims=args.dims, heads=args.heads)
    ds = InputShiftBump(
        cfg,
        seed=args.seed,
        train_size=args.train_size,
        test_size=args.test_size,
        device=device,
        dtype=dtype,
    )

    print(f"\nDataset: InputShiftBump(dims={args.dims}, heads={args.heads}), test_size={args.test_size}")

    sn_hidden = parse_arch(args.sn_arch)
    kan_hidden = parse_arch(args.kan_arch)

    # Param matching
    sn_params_target = sn_param_count(args.dims, sn_hidden, args.heads, args.sn_phi_knots, args.sn_Phi_knots)

    if args.kan_K is None:
        if args.equalize_params:
            K = choose_kan_K_for_budget(args.dims, kan_hidden, args.heads, sn_params_target, K_min=2, K_max=16)
        else:
            K = 4
    else:
        K = int(args.kan_K)

    kan_params = kan_param_count(args.dims, kan_hidden, args.heads, K)

    print("\n=== Model configs (extras off) ===")
    print(f"SN : arch={sn_hidden} + out={args.heads}, phi_knots={args.sn_phi_knots}, Phi_knots={args.sn_Phi_knots}")
    print(f"     params={sn_params_target}")
    print(f"KAN: arch={kan_hidden} + out={args.heads}, cubic_knots(K)={K}")
    print(f"     params={kan_params}")
    print(f"Parity: target(SN)={sn_params_target}, chosen(KAN)={kan_params}, diff(KAN-SN)={kan_params - sn_params_target:+d}")

    # Build models
    sn = SprecherNetwork(
        args.dims,
        sn_hidden,
        args.heads,
        phi_knots=args.sn_phi_knots,
        Phi_knots=args.sn_Phi_knots,
    ).to(device=device, dtype=dtype)
    kan = KANNetwork(args.dims, kan_hidden, args.heads, K=K).to(device=device, dtype=dtype)

    # Sanity check param counts match printed counts
    sn_actual = count_params(sn)
    kan_actual = count_params(kan)
    if sn_actual != sn_params_target:
        print(f"[warn] SN param count mismatch: computed={sn_params_target}, actual={sn_actual}")
    if kan_actual != kan_params:
        print(f"[warn] KAN param count mismatch: computed={kan_params}, actual={kan_actual}")

    # Train
    print("\n=== Training SN (domain warmup) ===")
    _ = train_model(
        "SN",
        sn,
        ds.x_train,
        ds.y_train,
        epochs=args.epochs,
        lr=args.lr,
        warmup_domains=True,
        warmup_frac=args.sn_warmup_frac,
    )

    print("\n=== Training KAN (cubic splines) ===")
    _ = train_model(
        "KAN",
        kan,
        ds.x_train,
        ds.y_train,
        epochs=args.epochs,
        lr=args.lr,
        warmup_domains=False,
    )

    # Eval
    sn_train_rmse, sn_train_mse = eval_model(sn, ds.x_train, ds.y_train)
    sn_test_rmse, sn_test_mse = eval_model(sn, ds.x_test, ds.y_test)

    kan_train_rmse, kan_train_mse = eval_model(kan, ds.x_train, ds.y_train)
    kan_test_rmse, kan_test_mse = eval_model(kan, ds.x_test, ds.y_test)

    print("\n=== Results ===")
    print(
        f"SN : train RMSE={sn_train_rmse:.6f}  train MSE={sn_train_mse:.6e} | test RMSE={sn_test_rmse:.6f}  test MSE={sn_test_mse:.6e}"
    )
    print(
        f"KAN: train RMSE={kan_train_rmse:.6f}  train MSE={kan_train_mse:.6e} | test RMSE={kan_test_rmse:.6f}  test MSE={kan_test_mse:.6e}"
    )

    print("\n=== One-line summary (for tables) ===")
    print(
        f"seed={args.seed}  SN_test_RMSE={sn_test_rmse:.6f}  KAN_test_RMSE={kan_test_rmse:.6f}  "
        f"SN_params={sn_actual}  KAN_params={kan_actual}  K={K}"
    )


if __name__ == "__main__":
    main()
