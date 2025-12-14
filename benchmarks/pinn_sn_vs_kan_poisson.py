#!/usr/bin/env python3
"""
benchmarks/pinn_sn_vs_kan_poisson.py

Apples-to-apples PINN benchmark: Sprecher Network (SN) vs KAN on a 2D Poisson problem
over Ω = [-1,1]^2 with homogeneous Dirichlet boundary conditions u|∂Ω = 0.

Manufactured solution:
    u(x,y) = sin(pi*x) * sin(pi*y^2)

PDE:
    Δu = g(x,y)  (g is derived analytically from the manufactured solution)

Benchmark goals:
- Same collocation points, optimizer, LR schedule, loss weights, dtype for both models.
- No SN residual weights, no SN lateral mixing, no normalization layers.
- Both models use cubic (PCHIP) splines.
- SN uses fixed spline domains (no domain updates during training).

CLI example:
    python -m benchmarks.pinn_sn_vs_kan_poisson --model both --epochs 5000 --seed 0 --target_params 1200
"""

from __future__ import annotations

import argparse
import math
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Callable

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

# SN core (project-local)
from sn_core import SprecherMultiLayerNetwork, CONFIG, Q_VALUES_FACTOR

# -----------------------------
# Problem setup
# -----------------------------

DOMAIN_MIN = -1.0
DOMAIN_MAX = 1.0
PI = math.pi


def set_seed(seed: int) -> None:
    seed = int(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def count_trainable_params(model: nn.Module) -> int:
    return int(sum(p.numel() for p in model.parameters() if p.requires_grad))


def manufactured_solution(xy: torch.Tensor) -> torch.Tensor:
    """
    u(x,y) = sin(pi*x) * sin(pi*y^2)
    xy: [N,2]
    returns: [N]
    """
    x = xy[:, 0]
    y = xy[:, 1]
    return torch.sin(PI * x) * torch.sin(PI * (y ** 2))


def poisson_forcing(xy: torch.Tensor) -> torch.Tensor:
    """
    g(x,y) = Δu for u(x,y)=sin(pi*x)*sin(pi*y^2)
    xy: [N,2]
    returns: [N]
    """
    x = xy[:, 0]
    y = xy[:, 1]
    sin_px = torch.sin(PI * x)

    t = PI * (y ** 2)
    sin_t = torch.sin(t)
    cos_t = torch.cos(t)

    # u_xx = -pi^2 sin(pi x) sin(pi y^2)
    u_xx = -(PI ** 2) * sin_px * sin_t

    # u_yy:
    # u_y = sin(pi x) * cos(pi y^2) * (2 pi y)
    # u_yy = sin(pi x) * [ -sin(pi y^2)*(2 pi y)*(2 pi y) + cos(pi y^2)*(2 pi) ]
    u_yy = sin_px * (-(sin_t) * (2 * PI * y) * (2 * PI * y) + cos_t * (2 * PI))

    return u_xx + u_yy


# -----------------------------
# Collocation sampling
# -----------------------------

def sample_interior_points(n: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    """
    Uniform random interior points in (-1,1)^2 (excluding boundary only in measure-zero sense).
    """
    pts = (DOMAIN_MAX - DOMAIN_MIN) * torch.rand(int(n), 2, device=device, dtype=dtype) + DOMAIN_MIN
    return pts


def sample_boundary_points(
    n_per_edge: int,
    device: torch.device,
    dtype: torch.dtype,
    include_corners: bool = False,
) -> torch.Tensor:
    """
    Deterministic boundary points on the 4 edges of the square.
    Returns [N_bnd,2].
    """
    n = int(n_per_edge)
    xs = torch.linspace(DOMAIN_MIN, DOMAIN_MAX, n, device=device, dtype=dtype)
    ys = torch.linspace(DOMAIN_MIN, DOMAIN_MAX, n, device=device, dtype=dtype)

    top = torch.stack([xs, torch.full_like(xs, DOMAIN_MAX)], dim=-1)
    bottom = torch.stack([xs, torch.full_like(xs, DOMAIN_MIN)], dim=-1)
    left = torch.stack([torch.full_like(ys, DOMAIN_MIN), ys], dim=-1)
    right = torch.stack([torch.full_like(ys, DOMAIN_MAX), ys], dim=-1)

    if include_corners:
        bnd = torch.cat([top, bottom, left, right], dim=0)
    else:
        # remove corners from left/right to avoid duplicates
        bnd = torch.cat([top, bottom, left[1:-1], right[1:-1]], dim=0)
    return bnd


# -----------------------------
# Autograd utilities
# -----------------------------

def laplacian_u(model: nn.Module, xy: torch.Tensor) -> torch.Tensor:
    """
    Compute Laplacian of model output u(x,y) wrt (x,y) using autograd.

    xy must require_grad=True.
    Returns: [N] tensor of Δu
    """
    u = model(xy)
    if u.ndim == 2 and u.shape[1] == 1:
        u = u[:, 0]

    grads = torch.autograd.grad(
        outputs=u,
        inputs=xy,
        grad_outputs=torch.ones_like(u),
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
    )[0]  # [N,2]

    u_x = grads[:, 0]
    u_y = grads[:, 1]

    u_xx = torch.autograd.grad(
        outputs=u_x,
        inputs=xy,
        grad_outputs=torch.ones_like(u_x),
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
    )[0][:, 0]

    u_yy = torch.autograd.grad(
        outputs=u_y,
        inputs=xy,
        grad_outputs=torch.ones_like(u_y),
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
    )[0][:, 1]

    return u_xx + u_yy


# =============================================================================
# KAN: cubic (PCHIP) splines
# =============================================================================

class CubicSpline(nn.Module):
    """
    1D cubic Hermite spline with PCHIP slopes.
    - Learnable y-values at knots (coeffs).
    - Uniform knots over [x_min, x_max].
    - Linear extrapolation outside domain using endpoint slopes.

    Designed to be differentiable wrt both x and coeffs for autograd PINNs.
    """

    def __init__(self, num_knots: int, x_min: float = -1.0, x_max: float = 1.0, dtype: torch.dtype = torch.float64):
        super().__init__()
        self.num_knots = int(num_knots)
        self.register_buffer("knots", torch.linspace(float(x_min), float(x_max), self.num_knots, dtype=dtype))
        self.coeffs = nn.Parameter(torch.zeros(self.num_knots, dtype=dtype))

    @staticmethod
    def _pchip_slopes(y: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """
        Vector PCHIP slopes for 1D y over strictly increasing x.
        Returns d with same shape as y.
        """
        h = x[1:] - x[:-1]  # [K-1]
        delta = (y[1:] - y[:-1]) / (h + 1e-12)  # [K-1]

        d = torch.zeros_like(y)

        # Endpoints
        d0 = delta[0]
        dN = delta[-1]
        d = d.clone()
        d[0] = d0
        d[-1] = dN

        # Interior slopes
        delta_prev = delta[:-1]
        delta_next = delta[1:]
        same_sign = (delta_prev * delta_next) > 0

        w1 = 2 * h[1:] + h[:-1]
        w2 = h[1:] + 2 * h[:-1]
        d_interior = (w1 + w2) / (w1 / (delta_prev + 1e-12) + w2 / (delta_next + 1e-12) + 1e-12)
        d[1:-1] = torch.where(same_sign, d_interior, torch.zeros_like(d_interior))
        return d

    @staticmethod
    def _hermite_eval(xq: torch.Tensor, x: torch.Tensor, y: torch.Tensor, d: torch.Tensor) -> torch.Tensor:
        """
        Evaluate cubic Hermite on query points xq (any shape).
        Uses piecewise intervals defined by x, y, slopes d.
        """
        xq_clamped = torch.clamp(xq, x[0], x[-1])
        idx = torch.searchsorted(x, xq_clamped) - 1
        idx = torch.clamp(idx, 0, x.numel() - 2)

        x0 = x[idx]
        x1 = x[idx + 1]
        y0 = y[idx]
        y1 = y[idx + 1]
        d0 = d[idx]
        d1 = d[idx + 1]

        h = (x1 - x0) + 1e-12
        t = (xq_clamped - x0) / h

        h00 = (2 * t ** 3) - (3 * t ** 2) + 1
        h10 = (t ** 3) - (2 * t ** 2) + t
        h01 = (-2 * t ** 3) + (3 * t ** 2)
        h11 = (t ** 3) - (t ** 2)

        return h00 * y0 + h10 * h * d0 + h01 * y1 + h11 * h * d1

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        knots = self.knots
        y = self.coeffs
        d = self._pchip_slopes(y, knots)

        below = x < knots[0]
        above = x > knots[-1]

        interp = self._hermite_eval(x, knots, y, d)

        # Linear extrapolation outside domain
        left_val = y[0]
        right_val = y[-1]
        left_slope = d[0]
        right_slope = d[-1]

        out = interp
        out = torch.where(below, left_val + left_slope * (x - knots[0]), out)
        out = torch.where(above, right_val + right_slope * (x - knots[-1]), out)
        return out


class KANLayer(nn.Module):
    """
    KAN layer: each output is sum_j spline_{out,in}(x_in_j) + bias_out
    Uses cubic splines (PCHIP).
    """

    def __init__(
        self,
        d_in: int,
        d_out: int,
        num_knots: int,
        x_min: float = -1.0,
        x_max: float = 1.0,
        dtype: torch.dtype = torch.float64,
        init_scale: Optional[float] = None,
    ):
        super().__init__()
        self.d_in = int(d_in)
        self.d_out = int(d_out)
        self.num_knots = int(num_knots)

        if init_scale is None:
            init_scale = 0.1 / math.sqrt(max(1, self.d_in))

        self.splines = nn.ModuleList()
        for _o in range(self.d_out):
            row = nn.ModuleList()
            for _i in range(self.d_in):
                sp = CubicSpline(num_knots=self.num_knots, x_min=x_min, x_max=x_max, dtype=dtype)
                with torch.no_grad():
                    sp.coeffs.normal_(0.0, float(init_scale))
                row.append(sp)
            self.splines.append(row)

        self.bias = nn.Parameter(torch.zeros(self.d_out, dtype=dtype))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        outs = []
        for o in range(self.d_out):
            acc = torch.zeros(x.shape[0], device=x.device, dtype=x.dtype)
            for i in range(self.d_in):
                acc = acc + self.splines[o][i](x[:, i])
            acc = acc + self.bias[o]
            outs.append(acc)
        return torch.stack(outs, dim=-1)  # [B,d_out]


class KAN(nn.Module):
    """
    Simple stacked KAN: hidden layers of fixed width, final layer to scalar output.
    """

    def __init__(self, input_dim: int, hidden_width: int, num_layers: int, num_knots: int, dtype: torch.dtype = torch.float64):
        super().__init__()
        self.input_dim = int(input_dim)
        self.hidden_width = int(hidden_width)
        self.num_layers = int(num_layers)
        self.num_knots = int(num_knots)

        layers: List[nn.Module] = []
        d_in = self.input_dim
        for li in range(self.num_layers):
            d_out = 1 if li == self.num_layers - 1 else self.hidden_width
            layers.append(KANLayer(d_in=d_in, d_out=d_out, num_knots=self.num_knots, dtype=dtype))
            d_in = d_out

        self.layers = nn.ModuleList(layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x)
        return x


def kan_param_count_formula(input_dim: int, hidden_width: int, num_layers: int, num_knots: int) -> int:
    total = 0
    d_in = int(input_dim)
    for li in range(int(num_layers)):
        d_out = 1 if li == int(num_layers) - 1 else int(hidden_width)
        total += d_in * d_out * int(num_knots) + d_out
        d_in = d_out
    return int(total)


@dataclass
class KANConfig:
    width: int
    knots: int
    num_layers: int
    est_params: int
    diff: int


def choose_kan_config(target_params: int, input_dim: int = 2, num_layers: int = 2) -> KANConfig:
    width_candidates = list(range(4, 129))
    knot_candidates = list(range(4, 65))

    best: Optional[KANConfig] = None
    best_key = None

    for w in width_candidates:
        for K in knot_candidates:
            est = kan_param_count_formula(input_dim=input_dim, hidden_width=w, num_layers=num_layers, num_knots=K)
            diff = abs(est - int(target_params))
            key = (diff, w, K)
            if best is None or key < best_key:
                best = KANConfig(width=w, knots=K, num_layers=num_layers, est_params=est, diff=diff)
                best_key = key

    assert best is not None
    return best


# =============================================================================
# SN config search + fixed-domain strategy
# =============================================================================

class SNInputNormalizer(nn.Module):
    """
    Wrap an SN to accept x in [-1,1]^2 by mapping to [0,1]^2.
    """

    def __init__(self, sn_model: nn.Module):
        super().__init__()
        self.sn_model = sn_model

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x01 = (x - DOMAIN_MIN) / (DOMAIN_MAX - DOMAIN_MIN)
        return self.sn_model(x01)


def get_sn_core_from_wrapper(model: nn.Module) -> SprecherMultiLayerNetwork:
    if isinstance(model, SNInputNormalizer):
        return model.sn_model  # type: ignore[return-value]
    raise TypeError("Expected SNInputNormalizer wrapper.")


@dataclass
class SNConfigChoice:
    target_params: int
    arch: List[int]
    phi_knots: int
    Phi_knots: int
    est_params: int
    diff: int
    compute_proxy: int
    knot_sum: int
    train_codomain: bool


def sn_compute_proxy(arch: List[int], input_dim: int = 2) -> int:
    proxy = 0
    d_in = int(input_dim)
    for d_out in arch:
        proxy += d_in * int(d_out)
        d_in = int(d_out)
    return int(proxy)


def sn_estimate_params(
    arch: List[int],
    phi_knots: int,
    Phi_knots: int,
    input_dim: int = 2,
    train_codomain: bool = False,
) -> int:
    n_layers = len(arch)

    sum_din = input_dim + sum(arch[:-1]) if n_layers >= 2 else input_dim
    spline_params = n_layers * (int(phi_knots) + int(Phi_knots))
    eta_params = n_layers
    codomain_params = 2 * n_layers if bool(train_codomain) else 0
    output_params = 2
    return int(sum_din + spline_params + eta_params + codomain_params + output_params)


def _split_knot_sum(knot_sum: int) -> Tuple[int, int]:
    ksum = int(knot_sum)
    phi_k = int(ksum // 2)
    Phi_k = int(ksum - phi_k)
    return phi_k, Phi_k


def choose_sn_config(
    target_params: int,
    input_dim: int = 2,
    min_width: int = 4,
    max_width: int = 64,
    min_layers: int = 2,
    max_layers: int = 2,
    min_knots_each: int = 8,
    train_codomain: bool = False,
) -> SNConfigChoice:
    if int(target_params) < 50:
        raise ValueError("--target_params is too small to make sense for this benchmark.")

    candidates: List[SNConfigChoice] = []

    for L in range(int(min_layers), int(max_layers) + 1):
        for w in range(int(min_width), int(max_width) + 1):
            arch = [int(w)] * int(L)
            n_layers = len(arch)

            sum_din = input_dim + sum(arch[:-1]) if n_layers >= 2 else input_dim
            base = sum_din + n_layers + (2 * n_layers if train_codomain else 0) + 2  # lambdas + eta + codomain + output

            remaining = int(target_params) - int(base)
            if remaining < n_layers * (2 * int(min_knots_each)):
                continue

            knot_sum_float = remaining / float(n_layers)
            for knot_sum_int in {int(math.floor(knot_sum_float)), int(math.ceil(knot_sum_float))}:
                if knot_sum_int < 2 * int(min_knots_each):
                    continue
                phi_k, Phi_k = _split_knot_sum(knot_sum_int)
                if phi_k < int(min_knots_each) or Phi_k < int(min_knots_each):
                    continue

                est = sn_estimate_params(
                    arch=arch,
                    phi_knots=phi_k,
                    Phi_knots=Phi_k,
                    input_dim=input_dim,
                    train_codomain=train_codomain,
                )
                diff = int(est) - int(target_params)

                candidates.append(
                    SNConfigChoice(
                        target_params=int(target_params),
                        arch=arch,
                        phi_knots=phi_k,
                        Phi_knots=Phi_k,
                        est_params=int(est),
                        diff=int(diff),
                        compute_proxy=sn_compute_proxy(arch, input_dim=input_dim),
                        knot_sum=int(phi_k + Phi_k),
                        train_codomain=bool(train_codomain),
                    )
                )

    if not candidates:
        raise RuntimeError(
            "No SN configs found. Try expanding width/layer ranges or decreasing --sn_min_knots_each."
        )

    candidates.sort(key=lambda c: (abs(c.diff), c.compute_proxy, c.knot_sum))
    return candidates[0]


# -----------------------------
# SN fixed domains (no updates)
# -----------------------------

@dataclass
class SNFixedDomainKnobs:
    eta_bound: float = 0.25
    lambda_sigma: float = 6.0
    phi_pad_abs: float = 0.25
    phi_pad_rel: float = 0.05
    Phi_pad_abs: float = 1.0
    Phi_pad_rel: float = 0.05


def _pad_interval(a: float, b: float, pad_abs: float, pad_rel: float) -> Tuple[float, float]:
    a = float(a)
    b = float(b)
    if a > b:
        a, b = b, a
    width = max(1e-12, b - a)
    pad = float(pad_abs) + float(pad_rel) * width
    return (a - pad, b + pad)


def _lambda_bound(d_in: int, lambda_sigma: float) -> float:
    d_in = max(1, int(d_in))
    return float(lambda_sigma) * math.sqrt(2.0 / float(d_in))


def sn_set_fixed_domains(sn_core: SprecherMultiLayerNetwork, knobs: SNFixedDomainKnobs) -> None:
    """
    Set conservative fixed input domains for each layer's φ and Φ splines.

    Uses only update_domain(...) (SimpleSpline API) and does NOT perform any domain updates during training.
    """
    in_min = 0.0
    in_max = 1.0

    for layer in sn_core.layers:
        d_in = int(layer.d_in)
        d_out = int(layer.d_out)
        q_span = max(0, d_out - 1)

        phi_lo = in_min - float(knobs.eta_bound) * q_span
        phi_hi = in_max + float(knobs.eta_bound) * q_span
        phi_lo, phi_hi = _pad_interval(phi_lo, phi_hi, knobs.phi_pad_abs, knobs.phi_pad_rel)

        lam_b = _lambda_bound(d_in, knobs.lambda_sigma)
        s_lo = -d_in * lam_b + float(Q_VALUES_FACTOR) * 0.0
        s_hi = +d_in * lam_b + float(Q_VALUES_FACTOR) * float(q_span)
        Phi_lo, Phi_hi = _pad_interval(s_lo, s_hi, knobs.Phi_pad_abs, knobs.Phi_pad_rel)

        # IMPORTANT FIX: SimpleSpline uses update_domain(...), not set_domain(...)
        layer.phi.update_domain((phi_lo, phi_hi), allow_resampling=False, force_resample=False)
        layer.Phi.update_domain((Phi_lo, Phi_hi), allow_resampling=False, force_resample=False)

        in_min, in_max = Phi_lo, Phi_hi


def sn_apply_param_clamps(sn_core: SprecherMultiLayerNetwork, knobs: SNFixedDomainKnobs) -> None:
    with torch.no_grad():
        for layer in sn_core.layers:
            if hasattr(layer, "eta"):
                layer.eta.data.clamp_(-float(knobs.eta_bound), float(knobs.eta_bound))
            lam_b = _lambda_bound(int(layer.d_in), knobs.lambda_sigma)
            if hasattr(layer, "lambdas"):
                layer.lambdas.data.clamp_(-lam_b, lam_b)


# -----------------------------
# SN builder
# -----------------------------

def build_sn_model_from_arch(
    arch: List[int],
    phi_knots: int,
    Phi_knots: int,
    device: torch.device,
    dtype: torch.dtype,
    train_codomain: bool,
    domain_strategy: str,
    fixed_knobs: SNFixedDomainKnobs,
    init_Phi_std: float = 1e-3,
) -> nn.Module:
    """
    Build an SN with a fixed-domain strategy (no domain updates in training).
    """
    # Fairness knobs: disable SN-only extras
    CONFIG["use_residual_weights"] = False
    CONFIG["use_lateral_mixing"] = False
    CONFIG["use_normalization"] = False

    CONFIG["train_phi_codomain"] = bool(train_codomain)

    # IMPORTANT FIX: SprecherMultiLayerNetwork expects phi_knots / Phi_knots
    sn_core = SprecherMultiLayerNetwork(
        input_dim=2,
        architecture=list(arch),
        final_dim=1,
        phi_knots=int(phi_knots),
        Phi_knots=int(Phi_knots),
        norm_type="none",
        norm_position="after",
        norm_skip_first=True,
        initialize_domains=False,
        domain_ranges=None,
        phi_spline_type="cubic",
        Phi_spline_type="cubic",
        phi_spline_order=None,
        Phi_spline_order=None,
    ).to(device=device, dtype=dtype)

    if str(domain_strategy).lower() == "fixed":
        sn_set_fixed_domains(sn_core, fixed_knobs)
    else:
        raise ValueError(f"Unsupported SN domain strategy: {domain_strategy!r} (expected 'fixed').")

    if float(init_Phi_std) > 0:
        with torch.no_grad():
            for layer in sn_core.layers:
                if hasattr(layer, "Phi") and hasattr(layer.Phi, "coeffs"):
                    layer.Phi.coeffs.normal_(0.0, float(init_Phi_std))

    return SNInputNormalizer(sn_core)


# =============================================================================
# PINN losses + training
# =============================================================================

def compute_pinn_loss(
    model: nn.Module,
    interior_points: torch.Tensor,
    boundary_points: torch.Tensor,
    interior_weight: float,
    boundary_weight: float,
    pde_norm_denom: float,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    xy_int = interior_points.clone().detach().requires_grad_(True)
    lap = laplacian_u(model, xy_int)
    g = poisson_forcing(xy_int.detach())
    res = lap - g
    pde_raw = (res ** 2).mean()
    pde_norm = pde_raw / float(pde_norm_denom)

    u_bnd = model(boundary_points)
    if u_bnd.ndim == 2 and u_bnd.shape[1] == 1:
        u_bnd = u_bnd[:, 0]
    bnd = (u_bnd ** 2).mean()

    total = float(interior_weight) * pde_norm + float(boundary_weight) * bnd
    logs = {
        "loss": float(total.detach().cpu()),
        "pde_norm": float(pde_norm.detach().cpu()),
        "pde_raw": float(pde_raw.detach().cpu()),
        "bnd": float(bnd.detach().cpu()),
    }
    return total, logs


def train_pinn(
    model: nn.Module,
    interior_points: torch.Tensor,
    boundary_points: torch.Tensor,
    epochs: int,
    lr: float,
    weight_decay: float,
    grad_clip: float,
    use_scheduler: bool,
    scheduler_min_lr: float,
    interior_weight: float,
    boundary_weight: float,
    pde_norm_denom: float,
    label: str,
    post_step_hook: Optional[Callable[[], None]] = None,
    log_every: int = 500,
) -> Dict[str, float]:
    model.train()
    opt = optim.Adam(model.parameters(), lr=float(lr), weight_decay=float(weight_decay))

    scheduler = None
    if bool(use_scheduler):
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            opt, T_max=int(epochs), eta_min=float(scheduler_min_lr)
        )

    pbar = tqdm(range(1, int(epochs) + 1), desc=f"Training {label}", ncols=120)
    last_logs: Dict[str, float] = {}

    for ep in pbar:
        opt.zero_grad(set_to_none=True)
        loss, logs = compute_pinn_loss(
            model=model,
            interior_points=interior_points,
            boundary_points=boundary_points,
            interior_weight=float(interior_weight),
            boundary_weight=float(boundary_weight),
            pde_norm_denom=float(pde_norm_denom),
        )
        loss.backward()

        if float(grad_clip) is not None and float(grad_clip) > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=float(grad_clip))

        opt.step()
        if post_step_hook is not None:
            post_step_hook()

        if scheduler is not None:
            scheduler.step()

        last_logs = logs
        if (ep == 1) or (ep % int(log_every) == 0) or (ep == int(epochs)):
            lr_now = opt.param_groups[0]["lr"]
            pbar.set_description(
                f"Training {label}: [{label}] ep {ep:5d}/{epochs}: "
                f"loss={logs['loss']:.4e}  pde(norm)={logs['pde_norm']:.4e}  "
                f"pde(raw)={logs['pde_raw']:.4e}  bnd={logs['bnd']:.4e}  lr={lr_now:.2e}"
            )

    return last_logs


# =============================================================================
# Evaluation
# =============================================================================

def make_evaluation_grids(
    grid_size: int, device: torch.device, dtype: torch.dtype
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    G = int(grid_size)
    xs = torch.linspace(DOMAIN_MIN, DOMAIN_MAX, G, device=device, dtype=dtype)
    ys = torch.linspace(DOMAIN_MIN, DOMAIN_MAX, G, device=device, dtype=dtype)

    try:
        X, Y = torch.meshgrid(xs, ys, indexing="ij")
    except TypeError:
        X, Y = torch.meshgrid(xs, ys)

    all_points = torch.stack([X.reshape(-1), Y.reshape(-1)], dim=-1)

    eps = 1e-12
    interior_mask = (all_points[:, 0] > DOMAIN_MIN + eps) & (all_points[:, 0] < DOMAIN_MAX - eps) & \
                    (all_points[:, 1] > DOMAIN_MIN + eps) & (all_points[:, 1] < DOMAIN_MAX - eps)
    interior_points = all_points[interior_mask]

    top = torch.stack([xs, torch.full_like(xs, DOMAIN_MAX)], dim=-1)
    bottom = torch.stack([xs, torch.full_like(xs, DOMAIN_MIN)], dim=-1)
    left = torch.stack([torch.full_like(ys[1:-1], DOMAIN_MIN), ys[1:-1]], dim=-1)
    right = torch.stack([torch.full_like(ys[1:-1], DOMAIN_MAX), ys[1:-1]], dim=-1)
    boundary_points = torch.cat([top, bottom, left, right], dim=0)

    return all_points, interior_points, boundary_points


def evaluate_model_on_grid(model: nn.Module, device: torch.device, dtype: torch.dtype, grid_size: int) -> Dict[str, float]:
    model.eval()

    all_points, interior_points, boundary_points = make_evaluation_grids(int(grid_size), device=device, dtype=dtype)

    interior_points_rg = interior_points.clone().detach().requires_grad_(True)
    lap = laplacian_u(model, interior_points_rg)
    g = poisson_forcing(interior_points_rg.detach())
    res = lap - g
    pde_mse = float((res ** 2).mean().detach().cpu())

    with torch.no_grad():
        u_bnd = model(boundary_points)
        if u_bnd.ndim == 2 and u_bnd.shape[1] == 1:
            u_bnd = u_bnd[:, 0]
        bnd_mse = float((u_bnd ** 2).mean().detach().cpu())

        u_pred = model(all_points)
        if u_pred.ndim == 2 and u_pred.shape[1] == 1:
            u_pred = u_pred[:, 0]
        u_true = manufactured_solution(all_points)
        l2_mse = float(((u_pred - u_true) ** 2).mean().detach().cpu())

    return {"pde_mse": pde_mse, "boundary_mse": bnd_mse, "l2_mse": l2_mse}


# =============================================================================
# CLI
# =============================================================================

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="PINN benchmark: SN vs KAN on 2D Poisson.")

    p.add_argument("--model", type=str, default="both", choices=["sn", "kan", "both"])
    p.add_argument("--epochs", type=int, default=5000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--force_cpu", action="store_true")

    p.add_argument("--target_params", type=int, default=1200)

    p.add_argument("--n_interior", type=int, default=2048)
    p.add_argument("--n_boundary_per_edge", type=int, default=257)  # 4*257=1028 when corners excluded
    p.add_argument("--no_boundary_corners", action="store_true")

    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight_decay", type=float, default=0.0)
    p.add_argument("--grad_clip", type=float, default=1.0)
    p.add_argument("--no_scheduler", action="store_true")
    p.add_argument("--scheduler_min_lr", type=float, default=1e-5)
    p.add_argument("--interior_weight", type=float, default=1.0)
    p.add_argument("--boundary_weight", type=float, default=1.0)

    p.add_argument("--no_pde_normalize", action="store_true")
    p.add_argument("--eval_grid", type=int, default=101)

    p.add_argument("--kan_layers", type=int, default=2)
    p.add_argument("--kan_width", type=int, default=None)
    p.add_argument("--kan_knots", type=int, default=None)

    p.add_argument("--sn_min_width", type=int, default=4)
    p.add_argument("--sn_max_width", type=int, default=32)
    p.add_argument("--sn_min_layers", type=int, default=2)
    p.add_argument("--sn_max_layers", type=int, default=2)
    p.add_argument("--sn_min_knots_each", type=int, default=8)

    p.add_argument("--sn_train_codomain", action="store_true")
    p.add_argument("--sn_domain_strategy", type=str, default="fixed", choices=["fixed"])
    p.add_argument("--sn_clamp_params", action="store_true")
    p.add_argument("--sn_init_Phi_std", type=float, default=1e-3)

    p.add_argument("--sn_eta_bound", type=float, default=0.25)
    p.add_argument("--sn_lambda_sigma", type=float, default=6.0)
    p.add_argument("--sn_phi_pad_abs", type=float, default=0.25)
    p.add_argument("--sn_phi_pad_rel", type=float, default=0.05)
    p.add_argument("--sn_Phi_pad_abs", type=float, default=1.0)
    p.add_argument("--sn_Phi_pad_rel", type=float, default=0.05)

    return p.parse_args()


def main() -> None:
    args = parse_args()

    if (not args.force_cpu) and torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    dtype = torch.float64

    print(f"Device: {device} | dtype: {dtype}")

    set_seed(args.seed)
    CONFIG["seed"] = int(args.seed)

    sn_choice = choose_sn_config(
        target_params=int(args.target_params),
        input_dim=2,
        min_width=int(args.sn_min_width),
        max_width=int(args.sn_max_width),
        min_layers=int(args.sn_min_layers),
        max_layers=int(args.sn_max_layers),
        min_knots_each=int(args.sn_min_knots_each),
        train_codomain=bool(args.sn_train_codomain),
    )

    fixed_knobs = SNFixedDomainKnobs(
        eta_bound=float(args.sn_eta_bound),
        lambda_sigma=float(args.sn_lambda_sigma),
        phi_pad_abs=float(args.sn_phi_pad_abs),
        phi_pad_rel=float(args.sn_phi_pad_rel),
        Phi_pad_abs=float(args.sn_Phi_pad_abs),
        Phi_pad_rel=float(args.sn_Phi_pad_rel),
    )

    print("\n" + "=" * 80)
    print("SN configuration:")
    print(f"  target_params        = {sn_choice.target_params}")
    print(f"  arch                 = {sn_choice.arch}")
    print(f"  phi_knots, Phi_knots = {sn_choice.phi_knots}, {sn_choice.Phi_knots}")
    diff_str = f"{sn_choice.diff:+d}"
    print(f"  est params           = {sn_choice.est_params} (diff={diff_str})")
    print(f"  compute proxy        = {sn_choice.compute_proxy} (lower is faster)")
    print(f"  train_codomain       = {sn_choice.train_codomain}")
    print("  residual/lateral     = OFF (fair)")
    print(f"  domain strategy      = {args.sn_domain_strategy}")
    print(
        "  fixed-domain knobs   = "
        f"eta_bound={fixed_knobs.eta_bound}, lambda_sigma={fixed_knobs.lambda_sigma}, "
        f"phi_pad_abs={fixed_knobs.phi_pad_abs}, phi_pad_rel={fixed_knobs.phi_pad_rel}, "
        f"Phi_pad_abs={fixed_knobs.Phi_pad_abs}, Phi_pad_rel={fixed_knobs.Phi_pad_rel}"
    )
    print("=" * 80)

    sn_model = build_sn_model_from_arch(
        arch=sn_choice.arch,
        phi_knots=sn_choice.phi_knots,
        Phi_knots=sn_choice.Phi_knots,
        device=device,
        dtype=dtype,
        train_codomain=bool(sn_choice.train_codomain),
        domain_strategy=str(args.sn_domain_strategy),
        fixed_knobs=fixed_knobs,
        init_Phi_std=float(args.sn_init_Phi_std),
    )
    sn_params_actual = count_trainable_params(sn_model)
    if sn_params_actual != sn_choice.est_params:
        print(f"[WARN] SN actual params ({sn_params_actual}) != estimate ({sn_choice.est_params}). Proceeding.")

    kan_model: Optional[nn.Module] = None
    kan_cfg: Optional[KANConfig] = None
    kan_params_actual: Optional[int] = None

    set_seed(args.seed)

    if args.model in ("kan", "both"):
        if args.kan_width is not None and args.kan_knots is not None:
            est = kan_param_count_formula(2, int(args.kan_width), int(args.kan_layers), int(args.kan_knots))
            kan_cfg = KANConfig(
                width=int(args.kan_width),
                knots=int(args.kan_knots),
                num_layers=int(args.kan_layers),
                est_params=int(est),
                diff=int(abs(est - sn_params_actual)),
            )
        else:
            kan_cfg = choose_kan_config(target_params=sn_params_actual, input_dim=2, num_layers=int(args.kan_layers))

        print("\n" + "=" * 80)
        print("KAN configuration:")
        print(f"  target_params (SN actual) = {sn_params_actual}")
        print(f"  width                     = {kan_cfg.width}")
        print(f"  knots                     = {kan_cfg.knots}")
        print(f"  layers                    = {kan_cfg.num_layers}")
        print(f"  est params                = {kan_cfg.est_params} (abs diff={kan_cfg.diff})")
        print("=" * 80)

        kan_model = KAN(
            input_dim=2,
            hidden_width=kan_cfg.width,
            num_layers=kan_cfg.num_layers,
            num_knots=kan_cfg.knots,
            dtype=dtype,
        ).to(device=device, dtype=dtype)

        kan_params_actual = count_trainable_params(kan_model)

    set_seed(args.seed)
    interior_points = sample_interior_points(int(args.n_interior), device=device, dtype=dtype)
    boundary_points = sample_boundary_points(
        int(args.n_boundary_per_edge),
        device=device,
        dtype=dtype,
        include_corners=(not args.no_boundary_corners),
    )

    print(f"\n[Data] Interior points: {interior_points.shape[0]}")
    print(f"[Data] Boundary points: {boundary_points.shape[0]}")

    pde_norm_denom = 1.0
    if not args.no_pde_normalize:
        with torch.no_grad():
            g_vals = poisson_forcing(interior_points)
            forcing_rms = float(torch.sqrt(torch.mean(g_vals ** 2)).cpu())
            pde_norm_denom = max(1e-12, forcing_rms ** 2)
        print(f"[PDE] Normalization: forcing RMS = {forcing_rms:.4e} (PDE loss divided by RMS^2)")

    use_scheduler = not args.no_scheduler

    results: Dict[str, Dict[str, float]] = {}

    if args.model in ("sn", "both"):
        sn_core = get_sn_core_from_wrapper(sn_model)

        post_step_hook = None
        if bool(args.sn_clamp_params):
            post_step_hook = lambda: sn_apply_param_clamps(sn_core, fixed_knobs)

        t0 = time.time()
        _ = train_pinn(
            model=sn_model,
            interior_points=interior_points,
            boundary_points=boundary_points,
            epochs=int(args.epochs),
            lr=float(args.lr),
            weight_decay=float(args.weight_decay),
            grad_clip=float(args.grad_clip),
            use_scheduler=use_scheduler,
            scheduler_min_lr=float(args.scheduler_min_lr),
            interior_weight=float(args.interior_weight),
            boundary_weight=float(args.boundary_weight),
            pde_norm_denom=float(pde_norm_denom),
            label="SN",
            post_step_hook=post_step_hook,
            log_every=500,
        )
        sn_train_time = time.time() - t0

        print("\n[SN] Evaluating on dense grid...")
        sn_eval = evaluate_model_on_grid(sn_model, device=device, dtype=dtype, grid_size=int(args.eval_grid))
        results["sn"] = {**sn_eval, "params": float(sn_params_actual), "time_s": float(sn_train_time)}
        print(f"  PDE residual MSE = {sn_eval['pde_mse']:.4e}")
        print(f"  boundary MSE     = {sn_eval['boundary_mse']:.4e}")
        print(f"  L2 solution MSE  = {sn_eval['l2_mse']:.4e}")
        print(f"  params           = {sn_params_actual}")
        print(f"  time             = {sn_train_time:.1f}s ({sn_train_time/60.0:.1f}m)")

    if args.model in ("kan", "both") and kan_model is not None:
        t0 = time.time()
        _ = train_pinn(
            model=kan_model,
            interior_points=interior_points,
            boundary_points=boundary_points,
            epochs=int(args.epochs),
            lr=float(args.lr),
            weight_decay=float(args.weight_decay),
            grad_clip=float(args.grad_clip),
            use_scheduler=use_scheduler,
            scheduler_min_lr=float(args.scheduler_min_lr),
            interior_weight=float(args.interior_weight),
            boundary_weight=float(args.boundary_weight),
            pde_norm_denom=float(pde_norm_denom),
            label="KAN",
            post_step_hook=None,
            log_every=500,
        )
        kan_train_time = time.time() - t0

        print("\n[KAN] Evaluating on dense grid...")
        kan_eval = evaluate_model_on_grid(kan_model, device=device, dtype=dtype, grid_size=int(args.eval_grid))
        results["kan"] = {**kan_eval, "params": float(kan_params_actual or 0), "time_s": float(kan_train_time)}
        print(f"  PDE residual MSE = {kan_eval['pde_mse']:.4e}")
        print(f"  boundary MSE     = {kan_eval['boundary_mse']:.4e}")
        print(f"  L2 solution MSE  = {kan_eval['l2_mse']:.4e}")
        print(f"  params           = {kan_params_actual}")
        print(f"  time             = {kan_train_time:.1f}s ({kan_train_time/60.0:.1f}m)")

    if args.model == "both" and ("sn" in results) and ("kan" in results):
        sn = results["sn"]
        kan = results["kan"]

        print("\n" + "=" * 80)
        print("Final comparison on dense grid:")
        print("=" * 80)
        print(
            f"SN  : PDE MSE = {sn['pde_mse']:.4e}, boundary MSE = {sn['boundary_mse']:.4e}, "
            f"L2 MSE = {sn['l2_mse']:.4e}, params = {int(sn['params'])}, time = {sn['time_s']:.1f}s"
        )
        print(
            f"KAN : PDE MSE = {kan['pde_mse']:.4e}, boundary MSE = {kan['boundary_mse']:.4e}, "
            f"L2 MSE = {kan['l2_mse']:.4e}, params = {int(kan['params'])}, time = {kan['time_s']:.1f}s"
        )
        print("=" * 80)

    print("\nDone.")


if __name__ == "__main__":
    main()