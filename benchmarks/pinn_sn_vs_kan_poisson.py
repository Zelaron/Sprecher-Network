"""
pinn_sn_vs_kan_poisson8.py

Benchmarking Sprecher Networks (SNs) vs Kolmogorov–Arnold Networks (KANs) on a 2D Poisson PINN.

Key improvements in this version:
1. Target ~2000 parameters for both networks (parameter parity)
2. Fix KAN boundary MSE = 0 issue by proper input scaling and initialization
3. Fair comparison: disable residual connections and lateral mixing in SN
4. Use PCHIP cubic splines for both networks
5. Proper domain handling for KAN to match the problem domain [-1, 1]²
"""

# Short benchmark description:
# We solve a manufactured Poisson problem on Ω = [-1, 1]^2 with zero Dirichlet boundary conditions, using a PINN
# built either from a Sprecher Network (SN) or a KAN-style spline network. The exact solution is
# u(x, y) = sin(π x) sin(π y²), which induces an anisotropic forcing term with rapidly varying curvature in y,
# especially near |y| ≈ 1. This is a natural, smooth, yet nontrivial test that has appeared (in nearly identical
# form) in the KAN literature and is representative of elliptic PDE benchmarks.

# Fairness note:
# This script implements an apples-to-apples Poisson PINN benchmark: SN and KAN use the same domain, PDE,
# collocation points, optimizer, learning rate, and PINN loss (with identical weights on interior and boundary terms).
# Both networks use cubic splines (PCHIP for SN, standard cubic for KAN). Residual connections and lateral
# mixing are DISABLED in the SN to ensure a fair comparison since KAN lacks these features.

import argparse
import os
import json
import math
import time
from dataclasses import dataclass
from typing import Tuple, Optional, List, Dict

import numpy as np
import torch
import torch.nn as nn
import torch.autograd as autograd

from tqdm import trange

from sn_core import (
    SprecherMultiLayerNetwork,
    SimpleSpline,
    CONFIG,
)

# ---------------------------------------------------------------------------
# Utilities: seeding and device handling
# ---------------------------------------------------------------------------


def set_seed(seed: int) -> None:
    """Set random seeds for reproducibility."""
    import random

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device(device_str: str) -> torch.device:
    if device_str == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_str)


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# ---------------------------------------------------------------------------
# Manufactured solution and Poisson operator
# ---------------------------------------------------------------------------

DOMAIN_MIN = -1.0
DOMAIN_MAX = 1.0


def manufactured_solution(xy: torch.Tensor) -> torch.Tensor:
    """
    Analytic solution u(x, y) = sin(pi x) sin(pi y^2)
    defined on Ω = [-1, 1]^2, with homogeneous Dirichlet boundary conditions.
    """
    x = xy[..., 0]
    y = xy[..., 1]
    pi = math.pi
    return torch.sin(pi * x) * torch.sin(pi * y ** 2)


def poisson_forcing(xy: torch.Tensor) -> torch.Tensor:
    """
    Forcing term g(x, y) such that Δu - g = 0 with u = manufactured_solution.

    From the derivation (and matching Liu et al. 2024, Section 3.4),
    Δu(x, y) = -π²(1 + 4 y²) sin(π x) sin(π y²) + 2π sin(π x) cos(π y²).
    We define g(x, y) = Δu(x, y), so the PDE is Δu - g = 0.
    """
    x = xy[..., 0]
    y = xy[..., 1]
    pi = math.pi
    return (
        -pi ** 2 * (1.0 + 4.0 * y ** 2) * torch.sin(pi * x) * torch.sin(pi * y ** 2)
        + 2.0 * pi * torch.sin(pi * x) * torch.cos(pi * y ** 2)
    )


def laplacian_u(model: nn.Module, xy: torch.Tensor) -> torch.Tensor:
    """
    Compute Laplacian Δu of the model output u(x, y) with respect to x and y.

    xy: tensor of shape [N, 2] with requires_grad=True.
    Returns: tensor of shape [N] containing Δu at each point.
    """
    u = model(xy)  # [N, 1] or [N]
    if u.ndim == 2 and u.shape[1] == 1:
        u = u[:, 0]
    elif u.ndim != 1:
        raise ValueError("Model output must be shape [N] or [N, 1]")

    grads = autograd.grad(
        outputs=u,
        inputs=xy,
        grad_outputs=torch.ones_like(u),
        create_graph=True,
        retain_graph=True,
    )[0]  # [N, 2]
    u_x = grads[:, 0]
    u_y = grads[:, 1]

    u_xx = autograd.grad(
        outputs=u_x,
        inputs=xy,
        grad_outputs=torch.ones_like(u_x),
        create_graph=True,
        retain_graph=True,
    )[0][:, 0]

    u_yy = autograd.grad(
        outputs=u_y,
        inputs=xy,
        grad_outputs=torch.ones_like(u_y),
        create_graph=True,
        retain_graph=True,
    )[0][:, 1]

    return u_xx + u_yy


# ---------------------------------------------------------------------------
# KAN implementation with cubic splines (improved version)
# ---------------------------------------------------------------------------


class CubicSpline(nn.Module):
    """
    A cubic spline with PCHIP-style slope computation for smooth interpolation.
    Uses trainable coefficients at knot positions.
    
    This matches the SN's cubic spline implementation for fair comparison.
    """
    
    def __init__(
        self, 
        num_knots: int = 8, 
        in_min: float = -1.0, 
        in_max: float = 1.0,
        init_scale: float = 0.1
    ):
        super().__init__()
        self.num_knots = num_knots
        self.in_min = in_min
        self.in_max = in_max
        
        # Fixed knot positions
        knots = torch.linspace(in_min, in_max, num_knots)
        self.register_buffer('knots', knots)
        
        # Trainable coefficients at each knot
        # Initialize with small random values for good gradient flow
        self.coeffs = nn.Parameter(torch.randn(num_knots) * init_scale)
    
    def _pchip_slopes(self, y: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """Compute PCHIP slopes for monotonicity-preserving cubic interpolation."""
        K = x.shape[0]
        if K < 2:
            return torch.zeros_like(y)
        
        h = x[1:] - x[:-1]  # (K-1,)
        delta = (y[1:] - y[:-1]) / (h + 1e-12)  # (K-1,)
        
        d = torch.zeros_like(y)  # (K,)
        
        if K == 2:
            d[0] = delta[0]
            d[1] = delta[0]
            return d
        
        # Interior points: weighted harmonic mean
        for i in range(1, K - 1):
            if delta[i-1] * delta[i] > 0:
                w1 = 2 * h[i] + h[i-1]
                w2 = h[i] + 2 * h[i-1]
                d[i] = (w1 + w2) / (w1 / (delta[i-1] + 1e-12) + w2 / (delta[i] + 1e-12) + 1e-12)
            else:
                d[i] = 0.0
        
        # Boundary slopes: one-sided
        d[0] = delta[0]
        d[-1] = delta[-1]
        
        return d
    
    def _hermite_eval(
        self, 
        x_vals: torch.Tensor, 
        x: torch.Tensor, 
        y: torch.Tensor, 
        d: torch.Tensor
    ) -> torch.Tensor:
        """Evaluate cubic Hermite interpolant."""
        K = x.shape[0]
        
        # Clamp to domain
        x_clamped = torch.clamp(x_vals, x[0], x[-1])
        
        # Find interval indices
        idx = torch.searchsorted(x, x_clamped) - 1
        idx = torch.clamp(idx, 0, K - 2)
        
        xk = x[idx]
        xk1 = x[idx + 1]
        yk = y[idx]
        yk1 = y[idx + 1]
        dk = d[idx]
        dk1 = d[idx + 1]
        
        h = xk1 - xk
        safe_h = torch.where(h.abs() < 1e-12, torch.ones_like(h), h)
        t = torch.where(h.abs() < 1e-12, torch.zeros_like(x_vals), (x_clamped - xk) / safe_h)
        
        t2 = t * t
        t3 = t2 * t
        h00 = 2 * t3 - 3 * t2 + 1
        h10 = t3 - 2 * t2 + t
        h01 = -2 * t3 + 3 * t2
        h11 = t3 - t2
        
        return h00 * yk + h10 * safe_h * dk + h01 * yk1 + h11 * safe_h * dk1
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Evaluate spline at given points."""
        original_shape = x.shape
        x_flat = x.reshape(-1)
        
        y = self.coeffs
        d = self._pchip_slopes(y, self.knots)
        
        # Interpolate within domain
        result = self._hermite_eval(x_flat, self.knots, y, d)
        
        # Linear extrapolation outside domain
        below_domain = x_flat < self.in_min
        above_domain = x_flat > self.in_max
        
        left_slope = d[0]
        right_slope = d[-1]
        
        result = torch.where(
            below_domain,
            y[0] + left_slope * (x_flat - self.in_min),
            result
        )
        result = torch.where(
            above_domain,
            y[-1] + right_slope * (x_flat - self.in_max),
            result
        )
        
        return result.reshape(original_shape)


class KANLayer(nn.Module):
    """
    A Kolmogorov–Arnold Network (KAN) layer with cubic splines:

        h_i(x) = sum_j s_ij(x_j) + b_i,

    where each s_ij is a univariate cubic spline with PCHIP slopes.
    
    Note: We use the spline outputs directly (no additional linear weights)
    to match standard KAN architecture. The spline coefficients already
    provide sufficient expressive power.
    """

    def __init__(self, d_in: int, d_out: int, num_knots: int = 8):
        super().__init__()
        self.d_in = d_in
        self.d_out = d_out
        self.num_knots = num_knots

        # One spline per edge (i, j)
        self.splines = nn.ModuleList([
            nn.ModuleList([
                CubicSpline(
                    num_knots=num_knots,
                    in_min=DOMAIN_MIN,
                    in_max=DOMAIN_MAX,
                    init_scale=0.1 / math.sqrt(d_in)  # Scale initialization
                )
                for _ in range(d_in)
            ])
            for _ in range(d_out)
        ])

        # Bias per output
        self.bias = nn.Parameter(torch.zeros(d_out))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [N, d_in]
        N, d_in = x.shape
        assert d_in == self.d_in

        outputs = []
        for i in range(self.d_out):
            acc = torch.zeros(N, device=x.device, dtype=x.dtype)
            for j in range(self.d_in):
                x_j = x[:, j]  # [N]
                s_val = self.splines[i][j](x_j)  # [N]
                acc = acc + s_val
            outputs.append(acc + self.bias[i])
        
        return torch.stack(outputs, dim=-1)  # [N, d_out]


class KAN(nn.Module):
    """
    Multi-layer KAN with configurable depth and width.
    
    Architecture: input_dim -> [hidden_width] * num_layers -> 1
    """

    def __init__(
        self, 
        input_dim: int = 2,
        hidden_width: int = 8, 
        num_layers: int = 2,
        num_knots: int = 6
    ):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_width = hidden_width
        self.num_layers = num_layers
        self.num_knots = num_knots

        layers = []
        d_in = input_dim
        
        for i in range(num_layers):
            if i == num_layers - 1:
                # Final layer outputs to 1
                d_out = 1
            else:
                d_out = hidden_width
            
            layers.append(KANLayer(d_in=d_in, d_out=d_out, num_knots=num_knots))
            d_in = d_out
        
        self.layers = nn.ModuleList(layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x)
        return x  # [N, 1]


def approximate_kan_params(
    input_dim: int, 
    hidden_width: int, 
    num_layers: int, 
    num_knots: int
) -> int:
    """
    Compute the parameter count for a KAN with given configuration.
    
    For each KANLayer with d_in -> d_out:
      - Splines: d_in * d_out * num_knots parameters (coeffs)
      - Biases: d_out
    Total per layer: d_in * d_out * num_knots + d_out
    """
    total = 0
    d_in = input_dim
    
    for i in range(num_layers):
        if i == num_layers - 1:
            d_out = 1
        else:
            d_out = hidden_width
        
        # Spline coeffs + bias
        total += d_in * d_out * num_knots + d_out
        d_in = d_out
    
    return total


def build_kan_model_for_params(
    target_params: int,
    input_dim: int = 2,
    num_layers: int = 2,
    width_candidates: Optional[List[int]] = None,
    knot_candidates: Optional[List[int]] = None,
) -> Tuple[KAN, int, int]:
    """
    Given a target parameter count, choose (width, num_knots) for the KAN
    so that its total trainable parameters are as close as possible to target_params.

    Returns (model, width, num_knots).
    """
    if width_candidates is None:
        width_candidates = list(range(4, 64, 2))  # 4, 6, 8, ..., 62
    if knot_candidates is None:
        knot_candidates = list(range(4, 32, 2))  # 4, 6, 8, ..., 30

    best_cfg = None
    best_diff = float("inf")

    for w in width_candidates:
        for K in knot_candidates:
            approx_params = approximate_kan_params(input_dim, w, num_layers, K)
            diff = abs(approx_params - target_params)
            if diff < best_diff:
                best_diff = diff
                best_cfg = (w, K, approx_params)

    if best_cfg is None:
        # Fallback
        best_cfg = (16, 10, approximate_kan_params(input_dim, 16, num_layers, 10))

    width, num_knots, approx = best_cfg
    model = KAN(
        input_dim=input_dim,
        hidden_width=width,
        num_layers=num_layers,
        num_knots=num_knots
    )
    exact_params = count_parameters(model)
    
    print(
        f"[KAN] Auto-matched hyperparameters to SN params={target_params}: "
        f"width={width}, num_knots={num_knots}, num_layers={num_layers}, "
        f"KAN_params={exact_params} (abs diff={abs(exact_params - target_params)})"
    )
    return model, width, num_knots


# ---------------------------------------------------------------------------
# SN model builder (fair configuration)
# ---------------------------------------------------------------------------


def count_sn_params(
    input_dim: int,
    architecture: List[int],
    final_dim: int,
    phi_knots: int,
    Phi_knots: int,
    use_residual: bool,
    use_lateral: bool,
) -> int:
    """
    Estimate parameter count for a Sprecher Network.
    
    Per layer (d_in -> d_out):
    - phi spline (monotonic): num_knots log_increments
    - Phi spline: num_knots coeffs
    - lambdas: d_in
    - eta: 1
    - Codomain params (if enabled): 2 (cc, cr)
    - Residual (if enabled): varies
    - Lateral mixing (if enabled): d_out
    """
    total = 0
    dims = [input_dim] + architecture
    
    # Check if final layer is included in architecture
    if len(architecture) > 0:
        n_layers = len(architecture)
        # If final_dim == 1 and architecture[-1] != 1, the last layer is final
        if final_dim == 1:
            pass  # Last layer in architecture is the final layer
        else:
            # Need an additional layer for final_dim > 1
            dims.append(final_dim)
            n_layers += 1
    else:
        # No hidden layers, just input -> output
        dims.append(final_dim)
        n_layers = 1
    
    for i in range(n_layers):
        d_in = dims[i]
        d_out = dims[i + 1] if i + 1 < len(dims) else final_dim
        
        # phi and Phi splines
        total += phi_knots  # phi log_increments
        total += Phi_knots  # Phi coeffs
        
        # lambdas and eta
        total += d_in + 1
        
        # Codomain params (cc, cr)
        total += 2
        
        # Residual connections
        if use_residual:
            if d_in == d_out:
                total += 1  # scalar weight
            elif d_in > d_out:
                total += d_in  # pooling weights
            else:
                total += d_out  # broadcast weights
        
        # Lateral mixing
        if use_lateral:
            total += d_out  # cyclic mixing weights
            total += 1  # lateral scale
    
    # Output scale and bias
    total += 2
    
    return total


def build_sn_model(
    sn_width: int,
    sn_layers: int,
    phi_knots: int,
    Phi_knots: int,
    use_residual: bool = False,
    use_lateral: bool = False,
) -> SprecherMultiLayerNetwork:
    """
    Build a Sprecher Network configured for fair PINN comparison.

    For apples-to-apples comparison with KAN:
    - Disable normalization (norm_type='none')
    - Use cubic splines (PCHIP) for both φ and Φ
    - Optionally disable residual connections and lateral mixing
    """
    # Configure global settings
    CONFIG['use_residual_weights'] = use_residual
    CONFIG['use_lateral_mixing'] = use_lateral
    CONFIG['train_phi_codomain'] = True
    CONFIG['norm_type'] = 'none'
    
    architecture = [sn_width] * sn_layers
    
    model = SprecherMultiLayerNetwork(
        input_dim=2,
        architecture=architecture,
        final_dim=1,
        phi_knots=phi_knots,
        Phi_knots=Phi_knots,
        norm_type="none",
        norm_position="after",
        norm_skip_first=True,
        initialize_domains=True,
        domain_ranges=None,
        phi_spline_type="cubic",  # PCHIP cubic splines
        Phi_spline_type="cubic",  # PCHIP cubic splines
        phi_spline_order=3,
        Phi_spline_order=3,
    )
    return model


def search_sn_config_for_params(
    target_params: int,
    width_candidates: Optional[List[int]] = None,
    layer_candidates: Optional[List[int]] = None,
    knot_candidates: Optional[List[int]] = None,
    use_residual: bool = False,
    use_lateral: bool = False,
) -> Tuple[int, int, int, int]:
    """
    Search for SN configuration that achieves approximately target_params.
    
    Returns (width, num_layers, phi_knots, Phi_knots).
    """
    if width_candidates is None:
        width_candidates = list(range(8, 64, 4))
    if layer_candidates is None:
        layer_candidates = [1, 2, 3]
    if knot_candidates is None:
        knot_candidates = list(range(16, 128, 8))
    
    best_cfg = None
    best_diff = float("inf")
    
    for width in width_candidates:
        for num_layers in layer_candidates:
            for knots in knot_candidates:
                params = count_sn_params(
                    input_dim=2,
                    architecture=[width] * num_layers,
                    final_dim=1,
                    phi_knots=knots,
                    Phi_knots=knots,
                    use_residual=use_residual,
                    use_lateral=use_lateral,
                )
                diff = abs(params - target_params)
                if diff < best_diff:
                    best_diff = diff
                    best_cfg = (width, num_layers, knots, knots)
    
    return best_cfg


# ---------------------------------------------------------------------------
# Collocation sampling
# ---------------------------------------------------------------------------


def sample_interior_points(n_points: int, device: torch.device) -> torch.Tensor:
    """Uniformly sample interior collocation points in Ω = [-1, 1]^2."""
    xy = (DOMAIN_MAX - DOMAIN_MIN) * torch.rand(n_points, 2, device=device) + DOMAIN_MIN
    return xy


def sample_boundary_points(n_per_edge: int, device: torch.device) -> torch.Tensor:
    """
    Sample boundary points on ∂Ω = {x = ±1} ∪ {y = ±1}, uniformly along each edge.
    Total boundary points = 4 * n_per_edge (corners are duplicated, which is fine).
    """
    xs = (DOMAIN_MAX - DOMAIN_MIN) * torch.rand(n_per_edge, device=device) + DOMAIN_MIN
    ys = xs.clone()

    # Top y = 1 and bottom y = -1 edges
    top = torch.stack([xs, torch.full_like(xs, DOMAIN_MAX)], dim=-1)
    bottom = torch.stack([xs, torch.full_like(xs, DOMAIN_MIN)], dim=-1)

    # Left x = -1 and right x = 1 edges
    left = torch.stack([torch.full_like(ys, DOMAIN_MIN), ys], dim=-1)
    right = torch.stack([torch.full_like(ys, DOMAIN_MAX), ys], dim=-1)

    boundary = torch.cat([top, bottom, left, right], dim=0)
    return boundary


# ---------------------------------------------------------------------------
# Input normalization wrapper for SN
# ---------------------------------------------------------------------------


class SNInputNormalizer(nn.Module):
    """
    Wrapper to normalize inputs from [-1, 1] to [0, 1] for SN.
    
    SN internally uses [0, 1]^n domain, so we need to transform inputs.
    """
    def __init__(self, sn_model: nn.Module):
        super().__init__()
        self.sn_model = sn_model
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Transform from [-1, 1] to [0, 1]
        x_normalized = (x - DOMAIN_MIN) / (DOMAIN_MAX - DOMAIN_MIN)
        return self.sn_model(x_normalized)


# ---------------------------------------------------------------------------
# PINN loss and training
# ---------------------------------------------------------------------------


@dataclass
class PinnLossComponents:
    total: float
    interior: float
    boundary: float


def compute_pinn_loss(
    model: nn.Module,
    interior_points: torch.Tensor,
    boundary_points: torch.Tensor,
    interior_weight: float = 1.0,
    boundary_weight: float = 1.0,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Compute the PINN loss:
      L = w_int * E[|Δu - g|²] + w_bnd * E[|u|² on ∂Ω].
    """
    # Interior residual
    xy_int = interior_points.clone().detach().requires_grad_(True)
    lap = laplacian_u(model, xy_int)
    g = poisson_forcing(xy_int)
    residual = lap - g
    interior_loss = residual.pow(2).mean()

    # Boundary term (Dirichlet u=0)
    xy_bnd = boundary_points
    u_bnd = model(xy_bnd)
    if u_bnd.ndim == 2 and u_bnd.shape[1] == 1:
        u_bnd = u_bnd[:, 0]
    boundary_loss = u_bnd.pow(2).mean()

    total_loss = interior_weight * interior_loss + boundary_weight * boundary_loss
    return total_loss, interior_loss, boundary_loss


def train_pinn(
    model: nn.Module,
    interior_points: torch.Tensor,
    boundary_points: torch.Tensor,
    epochs: int,
    lr: float,
    weight_decay: float,
    interior_weight: float,
    boundary_weight: float,
    device: torch.device,
    label: str,
    log_interval: int = 200,
    use_scheduler: bool = True,
) -> PinnLossComponents:
    """
    Full-batch PINN training loop for a single model.
    """
    model.to(device)
    model.train()

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    
    # Learning rate scheduler for better convergence
    if use_scheduler:
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=epochs, eta_min=lr * 0.01
        )
    else:
        scheduler = None

    interior_points = interior_points.to(device)
    boundary_points = boundary_points.to(device)

    pbar = trange(1, epochs + 1, desc=f"Training {label} PINN", ncols=100)
    final_total = final_int = final_bnd = None

    for epoch in pbar:
        optimizer.zero_grad()
        total_loss, interior_loss, boundary_loss = compute_pinn_loss(
            model,
            interior_points,
            boundary_points,
            interior_weight=interior_weight,
            boundary_weight=boundary_weight,
        )
        total_loss.backward()
        
        # Gradient clipping for stability
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()
        
        if scheduler is not None:
            scheduler.step()

        final_total = total_loss.item()
        final_int = interior_loss.item()
        final_bnd = boundary_loss.item()

        if epoch == 1 or epoch % log_interval == 0 or epoch == epochs:
            current_lr = optimizer.param_groups[0]['lr']
            pbar.write(
                f"[{label}] Epoch {epoch:5d}/{epochs}: "
                f"loss={final_total:.4e}, interior={final_int:.4e}, boundary={final_bnd:.4e}, lr={current_lr:.2e}"
            )

    print(f"\n[{label}] Final training losses:")
    print(f"  total     = {final_total:.4e}")
    print(f"  interior  = {final_int:.4e}")
    print(f"  boundary  = {final_bnd:.4e}")

    return PinnLossComponents(total=final_total, interior=final_int, boundary=final_bnd)


# ---------------------------------------------------------------------------
# Evaluation on dense grid
# ---------------------------------------------------------------------------


def make_evaluation_grids(
    grid_size: int, device: torch.device
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Create evaluation grids:
      - interior grid points (excluding boundary),
      - boundary grid points,
      - full grid for L2 error vs analytic solution.
    """
    xs = torch.linspace(DOMAIN_MIN, DOMAIN_MAX, grid_size, device=device)
    X, Y = torch.meshgrid(xs, xs, indexing="ij")
    all_points = torch.stack([X.reshape(-1), Y.reshape(-1)], dim=-1)

    # Interior: remove first/last indices along each axis
    xs_inner = xs[1:-1]
    X_inner, Y_inner = torch.meshgrid(xs_inner, xs_inner, indexing="ij")
    interior_points = torch.stack([X_inner.reshape(-1), Y_inner.reshape(-1)], dim=-1)

    # Boundary: four edges of the grid
    # Top and bottom edges
    top = torch.stack([xs, torch.full_like(xs, DOMAIN_MAX)], dim=-1)
    bottom = torch.stack([xs, torch.full_like(xs, DOMAIN_MIN)], dim=-1)
    # Left and right edges (excluding corners to avoid duplicates)
    ys_mid = xs[1:-1]
    left = torch.stack([torch.full_like(ys_mid, DOMAIN_MIN), ys_mid], dim=-1)
    right = torch.stack([torch.full_like(ys_mid, DOMAIN_MAX), ys_mid], dim=-1)
    boundary_points = torch.cat([top, bottom, left, right], dim=0)

    return interior_points, boundary_points, all_points


def evaluate_model_on_grid(
    model: nn.Module,
    device: torch.device,
    grid_size: int = 101,
) -> Dict[str, float]:
    """
    Evaluate PDE residual, boundary violation, and L2 solution error on a dense grid.
    """
    model.to(device)
    model.eval()

    interior_points, boundary_points, all_points = make_evaluation_grids(
        grid_size=grid_size, device=device
    )

    # PDE residual on interior
    interior_points = interior_points.requires_grad_(True)
    lap = laplacian_u(model, interior_points)
    g = poisson_forcing(interior_points)
    residual = lap - g
    pde_mse = residual.pow(2).mean().item()

    # Boundary violation
    with torch.no_grad():
        u_bnd = model(boundary_points)
        if u_bnd.ndim == 2 and u_bnd.shape[1] == 1:
            u_bnd = u_bnd[:, 0]
        boundary_mse = u_bnd.pow(2).mean().item()

        # L2 error vs analytic solution on full grid
        u_pred = model(all_points)
        if u_pred.ndim == 2 and u_pred.shape[1] == 1:
            u_pred = u_pred[:, 0]
        u_true = manufactured_solution(all_points)
        l2_mse = (u_pred - u_true).pow(2).mean().item()

    return {
        "pde_residual_mse": pde_mse,
        "boundary_mse": boundary_mse,
        "l2_solution_mse": l2_mse,
    }


# ---------------------------------------------------------------------------
# Argument parsing and main driver
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="PINN Poisson benchmark: Sprecher Network vs KAN (v8 - improved)"
    )

    parser.add_argument(
        "--model",
        type=str,
        choices=["sn", "kan", "both"],
        default="both",
        help="Which model(s) to train: sn, kan, or both (default: both).",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=5000,
        help="Number of training epochs for each model (default: 5000).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random seed (default: 0).",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cpu", "cuda"],
        help="Device to use (default: auto).",
    )

    # Target parameter count
    parser.add_argument(
        "--target_params",
        type=int,
        default=2000,
        help="Target number of parameters for both networks (default: 2000).",
    )

    # Collocation and evaluation
    parser.add_argument(
        "--n_interior",
        type=int,
        default=2048,
        help="Number of interior collocation points (default: 2048).",
    )
    parser.add_argument(
        "--n_boundary_per_edge",
        type=int,
        default=256,
        help="Number of boundary collocation points per edge (default: 256; total = 4 * this).",
    )
    parser.add_argument(
        "--grid_size",
        type=int,
        default=101,
        help="Evaluation grid size along one dimension (default: 101).",
    )

    # Shared optimizer hyperparameters
    parser.add_argument(
        "--lr",
        type=float,
        default=1e-3,
        help="Learning rate for Adam optimizer (default: 1e-3).",
    )
    parser.add_argument(
        "--weight_decay",
        type=float,
        default=1e-6,
        help="Weight decay for Adam optimizer (default: 1e-6).",
    )
    parser.add_argument(
        "--interior_weight",
        type=float,
        default=1.0,
        help="Weight on interior PDE residual loss (default: 1.0).",
    )
    parser.add_argument(
        "--boundary_weight",
        type=float,
        default=1.0,
        help="Weight on boundary condition loss (default: 1.0).",
    )
    parser.add_argument(
        "--log_interval",
        type=int,
        default=500,
        help="Epoch interval for logging training progress (default: 500).",
    )

    # Fairness options
    parser.add_argument(
        "--sn_residual",
        action="store_true",
        help="Enable residual connections in SN (default: disabled for fairness).",
    )
    parser.add_argument(
        "--sn_lateral",
        action="store_true",
        help="Enable lateral mixing in SN (default: disabled for fairness).",
    )

    # Manual SN hyperparameters (optional overrides)
    parser.add_argument(
        "--sn_width",
        type=int,
        default=None,
        help="Hidden width for SN. If not set, auto-chosen to match target_params.",
    )
    parser.add_argument(
        "--sn_layers",
        type=int,
        default=None,
        help="Number of hidden layers for SN. If not set, auto-chosen.",
    )
    parser.add_argument(
        "--sn_phi_knots",
        type=int,
        default=None,
        help="Number of knots for SN φ-splines. If not set, auto-chosen.",
    )
    parser.add_argument(
        "--sn_Phi_knots",
        type=int,
        default=None,
        help="Number of knots for SN Φ-splines. If not set, auto-chosen.",
    )

    # KAN hyperparameters (optional overrides)
    parser.add_argument(
        "--kan_width",
        type=int,
        default=None,
        help="Hidden width for KAN. If not set, auto-chosen to match SN parameter count.",
    )
    parser.add_argument(
        "--kan_knots",
        type=int,
        default=None,
        help="Number of spline knots for KAN. If not set, auto-chosen.",
    )
    parser.add_argument(
        "--kan_layers",
        type=int,
        default=2,
        help="Number of layers in KAN (default: 2).",
    )

    # Logging
    parser.add_argument(
        "--log_json",
        action="store_true",
        help="If set, write final metrics to a JSON file in --log_dir.",
    )
    parser.add_argument(
        "--log_dir",
        type=str,
        default="logs_pinn_poisson",
        help="Directory to store JSON logs (default: logs_pinn_poisson).",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    device = get_device(args.device)
    print(f"\nUsing device: {device}\n")

    set_seed(args.seed)
    CONFIG["seed"] = args.seed

    print("=" * 80)
    print(f"Running PINN Poisson benchmark (v8) with seed={args.seed}")
    print(f"Target parameters: ~{args.target_params}")
    print(f"Fairness settings: SN residual={args.sn_residual}, SN lateral={args.sn_lateral}")
    print("=" * 80)

    # ----------------------------------------------------------------------
    # Build SN model
    # ----------------------------------------------------------------------
    print("\n[SN] Building Sprecher Network model...")
    
    if args.sn_width is not None and args.sn_layers is not None and args.sn_phi_knots is not None:
        # Use manual configuration
        sn_width = args.sn_width
        sn_layers = args.sn_layers
        sn_phi_knots = args.sn_phi_knots
        sn_Phi_knots = args.sn_Phi_knots or args.sn_phi_knots
    else:
        # Auto-search for configuration
        sn_cfg = search_sn_config_for_params(
            target_params=args.target_params,
            use_residual=args.sn_residual,
            use_lateral=args.sn_lateral,
        )
        sn_width, sn_layers, sn_phi_knots, sn_Phi_knots = sn_cfg
        print(f"[SN] Auto-selected config: width={sn_width}, layers={sn_layers}, "
              f"phi_knots={sn_phi_knots}, Phi_knots={sn_Phi_knots}")
    
    sn_model_raw = build_sn_model(
        sn_width=sn_width,
        sn_layers=sn_layers,
        phi_knots=sn_phi_knots,
        Phi_knots=sn_Phi_knots,
        use_residual=args.sn_residual,
        use_lateral=args.sn_lateral,
    )
    
    # Wrap SN with input normalizer ([-1,1] -> [0,1])
    sn_model = SNInputNormalizer(sn_model_raw)
    sn_num_params = count_parameters(sn_model)
    print(f"[SN] Total trainable parameters: {sn_num_params}")

    # ----------------------------------------------------------------------
    # Build KAN model, matching parameter count as closely as possible
    # ----------------------------------------------------------------------
    if args.model in ("kan", "both"):
        set_seed(args.seed)
        CONFIG["seed"] = args.seed

        print("\n[KAN] Building KAN model...")
        if args.kan_width is not None and args.kan_knots is not None:
            kan_model = KAN(
                input_dim=2,
                hidden_width=args.kan_width, 
                num_layers=args.kan_layers,
                num_knots=args.kan_knots
            )
            kan_num_params = count_parameters(kan_model)
            print(
                f"[KAN] Using user-specified hyperparameters: width={args.kan_width}, "
                f"num_knots={args.kan_knots}, num_layers={args.kan_layers}, params={kan_num_params}"
            )
        else:
            kan_model, auto_width, auto_knots = build_kan_model_for_params(
                target_params=sn_num_params,
                input_dim=2,
                num_layers=args.kan_layers,
            )
            args.kan_width = auto_width
            args.kan_knots = auto_knots
            kan_num_params = count_parameters(kan_model)
        print(f"[KAN] Total trainable parameters: {kan_num_params}")
    else:
        kan_model = None
        kan_num_params = None

    # ----------------------------------------------------------------------
    # Sample collocation points (shared between models)
    # ----------------------------------------------------------------------
    set_seed(args.seed)  # Reset seed for consistent collocation sampling
    interior_points = sample_interior_points(args.n_interior, device=device)
    boundary_points = sample_boundary_points(args.n_boundary_per_edge, device=device)
    
    print(f"\n[Data] Interior points: {interior_points.shape[0]}")
    print(f"[Data] Boundary points: {boundary_points.shape[0]}")

    # ----------------------------------------------------------------------
    # Train SN
    # ----------------------------------------------------------------------
    sn_metrics: Optional[PinnLossComponents] = None
    sn_eval: Optional[Dict[str, float]] = None
    sn_train_time = 0.0

    if args.model in ("sn", "both"):
        set_seed(args.seed)  # Reset for reproducible training
        start_time = time.time()
        sn_metrics = train_pinn(
            model=sn_model,
            interior_points=interior_points,
            boundary_points=boundary_points,
            epochs=args.epochs,
            lr=args.lr,
            weight_decay=args.weight_decay,
            interior_weight=args.interior_weight,
            boundary_weight=args.boundary_weight,
            device=device,
            label="SN",
            log_interval=args.log_interval,
        )
        sn_train_time = time.time() - start_time

        print("[SN] Evaluation on dense grid:")
        sn_eval = evaluate_model_on_grid(
            model=sn_model,
            device=device,
            grid_size=args.grid_size,
        )
        print(f"  PDE residual MSE   = {sn_eval['pde_residual_mse']:.4e}")
        print(f"  boundary MSE       = {sn_eval['boundary_mse']:.4e}")
        print(f"  L2 solution MSE    = {sn_eval['l2_solution_mse']:.4e}")
        print(f"  parameters         = {sn_num_params}")
        print(f"  training time (s)  = {sn_train_time:.2f}")

    # ----------------------------------------------------------------------
    # Train KAN
    # ----------------------------------------------------------------------
    kan_metrics: Optional[PinnLossComponents] = None
    kan_eval: Optional[Dict[str, float]] = None
    kan_train_time = 0.0

    if args.model in ("kan", "both") and kan_model is not None:
        set_seed(args.seed)  # Reset for reproducible training
        start_time = time.time()
        kan_metrics = train_pinn(
            model=kan_model,
            interior_points=interior_points,
            boundary_points=boundary_points,
            epochs=args.epochs,
            lr=args.lr,
            weight_decay=args.weight_decay,
            interior_weight=args.interior_weight,
            boundary_weight=args.boundary_weight,
            device=device,
            label="KAN",
            log_interval=args.log_interval,
        )
        kan_train_time = time.time() - start_time

        print("[KAN] Evaluation on dense grid:")
        kan_eval = evaluate_model_on_grid(
            model=kan_model,
            device=device,
            grid_size=args.grid_size,
        )
        print(f"  PDE residual MSE   = {kan_eval['pde_residual_mse']:.4e}")
        print(f"  boundary MSE       = {kan_eval['boundary_mse']:.4e}")
        print(f"  L2 solution MSE    = {kan_eval['l2_solution_mse']:.4e}")
        print(f"  parameters         = {kan_num_params}")
        print(f"  training time (s)  = {kan_train_time:.2f}")

    # ----------------------------------------------------------------------
    # Final summary (if both models were run)
    # ----------------------------------------------------------------------
    if args.model == "both" and sn_eval is not None and kan_eval is not None:
        print("\n" + "=" * 80)
        print("Final comparison on dense grid:")
        print("=" * 80)
        print(
            f"SN  : PDE MSE = {sn_eval['pde_residual_mse']:.4e}, "
            f"boundary MSE = {sn_eval['boundary_mse']:.4e}, "
            f"L2 MSE = {sn_eval['l2_solution_mse']:.4e}, "
            f"params = {sn_num_params}, "
            f"time = {sn_train_time:.1f}s"
        )
        print(
            f"KAN : PDE MSE = {kan_eval['pde_residual_mse']:.4e}, "
            f"boundary MSE = {kan_eval['boundary_mse']:.4e}, "
            f"L2 MSE = {kan_eval['l2_solution_mse']:.4e}, "
            f"params = {kan_num_params}, "
            f"time = {kan_train_time:.1f}s"
        )
        print("=" * 80)
        
        # Determine winner
        if sn_eval['l2_solution_mse'] < kan_eval['l2_solution_mse']:
            winner = "SN"
            ratio = kan_eval['l2_solution_mse'] / (sn_eval['l2_solution_mse'] + 1e-12)
        else:
            winner = "KAN"
            ratio = sn_eval['l2_solution_mse'] / (kan_eval['l2_solution_mse'] + 1e-12)
        print(f"\nWinner (by L2 solution MSE): {winner} ({ratio:.2f}x better)")

    # ----------------------------------------------------------------------
    # Optional JSON logging
    # ----------------------------------------------------------------------
    if args.log_json:
        os.makedirs(args.log_dir, exist_ok=True)
        log_data = {
            "seed": args.seed,
            "epochs": args.epochs,
            "target_params": args.target_params,
            "n_interior": args.n_interior,
            "n_boundary_per_edge": args.n_boundary_per_edge,
            "lr": args.lr,
            "sn_residual": args.sn_residual,
            "sn_lateral": args.sn_lateral,
            "sn": {
                "params": sn_num_params,
                "width": sn_width,
                "layers": sn_layers,
                "phi_knots": sn_phi_knots,
                "Phi_knots": sn_Phi_knots,
                "train_time": sn_train_time,
                "train_loss_total": getattr(sn_metrics, "total", None) if sn_metrics else None,
                "train_loss_interior": getattr(sn_metrics, "interior", None) if sn_metrics else None,
                "train_loss_boundary": getattr(sn_metrics, "boundary", None) if sn_metrics else None,
                "eval": sn_eval,
            },
            "kan": {
                "params": kan_num_params,
                "width": args.kan_width,
                "layers": args.kan_layers,
                "num_knots": args.kan_knots,
                "train_time": kan_train_time,
                "train_loss_total": getattr(kan_metrics, "total", None) if kan_metrics else None,
                "train_loss_interior": getattr(kan_metrics, "interior", None) if kan_metrics else None,
                "train_loss_boundary": getattr(kan_metrics, "boundary", None) if kan_metrics else None,
                "eval": kan_eval,
            },
        }
        log_fname = os.path.join(
            args.log_dir,
            f"pinn_sn_vs_kan_poisson8_seed{args.seed}_params{args.target_params}.json",
        )
        with open(log_fname, "w") as f:
            json.dump(log_data, f, indent=2)
        print(f"\nSaved metrics to {log_fname}")

    print("\nDone.\n")


if __name__ == "__main__":
    main()