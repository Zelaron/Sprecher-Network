"""
pinn_sn_vs_kan_poisson1.py

Benchmarking Sprecher Networks (SNs) vs Kolmogorov–Arnold Networks (KANs) on a 2D Poisson PINN.
"""

# Short benchmark description:
# We solve a manufactured Poisson problem on Ω = [-1, 1]^2 with zero Dirichlet boundary conditions, using a PINN
# built either from a Sprecher Network (SN) or a KAN-style spline network. The exact solution is
# u(x, y) = sin(π x) sin(π y²), which induces an anisotropic forcing term with rapidly varying curvature in y,
# especially near |y| ≈ 1. This is a natural, smooth, yet nontrivial test that has appeared (in nearly identical
# form) in the KAN literature and is representative of elliptic PDE benchmarks. SNs might have a mild structural
# advantage here because their monotone inner splines and dynamically updated spline domains are well suited to
# handling warped coordinate directions and nonuniform gradients, while the KAN uses fixed-knot cubic splines;
# nevertheless, both models are trained with the same PINN loss, optimizer, and comparable parameter budgets.

# Physics-Informed Neural Networks (PINNs) comment block (copied verbatim as requested):
# \textbf{Physics-Informed Neural Networks (PINNs)}:  for a differential operator $D$ on a domain $\Omega$, we want the network $f$ to solve the PDE with Dirichlet boundary conditions:
# $$
# \begin{cases}
#  D(f) = 0 & \textnormal{in } \Omega, \\
#  f = 0 & \textnormal{in } \partial \Omega.  
# \end{cases}
# $$
# This end, the network is trained on the loss 
# $$
# \frac{1}{|S_1|}\sum_{\mathbf{x} \in S_1} | D(f) (\mathbf{x}) |^2 + \frac{1}{|S_2|} \sum_{\mathbf{x} \in S_2} f(\mathbf{x})^2,
# $$
# where $S_1 \subset \Omega$ and $S_2 \subset \partial \Omega$ are datasets. In \cite[Section 3.4]{liu2024kan}, $\Omega = [-1, 1]^2 \subset \mathbb{R}^2$, and $D = \Delta - g $ is the Poisson operator, where $\Delta$ is the Laplacian, and $f=-\pi^2(1 + 4y^2) \sin(\pi x)\sin(\pi y^2) + 2\pi \sin(\pi x) \cos(\pi y^2)$.

# Fairness note:
# This script implements an apples-to-apples Poisson PINN benchmark: SN and KAN use the same domain, PDE,
# collocation points, optimizer, learning rate, and PINN loss (with identical weights on interior and boundary terms).
# The KAN uses cubic splines with a parameterization comparable in expressive power to the SN, and its width/knots
# are auto-chosen to closely match the SN's total trainable parameter count. SNs may still have a mild structural
# edge because of their monotone φ-splines and domain-updating spline ranges, which naturally align with the warped
# y²-geometry of the manufactured solution, but we do not introduce any model-specific training tricks that would
# artificially favor either side.

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
# KAN implementation with cubic splines
# ---------------------------------------------------------------------------


class KANLayer(nn.Module):
    """
    A simple Kolmogorov–Arnold Network (KAN) layer:

        h_i(x) = sum_j w_ij * s_ij(x_j) + b_i,

    where each s_ij is a univariate cubic spline implemented via sn_core.SimpleSpline
    with fixed input domain [-1, 1] and no codomain scaling.

    This is intentionally close to the standard KAN design (per-edge univariate splines),
    but uses the same spline primitive as the Sprecher Network for a clean comparison.
    """

    def __init__(self, d_in: int, d_out: int, num_knots: int = 8):
        super().__init__()
        self.d_in = d_in
        self.d_out = d_out
        self.num_knots = num_knots

        # One spline per edge (i, j)
        self.splines = nn.ModuleList(
            [
                nn.ModuleList(
                    [
                        SimpleSpline(
                            num_knots=num_knots,
                            in_range=(DOMAIN_MIN, DOMAIN_MAX),
                            out_range=(DOMAIN_MIN, DOMAIN_MAX),
                            monotonic=False,
                            train_codomain=False,
                            codomain_params=None,
                            spline_kind="cubic",
                            order=3,
                        )
                        for _ in range(d_in)
                    ]
                )
                for _ in range(d_out)
            ]
        )

        # Linear mixing weights and biases
        self.weights = nn.Parameter(torch.randn(d_out, d_in) * math.sqrt(2.0 / d_in))
        self.bias = nn.Parameter(torch.zeros(d_out))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [N, d_in]
        N, d_in = x.shape
        assert d_in == self.d_in

        # Evaluate splines edge-wise
        outputs = []
        for i in range(self.d_out):
            acc = 0.0
            for j in range(self.d_in):
                # SimpleSpline supports arbitrary shapes; keep broadcasting simple
                s_ij = self.splines[i][j]
                x_j = x[:, j]  # [N]
                s_val = s_ij(x_j)  # [N]
                acc = acc + self.weights[i, j] * s_val
            outputs.append(acc + self.bias[i])
        return torch.stack(outputs, dim=-1)  # [N, d_out]


class KAN(nn.Module):
    """
    Two-layer KAN: 2 -> width -> 1, with cubic spline activations on each edge.
    """

    def __init__(self, hidden_width: int = 8, num_knots: int = 6):
        super().__init__()
        self.hidden_width = hidden_width
        self.num_knots = num_knots

        self.layer1 = KANLayer(d_in=2, d_out=hidden_width, num_knots=num_knots)
        self.layer2 = KANLayer(d_in=hidden_width, d_out=1, num_knots=num_knots)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.layer1(x)
        out = self.layer2(h)
        return out  # [N, 1]


def approximate_kan_params_for_config(hidden_width: int, num_knots: int) -> int:
    """
    Closed-form parameter count for the KAN defined above, mainly for hyperparameter search.

    For each KANLayer with d_in, d_out:
      - Splines: d_in * d_out * num_knots parameters (coeffs);
      - Weights: d_in * d_out;
      - Biases: d_out.
    Total per layer: d_in * d_out * (num_knots + 1) + d_out.

    For 2 -> w -> 1:
      N = 2*w*(K+1) + w  +  w*1*(K+1) + 1  = w*(3K + 4) + 1.
    """
    K = num_knots
    w = hidden_width
    return w * (3 * K + 4) + 1


def build_kan_model_for_params(
    target_params: int,
    width_candidates: Optional[List[int]] = None,
    knot_candidates: Optional[List[int]] = None,
) -> Tuple[KAN, int, int]:
    """
    Given a target parameter count (from the SN), choose (width, num_knots) for the KAN
    so that its total trainable parameters are as close as possible to target_params.

    Returns (model, width, num_knots).
    """
    if width_candidates is None:
        # Moderate widths: KANs tend to be parameter-heavy because of per-edge splines.
        width_candidates = [2, 4, 6, 8, 10, 12, 16, 24]
    if knot_candidates is None:
        # Reasonable cubic-spline resolutions
        knot_candidates = [4, 6, 8, 10, 12]

    best_cfg = None
    best_diff = float("inf")
    best_model = None

    for w in width_candidates:
        for K in knot_candidates:
            approx_params = approximate_kan_params_for_config(w, K)
            diff = abs(approx_params - target_params)
            if diff < best_diff:
                best_diff = diff
                best_cfg = (w, K)
                # Build a model to get the *exact* parameter count (in case of minor discrepancies)
                candidate = KAN(hidden_width=w, num_knots=K)
                exact_params = count_parameters(candidate)
                diff_exact = abs(exact_params - target_params)
                if diff_exact <= diff:
                    best_model = candidate
                    best_diff = diff_exact
                    best_cfg = (w, K)

    if best_model is None:
        # Fallback: just use a small default KAN
        best_model = KAN(hidden_width=8, num_knots=6)
        best_cfg = (8, 6)
        best_diff = abs(count_parameters(best_model) - target_params)

    width, num_knots = best_cfg
    print(
        f"[KAN] Auto-matched hyperparameters to SN params={target_params}: "
        f"width={width}, num_knots={num_knots}, approx_KAN_params={count_parameters(best_model)} "
        f"(abs diff={best_diff})"
    )
    return best_model, width, num_knots


# ---------------------------------------------------------------------------
# SN model builder
# ---------------------------------------------------------------------------


def build_sn_model(
    sn_width: int,
    sn_layers: int,
    phi_knots: int,
    Phi_knots: int,
) -> SprecherMultiLayerNetwork:
    """
    Build a Sprecher Network configured for smooth PINN tasks.

    We disable normalization (norm_type='none') and select cubic splines for both φ and Φ.
    """
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
        phi_spline_type="cubic",
        Phi_spline_type="cubic",
        phi_spline_order=3,
        Phi_spline_order=3,
    )
    return model


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
) -> PinnLossComponents:
    """
    Full-batch PINN training loop for a single model.
    """
    model.to(device)
    model.train()

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

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
        optimizer.step()

        final_total = total_loss.item()
        final_int = interior_loss.item()
        final_bnd = boundary_loss.item()

        if epoch == 1 or epoch % log_interval == 0 or epoch == epochs:
            pbar.write(
                f"[{label}] Epoch {epoch:5d}/{epochs}: "
                f"loss={final_total:.4e}, interior={final_int:.4e}, boundary={final_bnd:.4e}"
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
        description="PINN Poisson benchmark: Sprecher Network vs KAN"
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
        default=2000,
        help="Number of training epochs for each model (default: 2000).",
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
        help="Number of boundary collocation points per edge (default: 256; total boundary points = 4 * this).",
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
        default=200,
        help="Epoch interval for logging training progress (default: 200).",
    )

    # SN hyperparameters
    parser.add_argument(
        "--sn_width",
        type=int,
        default=16,
        help="Hidden width for SN (same for all layers; default: 16).",
    )
    parser.add_argument(
        "--sn_layers",
        type=int,
        default=2,
        help="Number of hidden layers for SN (default: 2).",
    )
    parser.add_argument(
        "--sn_phi_knots",
        type=int,
        default=32,
        help="Number of knots for SN φ-splines (default: 32).",
    )
    parser.add_argument(
        "--sn_Phi_knots",
        type=int,
        default=32,
        help="Number of knots for SN Φ-splines (default: 32).",
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
        help="Number of spline knots for KAN. If not set, auto-chosen to match SN parameter count.",
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
    print(f"Running PINN Poisson benchmark with seed={args.seed}")
    print("=" * 80)

    # ----------------------------------------------------------------------
    # Build SN model (always, to define a fair parameter budget)
    # ----------------------------------------------------------------------
    print("\n[SN] Building Sprecher Network model...")
    sn_model = build_sn_model(
        sn_width=args.sn_width,
        sn_layers=args.sn_layers,
        phi_knots=args.sn_phi_knots,
        Phi_knots=args.sn_Phi_knots,
    )
    sn_num_params = count_parameters(sn_model)
    print(f"[SN] Total trainable parameters: {sn_num_params}")

    # ----------------------------------------------------------------------
    # Build KAN model, matching parameter count as closely as possible
    # ----------------------------------------------------------------------
    if args.model in ("kan", "both"):
        # Re-seed before constructing KAN so that its random initialization
        # (weights, splines) is comparable to the SN initialization.
        set_seed(args.seed)
        CONFIG["seed"] = args.seed

        print("\n[KAN] Building KAN model...")
        if args.kan_width is not None and args.kan_knots is not None:
            kan_model = KAN(hidden_width=args.kan_width, num_knots=args.kan_knots)
            kan_num_params = count_parameters(kan_model)
            print(
                f"[KAN] Using user-specified hyperparameters: width={args.kan_width}, "
                f"num_knots={args.kan_knots}, params={kan_num_params}"
            )
        else:
            kan_model, auto_width, auto_knots = build_kan_model_for_params(sn_num_params)
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
    interior_points = sample_interior_points(args.n_interior, device=device)
    boundary_points = sample_boundary_points(args.n_boundary_per_edge, device=device)

    # ----------------------------------------------------------------------
    # Train SN
    # ----------------------------------------------------------------------
    sn_metrics: Optional[PinnLossComponents] = None
    sn_eval: Optional[Dict[str, float]] = None

    if args.model in ("sn", "both"):
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

    if args.model in ("kan", "both") and kan_model is not None:
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
            f"params = {sn_num_params}"
        )
        print(
            f"KAN : PDE MSE = {kan_eval['pde_residual_mse']:.4e}, "
            f"boundary MSE = {kan_eval['boundary_mse']:.4e}, "
            f"L2 MSE = {kan_eval['l2_solution_mse']:.4e}, "
            f"params = {kan_num_params}"
        )
        print("=" * 80)

    # ----------------------------------------------------------------------
    # Optional JSON logging
    # ----------------------------------------------------------------------
    if args.log_json:
        os.makedirs(args.log_dir, exist_ok=True)
        log_data = {
            "seed": args.seed,
            "epochs": args.epochs,
            "n_interior": args.n_interior,
            "n_boundary_per_edge": args.n_boundary_per_edge,
            "sn_params": sn_num_params,
            "kan_params": kan_num_params,
            "sn": {
                "train_loss_total": getattr(sn_metrics, "total", None)
                if sn_metrics is not None
                else None,
                "train_loss_interior": getattr(sn_metrics, "interior", None)
                if sn_metrics is not None
                else None,
                "train_loss_boundary": getattr(sn_metrics, "boundary", None)
                if sn_metrics is not None
                else None,
                "eval": sn_eval,
            },
            "kan": {
                "train_loss_total": getattr(kan_metrics, "total", None)
                if kan_metrics is not None
                else None,
                "train_loss_interior": getattr(kan_metrics, "interior", None)
                if kan_metrics is not None
                else None,
                "train_loss_boundary": getattr(kan_metrics, "boundary", None)
                if kan_metrics is not None
                else None,
                "eval": kan_eval,
            },
        }
        log_fname = os.path.join(
            args.log_dir,
            f"pinn_sn_vs_kan_poisson1_seed{args.seed}_model{args.model}.json",
        )
        with open(log_fname, "w") as f:
            json.dump(log_data, f, indent=2)
        print(f"\nSaved metrics to {log_fname}")

    print("\nDone.\n")


if __name__ == "__main__":
    main()
