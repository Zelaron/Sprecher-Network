"""
benchmarks/spline_comparison2A.py

Side-by-side spline visualizer: Sprecher Network (SN) vs a (simple) Kolmogorov–Arnold Network (KAN).

What this does
--------------
1) Trains an SN and a KAN on a toy regression function from the KAN paper:
       Toy2D in sn_core.data:  f(x1,x2) = exp(sin(pi x1)) + x2^2)

2) Saves ONE figure containing, for each model:
   - A 3D surface plot showing the learned function
   - ALL trained spline functions:
       * SN: φ^(l) and Φ^(l) per block l
       * KAN: one cubic spline per edge for every layer

Run
---
    python -m benchmarks.spline_comparison2A

Outputs
-------
By default:
    benchmarks/outputs/spline_comparison.png

Optionally also PDF:
    --pdf

Notes / design choices
----------------------
- Both models use *cubic (PCHIP)* splines for smoothness.
- To keep the comparison focused on spline behavior, we disable SN-only add-ons by default:
  residual weights, lateral mixing, normalization, and theoretical-domain updates.
- Parameter matching: we choose (φ_knots, Φ_knots) for SN and (KAN knots) for KAN so both
  models are close to --target_params, while using the same number of hidden layers.
- The KAN implementation here is intentionally minimal (spline-only edges, no grid updates,
  no extra base activation) to keep the splines directly visible and comparable.

If you want to stress oscillations more, try:
    --target_params 2000 --epochs 4000 --hidden_layers 3 --hidden_width 4
"""

from __future__ import annotations

import argparse
import math
import os
from dataclasses import dataclass
from typing import List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

# Headless-safe plotting
import matplotlib

if os.environ.get("DISPLAY", "") == "":
    matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec
from mpl_toolkits.mplot3d import Axes3D

# Project-local SN code
from sn_core import CONFIG, SprecherMultiLayerNetwork
from sn_core.data import Toy2D


# -------------------------
# Utilities
# -------------------------


def set_seed(seed: int) -> None:
    seed = int(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def count_trainable_params(model: nn.Module) -> int:
    return int(sum(p.numel() for p in model.parameters() if p.requires_grad))


def mse(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    pred = pred.reshape(-1)
    target = target.reshape(-1)
    return torch.mean((pred - target) ** 2)


# -------------------------
# Fast cubic PCHIP spline evaluator (vectorized)
# -------------------------


def pchip_slopes(y: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    """
    Vectorized PCHIP slopes (Fritsch–Carlson) along the last dimension.
    y: (..., K)
    x: (K,)
    returns d: (..., K)
    """
    K = y.shape[-1]
    if K < 2:
        return torch.zeros_like(y)

    h = x[1:] - x[:-1]  # (K-1)
    delta = (y[..., 1:] - y[..., :-1]) / (h + 1e-12)  # (..., K-1)

    d = torch.zeros_like(y)

    if K == 2:
        d[..., 0] = delta[..., 0]
        d[..., 1] = delta[..., 0]
        return d

    # Endpoints (limited)
    d0 = ((2 * h[0] + h[1]) * delta[..., 0] - h[0] * delta[..., 1]) / (h[0] + h[1] + 1e-12)
    dN = ((2 * h[-1] + h[-2]) * delta[..., -1] - h[-1] * delta[..., -2]) / (h[-1] + h[-2] + 1e-12)

    def limit(di: torch.Tensor, deltai: torch.Tensor) -> torch.Tensor:
        di = torch.where(di * deltai <= 0, torch.zeros_like(di), di)
        di = torch.where(torch.abs(di) > 3 * torch.abs(deltai), 3 * deltai, di)
        return di

    d[..., 0] = limit(d0, delta[..., 0])
    d[..., -1] = limit(dN, delta[..., -1])

    # Interior (harmonic mean where slopes have same sign)
    delta_prev = delta[..., :-1]  # (..., K-2)
    delta_next = delta[..., 1:]  # (..., K-2)
    same_sign = (delta_prev * delta_next) > 0

    w1 = 2 * h[1:] + h[:-1]  # (K-2)
    w2 = h[1:] + 2 * h[:-1]  # (K-2)

    d_int = (w1 + w2) / (w1 / (delta_prev + 1e-12) + w2 / (delta_next + 1e-12) + 1e-12)
    d[..., 1:-1] = torch.where(same_sign, d_int, torch.zeros_like(d_int))
    return d


def pchip_eval(
    xq: torch.Tensor,
    knots: torch.Tensor,
    y: torch.Tensor,
    d: torch.Tensor,
) -> torch.Tensor:
    """
    Evaluate cubic Hermite spline with PCHIP slopes at query points.
    Vectorized over leading dims of y/d.

    xq: (B,)
    knots: (K,)
    y: (..., K)
    d: (..., K)
    returns: (..., B)
    """
    K = knots.numel()
    xq = xq.to(knots.dtype)

    below = xq < knots[0]
    above = xq > knots[-1]

    xq_clamped = torch.clamp(xq, knots[0], knots[-1])
    idx = torch.searchsorted(knots, xq_clamped) - 1
    idx = torch.clamp(idx, 0, K - 2)  # (B,)

    x0 = knots[idx]  # (B,)
    x1 = knots[idx + 1]  # (B,)
    h = (x1 - x0) + 1e-12  # (B,)
    t = (xq_clamped - x0) / h  # (B,)

    # Hermite basis
    h00 = (2 * t**3) - (3 * t**2) + 1
    h10 = (t**3) - (2 * t**2) + t
    h01 = (-2 * t**3) + (3 * t**2)
    h11 = (t**3) - (t**2)

    # Gather y0,y1,d0,d1 from last dim
    # y: (...,K) -> gather into (...,B)
    idx_g = idx.view((1,) * (y.ndim - 1) + (-1,)).expand(*y.shape[:-1], idx.shape[0])
    y0 = torch.gather(y, dim=-1, index=idx_g)
    y1 = torch.gather(y, dim=-1, index=torch.clamp(idx_g + 1, 0, K - 1))
    d0 = torch.gather(d, dim=-1, index=idx_g)
    d1 = torch.gather(d, dim=-1, index=torch.clamp(idx_g + 1, 0, K - 1))

    # Broadcast scalar basis to (...,B)
    for _ in range(y.ndim - 1):
        h00 = h00.unsqueeze(0)
        h10 = h10.unsqueeze(0)
        h01 = h01.unsqueeze(0)
        h11 = h11.unsqueeze(0)
        h = h.unsqueeze(0)

    out = h00 * y0 + h10 * h * d0 + h01 * y1 + h11 * h * d1

    # Linear extrapolation (use endpoint slopes). We keep shapes consistent:
    # out has shape (..., B). below/above are (B,), so we reshape them to (..., B).
    if below.any():
        left_val = y[..., 0].unsqueeze(-1)  # (..., 1)
        left_slope = d[..., 0].unsqueeze(-1)  # (..., 1)
        dx = (xq - knots[0]).view((1,) * (y.ndim - 1) + (-1,))  # (..., B)
        cond = below.view((1,) * (y.ndim - 1) + (-1,))  # (..., B)
        out = torch.where(cond, left_val + left_slope * dx, out)

    if above.any():
        right_val = y[..., -1].unsqueeze(-1)  # (..., 1)
        right_slope = d[..., -1].unsqueeze(-1)  # (..., 1)
        dx = (xq - knots[-1]).view((1,) * (y.ndim - 1) + (-1,))  # (..., B)
        cond = above.view((1,) * (y.ndim - 1) + (-1,))  # (..., B)
        out = torch.where(cond, right_val + right_slope * dx, out)

    return out


# -------------------------
# KAN (fast, tensorized)
# -------------------------


class KANLayerFast(nn.Module):
    """
    Minimal KAN layer with one cubic spline per edge:

        y_o = sum_i spline_{o,i}(x_i) + bias_o

    Tensorized parameters:
        coeffs: (d_out, d_in, K)
        bias:   (d_out,)
    """

    def __init__(
        self,
        d_in: int,
        d_out: int,
        num_knots: int,
        *,
        x_min: float = 0.0,
        x_max: float = 1.0,
        dtype: torch.dtype = torch.float64,
        init_scale: float | None = None,
    ):
        super().__init__()
        self.d_in = int(d_in)
        self.d_out = int(d_out)
        self.num_knots = int(num_knots)

        self.register_buffer("knots", torch.linspace(float(x_min), float(x_max), self.num_knots, dtype=dtype))

        if init_scale is None:
            init_scale = 0.1 / math.sqrt(max(1, self.d_in))

        coeffs = torch.zeros(self.d_out, self.d_in, self.num_knots, dtype=dtype)
        coeffs.normal_(0.0, float(init_scale))
        self.coeffs = nn.Parameter(coeffs)

        self.bias = nn.Parameter(torch.zeros(self.d_out, dtype=dtype))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, d_in)
        B = x.shape[0]

        # slopes: (d_out, d_in, K)
        d = pchip_slopes(self.coeffs, self.knots)

        # accumulate contributions per input dimension (loop over d_in only; usually small)
        acc = torch.zeros(B, self.d_out, device=x.device, dtype=x.dtype)
        for i in range(self.d_in):
            y_i = self.coeffs[:, i, :]  # (d_out, K)
            d_i = d[:, i, :]  # (d_out, K)
            vals = pchip_eval(x[:, i], self.knots, y_i, d_i)  # (d_out, B)
            acc = acc + vals.transpose(0, 1)  # (B, d_out)

        acc = acc + self.bias.view(1, -1)
        return acc


class KANFast(nn.Module):
    """
    Stacked KAN with fixed hidden width and scalar output.
    num_layers includes the final output layer.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_width: int,
        num_layers: int,
        num_knots: int,
        *,
        dtype: torch.dtype = torch.float64,
        x_min: float = 0.0,
        x_max: float = 1.0,
    ):
        super().__init__()
        self.input_dim = int(input_dim)
        self.hidden_width = int(hidden_width)
        self.num_layers = int(num_layers)
        self.num_knots = int(num_knots)

        layers: List[nn.Module] = []
        d_in = self.input_dim
        for li in range(self.num_layers):
            d_out = 1 if li == self.num_layers - 1 else self.hidden_width
            layers.append(
                KANLayerFast(
                    d_in=d_in,
                    d_out=d_out,
                    num_knots=self.num_knots,
                    x_min=x_min,
                    x_max=x_max,
                    dtype=dtype,
                )
            )
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


# -------------------------
# Parameter matching
# -------------------------


def sn_param_estimate(
    input_dim: int,
    arch: List[int],
    phi_knots: int,
    Phi_knots: int,
    train_codomain: bool,
) -> int:
    """
    Conservative SN param estimate for barebones config:
      lambdas + (phi+Phi) coeffs per block + eta per block + (optional) codomain + output scale/bias
    """
    n_layers = len(arch) if len(arch) > 0 else 1
    sum_din = input_dim + sum(arch[:-1]) if n_layers >= 2 else input_dim
    spline_params = n_layers * (int(phi_knots) + int(Phi_knots))
    eta_params = n_layers
    codomain_params = (2 * n_layers) if bool(train_codomain) else 0
    output_params = 2  # output_scale, output_bias
    return int(sum_din + spline_params + eta_params + codomain_params + output_params)


@dataclass
class MatchedConfigs:
    sn_arch: List[int]
    sn_phi_knots: int
    sn_Phi_knots: int
    kan_hidden_width: int
    kan_num_layers: int
    kan_knots: int
    target_params: int


def choose_configs(
    target_params: int,
    input_dim: int,
    hidden_width: int,
    hidden_layers: int,
    train_codomain: bool,
) -> MatchedConfigs:
    """
    Keep the same hidden depth for both models:
      SN blocks = hidden_layers
      KAN num_layers = hidden_layers + 1 (extra output layer)

    Choose knot counts to hit target_params as closely as possible.
    """
    W = int(hidden_width)
    L = int(hidden_layers)
    if L < 1:
        raise ValueError("--hidden_layers must be >= 1")
    if W < 1:
        raise ValueError("--hidden_width must be >= 1")

    sn_arch = [W] * L
    kan_layers = L + 1

    # SN knots: target ≈ base + L*(phi+Phi)
    base = sn_param_estimate(input_dim, sn_arch, 0, 0, train_codomain)
    remaining = int(target_params) - int(base)
    if remaining <= 0:
        raise ValueError(f"target_params={target_params} too small; SN base={base} for W={W}, L={L}.")

    knot_sum = max(16, int(round(remaining / float(L))))
    sn_phi = max(8, knot_sum // 2)
    sn_Phi = max(8, knot_sum - sn_phi)

    # local refine
    best_phi, best_Phi, best_diff = sn_phi, sn_Phi, 10**18
    for dphi in range(-20, 21):
        phi_try = max(8, sn_phi + dphi)
        Phi_try = max(8, sn_Phi - dphi)
        est = sn_param_estimate(input_dim, sn_arch, phi_try, Phi_try, train_codomain)
        diff = abs(est - target_params)
        if diff < best_diff:
            best_phi, best_Phi, best_diff = phi_try, Phi_try, diff
    sn_phi, sn_Phi = best_phi, best_Phi

    # KAN knots: target ≈ edges*K + biases (solve then local search)
    d_in = input_dim
    edges = 0
    biases = 0
    for li in range(kan_layers):
        d_out = 1 if li == kan_layers - 1 else W
        edges += d_in * d_out
        biases += d_out
        d_in = d_out

    K0 = int(round((target_params - biases) / float(edges)))
    K0 = max(4, K0)

    best_K, best_Kdiff = K0, 10**18
    for K in range(max(4, K0 - 50), K0 + 51):
        est = kan_param_count_formula(input_dim, W, kan_layers, K)
        diff = abs(est - target_params)
        if diff < best_Kdiff:
            best_K, best_Kdiff = K, diff

    return MatchedConfigs(
        sn_arch=sn_arch,
        sn_phi_knots=int(sn_phi),
        sn_Phi_knots=int(sn_Phi),
        kan_hidden_width=W,
        kan_num_layers=kan_layers,
        kan_knots=int(best_K),
        target_params=int(target_params),
    )


# -------------------------
# Training
# -------------------------


def train_regression(
    model: nn.Module,
    x_train: torch.Tensor,
    y_train: torch.Tensor,
    *,
    epochs: int,
    lr: float,
    weight_decay: float,
    batch_size: int,
    label: str,
    device: torch.device,
) -> List[float]:
    model.train()
    opt = optim.AdamW(model.parameters(), lr=float(lr), weight_decay=float(weight_decay))
    sched = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=int(epochs), eta_min=float(lr) * 0.05)

    n = x_train.shape[0]
    losses: List[float] = []
    for ep in range(1, int(epochs) + 1):
        perm = torch.randperm(n, device=device)
        ep_loss = 0.0
        nb = 0
        for i in range(0, n, int(batch_size)):
            idx = perm[i : i + int(batch_size)]
            xb = x_train[idx]
            yb = y_train[idx]
            opt.zero_grad(set_to_none=True)
            pred = model(xb)
            loss = mse(pred, yb)
            loss.backward()
            opt.step()
            ep_loss += float(loss.detach().cpu())
            nb += 1
        sched.step()
        losses.append(ep_loss / max(1, nb))

        if ep == 1 or ep % max(1, epochs // 10) == 0 or ep == epochs:
            print(f"[{label}] epoch {ep:5d}/{epochs}  loss={losses[-1]:.4e}  lr={opt.param_groups[0]['lr']:.2e}")
    return losses


# -------------------------
# Plotting helpers
# -------------------------


@torch.no_grad()
def eval_on_grid(
    model: nn.Module,
    dataset: Toy2D,
    *,
    n_grid: int,
    device: torch.device,
    dtype: torch.dtype,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    xs = torch.linspace(0.0, 1.0, int(n_grid), device=device, dtype=dtype)
    ys = torch.linspace(0.0, 1.0, int(n_grid), device=device, dtype=dtype)
    X, Y = torch.meshgrid(xs, ys, indexing="ij")
    pts = torch.stack([X.flatten(), Y.flatten()], dim=-1)

    model.eval()
    pred = model(pts)
    true = dataset.evaluate(pts)

    pred = pred.reshape(int(n_grid), int(n_grid)).detach().cpu().numpy()
    true = true.reshape(int(n_grid), int(n_grid)).detach().cpu().numpy()
    return X.detach().cpu().numpy(), Y.detach().cpu().numpy(), true, pred


@torch.no_grad()
def collect_sn_splines(
    sn: SprecherMultiLayerNetwork,
    *,
    device: torch.device,
    dtype: torch.dtype,
    n_points: int = 200,
) -> List[Tuple[str, np.ndarray, np.ndarray]]:
    out: List[Tuple[str, np.ndarray, np.ndarray]] = []
    for li, layer in enumerate(sn.layers):
        x = torch.linspace(float(layer.phi.in_min), float(layer.phi.in_max), int(n_points), device=device, dtype=dtype)
        y = layer.phi(x)
        out.append((f"SN  φ^{li+1}", x.cpu().numpy(), y.cpu().numpy()))

        x = torch.linspace(float(layer.Phi.in_min), float(layer.Phi.in_max), int(n_points), device=device, dtype=dtype)
        y = layer.Phi(x)
        out.append((f"SN  Φ^{li+1}", x.cpu().numpy(), y.cpu().numpy()))
    return out


@torch.no_grad()
def collect_kan_edge_splines(
    kan: KANFast,
    *,
    device: torch.device,
    dtype: torch.dtype,
    n_points: int = 200,
) -> List[Tuple[str, np.ndarray, np.ndarray]]:
    out: List[Tuple[str, np.ndarray, np.ndarray]] = []
    for li, layer in enumerate(kan.layers):
        assert isinstance(layer, KANLayerFast)
        knots = layer.knots.to(device=device, dtype=dtype)
        # slopes for all edges in this layer
        d = pchip_slopes(layer.coeffs.to(device=device, dtype=dtype), knots)  # (d_out,d_in,K)

        for o in range(layer.d_out):
            for i in range(layer.d_in):
                x = torch.linspace(float(knots[0]), float(knots[-1]), int(n_points), device=device, dtype=dtype)
                y_edge = pchip_eval(x, knots, layer.coeffs[o, i, :], d[o, i, :])  # (B,)
                out.append((f"KAN L{li+1} o{o}←i{i}", x.cpu().numpy(), y_edge.cpu().numpy()))
    return out


def plot_spline_grid(
    fig: plt.Figure,
    parent_spec,
    splines: List[Tuple[str, np.ndarray, np.ndarray]],
    *,
    ncols: int,
    title: str,
    title_fontsize: int = 10,
    label_fontsize: int = 6,
):
    n = len(splines)
    ncols = max(1, int(ncols))
    nrows = int(math.ceil(n / float(ncols)))
    gs = GridSpecFromSubplotSpec(nrows, ncols, subplot_spec=parent_spec, wspace=0.35, hspace=0.55)

    bbox = parent_spec.get_position(fig)
    # Move title up a bit more
    fig.text((bbox.x0 + bbox.x1) / 2, bbox.y1 + 0.025, title, ha="center", va="bottom", fontsize=title_fontsize)

    for k in range(nrows * ncols):
        r = k // ncols
        c = k % ncols
        ax = fig.add_subplot(gs[r, c])
        if k >= n:
            ax.axis("off")
            continue
        lab, x, y = splines[k]
        ax.plot(x, y, linewidth=1.0)
        ax.grid(True, alpha=0.2)
        ax.set_title(lab, fontsize=label_fontsize, pad=2)
        ax.tick_params(axis="both", labelsize=6)
    return gs


def plot_sn_phi_Phi_grid(
    fig: plt.Figure,
    parent_spec,
    sn: SprecherMultiLayerNetwork,
    *,
    device: torch.device,
    dtype: torch.dtype,
    n_points: int = 200,
    title: str = "SN splines",
):
    L = len(sn.layers)
    gs = GridSpecFromSubplotSpec(2, L, subplot_spec=parent_spec, wspace=0.35, hspace=0.55)

    bbox = parent_spec.get_position(fig)
    # Move title up a bit more
    fig.text((bbox.x0 + bbox.x1) / 2, bbox.y1 + 0.025, title, ha="center", va="bottom", fontsize=10)

    for li, layer in enumerate(sn.layers):
        ax_phi = fig.add_subplot(gs[0, li])
        x = torch.linspace(float(layer.phi.in_min), float(layer.phi.in_max), int(n_points), device=device, dtype=dtype)
        y = layer.phi(x)
        ax_phi.plot(x.detach().cpu().numpy(), y.detach().cpu().numpy(), linewidth=1.2)
        ax_phi.grid(True, alpha=0.2)
        ax_phi.set_title(f"φ^{li+1}", fontsize=8)
        ax_phi.tick_params(axis="both", labelsize=6)

        ax_Phi = fig.add_subplot(gs[1, li])
        x = torch.linspace(float(layer.Phi.in_min), float(layer.Phi.in_max), int(n_points), device=device, dtype=dtype)
        y = layer.Phi(x)
        ax_Phi.plot(x.detach().cpu().numpy(), y.detach().cpu().numpy(), linewidth=1.2)
        ax_Phi.grid(True, alpha=0.2)
        ax_Phi.set_title(f"Φ^{li+1}", fontsize=8)
        ax_Phi.tick_params(axis="both", labelsize=6)

    return gs


def add_surface_plot(
    ax: Axes3D,
    X: np.ndarray,
    Y: np.ndarray,
    Z: np.ndarray,
    *,
    title: str,
    cmap: str = "viridis",
):
    """Add a 3D surface plot."""
    surf = ax.plot_surface(X, Y, Z, cmap=cmap, edgecolor='none', alpha=0.9)
    ax.set_xlabel(r'$x_1$', fontsize=10, labelpad=5)
    ax.set_ylabel(r'$x_2$', fontsize=10, labelpad=5)
    ax.set_zlabel(r'$f(\mathbf{x})$', fontsize=10, labelpad=5)
    ax.set_title(title, fontsize=11, pad=10)
    ax.tick_params(axis='both', labelsize=7)
    # Spread out x and y tick marks to reduce crowding
    ax.set_xticks([0, 0.5, 1.0])
    ax.set_yticks([0, 0.5, 1.0])
    ax.view_init(elev=25, azim=45)
    return surf


# -------------------------
# Main
# -------------------------


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="SN vs KAN spline comparison on Toy2D (KAN paper toy function).")

    p.add_argument("--epochs", type=int, default=1200, help="Training epochs (default: 1200)")
    p.add_argument("--lr", type=float, default=1e-3, help="AdamW learning rate (default: 1e-3)")
    p.add_argument("--weight_decay", type=float, default=1e-6, help="AdamW weight_decay (default: 1e-6)")
    p.add_argument("--batch_size", type=int, default=256, help="Mini-batch size (default: 256)")
    p.add_argument("--seed", type=int, default=0, help="Random seed (default: 0)")

    p.add_argument("--target_params", type=int, default=1200, help="Approx param budget for each model (default: 1200)")
    p.add_argument("--hidden_width", type=int, default=3, help="Hidden width for both models (default: 3)")
    p.add_argument(
        "--hidden_layers",
        type=int,
        default=3,
        help="Hidden layers (SN blocks). KAN uses +1 output layer. (default: 3)",
    )

    p.add_argument(
        "--dtype",
        type=str,
        default="float64",
        choices=["float32", "float64"],
        help="dtype (default: float64)",
    )
    p.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"], help="Device (default: auto)")
    p.add_argument("--grid", type=int, default=60, help="Evaluation grid resolution for function plot (default: 60)")

    p.add_argument("--out", type=str, default="benchmarks/outputs/spline_comparison", help="Output path without extension")
    p.add_argument("--pdf", action="store_true", help="Also save a PDF")
    p.add_argument("--show", action="store_true", help="Call plt.show() after saving")

    return p.parse_args()


def main() -> None:
    args = parse_args()

    # Device / dtype
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    dtype = torch.float64 if args.dtype == "float64" else torch.float32

    set_seed(args.seed)

    # Focus the comparison: disable SN extras
    CONFIG["use_residual_weights"] = False
    CONFIG["use_lateral_mixing"] = False
    CONFIG["use_normalization"] = False
    CONFIG["use_theoretical_domains"] = False  # fixed domains (no update_all_domains)
    CONFIG["train_phi_codomain"] = False  # simplest Phi

    cfg = choose_configs(
        target_params=args.target_params,
        input_dim=2,
        hidden_width=args.hidden_width,
        hidden_layers=args.hidden_layers,
        train_codomain=bool(CONFIG["train_phi_codomain"]),
    )

    print("\n=== Chosen configs ===")
    print(f"Target params: {cfg.target_params}")
    print(f"SN  arch={cfg.sn_arch}, phi_knots={cfg.sn_phi_knots}, Phi_knots={cfg.sn_Phi_knots}")
    print(f"KAN width={cfg.kan_hidden_width}, num_layers={cfg.kan_num_layers}, knots={cfg.kan_knots}")

    # Data (fixed training batch)
    dataset = Toy2D()
    n_train = 32 * 32
    x_train, y_train = dataset.sample(n_train, device=device)
    x_train = x_train.to(dtype=dtype)
    y_train = y_train.to(dtype=dtype)

    # Models
    sn = SprecherMultiLayerNetwork(
        input_dim=2,
        architecture=list(cfg.sn_arch),
        final_dim=1,
        phi_knots=int(cfg.sn_phi_knots),
        Phi_knots=int(cfg.sn_Phi_knots),
        norm_type="none",
        norm_position="after",
        norm_skip_first=True,
        initialize_domains=False,  # keep (0,1) domain
        domain_ranges=None,
        phi_spline_type="cubic",
        Phi_spline_type="cubic",
        phi_spline_order=None,
        Phi_spline_order=None,
    ).to(device=device, dtype=dtype)

    kan = KANFast(
        input_dim=2,
        hidden_width=int(cfg.kan_hidden_width),
        num_layers=int(cfg.kan_num_layers),
        num_knots=int(cfg.kan_knots),
        dtype=dtype,
        x_min=0.0,
        x_max=1.0,
    ).to(device=device, dtype=dtype)

    sn_params = count_trainable_params(sn)
    kan_params = count_trainable_params(kan)

    print("\n=== Parameter counts (actual) ===")
    print(f"SN  params:  {sn_params}")
    print(f"KAN params:  {kan_params}")
    print(f"Diff:        {kan_params - sn_params}")

    # Train
    print("\n=== Training SN ===")
    train_regression(
        sn,
        x_train,
        y_train,
        epochs=args.epochs,
        lr=args.lr,
        weight_decay=args.weight_decay,
        batch_size=args.batch_size,
        label="SN",
        device=device,
    )

    print("\n=== Training KAN ===")
    train_regression(
        kan,
        x_train,
        y_train,
        epochs=args.epochs,
        lr=args.lr,
        weight_decay=args.weight_decay,
        batch_size=args.batch_size,
        label="KAN",
        device=device,
    )

    # Metrics
    with torch.no_grad():
        sn_mse = float(mse(sn(x_train), y_train).cpu())
        kan_mse = float(mse(kan(x_train), y_train).cpu())

    # Evaluate on grid
    X, Y, Z_true, Z_sn = eval_on_grid(sn, dataset, n_grid=args.grid, device=device, dtype=dtype)
    _, _, _, Z_kan = eval_on_grid(kan, dataset, n_grid=args.grid, device=device, dtype=dtype)

    # Collect splines
    sn_splines = collect_sn_splines(sn, device=device, dtype=dtype, n_points=200)
    kan_splines = collect_kan_edge_splines(kan, device=device, dtype=dtype, n_points=200)

    # Layout
    sn_layers = len(sn.layers)
    sn_col_units = max(3, sn_layers)
    n_kan_splines = len(kan_splines)
    kan_ncols = min(6, max(3, int(round(math.sqrt(n_kan_splines)))))
    kan_col_units = max(4, kan_ncols)

    kan_rows = int(math.ceil(n_kan_splines / float(kan_ncols)))
    sn_rows = 2

    fig_w = 2.1 * (sn_col_units + kan_col_units)
    fig_h = 4.2 + 1.25 * max(sn_rows, kan_rows)  # Increased top height for 3D plots
    fig = plt.figure(figsize=(fig_w, fig_h), dpi=140)

    outer = GridSpec(
        2,
        2,
        figure=fig,
        height_ratios=[1.4, max(sn_rows, kan_rows)],  # More space for 3D plots
        width_ratios=[sn_col_units, kan_col_units],
        wspace=0.25,
        hspace=0.30,  # More vertical space between surface plots and spline grid
    )

    # 3D Surface plots
    ax_sn_fun = fig.add_subplot(outer[0, 0], projection='3d')
    ax_kan_fun = fig.add_subplot(outer[0, 1], projection='3d')

    # SN surface plot
    add_surface_plot(
        ax_sn_fun,
        X,
        Y,
        Z_sn,
        title=f"Sprecher Network (SN)\narch={cfg.sn_arch}  (params={sn_params})\nMSE={sn_mse:.3e}",
        cmap="viridis",
    )

    # KAN surface plot
    add_surface_plot(
        ax_kan_fun,
        X,
        Y,
        Z_kan,
        title=f"KAN baseline\nwidth={cfg.kan_hidden_width}, layers={cfg.kan_num_layers}  (params={kan_params})\nMSE={kan_mse:.3e}",
        cmap="viridis",
    )

    # Splines
    plot_sn_phi_Phi_grid(
        fig,
        outer[1, 0],
        sn,
        device=device,
        dtype=dtype,
        n_points=200,
        title=f"SN splines: φ and Φ per block (total {len(sn_splines)})",
    )

    plot_spline_grid(
        fig,
        outer[1, 1],
        kan_splines,
        ncols=kan_ncols,
        title=f"KAN edge splines (total {len(kan_splines)})",
        title_fontsize=10,
        label_fontsize=6,
    )

    fig.suptitle(
        f"Spline comparison on Toy2D (KAN paper toy function)\n"
        f"Target params={cfg.target_params}  |  SN params={sn_params}  |  KAN params={kan_params}",
        fontsize=14,
        y=0.995,
    )

    # Save
    out_base = args.out
    out_dir = os.path.dirname(out_base)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    png_path = out_base + ".png"
    fig.savefig(png_path, bbox_inches="tight")
    print(f"\nSaved: {png_path}")

    if args.pdf:
        pdf_path = out_base + ".pdf"
        fig.savefig(pdf_path, bbox_inches="tight")
        print(f"Saved: {pdf_path}")

    if bool(args.show):
        plt.show()
    plt.close(fig)


if __name__ == "__main__":
    main()
