"""
KAN vs SN — Parameter‑Parity Benchmark (with BN recalc + eval progress)

This version equalizes linear residual semantics and ensures evaluation does not
mutate BatchNorm buffers:

- For **SN**: at test, we use **batch statistics** while **preventing BN buffer
  updates** (so no running_mean/var/num_batches_tracked mutations). This matches
  the SN behavior that produced reasonable test RMSE, without side effects.

- For **KAN**: at test, we recompute BN running stats on the training batch and
  then switch to eval() so running stats are used, as before.

Usage:
  python -m benchmarks.kan_sn_parity_bench [FLAGS]

Flags
  General:
    --device {auto,cpu,cuda}            (default: auto)
    --seed INT                           (default: 45)
    --epochs INT                         (default: 4000)
    --dataset STR                        (default: toy_4d_to_5d)

  Test / Evaluation:
    --n_test INT                         (default: 20000)
    --bn_eval_mode {off,recalc_eval}     (default: recalc_eval)
    --bn_recalc_passes INT               (default: 10)
    --eval_batch_size INT                (default: 8192)

  Fairness / Parity:
    --equalize_params                    Match KAN params to SN param count
    --prefer_leq                         When equalizing, prefer <= target params

  SN (Sprecher Network):
    --sn_arch STR                        e.g., "15,15"
    --sn_phi_knots INT
    --sn_Phi_knots INT
    --sn_norm_type {none,batch,layer}    (default: batch)
    --sn_norm_position {before,after}    (default: after)
    --sn_norm_skip_first                 (default: True)
    --sn_norm_first                      Include first norm layer (overrides skip_first)
    --sn_no_residual                     Disable residual path
    --sn_residual_style {node,linear,standard,matrix}  (default: CONFIG)
    --sn_no_lateral                      Disable lateral mixing
    --sn_freeze_domains_after INT        Warm‑up epochs with domain updates, then freeze (0 = never)
    --sn_domain_margin FLOAT             Safety margin on computed domains during warm‑up (e.g., 0.01)

  KAN:
    --kan_arch STR                       e.g., "4,4"
    --kan_degree {2,3}                   (default: 3)
    --kan_K INT                          Basis count (ignored if --equalize_params)
    --kan_bn_type {none,batch}           (default: batch)
    --kan_bn_position {before,after}     (default: after)
    --kan_bn_skip_first
    --kan_outside {linear,clamp}         (default: linear)
    --kan_residual_type {silu,linear,none}  (default: silu)
    --kan_lr FLOAT                       (default: 1e-3)
    --kan_wd FLOAT                       (default: 1e-6)
    --kan_impl {fast,slow}               Implementation switch used in this script

  Output:
    --outdir PATH                        (default: benchmarks/results)

Examples
  # CPU, parameter parity, BN recalc, fast KAN, large eval batches, with *linear* residuals on both:
  python -m benchmarks.kan_sn_parity_bench \
    --dataset toy_4d_to_5d --epochs 4000 --device cpu --n_test 20000 \
    --seed 0 \
    --sn_arch 15,15 --sn_phi_knots 60 --sn_Phi_knots 60 \
    --sn_norm_type batch --sn_norm_position after --sn_norm_skip_first \
    --sn_residual_style linear --sn_no_lateral \
    --sn_freeze_domains_after 1500 --sn_domain_margin 0.01 \
    --kan_arch 4,4 --kan_degree 3 \
    --kan_bn_type batch --kan_bn_position after --kan_bn_skip_first \
    --kan_residual_type linear --kan_outside linear \
    --equalize_params --prefer_leq \
    --bn_eval_mode recalc_eval --bn_recalc_passes 10 \
    --eval_batch_size 8192 --kan_impl fast
"""

import argparse, math, time, os, json
from dataclasses import dataclass
from contextlib import nullcontext
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

# --- Use the SN code directly ---
from sn_core import (
    get_dataset, train_network, CONFIG, has_batchnorm
)
# For safe BN evaluation (batch stats without buffer updates) for SN
from sn_core.train import use_batch_stats_without_updating_bn

# -------------------------------
# Utilities
# -------------------------------

def set_global_seeds(seed: int):
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

def count_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def rmse_per_head(y_true, y_pred):
    mse = torch.mean((y_true - y_pred) ** 2, dim=0)
    per_head = torch.sqrt(mse)
    mean_rmse = torch.sqrt(torch.mean((y_true - y_pred) ** 2))
    return per_head.cpu().tolist(), float(mean_rmse.cpu())

def corr_frobenius(y_true, y_pred, eps=1e-8):
    yt = y_true - y_true.mean(dim=0, keepdim=True)
    yp = y_pred - y_pred.mean(dim=0, keepdim=True)
    std_t = yt.std(dim=0, keepdim=True)
    std_p = yp.std(dim=0, keepdim=True)
    use = ((std_t > eps) & (std_p > eps)).squeeze(0)

    if use.sum() >= 2:
        yt = yt[:, use] / (std_t[:, use] + eps)
        yp = yp[:, use] / (std_p[:, use] + eps)
        Ct = yt.t().mm(yt) / (yt.shape[0] - 1)
        Cp = yp.t().mm(yp) / (yp.shape[0] - 1)
        fr = float(torch.norm(Ct - Cp, p='fro').cpu())
        heads_used = int(use.sum().item())
    else:
        fr = float('nan')
        heads_used = int(use.sum().item())

    return fr, heads_used

@dataclass
class RunResult:
    model_name: str
    params: int
    train_mse: float
    test_rmse_mean: float
    test_rmse_per_head: list
    corr_Frob: float
    corr_heads_used: int
    seconds: float
    notes: str = ""


# -------------------------------
# Cubic / Quadratic B-spline machinery (KAN)
# -------------------------------

def make_clamped_uniform_knots(in_min, in_max, n_basis, degree):
    """Open-uniform (clamped) knot vector: length = n_basis + degree + 1."""
    in_min = float(in_min); in_max = float(in_max)
    assert in_max > in_min
    n_interior = n_basis - degree - 1
    if n_interior < 0:
        n_basis = degree + 1
        n_interior = 0
    if n_interior == 0:
        interior = []
    else:
        interior = list(np.linspace(in_min, in_max, n_interior + 2)[1:-1])
    knots = [in_min] * (degree + 1) + interior + [in_max] * (degree + 1)
    return torch.tensor(knots, dtype=torch.float32)

def bspline_basis(x, knots, degree):
    """
    Cox–de Boor recursion for B-spline basis.
    x: [B] tensor, knots: [M] (M = n_basis + degree + 1)
    returns: [B, n_basis]
    """
    device = x.device; dtype = x.dtype
    knots = knots.to(device=device, dtype=dtype)
    n_basis = knots.numel() - degree - 1
    # Degree 0
    B = []
    for i in range(n_basis):
        t_i, t_ip1 = knots[i], knots[i+1]
        if i == n_basis - 1:
            B0 = ((x >= t_i) & (x <= t_ip1)).to(dtype)
        else:
            B0 = ((x >= t_i) & (x <  t_ip1)).to(dtype)
        B.append(B0)
    B = torch.stack(B, dim=1)  # [B, n_basis]

    for p in range(1, degree + 1):
        Bp = torch.zeros_like(B)
        for i in range(n_basis):
            denom1 = knots[i+p] - knots[i]
            if denom1.abs() > 1e-12:
                left = (x - knots[i]) / denom1 * (B[:, i] if i < n_basis else 0.0)
            else:
                left = 0.0
            denom2 = knots[i+p+1] - knots[i+1] if (i+1) < knots.numel() else torch.tensor(0., device=device, dtype=dtype)
            if (i+1) < n_basis and denom2.abs() > 1e-12:
                right = (knots[i+p+1] - x) / denom2 * B[:, i+1]
            else:
                right = 0.0
            Bp[:, i] = left + right
        B = Bp
    return B  # [B, n_basis]

# --- Original (slow) KAN ---------------------------------------------

class BSpline1D(nn.Module):
    def __init__(self, n_basis=10, degree=3, in_min=0.0, in_max=1.0, outside="linear"):
        super().__init__()
        assert degree in (2, 3)
        self.n_basis = int(n_basis)
        self.degree = int(degree)
        self.in_min = float(in_min); self.in_max = float(in_max)

        # FIX: ensure at least degree+1 basis functions
        if self.n_basis < self.degree + 1:
            self.n_basis = self.degree + 1

        self.register_buffer('knots', make_clamped_uniform_knots(self.in_min, self.in_max, self.n_basis, self.degree))
        self.coeffs = nn.Parameter(torch.zeros(self.n_basis))
        self.outside = outside

    def _eval_inside(self, x):
        B = bspline_basis(x, self.knots, self.degree)  # [B, n_basis]
        return (B * self.coeffs.view(1, -1)).sum(dim=1)

    def forward(self, x):
        x = x.view(-1)
        m_lo = x < self.in_min
        m_hi = x > self.in_max
        m_in = (~m_lo) & (~m_hi)
        y = torch.empty_like(x)
        if m_in.any():
            y[m_in] = self._eval_inside(x[m_in])
        if m_lo.any() or m_hi.any():
            if self.outside == "clamp":
                if m_lo.any():
                    y[m_lo] = self._eval_inside(torch.full_like(x[m_lo], self.in_min))
                if m_hi.any():
                    y[m_hi] = self._eval_inside(torch.full_like(x[m_hi], self.in_max))
            else:
                eps = 1e-3 * (self.in_max - self.in_min)
                if m_lo.any():
                    x0 = torch.full_like(x[m_lo], self.in_min)
                    y0 = self._eval_inside(x0)
                    x1 = torch.full_like(x[m_lo], min(self.in_min + eps, self.in_max))
                    y1 = self._eval_inside(x1)
                    slope = (y1 - y0) / (x1 - x0 + 1e-12)
                    y[m_lo] = y0 + slope * (x[m_lo] - self.in_min)
                if m_hi.any():
                    x0 = torch.full_like(x[m_hi], self.in_max)
                    y0 = self._eval_inside(x0)
                    x1 = torch.full_like(x[m_hi], max(self.in_max - eps, self.in_min))
                    y1 = self._eval_inside(x1)
                    slope = (y0 - y1) / (x0 - x1 + 1e-12)
                    y[m_hi] = y0 + slope * (x[m_hi] - self.in_max)
        return y.view(-1, 1)  # column

class KANUnivariate(nn.Module):
    def __init__(self, n_basis=10, degree=3, in_min=0.0, in_max=1.0, outside="linear", residual_type="silu"):
        super().__init__()
        self.spline = BSpline1D(n_basis=n_basis, degree=degree, in_min=in_min, in_max=in_max, outside=outside)
        # Per-edge scales
        self.ws = nn.Parameter(torch.tensor(1.0))  # spline scale
        self.residual_type = residual_type
        if residual_type in ("silu", "linear"):
            self.wb = nn.Parameter(torch.tensor(1.0))  # bypass scale (per edge)
        else:
            self.wb = None  # no bypass

    def forward(self, x):
        s = self.spline(x).view(-1)  # [B]
        if self.residual_type == "silu":
            return self.wb * F.silu(x) + self.ws * s
        elif self.residual_type == "linear":
            return self.wb * x + self.ws * s
        else:
            return self.ws * s

class KANLayerSlow(nn.Module):
    def __init__(self, d_in, d_out, n_basis=10, degree=3, in_min=0.0, in_max=1.0,
                 outside="linear", residual_type="silu"):
        super().__init__()
        self.d_in = d_in; self.d_out = d_out
        # Equalized linear residual: scalar α if dims match, else edge-wise wb
        self.equalized_linear_scalar = (residual_type == "linear" and d_in == d_out)
        edge_residual_type = residual_type if not self.equalized_linear_scalar else "none"

        self.phi = nn.ModuleList(
            [KANUnivariate(n_basis=n_basis, degree=degree, in_min=in_min, in_max=in_max,
                           outside=outside, residual_type=edge_residual_type)
             for _ in range(d_in * d_out)]
        )

        if self.equalized_linear_scalar:
            self.residual_alpha = nn.Parameter(torch.tensor(0.1))
        else:
            self.register_parameter("residual_alpha", None)

    def forward(self, x):
        B = x.shape[0]
        out = x.new_zeros(B, self.d_out)
        idx = 0
        for j in range(self.d_out):
            s = 0.0
            for i in range(self.d_in):
                s = s + self.phi[idx](x[:, i]).view(-1)
                idx += 1
            out[:, j] = s

        # Add α·x once per layer when d_in == d_out and residual_type == 'linear'
        if self.residual_alpha is not None:
            out = out + self.residual_alpha * x
        return out  # <-- FIX: return AFTER loop, not inside

# --- FAST vectorized KAN ---------------------------------------------

class FastKANLayer(nn.Module):
    """
    Vectorized KAN layer mapping d_in -> d_out:
      y[:, j] = sum_i [ residual_term(x[:,i]) * wb[i,j]  +  ws[i,j] * S_i,j(x[:,i]) ],
    where residual_term is SiLU(x) if residual_type='silu', x if 'linear', or 0 if 'none'.
    S_i,j is a degree-d clamped-uniform B-spline with n_basis coefficients.
    Outside behavior 'linear' or 'clamp' matches the slow implementation.

    Equalized linear residuals:
      - If residual_type == 'linear' and d_in == d_out, we DO NOT use per-edge wb.
        Instead we add a single scalar α per layer and add α·x after the spline sum.
      - If residual_type == 'linear' and d_in != d_out, behavior is unchanged (wb matrix).
    """
    def __init__(self, d_in, d_out, n_basis=10, degree=3, in_min=0.0, in_max=1.0,
                 outside="linear", residual_type="silu"):
        super().__init__()
        assert degree in (2, 3)
        self.d_in = int(d_in)
        self.d_out = int(d_out)
        self.degree = int(degree)
        self.n_basis = int(n_basis)
        self.in_min = float(in_min)
        self.in_max = float(in_max)
        self.outside = outside
        self.residual_type = residual_type
        self.equalized_linear_scalar = (residual_type == "linear" and self.d_in == self.d_out)

        # FIX: ensure at least degree+1 basis functions
        if self.n_basis < self.degree + 1:
            self.n_basis = self.degree + 1

        # Shared knot vector
        self.register_buffer('knots', make_clamped_uniform_knots(self.in_min, self.in_max, self.n_basis, self.degree))
        # Parameters: per-edge coefficients and scales
        self.coeffs = nn.Parameter(torch.zeros(self.d_in, self.d_out, self.n_basis))  # [d_in, d_out, K]
        self.ws     = nn.Parameter(torch.ones(self.d_in, self.d_out))                # spline scale

        # Residual parameters:
        #   silu: per-edge wb
        #   linear & d_in!=d_out: per-edge wb
        #   linear & d_in==d_out: single scalar residual_alpha
        if self.residual_type == "silu":
            self.wb = nn.Parameter(torch.ones(self.d_in, self.d_out))
            self.register_parameter("residual_alpha", None)
        elif self.residual_type == "linear":
            if self.equalized_linear_scalar:
                self.register_parameter("wb", None)
                self.residual_alpha = nn.Parameter(torch.tensor(0.1))
            else:
                self.wb = nn.Parameter(torch.ones(self.d_in, self.d_out))
                self.register_parameter("residual_alpha", None)
        else:
            self.register_parameter("wb", None)
            self.register_parameter("residual_alpha", None)

        # Precompute boundary bases
        with torch.no_grad():
            self._B_min_cpu = bspline_basis(torch.tensor([self.in_min]), self.knots.cpu(), self.degree)  # [1,K]
            self._B_max_cpu = bspline_basis(torch.tensor([self.in_max]), self.knots.cpu(), self.degree)
            eps = max(1e-3 * (self.in_max - self.in_min), 1e-6)
            self._B_min_eps_cpu = bspline_basis(torch.tensor([min(self.in_min + eps, self.in_max)]),
                                                self.knots.cpu(), self.degree)
            self._B_max_eps_cpu = bspline_basis(torch.tensor([max(self.in_max - eps, self.in_min)]),
                                                self.knots.cpu(), self.degree)
            self._eps = eps

    def _eval_inside_all_outputs(self, B_i, coeffs_i):
        # B_i: [B, K], coeffs_i: [d_out, K] -> [B, d_out]
        return B_i @ coeffs_i.t()

    def forward(self, x):
        # x: [B, d_in]
        device = x.device
        Bsize = x.shape[0]
        out = x.new_zeros(Bsize, self.d_out)

        Bmin  = self._B_min_cpu.to(device=device, dtype=x.dtype)
        Bmax  = self._B_max_cpu.to(device=device, dtype=x.dtype)
        BminE = self._B_min_eps_cpu.to(device=device, dtype=x.dtype)
        BmaxE = self._B_max_eps_cpu.to(device=device, dtype=x.dtype)
        eps = self._eps

        for i in range(self.d_in):
            xi = x[:, i]  # [B]

            # residual / projection term
            if self.wb is not None:
                if self.residual_type == "silu":
                    bypass = F.silu(xi).unsqueeze(1) * self.wb[i]  # [B, d_out]
                else:  # "linear" with d_in != d_out
                    bypass = xi.unsqueeze(1) * self.wb[i]          # [B, d_out]
            else:
                bypass = 0.0

            # Spline term
            m_lo = xi < self.in_min
            m_hi = xi > self.in_max
            m_in = (~m_lo) & (~m_hi)

            if m_in.any():
                Bi = bspline_basis(xi[m_in], self.knots.to(device=device, dtype=x.dtype), self.degree)  # [B_in,K]
                Si = self._eval_inside_all_outputs(Bi, self.coeffs[i])  # [B_in, d_out]
                spline_term_in = torch.zeros(Bsize, self.d_out, device=device, dtype=x.dtype)
                spline_term_in[m_in] = Si
            else:
                spline_term_in = torch.zeros(Bsize, self.d_out, device=device, dtype=x.dtype)

            if m_lo.any() or m_hi.any():
                if self.outside == "clamp":
                    if m_lo.any():
                        y0 = (Bmin @ self.coeffs[i].t()).squeeze(0)
                        spline_term_in[m_lo] = y0
                    if m_hi.any():
                        y1 = (Bmax @ self.coeffs[i].t()).squeeze(0)
                        spline_term_in[m_hi] = y1
                else:
                    if m_lo.any():
                        y0 = (Bmin  @ self.coeffs[i].t()).squeeze(0)
                        y1 = (BminE @ self.coeffs[i].t()).squeeze(0)
                        slope = (y1 - y0) / (eps + 1e-12)
                        dx = (xi[m_lo] - self.in_min).unsqueeze(1)
                        spline_term_in[m_lo] = y0 + dx * slope
                    if m_hi.any():
                        y0 = (Bmax  @ self.coeffs[i].t()).squeeze(0)
                        y1 = (BmaxE @ self.coeffs[i].t()).squeeze(0)
                        slope = (y0 - y1) / (eps + 1e-12)
                        dx = (xi[m_hi] - self.in_max).unsqueeze(1)
                        spline_term_in[m_hi] = y0 + dx * slope

            out = out + bypass + (spline_term_in * self.ws[i])

        # Add α·x once per layer for linear residual with matching dims
        if self.residual_alpha is not None:
            out = out + self.residual_alpha * x

        return out

class FastKANNet(nn.Module):
    """
    Multi-layer KAN with optional BN before/after each layer and per-head affine at the end.
    Uses FastKANLayer internally.
    """
    def __init__(self, input_dim, architecture, final_dim, n_basis=10, degree=3,
                 bn_type="batch", bn_position="after", bn_skip_first=True, outside="linear",
                 residual_type="silu"):
        super().__init__()
        self.deg = int(degree)
        self.n_basis = int(n_basis)
        self.bn_type = bn_type
        self.bn_position = bn_position
        self.bn_skip_first = bn_skip_first
        self.outside = outside
        self.residual_type = residual_type

        dims = [input_dim] + list(architecture) + [final_dim]
        self.layers = nn.ModuleList()
        self.bn_layers = nn.ModuleList()

        for li, (a, b) in enumerate(zip(dims[:-1], dims[1:])):
            if bn_type == "batch" and bn_position == "before" and not (bn_skip_first and li == 0):
                self.bn_layers.append(nn.BatchNorm1d(a))
            else:
                self.bn_layers.append(nn.Identity())

            self.layers.append(FastKANLayer(a, b, n_basis=self.n_basis, degree=self.deg,
                                            outside=self.outside, residual_type=self.residual_type))

            if bn_type == "batch" and bn_position == "after" and not (bn_skip_first and li == 0):
                self.bn_layers.append(nn.BatchNorm1d(b))
            else:
                self.bn_layers.append(nn.Identity())

        self.out_scale = nn.Parameter(torch.ones(final_dim))
        self.out_bias  = nn.Parameter(torch.zeros(final_dim))

    def forward(self, x):
        h = x
        for li, layer in enumerate(self.layers):
            bn_before = self.bn_layers[2*li + 0]
            bn_after  = self.bn_layers[2*li + 1]
            h = bn_before(h)
            h = layer(h)
            h = bn_after(h)
        return h * self.out_scale + self.out_bias

# --- Backward-compatible (slow) KANNet -------------------------------

class KANNetSlow(nn.Module):
    def __init__(self, input_dim, architecture, final_dim, n_basis=10, degree=3,
                 bn_type="batch", bn_position="after", bn_skip_first=True, outside="linear",
                 residual_type="silu"):
        super().__init__()
        self.deg = int(degree)
        self.n_basis = int(n_basis)
        self.bn_type = bn_type
        self.bn_position = bn_position
        self.bn_skip_first = bn_skip_first
        self.outside = outside
        self.residual_type = residual_type

        dims = [input_dim] + list(architecture) + [final_dim]
        self.layers = nn.ModuleList()
        self.bn_layers = nn.ModuleList()

        for li, (a, b) in enumerate(zip(dims[:-1], dims[1:])):
            if bn_type == "batch" and bn_position == "before" and not (bn_skip_first and li == 0):
                self.bn_layers.append(nn.BatchNorm1d(a))
            else:
                self.bn_layers.append(nn.Identity())

            self.layers.append(KANLayerSlow(a, b, n_basis=self.n_basis, degree=self.deg,
                                            outside=self.outside, residual_type=self.residual_type))

            if bn_type == "batch" and bn_position == "after" and not (bn_skip_first and li == 0):
                self.bn_layers.append(nn.BatchNorm1d(b))
            else:
                self.bn_layers.append(nn.Identity())

        self.out_scale = nn.Parameter(torch.ones(final_dim))
        self.out_bias  = nn.Parameter(torch.zeros(final_dim))

    def forward(self, x):
        h = x
        for li, layer in enumerate(self.layers):
            bn_before = self.bn_layers[2*li + 0]
            bn_after  = self.bn_layers[2*li + 1]
            h = bn_before(h)
            h = layer(h)
            h = bn_after(h)
        return h * self.out_scale + self.out_bias  # <-- FIX: return after loop

def kan_param_count(arch, input_dim, final_dim, n_basis, degree,
                    bn_type, bn_position, bn_skip_first, residual_type):
    """
    Count parameters for KAN under the chosen residual_type.
      - per-edge: n_basis coeffs + ws (+ wb if residual_type!='none')
      - BUT for residual_type='linear' with a==b, **omit wb per-edge** and add **1 scalar** per layer.
      - BN affine params
      - output affine
    """
    dims = [input_dim] + list(arch) + [final_dim]
    total = 0
    eff_basis = max(int(n_basis), int(degree) + 1)  # FIX: enforce minimum basis

    for a, b in zip(dims[:-1], dims[1:]):
        use_layer_scalar = (residual_type == "linear" and a == b)
        layer_add_wb = (residual_type != "none") and not use_layer_scalar
        per_edge = eff_basis + 1 + (1 if layer_add_wb else 0)  # coeffs + ws + (wb?)
        total += per_edge * a * b
        if use_layer_scalar:
            total += 1  # α scalar for this layer
    if bn_type == "batch":
        for li, (a, b) in enumerate(zip(dims[:-1], dims[1:])):
            if bn_position == "before" and not (bn_skip_first and li == 0):
                total += 2 * a
            if bn_position == "after"  and not (bn_skip_first and li == 0):
                total += 2 * b
    total += (final_dim * 2)
    return total


def choose_kan_basis_for_parity(target_params, arch, input_dim, final_dim,
                                degree, bn_type, bn_position, bn_skip_first,
                                residual_type, prefer_leq=True):
    min_K = int(degree) + 1  # FIX: search only valid K
    best_n = None; best_diff = float('inf'); best_count = None
    for n in range(min_K, 200):
        cnt = kan_param_count(arch, input_dim, final_dim, n, degree,
                              bn_type, bn_position, bn_skip_first, residual_type)
        diff = abs(cnt - target_params)
        if prefer_leq:
            if cnt <= target_params and diff < best_diff:
                best_diff = diff; best_n = n; best_count = cnt
        else:
            if diff < best_diff:
                best_diff = diff; best_n = n; best_count = cnt
    if best_n is None:
        for n in range(min_K, 200):
            cnt = kan_param_count(arch, input_dim, final_dim, n, degree,
                                  bn_type, bn_position, bn_skip_first, residual_type)
            if cnt >= target_params:
                return n, cnt
        return 199, kan_param_count(arch, input_dim, final_dim, 199, degree,
                                    bn_type, bn_position, bn_skip_first, residual_type)
    return best_n, best_count


# -------------------------------
# KAN BN recalc (with progress)
# -------------------------------

def recalc_bn_for_kan_progress(model: nn.Module, x_train: torch.Tensor, passes: int = 10, desc="KAN BN re-calc"):
    """Recompute KAN BatchNorm running stats using training data (with tqdm)."""
    was_training = model.training
    for m in model.modules():
        if isinstance(m, nn.BatchNorm1d):
            m.reset_running_stats()
            m.momentum = 0.1
    model.train()
    with torch.no_grad():
        for p in tqdm(range(passes), desc=desc):
            _ = model(x_train)
            if p >= passes - 3:
                for m in model.modules():
                    if isinstance(m, nn.BatchNorm1d):
                        m.momentum = 0.01
    if not was_training:
        model.eval()


def recalc_bn_for_sn_progress(model: nn.Module, x_train: torch.Tensor, passes: int = 10, desc="SN BN re-calc"):
    """SN BN recalc with progress bars."""
    was_training = model.training
    for module in model.modules():
        if isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
            module.reset_running_stats()
            module.momentum = 0.1
    model.train()
    with torch.no_grad():
        for p in tqdm(range(passes), desc=desc):
            _ = model(x_train)
            if p >= passes - 3:
                for module in model.modules():
                    if isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
                        module.momentum = 0.01
    if not was_training:
        model.eval()


# -------------------------------
# KAN training (fast/slow)
# -------------------------------

def build_kan(input_dim, final_dim, arch, n_basis, degree, bn_type, bn_position, bn_skip_first,
              outside, residual_type, impl="fast"):
    if impl == "fast":
        return FastKANNet(
            input_dim=input_dim, architecture=arch, final_dim=final_dim,
            n_basis=n_basis, degree=degree,
            bn_type=bn_type, bn_position=bn_position, bn_skip_first=bn_skip_first,
            outside=outside, residual_type=residual_type
        )
    else:
        return KANNetSlow(
            input_dim=input_dim, architecture=arch, final_dim=final_dim,
            n_basis=n_basis, degree=degree,
            bn_type=bn_type, bn_position=bn_position, bn_skip_first=bn_skip_first,
            outside=outside, residual_type=residual_type
        )

def train_kan(x_train, y_train, input_dim, final_dim, arch, n_basis, degree, device, epochs=4000,
              lr=1e-3, wd=1e-6, seed=0, bn_type="batch", bn_position="after", bn_skip_first=True,
              outside="linear", residual_type="silu", impl="fast"):
    set_global_seeds(seed)
    model = build_kan(
        input_dim=input_dim, final_dim=final_dim, arch=arch, n_basis=n_basis, degree=degree,
        bn_type=bn_type, bn_position=bn_position, bn_skip_first=bn_skip_first,
        outside=outside, residual_type=residual_type, impl=impl
    ).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
    loss_fn = nn.MSELoss()
    x_train = x_train.to(device); y_train = y_train.to(device)

    t0 = time.time()
    for _ in tqdm(range(epochs), desc=f"Training KAN ({impl})"):
        opt.zero_grad(set_to_none=True)
        yhat = model(x_train)
        loss = loss_fn(yhat, y_train)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
    secs = time.time() - t0

    with torch.no_grad():
        train_mse = float(loss_fn(model(x_train), y_train).cpu())
    return model, train_mse, secs


# -------------------------------
# SN warm-up then freeze helper
# -------------------------------

def continue_train_sn_no_domain_updates(model, x_train, y_train, epochs, device, seed,
                                        lr_other=3e-4, lr_codomain=1e-3, wd=1e-7, clip=1.0):
    set_global_seeds(seed)
    model = model.to(device)
    model.train()
    params = []
    if CONFIG.get('train_phi_codomain', True):
        params.append({"params": [p for n, p in model.named_parameters() if "phi_codomain_params" in n], "lr": lr_codomain})
    if CONFIG.get('use_lateral_mixing', False):
        params.append({"params": [p for n, p in model.named_parameters() if "lateral" in n], "lr": 5e-4})
    excluded = []
    if CONFIG.get('train_phi_codomain', True): excluded.append("phi_codomain_params")
    if CONFIG.get('use_lateral_mixing', False): excluded.append("lateral")
    params.append({"params": [p for n, p in model.named_parameters() if not any(e in n for e in excluded)], "lr": lr_other})
    opt = torch.optim.Adam(params, weight_decay=wd)

    loss_fn = nn.MSELoss()
    x_train = x_train.to(device); y_train = y_train.to(device)
    t0 = time.time()
    for _ in tqdm(range(max(0, int(epochs))), desc="Continue SN (domains frozen)"):
        opt.zero_grad(set_to_none=True)
        yhat = model(x_train)
        loss = loss_fn(yhat, y_train)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        opt.step()
    secs = time.time() - t0

    with torch.no_grad():
        train_mse = float(loss_fn(model(x_train), y_train).cpu())
    return model, train_mse, secs


# -------------------------------
# Batched evaluation with PROGRESS
# -------------------------------

def _is_sn_model(model: nn.Module) -> bool:
    """
    Heuristic: SN models in this codebase have `layers` AND `norm_layers` attributes.
    KAN nets have `layers` and `bn_layers` but no `norm_layers`.
    """
    return hasattr(model, "layers") and hasattr(model, "norm_layers")

def evaluate_with_progress(model, x_test, y_test, *,
                           bn_mode="off", bn_recalc_passes=10, x_train_for_bn=None,
                           name="Model", eval_batch_size=8192):
    """
    Evaluate `model` on (x_test, y_test) in batches with tqdm and **no BN state mutation**.

    - If `bn_mode == "recalc_eval"` and the model has BN:
        * SN:   Recompute BN stats (for completeness), then evaluate using **batch stats**
                while **disabling BN buffer updates** via a context manager. Restore the
                model's original train/eval state afterwards.
        * KAN:  Recompute BN stats, switch to eval() (running stats), evaluate, then restore.
    """
    N = x_test.shape[0]
    out_dim = y_test.shape[1]
    y_pred_cpu = torch.empty(N, out_dim, dtype=y_test.dtype)
    bs = max(1, int(eval_batch_size))
    device = next((p.device for p in model.parameters() if p.requires_grad), x_test.device)

    # Preserve original top-level train/eval state and set up evaluation policy
    was_training = model.training
    context = nullcontext()

    if bn_mode == "recalc_eval" and has_batchnorm(model):
        if _is_sn_model(model):
            # Recompute (not strictly required since we'll use batch stats), then
            # evaluate with BN using batch stats but without updating running buffers.
            if x_train_for_bn is not None:
                recalc_bn_for_sn_progress(model, x_train_for_bn, passes=bn_recalc_passes, desc=f"{name} BN re-calc")
            model.train(True)  # ensure BN layers are in train mode for batch stats
            context = use_batch_stats_without_updating_bn(model)
        else:
            # KAN: standard pattern — recalc running stats then eval() to use them
            if x_train_for_bn is not None:
                recalc_bn_for_kan_progress(model, x_train_for_bn, passes=bn_recalc_passes, desc=f"{name} BN re-calc")
            model.eval()
            context = nullcontext()
    else:
        # No special BN handling requested — just use current state
        context = nullcontext()

    with torch.no_grad(), context:
        rng = range(0, N, bs)
        for start in tqdm(rng, desc=f"Evaluating {name}", total=(N + bs - 1) // bs):
            end = min(N, start + bs)
            yb = model(x_test[start:end].to(device))
            y_pred_cpu[start:end] = yb.detach().cpu()

    # Restore original state
    model.train(was_training)

    per_head, mean_rmse = rmse_per_head(y_test.cpu(), y_pred_cpu)
    corrF, used = corr_frobenius(y_test.cpu(), y_pred_cpu)
    return per_head, mean_rmse, corrF, used


# -------------------------------
# Main
# -------------------------------

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"])
    p.add_argument("--seed", type=int, default=45)
    p.add_argument("--epochs", type=int, default=4000)

    # Dataset
    p.add_argument("--dataset", type=str, default="toy_4d_to_5d")

    # --- SN config ---
    p.add_argument("--sn_arch", type=str, default="15,15")
    p.add_argument("--sn_phi_knots", type=int, default=60)
    p.add_argument("--sn_Phi_knots", type=int, default=60)
    p.add_argument("--sn_norm_type", type=str, default="batch", choices=["none", "batch", "layer"])
    p.add_argument("--sn_norm_position", type=str, default="after", choices=["before", "after"])
    p.add_argument("--sn_norm_skip_first", action="store_true", default=True)
    p.add_argument("--sn_norm_first", action="store_true")
    p.add_argument("--sn_no_residual", action="store_true")
    # FIX: the 'type' must be passed as a keyword argument, not as a positional value.
    p.add_argument("--sn_residual_style", type=str, default=None,
                   choices=["node", "linear", "standard", "matrix"])
    p.add_argument("--sn_no_lateral", action="store_true")
    # Domain policy
    p.add_argument("--sn_freeze_domains_after", type=int, default=0)
    p.add_argument("--sn_domain_margin", type=float, default=0.0)

    # --- KAN config ---
    p.add_argument("--kan_arch", type=str, default="4,4")
    p.add_argument("--kan_degree", type=int, default=3, choices=[2, 3])
    p.add_argument("--kan_K", type=int, default=60, help="Initial #basis; ignored if --equalize_params is used.")
    p.add_argument("--kan_bn_type", type=str, default="batch", choices=["none", "batch"])
    p.add_argument("--kan_bn_position", type=str, default="after", choices=["before", "after"])
    p.add_argument("--kan_bn_skip_first", action="store_true", default=True)
    p.add_argument("--kan_outside", type=str, default="linear", choices=["linear", "clamp"])
    p.add_argument("--kan_residual_type", type=str, default="silu", choices=["silu", "linear", "none"])
    p.add_argument("--kan_lr", type=float, default=1e-3)
    p.add_argument("--kan_wd", type=float, default=1e-6)
    p.add_argument("--kan_impl", type=str, default="fast", choices=["fast", "slow"],
                   help="Use fast vectorized KAN or the original per-edge ModuleList version.")

    # Global fairness / parity
    p.add_argument("--equalize_params", action="store_true")
    p.add_argument("--prefer_leq", action="store_true")

    # BN at test
    p.add_argument("--bn_eval_mode", type=str, default="recalc_eval", choices=["off", "recalc_eval"])
    p.add_argument("--bn_recalc_passes", type=int, default=10)

    # Test set size + eval batching
    p.add_argument("--n_test", type=int, default=20000)
    p.add_argument("--eval_batch_size", type=int, default=8192)

    # Output
    p.add_argument("--outdir", type=str, default="benchmarks/results")

    args = p.parse_args()
    device = ("cuda" if (args.device == "auto" and torch.cuda.is_available()) else
              ("cpu" if args.device == "auto" else args.device))
    set_global_seeds(args.seed)

    # dataset & dims
    dataset = get_dataset(args.dataset)
    input_dim = dataset.input_dim
    final_dim = dataset.output_dim

    # fixed test set
    torch.manual_seed(args.seed + 2025)
    with torch.no_grad():
        x_test, y_test = dataset.sample(args.n_test, device=device)

    # ------------------------------
    # Configure SN fairness knobs
    # ------------------------------
    if args.sn_no_residual:
        CONFIG['use_residual_weights'] = False
    else:
        # residual style override if provided
        if args.sn_residual_style is not None:
            style = args.sn_residual_style.lower()
            if style in ("standard", "matrix"):
                style = "linear"
            CONFIG['residual_style'] = style  # used when constructing SN blocks

    if args.sn_no_lateral:
        CONFIG['use_lateral_mixing'] = False

    sn_norm_skip = False if args.sn_norm_first else args.sn_norm_skip_first
    CONFIG['domain_safety_margin'] = float(args.sn_domain_margin)

    sn_arch = [int(x) for x in args.sn_arch.split(",")] if args.sn_arch.strip() else []

    # ------------------------------
    # SN training: warm-up then freeze (optional)
    # ------------------------------
    total_epochs = int(args.epochs)
    warmup_epochs = max(0, min(args.sn_freeze_domains_after, total_epochs))
    rest_epochs = total_epochs - warmup_epochs

    # Phase 1: warm-up WITH domain updates (standard trainer)
    t0_sn = time.time()
    # residual style to pass explicitly (if residuals enabled)
    residual_style_override = None if args.sn_no_residual else CONFIG.get('residual_style', 'node')
    plotting_snapshot, _ = train_network(
        dataset=dataset,
        architecture=sn_arch,
        total_epochs=warmup_epochs if warmup_epochs > 0 else total_epochs,
        print_every=max(1, (warmup_epochs if warmup_epochs > 0 else total_epochs) // 10),
        device=device,
        phi_knots=args.sn_phi_knots,
        Phi_knots=args.sn_Phi_knots,
        seed=args.seed,
        norm_type=args.sn_norm_type,
        norm_position=args.sn_norm_position,
        norm_skip_first=sn_norm_skip,
        no_load_best=False,
        bn_recalc_on_load=False,
        residual_style=residual_style_override
    )
    sn_secs = time.time() - t0_sn

    sn_model = plotting_snapshot["model"].to(device)
    x_train  = plotting_snapshot["x_train"].to(device)
    y_train  = plotting_snapshot["y_train"].to(device)

    if rest_epochs > 0:
        t1 = time.time()
        sn_model, sn_train_mse, secs2 = continue_train_sn_no_domain_updates(
            model=sn_model, x_train=x_train, y_train=y_train, epochs=rest_epochs,
            device=device, seed=args.seed
        )
        sn_secs += (time.time() - t1)
    else:
        with torch.no_grad():
            sn_train_mse = float(torch.mean((sn_model(x_train) - y_train) ** 2).cpu())

    sn_params = count_params(sn_model)

    # ------------------------------
    # Choose K for KAN if equalizing
    # ------------------------------
    kan_arch = [int(x) for x in args.kan_arch.split(",")] if args.kan_arch.strip() else []
    if args.equalize_params:
        chosen_K, est_cnt = choose_kan_basis_for_parity(
            target_params=sn_params, arch=kan_arch, input_dim=input_dim, final_dim=final_dim,
            degree=args.kan_degree, bn_type=args.kan_bn_type, bn_position=args.kan_bn_position,
            bn_skip_first=args.kan_bn_skip_first, residual_type=args.kan_residual_type,
            prefer_leq=args.prefer_leq
        )
        kan_K = chosen_K
        parity_note = f"[ParamMatch] Target SN params = {sn_params}. Chosen K for KAN = {kan_K} (est. {est_cnt} params)."
        print(parity_note)
    else:
        kan_K = args.kan_K
        parity_note = ""

    # ------------------------------
    # Train KAN on EXACT same (x_train, y_train)
    # ------------------------------
    t0_kan = time.time()
    kan_model, kan_train_mse, kan_secs = train_kan(
        x_train=x_train, y_train=y_train,
        input_dim=input_dim, final_dim=final_dim,
        arch=kan_arch, n_basis=kan_K, degree=args.kan_degree,
        device=device, epochs=args.epochs, seed=args.seed,
        lr=args.kan_lr, wd=args.kan_wd,
        bn_type=args.kan_bn_type, bn_position=args.kan_bn_position, bn_skip_first=args.kan_bn_skip_first,
        outside=args.kan_outside, residual_type=args.kan_residual_type, impl=args.kan_impl
    )
    kan_params = count_params(kan_model)

    # ------------------------------
    # Test evaluation with PROGRESS
    # ------------------------------
    sn_per_head, sn_rmse_mean, sn_corrF, sn_corr_used = evaluate_with_progress(
        sn_model, x_test, y_test,
        bn_mode=args.bn_eval_mode, bn_recalc_passes=args.bn_recalc_passes, x_train_for_bn=x_train,
        name="SN", eval_batch_size=args.eval_batch_size
    )
    kan_per_head, kan_rmse_mean, kan_corrF, kan_corr_used = evaluate_with_progress(
        kan_model, x_test, y_test,
        bn_mode=args.bn_eval_mode, bn_recalc_passes=args.bn_recalc_passes, x_train_for_bn=x_train,
        name=f"KAN-{args.kan_impl}", eval_batch_size=args.eval_batch_size
    )

    # ------------------------------
    # Print & save results
    # ------------------------------
    def pretty(title, params, train_mse, mean_rmse, per_head, corrF, used, secs, notes=""):
        out = {
            "model": title,
            "params": int(params),
            "train_mse": float(train_mse),
            "test_rmse_mean": float(mean_rmse),
            "test_rmse_per_head": [float(x) for x in per_head],
            "corr_Frob_error": (None if (corrF is None or not math.isfinite(corrF)) else float(corrF)),
            "corr_Frob_heads_used": int(used),
            "train_seconds": float(secs),
        }
        if notes:
            out["notes"] = notes
        return out

    sn_title = (
        f"SN(arch={sn_arch}, phi_knots={args.sn_phi_knots}, Phi_knots={args.sn_Phi_knots}, "
        f"norm={args.sn_norm_type}/{args.sn_norm_position}/"
        f"{'skip_first' if sn_norm_skip else 'include_first'}, "
        f"residuals={'off' if args.sn_no_residual else 'on(style='+CONFIG.get('residual_style','node')+')'}, "
        f"lateral={'off' if args.sn_no_lateral else 'on'}, "
        f"domains={'warmup+freeze' if args.sn_freeze_domains_after>0 else 'updated'})"
    )
    kan_title = (
        f"KAN[{args.kan_impl}](arch={kan_arch}, K={kan_K}, degree={args.kan_degree}, "
        f"BN={args.kan_bn_type}/{args.kan_bn_position}/"
        f"{'skip_first' if args.kan_bn_skip_first else 'include_first'}, "
        f"outside={args.kan_outside}, residual={args.kan_residual_type})"
    )

    sn_result = pretty(
        sn_title, sn_params, sn_train_mse, sn_rmse_mean, sn_per_head, sn_corrF, sn_corr_used, sn_secs,
        notes="SN timing includes warm-up + optional freeze phase. " + parity_note
    )
    kan_result = pretty(
        kan_title, kan_params, kan_train_mse, kan_rmse_mean, kan_per_head, kan_corrF, kan_corr_used, kan_secs,
        notes=("BN standardized at test" if args.bn_eval_mode == "recalc_eval" else "") + ("; " + parity_note if parity_note else "")
    )

    print(f"\n=== Head-to-Head Results ({args.dataset}) ===")
    for r in (sn_result, kan_result):
        print(json.dumps(r, indent=2))

    os.makedirs(args.outdir, exist_ok=True)
    out_path = os.path.join(args.outdir, f"kan_sn_parity_{args.dataset}_seed{args.seed}.json")
    with open(out_path, "w") as f:
        json.dump([sn_result, kan_result], f, indent=2)
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()