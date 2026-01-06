"""
Benchmark: Shared-motif regression (barebones PWL SN vs barebones PCHIP KAN)

This script trains:
  1) A *barebones* Sprecher Network (SN) with **piecewise-linear** splines,
     with theoretical **domain updates only during the first 10% of epochs**
     (400 warmup epochs out of 4000 total).
  2) A *barebones* KAN with **cubic PCHIP** splines (fixed knot grids; no grid updates).

No engineering extras:
  - No residual connections
  - No lateral mixing
  - No normalization layers
  - No grid updates / adaptive knot relocation for KAN

The target is a "shared-motif" synthetic function designed to highlight a core
strength of SNs: *re-using the same learned 1D motifs across many coordinates*.
The function is also mildly non-smooth (|sin|), which tends to favor PWL
representations over globally C¹ cubic splines under the same parameter budget.

Run (example):
  python -m benchmarks.benchmark_motif_chirp --seed 0

Multi-seed:
  for s in 0 1 2 3; do
    python -m benchmarks.benchmark_motif_chirp --seed $s --phi_knots 512 --Phi_knots 512
  done
"""

from __future__ import annotations

import argparse
import time
from typing import List, Tuple, Optional

import numpy as np
import torch
import torch.nn as nn

from sn_core import SprecherMultiLayerNetwork, CONFIG


# ----------------------------- utilities -----------------------------

def set_global_seed(seed: int) -> None:
    seed = int(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def count_trainable_params(model: nn.Module) -> int:
    return int(sum(p.numel() for p in model.parameters() if p.requires_grad))


@torch.no_grad()
def rmse(pred: torch.Tensor, target: torch.Tensor) -> float:
    return float(torch.sqrt(torch.mean((pred - target) ** 2)).item())


# -------------------------- target function --------------------------

@torch.no_grad()
def shared_motif_target(
    x: torch.Tensor,
    w1: int,
    w2: int,
    out_scale: float = 5.0,
    out_shift: float = -0.4,
) -> torch.Tensor:
    """
    A 2-block Sprecher-structured target on x in [0,1]^D.

    Structure:
      s1_q = sum_i λ1_i * φ*(x_i + η1*q) + q
      h1_q = Φ1*(s1_q)

      s2_r = sum_q λ2_q * φ2*(h1_q + η2*r) + r
      y    = (1/w2) * sum_r Φ2*(s2_r)

    where φ*, φ2* are monotone (sigmoids), and Φ1*, Φ2* include |sin| terms
    to introduce mild non-smoothness (kinks at the sin zero-crossings).
    """
    if x.ndim != 2:
        raise ValueError(f"x must be [B,D], got {tuple(x.shape)}")
    B, D = x.shape

    # Deterministic positive weights (same across runs; target is fixed)
    idx = torch.arange(D, device=x.device, dtype=x.dtype)
    lam1 = 0.30 + 0.70 * torch.abs(torch.cos(0.8 * idx + 0.1))
    lam1 = lam1 / lam1.sum() * 9.0

    eta1 = 0.06

    def phi_star(t: torch.Tensor) -> torch.Tensor:
        # steep monotone transition around 0.5 -> "hard-ish" gating
        return torch.sigmoid(30.0 * (t - 0.5))

    def Phi1_star(s: torch.Tensor) -> torch.Tensor:
        # bounded, mildly non-smooth because of abs(sin)
        return torch.sin(1.7 * s) + 0.28 * torch.abs(torch.sin(0.55 * s + 0.2))

    q = torch.arange(w1, device=x.device, dtype=x.dtype).view(1, w1)  # (1,w1)
    shifted = x.unsqueeze(-1) + eta1 * q.view(1, 1, w1)              # (B,D,w1)
    s1 = (phi_star(shifted) * lam1.view(1, D, 1)).sum(dim=1) + q     # (B,w1)
    h1 = Phi1_star(s1)                                               # (B,w1), ~[-1,1.3]

    jdx = torch.arange(w1, device=x.device, dtype=x.dtype)
    lam2 = 0.25 + 0.75 * torch.abs(torch.sin(0.6 * jdx + 0.4))
    lam2 = lam2 / lam2.sum() * 7.0

    eta2 = 0.05

    def phi2_star(t: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(12.0 * (t - 0.0))

    def Phi2_star(s: torch.Tensor) -> torch.Tensor:
        return torch.sin(1.15 * s) + 0.22 * torch.abs(torch.sin(0.9 * s + 0.7))

    r = torch.arange(w2, device=x.device, dtype=x.dtype).view(1, w2)
    shifted2 = h1.unsqueeze(-1) + eta2 * r.view(1, 1, w2)            # (B,w1,w2)
    s2 = (phi2_star(shifted2) * lam2.view(1, w1, 1)).sum(dim=1) + r  # (B,w2)

    y = Phi2_star(s2).sum(dim=1, keepdim=True) / float(w2)
    y = out_scale * y + out_shift
    return y


def make_dataset(
    n_train: int,
    n_test: int,
    input_dim: int,
    w1: int,
    w2: int,
    seed: int,
    device: torch.device,
    dtype: torch.dtype,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Fixed train/test sets (for fair comparison).
    """
    g = torch.Generator(device="cpu")
    g.manual_seed(int(seed))

    x_train = torch.rand((int(n_train), int(input_dim)), generator=g, dtype=dtype).to(device)
    x_test = torch.rand((int(n_test), int(input_dim)), generator=g, dtype=dtype).to(device)

    y_train = shared_motif_target(x_train, w1=w1, w2=w2).to(dtype=dtype)
    y_test = shared_motif_target(x_test, w1=w1, w2=w2).to(dtype=dtype)
    return x_train, y_train, x_test, y_test


# --------------------------- barebones KAN ---------------------------

def pchip_slopes(y: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    """
    Vectorized PCHIP slopes for y[..., K] on strictly increasing x[K].
    Returns d with same shape as y.

    Reference: Fritsch & Carlson (1980) monotone piecewise cubic interpolation.
    """
    if x.ndim != 1:
        raise ValueError("x must be 1D knots")
    K = x.numel()
    if y.shape[-1] != K:
        raise ValueError(f"y last dim must be K={K}, got {y.shape[-1]}")

    if K < 2:
        return torch.zeros_like(y)

    h = x[1:] - x[:-1]                      # (K-1)
    delta = (y[..., 1:] - y[..., :-1]) / (h + 1e-12)  # (..., K-1)

    d = torch.zeros_like(y)

    if K == 2:
        d[..., 0] = delta[..., 0]
        d[..., 1] = delta[..., 0]
        return d

    # Interior slopes (k = 1..K-2)
    hkm1 = h[:-1]             # (K-2)
    hk = h[1:]                # (K-2)
    deltam1 = delta[..., :-1] # (..., K-2)
    deltak = delta[..., 1:]   # (..., K-2)

    sign_ok = (deltam1 * deltak) > 0
    w1 = 2 * hk + hkm1
    w2 = hk + 2 * hkm1

    denom = (w1 / (deltam1 + 1e-12)) + (w2 / (deltak + 1e-12))
    d_int = (w1 + w2) / (denom + 1e-12)
    d_int = torch.where(sign_ok, d_int, torch.zeros_like(d_int))
    d[..., 1:-1] = d_int

    # Endpoints
    d0 = ((2 * h[0] + h[1]) * delta[..., 0] - h[0] * delta[..., 1]) / (h[0] + h[1] + 1e-12)
    dN = ((2 * h[-1] + h[-2]) * delta[..., -1] - h[-1] * delta[..., -2]) / (h[-1] + h[-2] + 1e-12)

    def _limit_endpoint(di: torch.Tensor, deltai: torch.Tensor, deltai1: torch.Tensor) -> torch.Tensor:
        # If di and deltai differ in sign => 0
        di = torch.where((di * deltai) <= 0, torch.zeros_like(di), di)
        # If deltai and deltai1 differ in sign, clamp magnitude to 3*|deltai|
        cond = (deltai * deltai1) < 0
        di = torch.where(cond & (torch.abs(di) > 3 * torch.abs(deltai)), 3 * deltai, di)
        return di

    d[..., 0] = _limit_endpoint(d0, delta[..., 0], delta[..., 1])
    d[..., -1] = _limit_endpoint(dN, delta[..., -1], delta[..., -2])

    return d


class KANLayer(nn.Module):
    """
    Barebones KAN layer:
        y_out[o] = sum_i spline_{o,i}(x_i) + bias[o]

    Uses cubic Hermite interpolation with PCHIP slopes (fixed grid).
    """

    def __init__(
        self,
        d_in: int,
        d_out: int,
        num_knots: int,
        x_min: float,
        x_max: float,
        dtype: torch.dtype = torch.float32,
        init_scale: float = 0.05,
    ):
        super().__init__()
        self.d_in = int(d_in)
        self.d_out = int(d_out)
        self.num_knots = int(num_knots)
        self.x_min = float(x_min)
        self.x_max = float(x_max)

        if self.num_knots < 4:
            raise ValueError("PCHIP cubic splines are intended for num_knots >= 4")

        knots = torch.linspace(self.x_min, self.x_max, self.num_knots, dtype=dtype)
        self.register_buffer("knots", knots)

        # coeffs[o, i, k] = y-value at knot k for edge i->o
        self.coeffs = nn.Parameter(torch.zeros(self.d_out, self.d_in, self.num_knots, dtype=dtype))
        with torch.no_grad():
            self.coeffs.normal_(mean=0.0, std=float(init_scale))

        self.bias = nn.Parameter(torch.zeros(self.d_out, dtype=dtype))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 2 or x.shape[1] != self.d_in:
            raise ValueError(f"Expected x shape [B,{self.d_in}], got {tuple(x.shape)}")

        B = x.shape[0]
        K = self.num_knots
        knots = self.knots

        # Compute per-edge slopes d[o,i,k]
        d = pchip_slopes(self.coeffs, knots)  # (d_out, d_in, K)

        # Prepare interval index/t for all x entries
        below = x < self.x_min
        above = x > self.x_max
        x_clamped = torch.clamp(x, self.x_min, self.x_max)

        idx = torch.searchsorted(knots, x_clamped) - 1
        idx = torch.clamp(idx, 0, K - 2)  # (B, d_in)

        xk = knots[idx]           # (B, d_in)
        xk1 = knots[idx + 1]      # (B, d_in)
        h = (xk1 - xk).clamp_min(1e-12)
        t = (x_clamped - xk) / h  # (B, d_in)

        # Hermite basis
        t2 = t * t
        t3 = t2 * t
        h00 = 2 * t3 - 3 * t2 + 1
        h10 = t3 - 2 * t2 + t
        h01 = -2 * t3 + 3 * t2
        h11 = t3 - t2

        # Gather y0,y1,d0,d1 for each interval index
        coeffs = self.coeffs.unsqueeze(0).expand(B, -1, -1, -1)  # (B, d_out, d_in, K)
        slopes = d.unsqueeze(0).expand(B, -1, -1, -1)            # (B, d_out, d_in, K)

        idx_exp = idx.unsqueeze(1).unsqueeze(-1).expand(B, self.d_out, self.d_in, 1)  # (B,d_out,d_in,1)
        idx1_exp = (idx + 1).unsqueeze(1).unsqueeze(-1).expand(B, self.d_out, self.d_in, 1)

        y0 = torch.gather(coeffs, dim=3, index=idx_exp).squeeze(-1)   # (B,d_out,d_in)
        y1 = torch.gather(coeffs, dim=3, index=idx1_exp).squeeze(-1)
        d0 = torch.gather(slopes, dim=3, index=idx_exp).squeeze(-1)
        d1 = torch.gather(slopes, dim=3, index=idx1_exp).squeeze(-1)

        # Broadcast basis and h to (B,1,d_in)
        h_b = h.unsqueeze(1)
        h00_b = h00.unsqueeze(1)
        h10_b = h10.unsqueeze(1)
        h01_b = h01.unsqueeze(1)
        h11_b = h11.unsqueeze(1)

        vals = h00_b * y0 + h10_b * (h_b * d0) + h01_b * y1 + h11_b * (h_b * d1)  # (B,d_out,d_in)

        # Linear extrapolation outside domain using endpoint slopes
        if below.any() or above.any():
            below_b = below.unsqueeze(1)
            above_b = above.unsqueeze(1)

            left_y = self.coeffs[:, :, 0].unsqueeze(0).expand(B, -1, -1)   # (B,d_out,d_in)
            right_y = self.coeffs[:, :, -1].unsqueeze(0).expand(B, -1, -1)

            left_s = d[:, :, 0].unsqueeze(0).expand(B, -1, -1)
            right_s = d[:, :, -1].unsqueeze(0).expand(B, -1, -1)

            vals = torch.where(below_b, left_y + left_s * (x.unsqueeze(1) - self.x_min), vals)
            vals = torch.where(above_b, right_y + right_s * (x.unsqueeze(1) - self.x_max), vals)

        out = vals.sum(dim=2) + self.bias.unsqueeze(0)  # (B,d_out)
        return out


class BareKAN(nn.Module):
    """2-hidden-layer KAN with cubic PCHIP splines, no grid updates."""

    def __init__(
        self,
        input_dim: int,
        widths: List[int],
        num_knots: int,
        dtype: torch.dtype,
        x_min_in: float = 0.0,
        x_max_in: float = 1.0,
        x_min_hidden: float = -2.0,
        x_max_hidden: float = 2.0,
        init_scale: float = 0.05,
    ):
        super().__init__()
        if len(widths) != 2:
            raise ValueError("This benchmark expects exactly two hidden layers (widths length == 2).")
        w1, w2 = int(widths[0]), int(widths[1])

        self.layers = nn.ModuleList([
            KANLayer(input_dim, w1, num_knots=num_knots, x_min=x_min_in, x_max=x_max_in, dtype=dtype, init_scale=init_scale),
            KANLayer(w1, w2, num_knots=num_knots, x_min=x_min_hidden, x_max=x_max_hidden, dtype=dtype, init_scale=init_scale),
            KANLayer(w2, 1,  num_knots=num_knots, x_min=x_min_hidden, x_max=x_max_hidden, dtype=dtype, init_scale=init_scale),
        ])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = x
        for layer in self.layers:
            h = layer(h)
        return h


def kan_param_count(input_dim: int, widths: List[int], num_knots: int) -> int:
    dims = [int(input_dim)] + [int(w) for w in widths] + [1]
    edges = sum(dims[i] * dims[i + 1] for i in range(len(dims) - 1))
    biases = sum(dims[i + 1] for i in range(len(dims) - 1))
    return int(edges * int(num_knots) + biases)


def choose_kan_knots_to_match(
    target_params: int,
    input_dim: int,
    widths: List[int],
    min_knots: int = 4,
    max_knots: int = 64,
    min_params: int = 2000,
) -> Tuple[int, int]:
    best_k: Optional[int] = None
    best_params: Optional[int] = None
    best_diff: Optional[int] = None

    for k in range(int(min_knots), int(max_knots) + 1):
        p = kan_param_count(input_dim=input_dim, widths=widths, num_knots=k)
        if p < int(min_params):
            continue
        diff = abs(int(p) - int(target_params))
        if best_k is None or diff < best_diff:
            best_k, best_params, best_diff = k, p, diff

    if best_k is None:
        raise ValueError(
            f"Could not find KAN knots in [{min_knots},{max_knots}] yielding >= {min_params} params "
            f"for widths={widths}, input_dim={input_dim}."
        )
    return int(best_k), int(best_params)


# ------------------------------ training ------------------------------

def train_fullbatch(
    model: nn.Module,
    x_train: torch.Tensor,
    y_train: torch.Tensor,
    x_test: torch.Tensor,
    y_test: torch.Tensor,
    *,
    epochs: int,
    lr: float,
    warmup_domain_updates: int = 0,
    is_sn: bool = False,
    print_every: int = 400,
) -> dict:
    mse = nn.MSELoss()
    opt = torch.optim.Adam(model.parameters(), lr=float(lr))

    # Baseline eval
    model.eval()
    with torch.no_grad():
        base_train = rmse(model(x_train), y_train)
        base_test = rmse(model(x_test), y_test)

    t0 = time.time()
    for ep in range(1, int(epochs) + 1):
        model.train()

        if is_sn and ep <= int(warmup_domain_updates):
            # resample-on-update during warmup (then freeze domains)
            model.update_all_domains(allow_resampling=True)

        pred = model(x_train)
        loss = mse(pred, y_train)

        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()

        if is_sn and ep <= int(warmup_domain_updates):
            # tighten bounds without resampling after the step
            model.update_all_domains(allow_resampling=False)

        if (ep % int(print_every) == 0) or (ep == 1) or (ep == int(epochs)):
            model.eval()
            with torch.no_grad():
                tr = rmse(model(x_train), y_train)
                te = rmse(model(x_test), y_test)
            elapsed = time.time() - t0
            print(
                f"[{model.__class__.__name__}] epoch {ep:4d}/{epochs} | loss {loss.item():.3e} | "
                f"train RMSE {tr:.5f} | test RMSE {te:.5f} | {elapsed:.1f}s"
            )

    total_time = time.time() - t0

    model.eval()
    with torch.no_grad():
        final_train = rmse(model(x_train), y_train)
        final_test = rmse(model(x_test), y_test)

    return {
        "base_train_rmse": base_train,
        "base_test_rmse": base_test,
        "final_train_rmse": final_train,
        "final_test_rmse": final_test,
        "wall_time_sec": total_time,
    }


# ------------------------------- main -------------------------------

def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--dtype", type=str, default="float32", choices=["float32", "float64"])

    p.add_argument("--input_dim", type=int, default=10)
    p.add_argument("--width1", type=int, default=16)
    p.add_argument("--width2", type=int, default=15)

    # SN knot controls
    p.add_argument("--phi_knots", type=int, default=512)
    p.add_argument("--Phi_knots", type=int, default=512)

    # KAN knot control (optional). If not provided, auto-match params to SN.
    p.add_argument("--kan_knots", type=int, default=-1)

    # Data & training
    p.add_argument("--train_samples", type=int, default=2048)
    p.add_argument("--test_samples", type=int, default=8192)
    p.add_argument("--epochs", type=int, default=4000)
    p.add_argument("--warmup_epochs", type=int, default=400)  # 10% of 4000
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--print_every", type=int, default=400)

    # KAN fixed domains (no grid updates)
    p.add_argument("--kan_xmin_in", type=float, default=0.0)
    p.add_argument("--kan_xmax_in", type=float, default=1.0)
    p.add_argument("--kan_xmin_hidden", type=float, default=-2.0)
    p.add_argument("--kan_xmax_hidden", type=float, default=2.0)

    args = p.parse_args()

    device = torch.device(args.device)
    dtype = torch.float32 if args.dtype == "float32" else torch.float64

    # ----------------- enforce "barebones" SN settings -----------------
    CONFIG["seed"] = int(args.seed)

    # Turn OFF all engineering extras
    CONFIG["use_residual_weights"] = False
    CONFIG["use_lateral_mixing"] = False
    CONFIG["use_normalization"] = False
    CONFIG["train_phi_codomain"] = False  # keep Φ unconstrained; fewer moving parts

    # We'll do manual warmup domain updates for SN
    CONFIG["use_theoretical_domains"] = True

    set_global_seed(args.seed)

    input_dim = int(args.input_dim)
    widths = [int(args.width1), int(args.width2)]

    # ------------------------------- data ------------------------------
    x_train, y_train, x_test, y_test = make_dataset(
        n_train=args.train_samples,
        n_test=args.test_samples,
        input_dim=input_dim,
        w1=widths[0],
        w2=widths[1],
        seed=args.seed,
        device=device,
        dtype=dtype,
    )

    # ------------------------- build SN (PWL) --------------------------
    sn = SprecherMultiLayerNetwork(
        input_dim=input_dim,
        architecture=widths,
        final_dim=1,
        phi_knots=int(args.phi_knots),
        Phi_knots=int(args.Phi_knots),
        norm_type="none",
        norm_position="after",
        norm_skip_first=True,
        initialize_domains=True,
        domain_ranges=None,
        phi_spline_type="linear",
        Phi_spline_type="linear",
    ).to(device=device, dtype=dtype)

    sn_params = count_trainable_params(sn)
    if sn_params < 2000:
        raise ValueError(
            f"SN has only {sn_params} trainable params (<2000). "
            f"Increase --phi_knots/--Phi_knots to satisfy the benchmark constraint."
        )

    # ------------------- choose KAN knots to match ---------------------
    if int(args.kan_knots) > 0:
        kan_knots = int(args.kan_knots)
        _ = kan_param_count(input_dim=input_dim, widths=widths, num_knots=kan_knots)
    else:
        kan_knots, _ = choose_kan_knots_to_match(
            target_params=sn_params,
            input_dim=input_dim,
            widths=widths,
            min_knots=4,
            max_knots=64,
            min_params=2000,
        )

    kan = BareKAN(
        input_dim=input_dim,
        widths=widths,
        num_knots=kan_knots,
        dtype=dtype,
        x_min_in=float(args.kan_xmin_in),
        x_max_in=float(args.kan_xmax_in),
        x_min_hidden=float(args.kan_xmin_hidden),
        x_max_hidden=float(args.kan_xmax_hidden),
        init_scale=0.05,
    ).to(device=device, dtype=dtype)

    kan_params = count_trainable_params(kan)
    if kan_params < 2000:
        raise ValueError(f"KAN has only {kan_params} trainable params (<2000). Adjust widths/knots.")

    # ------------------------ report setup -----------------------------
    print("\n=== Setup ===")
    print(f"seed={args.seed} device={device} dtype={dtype}")
    print(f"data: train={x_train.shape[0]} test={x_test.shape[0]} input_dim={input_dim}")
    print(f"hidden widths: {widths} (two hidden layers)")

    print("\n=== Parameter counts ===")
    print(f"SN  params: {sn_params}")
    print(f"KAN params: {kan_params} (knots={kan_knots})")
    rel = (kan_params - sn_params) / float(sn_params)
    print(f"KAN/SN param ratio: {kan_params / float(sn_params):.4f}  (relative diff {100.0*rel:+.2f}%)")

    # ----------------------------- train -------------------------------
    print("\n=== Training SN (PWL, warmup domain updates) ===")
    sn_metrics = train_fullbatch(
        sn,
        x_train, y_train,
        x_test, y_test,
        epochs=int(args.epochs),
        lr=float(args.lr),
        warmup_domain_updates=int(args.warmup_epochs),
        is_sn=True,
        print_every=int(args.print_every),
    )

    print("\n=== Training KAN (cubic PCHIP, fixed grid) ===")
    kan_metrics = train_fullbatch(
        kan,
        x_train, y_train,
        x_test, y_test,
        epochs=int(args.epochs),
        lr=float(args.lr),
        warmup_domain_updates=0,
        is_sn=False,
        print_every=int(args.print_every),
    )

    # ----------------------------- summary -----------------------------
    print("\n=== Final results (test RMSE is the key metric) ===")
    print(f"SN  test RMSE: {sn_metrics['final_test_rmse']:.6f}")
    print(f"KAN test RMSE: {kan_metrics['final_test_rmse']:.6f}")

    print("\n(Also reporting train RMSE for completeness)")
    print(f"SN  train RMSE: {sn_metrics['final_train_rmse']:.6f}")
    print(f"KAN train RMSE: {kan_metrics['final_train_rmse']:.6f}")


if __name__ == "__main__":
    main()