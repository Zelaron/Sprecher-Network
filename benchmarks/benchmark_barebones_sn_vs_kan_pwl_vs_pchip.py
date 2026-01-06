# benchmarks/benchmark_barebones_sn_vs_kan_pwl_vs_pchip.py
"""
Barebones SN vs Barebones KAN benchmark (PWL + domain warmup vs cubic PCHIP).

Run (example):

  # single seed
  python -m benchmarks.benchmark_barebones_sn_vs_kan_pwl_vs_pchip --seed 0

  # sweep seeds
  for s in 0 1 2 3; do
    python -m benchmarks.benchmark_barebones_sn_vs_kan_pwl_vs_pchip --seed $s
  done

What this benchmark is designed to show
---------------------------------------
This benchmark intentionally uses a *non-smooth* single-index target function:
    y = triangle_wave(freq * mean(x))

Why this can favor the SN:
- The target is continuous but has many sharp corners (derivative discontinuities).
- The SN here uses *piecewise-linear* (PWL) splines, which can represent corners directly.
- The KAN here uses *cubic PCHIP* splines, which are C1 (continuously differentiable),
  so corners must be approximated by smoothing them out.

To keep the comparison "barebones", we disable SN engineering extras (residuals,
lateral mixing, normalization, grid updates, etc.). The only SN "extra" allowed
is domain updates during the first 10% of training (400 / 4000 epochs). The KAN
gets cubic PCHIP splines and otherwise no extras.

Parameter matching
------------------
The script parameter-matches SN and KAN *by trainable parameter count*.
By default, it chooses KAN knot count to match the SN as closely as possible.
"""

from __future__ import annotations

import argparse
import math
import time
from dataclasses import dataclass
from typing import List, Tuple, Optional

import torch
import torch.nn as nn

from sn_core import SprecherMultiLayerNetwork, CONFIG


# -----------------------------------------------------------------------------
# Utilities
# -----------------------------------------------------------------------------

def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def parse_int_list(csv: str) -> List[int]:
    parts = [p.strip() for p in csv.split(",") if p.strip()]
    if not parts:
        raise ValueError("Empty list.")
    return [int(p) for p in parts]


def count_trainable_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def rmse(pred: torch.Tensor, target: torch.Tensor) -> float:
    return float(torch.sqrt(torch.mean((pred - target) ** 2)).detach().cpu())


# -----------------------------------------------------------------------------
# Target function: triangle wave on a single index (mean of inputs)
# -----------------------------------------------------------------------------

def triangle_wave(t: torch.Tensor) -> torch.Tensor:
    """
    Period-1 triangle wave in [-1, 1] with kinks at half-integers.

    For t >= 0:
        tri(t) = 2*abs(2*frac(t) - 1) - 1
    """
    frac = t - torch.floor(t)
    return 2.0 * torch.abs(2.0 * frac - 1.0) - 1.0


def target_function(x: torch.Tensor, freq: int) -> torch.Tensor:
    """
    x: [N, D] in [0, 1]
    Returns y: [N, 1]
    """
    s = x.mean(dim=1, keepdim=True)  # in [0, 1]
    y = triangle_wave(freq * s)
    # tiny linear term to break perfect symmetry (doesn't change "kinky" nature)
    y = y + 0.05 * (s - 0.5)
    return y


# -----------------------------------------------------------------------------
# Barebones cubic PCHIP "KAN" (vectorized per layer)
# -----------------------------------------------------------------------------

def _pchip_slopes_uniform(y: torch.Tensor, h: float) -> torch.Tensor:
    """
    Vectorized PCHIP slope computation for uniformly spaced knots.

    y: [..., K]
    returns d: same shape as y

    This implements the Fritsch–Carlson / PCHIP slope rules (SciPy-like).
    """
    if y.ndim < 1:
        raise ValueError("y must have at least 1 dimension.")
    K = y.shape[-1]
    d = torch.zeros_like(y)

    if K == 1:
        return d
    if K == 2:
        # With 2 points, it's just a line: slope is the secant.
        delta = (y[..., 1] - y[..., 0]) / h
        d[..., 0] = delta
        d[..., 1] = delta
        return d

    # Secant slopes between knots
    delta = (y[..., 1:] - y[..., :-1]) / h  # [..., K-1]

    # Endpoints (uniform h => simplified)
    d0 = (3.0 * delta[..., 0] - delta[..., 1]) / 2.0
    dN = (3.0 * delta[..., -1] - delta[..., -2]) / 2.0

    # Enforce shape-preserving endpoint rules
    def _fix_endpoint(d_end: torch.Tensor, delta0: torch.Tensor, delta1: torch.Tensor) -> torch.Tensor:
        # If slope has opposite sign to first secant, set to 0
        bad_sign = (d_end * delta0) <= 0
        d_end = torch.where(bad_sign, torch.zeros_like(d_end), d_end)
        # If secants change sign and |d| > 3|delta0|, clamp
        sign_change = (delta0 * delta1) < 0
        too_big = torch.abs(d_end) > 3.0 * torch.abs(delta0)
        d_end = torch.where(sign_change & too_big, 3.0 * delta0, d_end)
        return d_end

    d0 = _fix_endpoint(d0, delta[..., 0], delta[..., 1])
    dN = _fix_endpoint(dN, delta[..., -1], delta[..., -2])

    d[..., 0] = d0
    d[..., -1] = dN

    # Interior slopes: harmonic mean where secants have same sign, else 0
    delta_prev = delta[..., :-1]  # [..., K-2]
    delta_next = delta[..., 1:]   # [..., K-2]
    same_sign = (delta_prev * delta_next) > 0

    eps = 1e-12
    denom = delta_prev + delta_next
    d_mid = torch.where(
        same_sign,
        2.0 * delta_prev * delta_next / (denom + eps),
        torch.zeros_like(denom),
    )
    d[..., 1:-1] = d_mid
    return d


def _hermite_eval_uniform(
    x: torch.Tensor,
    y: torch.Tensor,
    d: torch.Tensor,
    x_min: float,
    x_max: float,
) -> torch.Tensor:
    """
    Evaluate a batch of cubic Hermite splines (PCHIP slopes), for *uniform* knots.

    x: [B]
    y: [O, K]   knot values for O splines
    d: [O, K]   knot slopes for O splines
    returns: [B, O]
    """
    O, K = y.shape
    if K < 2:
        # Degenerate: constant
        return y[:, 0].unsqueeze(0).expand(x.shape[0], O)

    h = (x_max - x_min) / (K - 1)
    # scaled position
    t = (x - x_min) / h  # [B]
    idx = torch.floor(t).to(torch.long)  # [B]
    idx = torch.clamp(idx, 0, K - 2)     # [B]
    u = (t - idx.to(t.dtype))            # [B]

    # Gather y_i, y_{i+1}, d_i, d_{i+1} for all O splines
    y0 = y[:, idx]         # [O, B]
    y1 = y[:, idx + 1]     # [O, B]
    d0 = d[:, idx]         # [O, B]
    d1 = d[:, idx + 1]     # [O, B]

    u = u.unsqueeze(0)     # [1, B]
    u2 = u * u
    u3 = u2 * u

    h00 = 2 * u3 - 3 * u2 + 1
    h10 = u3 - 2 * u2 + u
    h01 = -2 * u3 + 3 * u2
    h11 = u3 - u2

    interp = h00 * y0 + h10 * (h * d0) + h01 * y1 + h11 * (h * d1)  # [O, B]

    # Linear extrapolation outside domain
    below = x < x_min
    above = x > x_max
    if below.any():
        x0 = x_min
        left = y[:, 0].unsqueeze(1) + d[:, 0].unsqueeze(1) * (x - x0).unsqueeze(0)
        interp[:, below] = left[:, below]
    if above.any():
        xN = x_max
        right = y[:, -1].unsqueeze(1) + d[:, -1].unsqueeze(1) * (x - xN).unsqueeze(0)
        interp[:, above] = right[:, above]

    return interp.transpose(0, 1)  # [B, O]


class BareKANLayer(nn.Module):
    """
    A single KAN layer:
        y_o = sum_i spline_{o,i}(x_i) + bias_o
    with cubic PCHIP splines (uniform knots).
    """

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        num_knots: int,
        x_min: float,
        x_max: float,
        init_scale: float = 1e-2,
        dtype: torch.dtype = torch.float32,
    ) -> None:
        super().__init__()
        assert num_knots >= 4, "Use >=4 knots for a meaningful cubic PCHIP spline."
        self.in_dim = int(in_dim)
        self.out_dim = int(out_dim)
        self.num_knots = int(num_knots)
        self.x_min = float(x_min)
        self.x_max = float(x_max)

        coeffs = init_scale * torch.randn(out_dim, in_dim, num_knots, dtype=dtype)
        self.coeffs = nn.Parameter(coeffs)
        self.bias = nn.Parameter(torch.zeros(out_dim, dtype=dtype))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [B, in_dim]
        returns: [B, out_dim]
        """
        B, D = x.shape
        assert D == self.in_dim
        h = (self.x_max - self.x_min) / (self.num_knots - 1)

        # Compute slopes for all splines in this layer: [O, I, K]
        slopes = _pchip_slopes_uniform(self.coeffs, h)

        out = torch.zeros(B, self.out_dim, device=x.device, dtype=x.dtype)

        # Sum over input dimensions (vectorized over output dimension)
        for i in range(self.in_dim):
            # y_i: [O, K], d_i: [O, K], x_i: [B]
            y_i = self.coeffs[:, i, :]
            d_i = slopes[:, i, :]
            out = out + _hermite_eval_uniform(x[:, i], y_i, d_i, self.x_min, self.x_max)

        out = out + self.bias.unsqueeze(0)
        return out


class BareKANNet(nn.Module):
    """
    A 2-hidden-layer KAN (3 layers total).
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dims: Tuple[int, int],
        num_knots: int,
        input_range: Tuple[float, float] = (0.0, 1.0),
        hidden_range: Tuple[float, float] = (-1.0, 1.0),
        init_scale: float = 1e-2,
        dtype: torch.dtype = torch.float32,
    ) -> None:
        super().__init__()
        h1, h2 = hidden_dims
        self.l1 = BareKANLayer(input_dim, h1, num_knots, input_range[0], input_range[1], init_scale, dtype=dtype)
        self.l2 = BareKANLayer(h1, h2, num_knots, hidden_range[0], hidden_range[1], init_scale, dtype=dtype)
        self.l3 = BareKANLayer(h2, 1, num_knots, hidden_range[0], hidden_range[1], init_scale, dtype=dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.l1(x)
        x = self.l2(x)
        x = self.l3(x)
        return x


# -----------------------------------------------------------------------------
# Parameter matching
# -----------------------------------------------------------------------------

@dataclass
class KANConfig:
    hidden_dims: Tuple[int, int]
    num_knots: int
    num_params: int
    rel_diff: float


def _kan_param_count(input_dim: int, hidden_dims: Tuple[int, int], num_knots: int) -> int:
    h1, h2 = hidden_dims
    edges = input_dim * h1 + h1 * h2 + h2 * 1
    biases = h1 + h2 + 1
    return edges * num_knots + biases


def choose_kan_config_to_match_sn(
    sn_params: int,
    input_dim: int,
    preferred_hidden: Tuple[int, int],
    min_width: int,
    max_width: int,
    min_knots: int,
    max_knots: int,
) -> KANConfig:
    """
    Find (h1,h2,num_knots) for KAN with 2 hidden layers that best matches sn_params.
    We search over widths in [min_width,max_width] and knot counts in [min_knots,max_knots].
    """
    best: Optional[KANConfig] = None

    # Helper to update best with tie-breaking toward preferred_hidden
    def consider(h1: int, h2: int, k: int) -> None:
        nonlocal best
        p = _kan_param_count(input_dim, (h1, h2), k)
        rel = abs(p - sn_params) / max(1, sn_params)
        cand = KANConfig(hidden_dims=(h1, h2), num_knots=k, num_params=p, rel_diff=rel)
        if best is None:
            best = cand
            return
        if cand.rel_diff < best.rel_diff - 1e-12:
            best = cand
            return
        # tie-break: closer to preferred widths, then higher knots
        if abs(cand.rel_diff - best.rel_diff) < 1e-12:
            pref_dist = abs(h1 - preferred_hidden[0]) + abs(h2 - preferred_hidden[1])
            best_dist = abs(best.hidden_dims[0] - preferred_hidden[0]) + abs(best.hidden_dims[1] - preferred_hidden[1])
            if pref_dist < best_dist:
                best = cand
            elif pref_dist == best_dist and cand.num_knots > best.num_knots:
                best = cand

    for h1 in range(min_width, max_width + 1):
        for h2 in range(min_width, max_width + 1):
            edges = input_dim * h1 + h1 * h2 + h2
            biases = h1 + h2 + 1
            if edges <= 0:
                continue
            k_float = (sn_params - biases) / edges
            k_candidates = {
                int(math.floor(k_float)),
                int(math.ceil(k_float)),
                min_knots,
                max_knots,
            }
            for k in sorted(k_candidates):
                if k < min_knots or k > max_knots:
                    continue
                consider(h1, h2, k)

    assert best is not None
    return best


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")

    # Data
    p.add_argument("--input_dim", type=int, default=10)
    p.add_argument("--train_n", type=int, default=2048)
    p.add_argument("--test_n", type=int, default=8192)
    p.add_argument("--freq", type=int, default=12)

    # Training
    p.add_argument("--epochs", type=int, default=4000)
    p.add_argument("--warmup_epochs", type=int, default=400)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--print_every", type=int, default=200)

    # Architectures (2 hidden layers)
    p.add_argument("--hidden_dims", type=str, default="9,9", help="Two hidden widths, e.g. '9,9'")

    # SN spline knots
    p.add_argument("--sn_phi_knots", type=int, default=359)
    p.add_argument("--sn_Phi_knots", type=int, default=359)

    # KAN knots (if not provided, we auto-match params)
    p.add_argument("--kan_knots", type=int, default=None)
    p.add_argument("--kan_min_knots", type=int, default=4)
    p.add_argument("--kan_max_knots", type=int, default=64)

    # KAN width search bounds (used only if auto-matching params and kan_knots is None)
    p.add_argument("--kan_min_width", type=int, default=2)
    p.add_argument("--kan_max_width", type=int, default=20)

    # Ranges for KAN splines
    p.add_argument("--kan_hidden_range", type=float, default=1.5, help="Hidden-layer spline domain is [-R, R].")

    args = p.parse_args()

    assert args.epochs == 4000, "Per spec: use 4000 epochs."
    assert args.warmup_epochs == 400, "Per spec: first 400 epochs have SN domain updates (10%)."

    hidden = tuple(parse_int_list(args.hidden_dims))
    assert len(hidden) == 2, "--hidden_dims must have exactly 2 integers, e.g. '9,9'"
    h1, h2 = hidden
    assert 1 <= h1 <= 20 and 1 <= h2 <= 20, "Please keep widths <= 20."

    device = torch.device(args.device)
    set_seed(args.seed)

    # -------------------------------------------------------------------------
    # Disable SN engineering extras (barebones)
    # -------------------------------------------------------------------------
    CONFIG["use_residual_weights"] = False
    CONFIG["use_lateral_mixing"] = False
    CONFIG["use_normalization"] = False
    CONFIG["norm_type"] = "none"
    CONFIG["train_phi_codomain"] = False
    CONFIG["train_Phi_codomain"] = False
    CONFIG["track_domain_violations"] = True   # enables safe resampling during domain updates
    CONFIG["verbose_domain_violations"] = False

    # -------------------------------------------------------------------------
    # Data
    # -------------------------------------------------------------------------
    x_train = torch.rand(args.train_n, args.input_dim, device=device)
    y_train = target_function(x_train, freq=args.freq)

    x_test = torch.rand(args.test_n, args.input_dim, device=device)
    y_test = target_function(x_test, freq=args.freq)

    # -------------------------------------------------------------------------
    # Build SN (PWL) with 2 hidden layers
    # -------------------------------------------------------------------------
    sn = SprecherMultiLayerNetwork(
        input_dim=args.input_dim,
        architecture=[h1, h2],
        final_dim=1,
        phi_knots=args.sn_phi_knots,
        Phi_knots=args.sn_Phi_knots,
        norm_type="none",
        initialize_domains=True,
        phi_spline_type="linear",
        Phi_spline_type="linear",
    ).to(device)

    # Freeze the extra affine output parameters to keep SN truly barebones.
    # (KAN can represent global scaling/shift via its splines anyway.)
    if hasattr(sn, "output_scale"):
        sn.output_scale.requires_grad_(False)
        with torch.no_grad():
            sn.output_scale.fill_(1.0)
    if hasattr(sn, "output_bias"):
        sn.output_bias.requires_grad_(False)
        with torch.no_grad():
            sn.output_bias.fill_(0.0)

    sn_params = count_trainable_params(sn)

    # -------------------------------------------------------------------------
    # Choose/build KAN (cubic PCHIP) to parameter-match the SN
    # -------------------------------------------------------------------------
    if args.kan_knots is not None:
        kan_hidden = (h1, h2)
        kan_knots = int(args.kan_knots)
        if kan_knots < args.kan_min_knots:
            raise ValueError(f"--kan_knots must be >= {args.kan_min_knots}")
        kan_params = _kan_param_count(args.input_dim, kan_hidden, kan_knots)
        rel_diff = abs(kan_params - sn_params) / max(1, sn_params)
        kan_cfg = KANConfig(hidden_dims=kan_hidden, num_knots=kan_knots, num_params=kan_params, rel_diff=rel_diff)
    else:
        # Auto-match: search widths+knots to best match sn_params.
        kan_cfg = choose_kan_config_to_match_sn(
            sn_params=sn_params,
            input_dim=args.input_dim,
            preferred_hidden=(h1, h2),
            min_width=args.kan_min_width,
            max_width=args.kan_max_width,
            min_knots=args.kan_min_knots,
            max_knots=args.kan_max_knots,
        )

    kan = BareKANNet(
        input_dim=args.input_dim,
        hidden_dims=kan_cfg.hidden_dims,
        num_knots=kan_cfg.num_knots,
        input_range=(0.0, 1.0),
        hidden_range=(-args.kan_hidden_range, args.kan_hidden_range),
        init_scale=1e-2,
        dtype=torch.float32,
    ).to(device)

    kan_params = count_trainable_params(kan)

    # -------------------------------------------------------------------------
    # Report setup
    # -------------------------------------------------------------------------
    print("=" * 80)
    print("Barebones SN (PWL + domain warmup) vs Barebones KAN (cubic PCHIP)")
    print(f"seed={args.seed} device={device.type} D={args.input_dim} freq={args.freq}")
    print(f"train_n={args.train_n} test_n={args.test_n} epochs={args.epochs} warmup={args.warmup_epochs}")
    print("-" * 80)
    print(f"SN hidden=[{h1},{h2}]  phi_knots={args.sn_phi_knots} Phi_knots={args.sn_Phi_knots}  params={sn_params:,}")
    print(f"KAN hidden={list(kan_cfg.hidden_dims)}  knots={kan_cfg.num_knots}  params={kan_params:,}  rel_diff={kan_cfg.rel_diff*100:.2f}%")
    if (h1, h2) != kan_cfg.hidden_dims:
        print("NOTE: KAN hidden dims were adjusted for parameter matching.")
    print("=" * 80)

    # -------------------------------------------------------------------------
    # Optimizers (same hyperparams, barebones)
    # -------------------------------------------------------------------------
    opt_sn = torch.optim.Adam([p for p in sn.parameters() if p.requires_grad], lr=args.lr)
    opt_kan = torch.optim.Adam(kan.parameters(), lr=args.lr)

    mse = nn.MSELoss()

    # -------------------------------------------------------------------------
    # Training loop
    # -------------------------------------------------------------------------
    sn_train_time = 0.0
    kan_train_time = 0.0

    for epoch in range(args.epochs):
        sn.train()
        kan.train()

        # SN domain updates only during warmup
        if epoch < args.warmup_epochs:
            sn.update_all_domains(allow_resampling=True)

        # SN step
        sn_start = time.perf_counter()
        opt_sn.zero_grad(set_to_none=True)
        pred_sn = sn(x_train)
        loss_sn = mse(pred_sn, y_train)
        loss_sn.backward()
        opt_sn.step()
        sn_train_time += time.perf_counter() - sn_start

        # KAN step
        kan_start = time.perf_counter()
        opt_kan.zero_grad(set_to_none=True)
        pred_kan = kan(x_train)
        loss_kan = mse(pred_kan, y_train)
        loss_kan.backward()
        opt_kan.step()
        kan_train_time += time.perf_counter() - kan_start

        if (epoch + 1) % args.print_every == 0 or epoch == 0:
            with torch.no_grad():
                sn_rmse = rmse(pred_sn, y_train)
                kan_rmse = rmse(pred_kan, y_train)
            print(f"epoch {epoch+1:4d}/{args.epochs} | train RMSE: SN={sn_rmse:.6f}  KAN={kan_rmse:.6f}")

    # -------------------------------------------------------------------------
    # Final evaluation (TEST RMSE is the main score)
    # -------------------------------------------------------------------------
    sn.eval()
    kan.eval()
    with torch.no_grad():
        pred_sn_test = sn(x_test)
        pred_kan_test = kan(x_test)
        sn_test_rmse = rmse(pred_sn_test, y_test)
        kan_test_rmse = rmse(pred_kan_test, y_test)

    print("-" * 80)
    print(f"TEST RMSE (lower is better):  SN={sn_test_rmse:.8f}   KAN={kan_test_rmse:.8f}")
    print(f"Training time:  SN={sn_train_time:.2f}s   KAN={kan_train_time:.2f}s")
    print("-" * 80)


if __name__ == "__main__":
    main()
