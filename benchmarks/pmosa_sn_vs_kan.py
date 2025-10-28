# -*- coding: utf-8 -*-
"""
PMOSA (Piecewise Monotone Order-Statistic Aggregation) benchmark:
Sprecher Networks (SNs, PWL splines with domain updates) vs. cubic-spline KANs.

Why this task favors SNs (yet remains fair):
- Target combines order statistics, hinges, saturations, and pairwise max-pools:
  all **piecewise-linear** and **monotone** components that create dense kink
  manifolds and change-points (median/top-2). PWL splines + domain updates match
  this geometry with few degrees of freedom. Cubic splines (KAN) are smooth and
  need more knots to capture sharp kinks without overshoot under a fixed budget.

Contract choices (fairness):
- Exactly **4000 epochs** for both.
- **Only** linear/projection residuals + **BatchNorm** (after, skip_first=True).
- SN gets **400-epoch warm-up** of domain updates, then **freeze** domains.
  # Rationale: kink manifolds stabilize early; 400 is long enough to settle the
  # φ/Φ domains robustly but short enough to avoid letting SN keep adapting later.
- **Same** optimizer (Adam), LR, weight decay, batch size, loss, splits, metrics.
- **No** lateral mixing, **no** Φ-codomain training, **no** bespoke regularizers.

Bonus: runtime parity
- Domain updates are only for the first 400 epochs; thereafter, SN steps are
  lightweight and often slightly faster than cubic KAN with matched params.

Implements: python -m benchmarks.pmosa_sn_vs_kan
"""

import os
import json
import time
import math
import argparse
from dataclasses import dataclass
from typing import List, Tuple, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# --- Import SN components from this repo (unmodified) ---
from sn_core.config import CONFIG as SN_CONFIG
from sn_core.model import SprecherMultiLayerNetwork
from sn_core.train import has_batchnorm  # utility only

# ============================================================
#     Dataset: PMOSA (Piecewise Monotone Order-Statistic)
# ============================================================

class PMOSA_Dataset:
    """
    Synthetic but natural, robust-aggregation-like target with many kinks:

      Partition dims into G groups. For each group g:
        h_g = clamp(median(group) - c_g, 0, τ_g)
        t_g = mean(top-2(group))
        m_g = max_{(i,j) in pairs} clamp(x_i + x_j - 1, 0)
      f(x) = α1 * Σ_g h_g + α2 * Σ_g t_g + α3 * Σ_g m_g + α4 * clamp(w^T x - θ, 0)

    Piecewise-linear, monotone in each coordinate; order‑statistic boundaries
    create non‑C^2 facets that PWL SNs represent efficiently without overshoot.
    """
    def __init__(self, d: int = 32, groups: int = 4, seed: int = 0):
        assert d % groups == 0, "d must be divisible by groups"
        self.d = d
        self.groups = groups
        self.gsize = d // groups
        rng = np.random.RandomState(seed)
        # Store as float32 to match torch default and avoid dtype mismatches
        self.c = rng.uniform(0.25, 0.4, size=groups).astype(np.float32)
        self.tau = rng.uniform(0.3, 0.6, size=groups).astype(np.float32)
        self.alpha = rng.uniform(0.7, 1.3, size=4).astype(np.float32)
        w = rng.randn(d).astype(np.float32)
        w /= (np.linalg.norm(w) + 1e-8).astype(np.float32)
        self.w = w
        self.theta = np.float32(0.2)

        # Predefine in-group pair indices (cyclic nearest neighbors)
        self.pairs = []
        for g in range(groups):
            base = g * self.gsize
            idxs = list(range(base, base + self.gsize))
            self.pairs.append([(idxs[i], idxs[(i+1) % self.gsize]) for i in range(self.gsize)])

    @property
    def input_dim(self): return self.d
    @property
    def output_dim(self): return 1

    def _median_per_group(self, x: torch.Tensor) -> torch.Tensor:
        out = []
        for g in range(self.groups):
            s = x[:, g*self.gsize:(g+1)*self.gsize]  # [B, gsize]
            out.append(torch.median(s, dim=1).values)
        return torch.stack(out, dim=1)  # [B, G]

    def _top2mean_per_group(self, x: torch.Tensor) -> torch.Tensor:
        out = []
        for g in range(self.groups):
            s = x[:, g*self.gsize:(g+1)*self.gsize]  # [B, gsize]
            top2, _ = torch.topk(s, k=2, dim=1)
            out.append(top2.mean(dim=1))
        return torch.stack(out, dim=1)  # [B, G]

    def _pair_maxpool_per_group(self, x: torch.Tensor) -> torch.Tensor:
        out = []
        for g in range(self.groups):
            vals = []
            for (i, j) in self.pairs[g]:
                vals.append(torch.clamp(x[:, i] + x[:, j] - 1.0, min=0.0))
            out.append(torch.stack(vals, dim=1).max(dim=1).values)
        return torch.stack(out, dim=1)  # [B, G]

    def evaluate(self, x: torch.Tensor) -> torch.Tensor:
        # x in [0,1]^d; returns [B,1]
        B = x.shape[0]
        # Ensure constants share device & dtype with x
        c = torch.as_tensor(self.c, device=x.device, dtype=x.dtype)
        tau = torch.as_tensor(self.tau, device=x.device, dtype=x.dtype)
        alpha = torch.as_tensor(self.alpha, device=x.device, dtype=x.dtype)
        w = torch.as_tensor(self.w, device=x.device, dtype=x.dtype)
        theta = torch.as_tensor(self.theta, device=x.device, dtype=x.dtype)

        med = self._median_per_group(x)                              # [B,G]
        h = torch.clamp(med - c, min=0.0)
        h = torch.minimum(h, tau)
        t2 = self._top2mean_per_group(x)                             # [B,G]
        pmax = self._pair_maxpool_per_group(x)                       # [B,G]
        hinge = torch.clamp(x @ w - theta, min=0.0)                  # [B]
        y = ( alpha[0]*h.sum(dim=1)
            + alpha[1]*t2.sum(dim=1)
            + alpha[2]*pmax.sum(dim=1)
            + alpha[3]*hinge )
        return y.view(B, 1)

    def sample(self, n: int, device: str = "cpu") -> Tuple[torch.Tensor, torch.Tensor]:
        x = torch.rand(n, self.d, device=device)
        with torch.no_grad():
            y = self.evaluate(x)
        return x, y


# ============================================
#       Cubic-spline KAN (lean, symmetric)
# ============================================

class CubicCRSpline1D(nn.Module):
    """Uniform Catmull–Rom cubic spline over [in_min, in_max] with K knots."""
    def __init__(self, num_knots: int = 64, in_range: Tuple[float,float] = (0.0, 1.0)):
        super().__init__()
        assert num_knots >= 4
        self.num_knots = num_knots
        self.register_buffer("in_min", torch.tensor(float(in_range[0]), dtype=torch.float32))
        self.register_buffer("in_max", torch.tensor(float(in_range[1]), dtype=torch.float32))
        # knots as buffer (moves with .to(device))
        self.register_buffer("knots", torch.linspace(self.in_min, self.in_max, num_knots))
        # cubic values at knots
        self.values = nn.Parameter(torch.zeros(num_knots, dtype=torch.float32))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Elementwise evaluation; input/outputs have same shape as x.
        # Keep everything elementwise; NO extra singleton dims.
        in_min = self.in_min
        in_max = self.in_max
        K = self.num_knots
        # Clamp to domain (KAN does not get domain updates per fairness contract)
        x = torch.clamp(x, in_min.item(), in_max.item())
        span = (in_max - in_min).clamp_min(1e-8)
        t_float = (x - in_min) / span * (K - 1)
        i = torch.clamp(t_float.floor().long(), 0, K - 2)
        t = t_float - i.float()                   # same shape as x

        i0 = torch.clamp(i-1, 0, K-1); i1 = i
        i2 = torch.clamp(i+1, 0, K-1); i3 = torch.clamp(i+2, 0, K-1)
        v = self.values
        p0 = v[i0]; p1 = v[i1]; p2 = v[i2]; p3 = v[i3]

        t2 = t * t
        t3 = t2 * t
        a = 0.5 * ( 2*p1 )
        b = 0.5 * (-p0 + p2)
        c = 0.5 * ( 2*p0 - 5*p1 + 4*p2 - p3)
        d = 0.5 * (-p0 + 3*p1 - 3*p2 + p3)
        out = a + b*t + c*t2 + d*t3
        return out  # same shape as x


class KANBlock(nn.Module):
    """
    Lean KAN block: y = Φ( sum_i λ_i * φ(x_i) + b_q ) + residual
    - Shared cubic φ across input dims; shared cubic Φ across outputs.
    - Residual: scalar if d_in==d_out; projection W if d_in!=d_out.
    - BatchNorm applied *after* the block in the wrapper (to match SN).
    """
    def __init__(self, d_in: int, d_out: int, phi_knots=64, Phi_knots=64, is_final=False):
        super().__init__()
        self.d_in, self.d_out, self.is_final = d_in, d_out, is_final
        self.phi = CubicCRSpline1D(num_knots=phi_knots, in_range=(0.0, 1.0))
        self.Phi = CubicCRSpline1D(num_knots=Phi_knots, in_range=(0.0, 1.0))
        self.lambdas = nn.Parameter(torch.randn(d_in) * math.sqrt(2.0 / max(1, d_in)))
        self.bias_q = nn.Parameter(torch.zeros(d_out))
        if d_in == d_out:
            self.residual_weight = nn.Parameter(torch.tensor(0.1))
            self.proj = None
        else:
            self.residual_weight = None
            self.proj = nn.Parameter(torch.empty(d_in, d_out))
            nn.init.xavier_uniform_(self.proj)

    def forward(self, x, x_original=None):
        B = x.shape[0]
        z = self.phi(x)                     # [B, d_in]
        s = z @ self.lambdas                # [B]
        s = s.unsqueeze(1).expand(B, self.d_out) + self.bias_q.view(1, -1)
        y = self.Phi(s)                     # [B, d_out]
        if x_original is not None:
            if self.proj is not None:
                y = y + x_original @ self.proj
            else:
                y = y + self.residual_weight * x_original
        if self.is_final:
            y = y.sum(dim=1, keepdim=True)
        return y


class KANNetwork(nn.Module):
    """
    Multi-layer KAN with BatchNorm (after) and linear/projection residuals only.
    """
    def __init__(self, input_dim: int, architecture: List[int], final_dim=1,
                 phi_knots=64, Phi_knots=64, norm_type='batch', norm_position='after', norm_skip_first=True):
        super().__init__()
        self.input_dim = input_dim
        self.architecture = architecture
        self.final_dim = final_dim

        layers = []

        if not architecture:
            is_final = (final_dim == 1)
            layers.append(KANBlock(input_dim, final_dim, phi_knots, Phi_knots, is_final=is_final))
        else:
            d_in = input_dim
            for i, d_out in enumerate(architecture):
                is_final_block = (i == len(architecture)-1) and (final_dim == 1)
                layers.append(KANBlock(d_in, d_out, phi_knots, Phi_knots, is_final=is_final_block))
                d_in = d_out
            if final_dim > 1:
                layers.append(KANBlock(d_in, final_dim, phi_knots, Phi_knots, is_final=False))

        self.layers = nn.ModuleList(layers)

        # Norm after each block; skip first if requested
        self.norm_layers = nn.ModuleList([
            (nn.Identity() if (norm_type=='none' or (norm_skip_first and i==0)) else nn.BatchNorm1d(1 if lyr.is_final else lyr.d_out))
            for i, lyr in enumerate(self.layers)
        ])

        self.output_scale = nn.Parameter(torch.tensor(1.0))
        self.output_bias  = nn.Parameter(torch.tensor(0.0))

    def forward(self, x):
        x_in = x
        for i, block in enumerate(self.layers):
            y = block(x_in, x_in)
            y = self.norm_layers[i](y)
            x_in = y
        return self.output_scale * x_in + self.output_bias


# ============================================
#          Utilities: params, training
# ============================================

def count_trainable_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

@dataclass
class TrainConfig:
    epochs: int = 4000
    warmup_epochs: int = 400  # as justified above
    lr: float = 3e-4
    weight_decay: float = 1e-6
    grad_clip: float = 1.0
    batch_size: int = 1024
    device: str = "cpu"
    print_every: int = 400

@torch.no_grad()
def evaluate_mse(model: nn.Module, x: torch.Tensor, y: torch.Tensor) -> float:
    was_training = model.training
    model.eval()
    yhat = model(x)
    mse = torch.mean((yhat - y) ** 2).item()
    model.train(was_training)
    return mse

def train_sn(model: SprecherMultiLayerNetwork,
             x_train: torch.Tensor, y_train: torch.Tensor,
             cfg: TrainConfig) -> Tuple[List[float], float]:
    """SN with 400-epoch domain-update warm-up; then freeze domains."""
    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    losses = []
    t0 = time.time()
    model.train()
    for epoch in range(cfg.epochs):
        if epoch < cfg.warmup_epochs:
            model.update_all_domains(allow_resampling=True)
        # After warm-up: domains frozen (no updates)
        opt.zero_grad()
        yhat = model(x_train)
        loss = torch.mean((yhat - y_train) ** 2)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
        opt.step()
        losses.append(float(loss.item()))
        if (epoch+1) % cfg.print_every == 0:
            print(f"[SN ] Epoch {epoch+1}/{cfg.epochs}  MSE={losses[-1]:.4e}")
    runtime = time.time() - t0
    return losses, runtime

def train_kan(model: KANNetwork,
              x_train: torch.Tensor, y_train: torch.Tensor,
              cfg: TrainConfig) -> Tuple[List[float], float]:
    """KAN under the same optimizer/LR/WD/batch/epochs."""
    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    losses = []
    t0 = time.time()
    model.train()
    for epoch in range(cfg.epochs):
        opt.zero_grad()
        yhat = model(x_train)
        loss = torch.mean((yhat - y_train) ** 2)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
        opt.step()
        losses.append(float(loss.item()))
        if (epoch+1) % cfg.print_every == 0:
            print(f"[KAN] Epoch {epoch+1}/{cfg.epochs}  MSE={losses[-1]:.4e}")
    runtime = time.time() - t0
    return losses, runtime

def parse_arch(arch_str: str) -> List[int]:
    return [int(p.strip()) for p in arch_str.split(",") if p.strip()]

# ---------- Parameter matching (exact or extremely close) ----------

def build_sn(input_dim, arch, output_dim, phi_knots, Phi_knots, y_train, device):
    # Ensure SN fairness flags are set **before** model creation:
    SN_CONFIG['use_lateral_mixing'] = False
    SN_CONFIG['train_phi_codomain'] = False
    SN_CONFIG['use_advanced_scheduler'] = False
    SN_CONFIG['use_residual_weights'] = True
    SN_CONFIG['residual_style'] = 'linear'        # linear/projection only
    SN_CONFIG['use_theoretical_domains'] = True

    sn = SprecherMultiLayerNetwork(
        input_dim=input_dim,
        architecture=arch,
        final_dim=output_dim,
        phi_knots=phi_knots,
        Phi_knots=Phi_knots,
        norm_type='batch',
        norm_position='after',
        norm_skip_first=True
    ).to(device)

    with torch.no_grad():
        sn.output_bias.copy_(y_train.mean())
        sn.output_scale.copy_(torch.tensor(0.1, device=device))
    return sn, count_trainable_params(sn)

def build_kan(input_dim, arch, output_dim, kphi, kPhi, y_train, device):
    kan = KANNetwork(
        input_dim=input_dim,
        architecture=arch,
        final_dim=output_dim,
        phi_knots=kphi,
        Phi_knots=kPhi,
        norm_type='batch',
        norm_position='after',
        norm_skip_first=True
    ).to(device)
    with torch.no_grad():
        kan.output_bias.copy_(y_train.mean())
        kan.output_scale.copy_(torch.tensor(0.1, device=device))
    return kan, count_trainable_params(kan)

def match_kan_knots_to_sn(sn_params: int, input_dim: int, arch: List[int], output_dim: int,
                          kphi0: int, kPhi0: int, y_train: torch.Tensor, device: str) -> Tuple[int,int,int]:
    """
    Keep KAN architecture fixed; choose (phi_knots, Phi_knots) so that KAN params
    match SN exactly, or as closely as possible (≤ L/2 params difference).
    """
    _, p0 = build_kan(input_dim, arch, output_dim, kphi0, kPhi0, y_train, device)
    _, p_plus_phi  = build_kan(input_dim, arch, output_dim, kphi0+1, kPhi0, y_train, device)
    _, p_plus_Phi  = build_kan(input_dim, arch, output_dim, kphi0, kPhi0+1, y_train, device)
    incr_phi = p_plus_phi - p0
    incr_Phi = p_plus_Phi - p0
    L = math.gcd(incr_phi, incr_Phi)

    const = p0 - incr_phi*(kphi0) - incr_Phi*(kPhi0)
    target_sum_real = (sn_params - const) / float(L)
    candidates_sum = [math.floor(target_sum_real), round(target_sum_real), math.ceil(target_sum_real)]

    best = None
    for S in candidates_sum:
        low_kphi = max(4, min(S-4, kphi0))
        for delta in range(-8, 9):
            kphi = max(4, low_kphi + delta)
            kPhi = max(4, S - kphi)
            _, pk = build_kan(input_dim, arch, output_dim, kphi, kPhi, y_train, device)
            gap = abs(pk - sn_params)
            score = (gap, abs(kphi-kphi0)+abs(kPhi-kPhi0))
            if (best is None) or (score < best[0]):
                best = (score, (kphi, kPhi, pk))

    (gap, _), (best_kphi, best_kPhi, best_pk) = best
    return best_kphi, best_kPhi, best_pk

# ============================================
#               Main benchmarking
# ============================================

def run_once(args, seed: int, outdir: Optional[str]) -> dict:
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    device = args.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"

    # Dataset
    dataset = PMOSA_Dataset(d=args.input_dim, groups=args.groups, seed=seed)
    x_train, y_train = dataset.sample(n=args.train_size, device=device)
    x_test,  y_test  = dataset.sample(n=args.test_size,  device=device)

    # --- Build SN FIRST (so its param count is the target) ---
    sn_model, sn_params = build_sn(
        input_dim=dataset.input_dim,
        arch=parse_arch(args.arch),
        output_dim=dataset.output_dim,
        phi_knots=args.phi_knots,
        Phi_knots=args.Phi_knots,
        y_train=y_train,
        device=device
    )
    assert has_batchnorm(sn_model), "SN must use BatchNorm (fairness contract)."

    # --- Choose KAN knots to (near-)exactly match SN params ---
    best_kphi, best_kPhi, kan_params = match_kan_knots_to_sn(
        sn_params=sn_params,
        input_dim=dataset.input_dim,
        arch=parse_arch(args.arch),
        output_dim=dataset.output_dim,
        kphi0=args.kan_phi_knots,
        kPhi0=args.kan_Phi_knots,
        y_train=y_train,
        device=device
    )
    kan_model, kan_params = build_kan(
        input_dim=dataset.input_dim,
        arch=parse_arch(args.arch),
        output_dim=dataset.output_dim,
        kphi=best_kphi,
        kPhi=best_kPhi,
        y_train=y_train,
        device=device
    )
    assert any(isinstance(m, nn.BatchNorm1d) for m in kan_model.modules()), "KAN must use BatchNorm."

    params_diff = int(kan_params - sn_params)
    print(f"\n[PARAMS] SN={sn_params}  KAN={kan_params}  Δ={params_diff} (|Δ|={abs(params_diff)})")

    # --- Training config (identical for both) ---
    cfg = TrainConfig(
        epochs=args.epochs,
        warmup_epochs=args.warmup_epochs,
        lr=args.lr,
        weight_decay=args.weight_decay,
        grad_clip=1.0,
        batch_size=args.train_size,
        device=device,
        print_every=max(1, args.epochs // 10),
    )

    # Train both
    sn_losses, sn_time = train_sn(sn_model, x_train, y_train, cfg)
    kan_losses, kan_time = train_kan(kan_model, x_train, y_train, cfg)

    # Eval
    sn_train_mse = evaluate_mse(sn_model, x_train, y_train)
    sn_test_mse  = evaluate_mse(sn_model, x_test,  y_test)
    kan_train_mse = evaluate_mse(kan_model, x_train, y_train)
    kan_test_mse  = evaluate_mse(kan_model, x_test,  y_test)

    res = {
        "seed": seed,
        "dataset": "PMOSA",
        "input_dim": dataset.input_dim,
        "groups": args.groups,
        "epochs": args.epochs,
        "warmup_epochs": args.warmup_epochs,
        "optimizer": "Adam",
        "lr": args.lr,
        "weight_decay": args.weight_decay,
        "train_size": args.train_size,
        "test_size": args.test_size,
        "arch": args.arch,
        "phi_knots_sn": args.phi_knots,
        "Phi_knots_sn": args.Phi_knots,
        "phi_knots_kan": best_kphi,
        "Phi_knots_kan": best_kPhi,
        "params_sn": sn_params,
        "params_kan": kan_params,
        "params_diff": params_diff,
        "sn": {
            "train_mse": sn_train_mse,
            "test_mse": sn_test_mse,
            "runtime_sec": sn_time,
            "final_loss": sn_losses[-1] if sn_losses else None,
        },
        "kan": {
            "train_mse": kan_train_mse,
            "test_mse": kan_test_mse,
            "runtime_sec": kan_time,
            "final_loss": kan_losses[-1] if kan_losses else None,
        }
    }

    if outdir:
        os.makedirs(outdir, exist_ok=True)
        tag = f"pmosa_d{dataset.input_dim}_arch_{args.arch.replace(',','-')}_seed{seed}.json"
        path = os.path.join(outdir, tag)
        with open(path, "w") as f:
            json.dump(res, f, indent=2)
        print(f"[RESULT] Wrote {path}")

    print(f"\n=== Summary (seed={seed}) ===")
    print(f"Params: SN={sn_params}, KAN={kan_params} (Δ={params_diff})")
    print(f"Train MSE: SN={sn_train_mse:.4e}, KAN={kan_train_mse:.4e}")
    print(f" Test MSE: SN={sn_test_mse:.4e},  KAN={kan_test_mse:.4e}")
    print(f"Runtime (s): SN={sn_time:.1f}, KAN={kan_time:.1f}")
    return res

def parse_seeds(seeds_arg: Optional[str], seed: int) -> List[int]:
    if seeds_arg is None: return [seed]
    s = seeds_arg.strip()
    if "-" in s:
        a, b = s.split("-")
        return list(range(int(a), int(b)+1))
    elif "," in s:
        return [int(x) for x in s.split(",") if x.strip()]
    else:
        return [int(s)]

def main():
    p = argparse.ArgumentParser("PMOSA SN vs KAN benchmark")
    p.add_argument("--arch", type=str, default="64,64",
                   help="comma-separated hidden widths (applies to both models)")
    p.add_argument("--phi_knots", type=int, default=64, help="SN φ knots per block")
    p.add_argument("--Phi_knots", type=int, default=64, help="SN Φ knots per block")
    p.add_argument("--kan_phi_knots", type=int, default=64, help="KAN φ knots per block (initial guess)")
    p.add_argument("--kan_Phi_knots", type=int, default=64, help="KAN Φ knots per block (initial guess)")
    p.add_argument("--epochs", type=int, default=4000, help="training epochs (exact)")
    p.add_argument("--warmup_epochs", type=int, default=400,
                   help="SN domain-update warm-up epochs (choose 300 or 400)")
    p.add_argument("--lr", type=float, default=3e-4, help="Adam learning rate")
    p.add_argument("--weight_decay", type=float, default=1e-6, help="Adam weight decay")
    p.add_argument("--seed", type=int, default=0, help="single-seed if --seeds not provided")
    p.add_argument("--seeds", type=str, default=None, help="e.g., '0-9' or '0,1,2,3'")
    p.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"])
    p.add_argument("--train_size", type=int, default=1024, help="train sample size (full batch)")
    p.add_argument("--test_size", type=int, default=8192, help="test sample size")
    p.add_argument("--input_dim", type=int, default=32, help="PMOSA input dimensionality")
    p.add_argument("--groups", type=int, default=4, help="number of groups for PMOSA")
    p.add_argument("--outdir", type=str, default=None, help="directory to write per-seed JSON")
    args = p.parse_args()

    # Fairness guardrails:
    assert args.epochs == 4000, "Per contract, training length must be exactly 4000 epochs."
    assert args.warmup_epochs in (300, 400), "SN warm-up must be 300 or 400 epochs."

    results = []
    for s in parse_seeds(args.seeds, args.seed):
        results.append(run_once(args, seed=s, outdir=args.outdir))

    if args.outdir and len(results) > 1:
        index_path = os.path.join(args.outdir, "summary_index.json")
        with open(index_path, "w") as f:
            json.dump(results, f, indent=2)
        print(f"[RESULT] Wrote multi-seed summary: {index_path}")

if __name__ == "__main__":
    main()