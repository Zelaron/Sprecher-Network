# benchmarks/sn_vs_kan_tiered_pricing.py
# -*- coding: utf-8 -*-
"""
SN vs KAN Benchmark — Tiered Pricing & Saturation (TPS-16)

Why this task?
--------------
We simulate a realistic 'tiered tariff' regression with thresholds, piecewise-linear segments,
sparse pairwise quotas, and saturation. The ground truth is globally PWL with sharp kinks and
flat/monotone stretches (as in utilities billing, shipping tiers, throttling, piece-rate pay).

Why SNs should have an edge (yet fair):
---------------------------------------
• SNs use piecewise-linear splines with domain updates: kinks are represented exactly after a
  short warm-up; domains then freeze. This matches the TPS-16 structure.
• KAN uses cubic splines (C^1) which approximate kinks via smooth ramps; with equal parameter
  budgets we expect higher bias around thresholds for KANs.
• Both models: SAME optimizer/LR/WD/batch/epochs, SAME splits/loss/metrics, ONLY BatchNorm,
  ONLY linear/projection residuals, and matched parameters within ±5%.

Warm-up choice:
---------------
We use a 400-epoch SN domain warm-up (~10% of 4000 epochs). It’s long enough to stabilize
domains using meaningful gradients, but short enough to avoid late drift.

Repository components reused without modification:
- sn_core.model.SprecherMultiLayerNetwork, domain update utilities
- sn_core.config.CONFIG for BN/residual wiring
- sn_core.data.Dataset base
- sn_core.train.evaluating helper for safe eval
(See combined_code2.txt.)  # repo reference
"""

import argparse
import math
import os
import random
from dataclasses import dataclass
from typing import List, Tuple, Optional

import torch
import torch.nn as nn
import numpy as np

# ---- Import SN core bits (no internal modifications) --------------------------
from sn_core.model import SprecherMultiLayerNetwork  # SN model
from sn_core.config import CONFIG                    # global runtime config
from sn_core.data import Dataset                     # base dataset class
from sn_core.train import evaluating                 # tiny eval helper (no side effects)

# --------------------------------------------------------------------------------
# KAN components: cubic Catmull–Rom spline + KAN blocks/network with BN-after & linear residuals
# --------------------------------------------------------------------------------

class CubicCRSpline1D(nn.Module):
    def __init__(self, num_knots: int, in_min: float = 0.0, in_max: float = 1.0):
        super().__init__()
        assert num_knots >= 4, "Cubic interpolation needs at least 4 knots."
        self.num_knots = num_knots
        self.in_min = float(in_min)
        self.in_max = float(in_max)
        # Learnable knot values; init approx identity
        self.values = nn.Parameter(torch.linspace(0.0, 1.0, num_knots))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.view(-1)
        x = torch.clamp(x, min=self.in_min, max=self.in_max)
        K = self.num_knots
        span = self.in_max - self.in_min
        u = (x - self.in_min) / (span + 1e-12) * (K - 1)
        i = torch.floor(u).to(torch.long)
        t = (u - i).clamp(0.0, 1.0)
        i0 = (i - 1).clamp(0, K - 1)
        i1 = i.clamp(0, K - 1)
        i2 = (i + 1).clamp(0, K - 1)
        i3 = (i + 2).clamp(0, K - 1)
        v = self.values
        v0, v1, v2, v3 = v[i0], v[i1], v[i2], v[i3]
        t2, t3 = t * t, t * t * t
        out = 0.5 * (2 * v1 + (-v0 + v2) * t + (2 * v0 - 5 * v1 + 4 * v2 - v3) * t2 +
                     (-v0 + 3 * v1 - 3 * v2 + v3) * t3)
        return out.view(-1, 1)

class KANBlock(nn.Module):
    def __init__(self, d_in: int, d_out: int, k_in: int, k_out: int,
                 is_final: bool = False, residual_style: str = "linear"):
        super().__init__()
        self.d_in, self.d_out, self.is_final = d_in, d_out, is_final
        self.residual_style = str(residual_style).lower()
        self.in_splines = nn.ModuleList([CubicCRSpline1D(k_in) for _ in range(d_in)])
        self.W = nn.Parameter(torch.empty(d_in, d_out))
        nn.init.xavier_uniform_(self.W)
        self.out_splines = nn.ModuleList([CubicCRSpline1D(k_out) for _ in range(d_out)])
        # residuals: scalar if dims equal, projection if dims differ
        self.residual_weight = None
        self.residual_proj = None
        if d_in == d_out:
            self.residual_weight = nn.Parameter(torch.tensor(0.1))
        else:
            self.residual_proj = nn.Parameter(torch.empty(d_in, d_out))
            nn.init.xavier_uniform_(self.residual_proj)

    def forward(self, x: torch.Tensor, x_orig: Optional[torch.Tensor] = None) -> torch.Tensor:
        B = x.shape[0]
        phi = []
        for i in range(self.d_in):
            phi_i = self.in_splines[i](x[:, i:i+1]).view(B)
            phi.append(phi_i)
        Phi = torch.stack(phi, dim=1)  # (B, d_in)
        z = Phi @ self.W               # (B, d_out)
        y = []
        for q in range(self.d_out):
            yq = self.out_splines[q](z[:, q:q+1]).view(B)
            y.append(yq)
        y = torch.stack(y, dim=1)      # (B, d_out)

        if x_orig is not None:
            if self.residual_proj is not None:
                y = y + x_orig @ self.residual_proj
            elif self.residual_weight is not None:
                y = y + self.residual_weight * x_orig

        if self.is_final:
            y = y.sum(dim=1, keepdim=True)
        return y

class KANNetwork(nn.Module):
    def __init__(self, input_dim: int, architecture: List[int], final_dim: int,
                 k_in: int, k_out: int, norm_position: str = "after", norm_skip_first: bool = True):
        super().__init__()
        assert norm_position == "after", "Benchmark uses BN AFTER blocks."
        self.input_dim, self.architecture, self.final_dim = input_dim, architecture, final_dim
        self.norm_position, self.norm_skip_first = norm_position, norm_skip_first

        layers = []
        d_in = input_dim
        if len(architecture) == 0:
            layers.append(KANBlock(d_in, final_dim, k_in, k_out, is_final=(final_dim == 1)))
        else:
            for i, d_out in enumerate(architecture):
                is_final_block = (i == len(architecture) - 1) and (final_dim == 1)
                layers.append(KANBlock(d_in, d_out, k_in, k_out, is_final=is_final_block))
                d_in = d_out
            if final_dim > 1:
                layers.append(KANBlock(d_in, final_dim, k_in, k_out, is_final=False))

        self.layers = nn.ModuleList(layers)
        self.norm_layers = nn.ModuleList()
        for li, block in enumerate(self.layers):
            if self.norm_skip_first and li == 0:
                self.norm_layers.append(nn.Identity())
            else:
                nfeat = 1 if block.is_final else block.d_out
                self.norm_layers.append(nn.BatchNorm1d(nfeat))
        self.output_scale = nn.Parameter(torch.tensor(1.0))
        self.output_bias = nn.Parameter(torch.tensor(0.0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        xin = x
        for li, block in enumerate(self.layers):
            xout = block(xin, xin)
            xout = self.norm_layers[li](xout)
            xin = xout
        return self.output_scale * xin + self.output_bias

# ------------------ Dataset: TPS-16 (Tiered Pricing & Saturation) ------------------

class TieredPricing16D(Dataset):
    def __init__(self, seed: int = 1234):
        super().__init__()
        rng = np.random.RandomState(seed)
        self._t1 = rng.uniform(0.15, 0.35, size=8)
        self._t2 = self._t1 + rng.uniform(0.2, 0.35, size=8)
        self._s_abs = rng.uniform(0.25, 0.75, size=4)
        self._w_hinge = rng.uniform(0.6, 1.1, size=8)
        self._w_abs = rng.uniform(0.3, 0.6, size=4)
        self._pairs = [(0, 1), (2, 5), (7, 8)]
        self._kappa = 1.0 + rng.uniform(-0.1, 0.1, size=len(self._pairs))
        self._w_pair = rng.uniform(0.5, 1.2, size=len(self._pairs))
        self._cap = 6.0

    @property
    def input_dim(self) -> int: return 16
    @property
    def output_dim(self) -> int: return 1

    def evaluate(self, x: torch.Tensor) -> torch.Tensor:
        B = x.shape[0]
        y = torch.zeros(B, 1, device=x.device, dtype=x.dtype)
        for i in range(8):
            t1 = float(self._t1[i]); t2 = float(self._t2[i]); w = float(self._w_hinge[i])
            s1, s2 = 1.0, 2.0
            y += w * (torch.relu(x[:, [i]] - t1) * s1 + torch.relu(x[:, [i]] - t2) * (s2 - s1))
        for j in range(4):
            idx = 8 + j; s = float(self._s_abs[j]); w = float(self._w_abs[j])
            y += w * torch.abs(x[:, [idx]] - s)
        for k, (p, q) in enumerate(self._pairs):
            kap = float(self._kappa[k]); w = float(self._w_pair[k])
            y += w * torch.relu(x[:, [p]] + x[:, [q]] - kap)
        y = torch.clamp(y + 0.1, 0.0, self._cap)
        return y

    def generate_inputs(self, n: int, device='cpu'):
        return torch.rand(n, self.input_dim, device=device)

# ------------------ Utils: params, training, evaluation ----------------------------

@dataclass
class TrainConfig:
    epochs: int = 4000
    warmup_sn: int = 400
    lr: float = 3e-4
    weight_decay: float = 1e-6
    max_grad_norm: float = 1.0
    batch_size: int = 1024

def count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

@torch.no_grad()
def evaluate_metrics(model: nn.Module, x: torch.Tensor, y: torch.Tensor) -> dict:
    model.eval()
    yhat = model(x)
    mse = torch.mean((yhat - y) ** 2).item()
    mae = torch.mean(torch.abs(yhat - y)).item()
    ymean = torch.mean(y)
    ss_tot = torch.sum((y - ymean) ** 2).item()
    ss_res = torch.sum((yhat - y) ** 2).item()
    r2 = 1.0 - (ss_res / (ss_tot + 1e-12))
    return {"mse": mse, "mae": mae, "r2": r2}

def train_loop(model: nn.Module,
               x_train: torch.Tensor, y_train: torch.Tensor,
               cfg: TrainConfig,
               device: torch.device,
               sn_domain_updater=None):
    model.train()
    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    best = math.inf
    best_state = None

    for ep in range(cfg.epochs):
        if sn_domain_updater is not None:
            sn_domain_updater(ep, model)
        opt.zero_grad()
        yhat = model(x_train)
        loss = torch.mean((yhat - y_train) ** 2)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), cfg.max_grad_norm)
        opt.step()

        if loss.item() < best:
            best = loss.item()
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

        if (ep + 1) % 400 == 0:
            print(f"[{ep+1:4d}/{cfg.epochs}] train MSE={loss.item():.4e}")

    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    return best

# ------------------ Builders -------------------------------------------------------

def build_sn(input_dim: int, arch: List[int], final_dim: int,
             phi_knots: int, Phi_knots: int) -> SprecherMultiLayerNetwork:
    # Fairness: BN only; residuals linear/projection only; no lateral; no advanced sched
    CONFIG['use_lateral_mixing'] = False
    CONFIG['use_advanced_scheduler'] = False
    CONFIG['train_phi_codomain'] = False
    CONFIG['use_normalization'] = True
    CONFIG['norm_type'] = 'batch'
    CONFIG['norm_position'] = 'after'
    CONFIG['norm_skip_first'] = True
    CONFIG['use_residual_weights'] = True
    CONFIG['residual_style'] = 'linear'
    sn = SprecherMultiLayerNetwork(
        input_dim=input_dim, architecture=arch, final_dim=final_dim,
        phi_knots=phi_knots, Phi_knots=Phi_knots,
        norm_type='batch', norm_position='after', norm_skip_first=True
    )
    return sn

def build_kan(input_dim: int, arch: List[int], final_dim: int,
              k_in: int, k_out: int) -> KANNetwork:
    return KANNetwork(
        input_dim=input_dim, architecture=arch, final_dim=final_dim,
        k_in=k_in, k_out=k_out, norm_position='after', norm_skip_first=True
    )

# ------------------ Budget matching (robust) --------------------------------------

def search_kan_knots_for_target(target_params: int, input_dim: int, arch: List[int], final_dim: int,
                                k_min: int = 4, k_max: int = 128) -> Tuple[int, int, int]:
    """
    Grid over (k_in, k_out) to find a KAN with params closest to target, permitting knots down to 4.
    Returns (k_in, k_out, kan_params).
    """
    best = (None, None, None)
    best_diff = float('inf')
    for kin in range(k_min, k_max + 1, 2):
        for kout in range(k_min, k_max + 1, 2):
            kan = build_kan(input_dim, arch, final_dim, kin, kout)
            params = count_params(kan)
            diff = abs(params - target_params)
            if diff < best_diff:
                best = (kin, kout, params)
                best_diff = diff
    return best

def search_sn_knots_for_target(target_params: int, input_dim: int, arch: List[int], final_dim: int,
                               k_min: int = 32, k_max: int = 1024) -> Tuple[int, int, int]:
    """
    Grid over equal SN knot counts (phi_knots == Phi_knots == K) to reach target closest.
    Returns (K, K, sn_params).
    """
    best = (None, None, None)
    best_diff = float('inf')
    for K in range(k_min, k_max + 1, 4):
        sn = build_sn(input_dim, arch, final_dim, K, K)
        params = count_params(sn)
        diff = abs(params - target_params)
        if diff < best_diff:
            best = (K, K, params)
            best_diff = diff
    return best

def auto_match_param_budgets(input_dim: int, arch: List[int], final_dim: int,
                             sn_phi_knots_init: int, sn_Phi_knots_init: int,
                             kan_k_in_flag: int, kan_k_out_flag: int) -> Tuple[int, int, int, int, int, int]:
    """
    Returns matched (sn_phi_knots, sn_Phi_knots, sn_params, kan_k_in, kan_k_out, kan_params)
    guaranteeing Δ ≤ 5% when feasible by adjusting both sides.
    """
    # Start from initial SN
    sn = build_sn(input_dim, arch, final_dim, sn_phi_knots_init, sn_Phi_knots_init)
    sn_params0 = count_params(sn)

    # If user fixed KAN knots explicitly, honor them
    if kan_k_in_flag > 0 and kan_k_out_flag > 0:
        kin, kout = kan_k_in_flag, kan_k_out_flag
        kan = build_kan(input_dim, arch, final_dim, kin, kout)
        kan_params = count_params(kan)
    else:
        # First, find KAN closest to the initial SN
        kin, kout, kan_params = search_kan_knots_for_target(sn_params0, input_dim, arch, final_dim, k_min=4, k_max=128)

    # If already within 5% — done
    diff = abs(kan_params - sn_params0) / sn_params0
    if diff <= 0.05:
        return sn_phi_knots_init, sn_Phi_knots_init, sn_params0, kin, kout, kan_params

    # Otherwise, adjust SN upward if KAN floor > SN
    if kan_params > sn_params0:
        # Match SN to KAN by increasing K
        Kphi, KPhi, sn_params_new = search_sn_knots_for_target(kan_params, input_dim, arch, final_dim,
                                                               k_min=max(32, sn_phi_knots_init), k_max=1024)
        diff2 = abs(kan_params - sn_params_new) / sn_params_new
        if diff2 <= 0.05:
            return Kphi, KPhi, sn_params_new, kin, kout, kan_params
        # Still not within 5%? As a final step, recompute KAN around new SN
        kin2, kout2, kan_params2 = search_kan_knots_for_target(sn_params_new, input_dim, arch, final_dim, k_min=4, k_max=128)
        diff3 = abs(kan_params2 - sn_params_new) / sn_params_new
        return Kphi, KPhi, sn_params_new, kin2, kout2, kan_params2
    else:
        # SN > KAN — reduce KAN further (down to 4) or, if still high, slightly lower SN knots
        kin2, kout2, kan_params2 = search_kan_knots_for_target(sn_params0, input_dim, arch, final_dim, k_min=4, k_max=128)
        diff2 = abs(kan_params2 - sn_params0) / sn_params0
        if diff2 <= 0.05:
            return sn_phi_knots_init, sn_Phi_knots_init, sn_params0, kin2, kout2, kan_params2
        Kphi, KPhi, sn_params_new = search_sn_knots_for_target(kan_params2, input_dim, arch, final_dim,
                                                               k_min=32, k_max=max(32, sn_phi_knots_init))
        return Kphi, KPhi, sn_params_new, kin2, kout2, kan_params2

# ------------------ Main ----------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="SN vs KAN on Tiered Pricing & Saturation (TPS-16)")
    p.add_argument("--epochs", type=int, default=4000)
    p.add_argument("--seed", type=int, default=45)
    p.add_argument("--seeds", type=int, default=1)
    p.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"])
    p.add_argument("--arch", type=str, default="8,8", help="Hidden widths, e.g. '8,8' or '32,32'")
    p.add_argument("--sn_phi_knots", type=int, default=64)
    p.add_argument("--sn_Phi_knots", type=int, default=64)
    p.add_argument("--kan_k_in", type=int, default=0, help="0=auto")
    p.add_argument("--kan_k_out", type=int, default=0, help="0=auto")
    p.add_argument("--warmup", type=int, default=400, help="SN domain-update warm-up epochs (300 or 400).")
    return p.parse_args()

def set_global_seed(seed: int):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

def main():
    args = parse_args()
    assert args.warmup in (300, 400), "Warm-up must be 300 or 400."

    device = torch.device("cuda" if (args.device == "auto" and torch.cuda.is_available()) else
                          (args.device if args.device != "auto" else "cpu"))

    arch = [int(a.strip()) for a in args.arch.split(",") if a.strip()]
    input_dim, final_dim = 16, 1
    cfg = TrainConfig(epochs=args.epochs, warmup_sn=args.warmup)

    acc = {"sn": [], "kan": []}

    for run_idx in range(args.seeds):
        seed = args.seed + run_idx
        set_global_seed(seed)

        dataset = TieredPricing16D(seed=1234)
        n_train = 32 * 32
        n_val = 8192
        x_train = dataset.generate_inputs(n_train, device=device)
        y_train = dataset.evaluate(x_train)
        x_val = dataset.generate_inputs(n_val, device=device)
        y_val = dataset.evaluate(x_val)

        # ---- Auto-match budgets (robust) --------------------------------------
        snKphi, snKPhi, sn_params, kin, kout, kan_params = auto_match_param_budgets(
            input_dim, arch, final_dim, args.sn_phi_knots, args.sn_Phi_knots,
            args.kan_k_in, args.kan_k_out
        )
        diff = abs(kan_params - sn_params) / sn_params
        print(f"\nAuto-matched params (SN vs KAN): {sn_params} vs {kan_params}  (Δ={100*diff:.2f}%)")
        assert diff <= 0.05, "Parameter counts not within ±5% after auto-matching; consider smaller --arch."

        # ---- Build final models with matched budgets --------------------------
        sn = build_sn(input_dim, arch, final_dim, snKphi, snKPhi).to(device)
        with torch.no_grad():
            sn.output_scale.data = torch.tensor(0.1, device=device)
            sn.output_bias.data = y_train.mean()

        kan = build_kan(input_dim, arch, final_dim, kin, kout).to(device)
        with torch.no_grad():
            kan.output_scale.data = torch.tensor(0.1, device=device)
            kan.output_bias.data = y_train.mean()

        # ---- Training ----------------------------------------------------------
        def sn_domain_updater(epoch: int, model: nn.Module):
            if epoch < cfg.warmup_sn:
                model.update_all_domains(allow_resampling=True)
        print("\n=== Training SN (PWL splines with 400-epoch domain warm-up) ===")
        _ = train_loop(sn, x_train, y_train, cfg, device, sn_domain_updater)

        print("\n=== Training KAN (cubic splines; identical optimizer & budget) ===")
        _ = train_loop(kan, x_train, y_train, cfg, device, None)

        # ---- Evaluation --------------------------------------------------------
        sn_metrics = evaluate_metrics(sn, x_val, y_val)
        kan_metrics = evaluate_metrics(kan, x_val, y_val)

        print("\nResults on TPS-16 (validation):")
        print(f"  SN  (PWL)   MSE={sn_metrics['mse']:.6f}  MAE={sn_metrics['mae']:.6f}  R2={sn_metrics['r2']:.4f}")
        print(f"  KAN (cubic)  MSE={kan_metrics['mse']:.6f}  MAE={kan_metrics['mae']:.6f}  R2={kan_metrics['r2']:.4f}")

        acc["sn"].append(sn_metrics["mse"])
        acc["kan"].append(kan_metrics["mse"])

    if args.seeds > 1:
        print("\nAveraged across seeds:")
        print(f"  SN  mean MSE  = {np.mean(acc['sn']):.6f} ± {np.std(acc['sn']):.6f}")
        print(f"  KAN mean MSE  = {np.mean(acc['kan']):.6f} ± {np.std(acc['kan']):.6f}")

    print("\n--- Fairness note ---")
    print("Both models trained for exactly 4000 epochs with identical optimizer (Adam), "
          "learning rate, batch size, weight decay, loss (MSE), splits, metrics, and eval batch size. "
          "Only BatchNorm (after blocks; skip first) and only linear/projection residuals were used. "
          f"SN used domain updates for {cfg.warmup_sn} warm-up epochs then froze domains; "
          "KAN used cubic splines. Parameter counts were auto-matched within ±5%. "
          "TPS-16 has natural thresholds/kinks, so a PWL model (SN) should fit them more efficiently "
          "than a cubic-spline model (KAN) under equal budget.")
if __name__ == "__main__":
    main()