# benchmarks/kan_sn_shiftshare_bench.py
"""
KAN vs SN — Shift-Shared Pre-Activation Benchmark (SSPAB)

Goal
----
Construct a multi-head regression task where all heads are horizontally shifted
copies of the same latent index, and the shift happens *before* the inner
monotone nonlinearity. This exactly matches the Sprecher construction
y_q(x) = Φ( Σ_i λ_i · φ(x_i + η·q) ), giving SNs a strong parameter-sharing prior:
φ(·) and λ are shared across heads, and q only enters through the η·q shift.

KANs, by contrast, use per-edge B-splines without this head-shift sharing—
so with many heads 'm' and matched total parameter budgets, they must lower
the univariate basis count K, degrading accuracy.

We keep the comparison "apples-to-apples":
  • Parameter parity: pick KAN's K to match the SN param count (<= target).
    Uses the same utility as your parity bench.  [choose_kan_basis_for_parity]
  • BN semantics at test: use 'batch_no_update' as in your parity bench
    (eval() for everything, BatchNorm layers use *batch* stats without updating buffers).
  • Residuals and normalization: both use batch norm (after), linear residuals,
    and conventional linear (matrix) residuals for SN per your config.

We also report in-distribution (ID) *and* a mild covariate-shift (OOD) test.
SN uses a warm-up period with theoretical domain updates then "freezes" domains,
exactly like your other scripts.

References into your code (used here verbatim):
  - Param matching (KAN K): choose_kan_basis_for_parity                     [parity bench]
  - KAN trainer (fast/slow impls supported): train_kan                      [parity bench]
  - BN-safe evaluation ('batch_no_update'): evaluate_with_progress          [parity bench]
  - SN training entrypoint with warm-up snapshot: train_network             [sn_core]
"""

import argparse, os, json, time
from dataclasses import dataclass
import numpy as np
import torch
import torch.nn as nn

# --- SN core + config ---
from sn_core import CONFIG, train_network  # trainer returns snapshot incl. (model, x_train, y_train)  # noqa
# BN presence checks and BN-safe eval come from parity bench imports below.

# --- Reuse parity-bench utilities to ensure identical training/eval accounting ---
from benchmarks.kan_sn_parity_bench import (  # noqa: F401
    set_global_seeds,            # seeds
    count_params,                # param counter
    train_kan,                   # KAN training loop (fast/slow)
    choose_kan_basis_for_parity, # K selection for param parity
    evaluate_with_progress,      # BN-safe test evaluation
    continue_train_sn_no_domain_updates,  # SN freeze-phase trainer
)

# -------------------------------
# Dataset: Shift-Shared Pre-Activation family
# -------------------------------

class ShiftSharedPreActDataset:
    """
    Multi-head target where the head index q enters *before* the inner monotone nonlinearity:

        y_q(x) = tanh( β · ( Σ_i λ_i · sigm(a_i · (x_i + η·q − c_i)) − μ ) )

    • x ∈ [0,1]^D (ID).  For OOD tests we sample x ∈ [-margin, 1+margin]^D (default 0.2).
    • λ_i ≥ 0 normalized; slopes a_i and centers c_i vary across dims.
    • η > 0 controls the horizontal shift per head (q = 0..m-1).
    • μ recenters the score to keep tanh in a sensitive (non-saturated) regime.

    Why this favors SNs under param parity:
      - SN shares φ across all heads and injects q via (+ η·q) inside φ (built-in).
      - A KAN needs separate per-edge splines S_{i,j} for each head j—it can *approximate*
        the horizontal shifts but must spend parameters K×(D×m) to do it. With m large,
        matching the SN's total parameter count forces K small -> larger error.

    The task is noise-free by default to make the approximation gap clean.
    """
    def __init__(self, d_in=24, n_heads=25, seed=0, *,
                 eta=0.08, beta=1.2, a_min=2.0, a_max=8.0,
                 center_low=0.15, center_high=0.85,
                 noise_std=0.0):
        self.D = int(d_in)
        self.m = int(n_heads)
        self.eta = float(eta)
        self.beta = float(beta)
        self.noise_std = float(noise_std)

        rs = np.random.RandomState(seed)
        # Per-dim sigmoid parameters and positive weights λ
        self.a = torch.tensor(rs.uniform(a_min, a_max, size=self.D), dtype=torch.float32)
        self.c = torch.tensor(rs.uniform(center_low, center_high, size=self.D), dtype=torch.float32)
        lam = rs.rand(self.D); lam = lam / (lam.sum() + 1e-12)
        self.lam = torch.tensor(lam, dtype=torch.float32)

        # Choose μ so that the pre-tanh score is roughly centered
        # over x ~ Uniform[0,1]^D and q ~ {0,...,m-1}.
        # A crude analytic centroid near 0.5 for sigm plus shift through heads:
        self.mu = float(self.lam.sum().item() * 0.5)

        # Fixed head offsets q
        self.q = torch.arange(self.m, dtype=torch.float32)  # [0,1,...,m-1]

    @property
    def input_dim(self):  return self.D
    @property
    def output_dim(self): return self.m

    def _evaluate_core(self, x):
        # x: [N, D] in any box; produce [N, m]
        N = x.shape[0]
        # Broadcast: (N,D) + (m,) -> (N,D,m)
        q = self.q.to(x.device).view(1, 1, -1)
        a = self.a.to(x.device).view(1, -1, 1)
        c = self.c.to(x.device).view(1, -1, 1)
        lam = self.lam.to(x.device).view(1, -1, 1)

        shifted = x.unsqueeze(-1) + self.eta * q        # (N,D,m)
        h = torch.sigmoid(a * (shifted - c))            # (N,D,m)  monotone in each coord
        s = (lam * h).sum(dim=1)                        # (N,m)    latent index per head
        y = torch.tanh(self.beta * (s - self.mu))       # monotone in q if β>0 and η>0
        return y

    def sample(self, n, device="cpu"):
        x = torch.rand(int(n), self.D, device=device)
        y = self._evaluate_core(x)
        if self.noise_std > 0:
            y = y + torch.randn_like(y) * self.noise_std
        return x, y

    def sample_ood(self, n, device="cpu", margin=0.2):
        x = (1 + 2*margin) * torch.rand(int(n), self.D, device=device) - margin
        y = self._evaluate_core(x)
        if self.noise_std > 0:
            y = y + torch.randn_like(y) * self.noise_std
        return x, y


# -------------------------------
# Simple extra metric: monotonicity across heads
# -------------------------------

def monotonicity_violation_rate(y_pred: torch.Tensor) -> float:
    """
    Fraction of samples where the predicted head sequence is not non-decreasing.
    (Useful when η>0 and β>0; lower is better.)
    """
    with torch.no_grad():
        diffs = y_pred[:, 1:] - y_pred[:, :-1]
        bad = (diffs < 0).any(dim=1).float().mean().item()
    return float(bad)


# -------------------------------
# Run summary
# -------------------------------

@dataclass
class RunSummary:
    model: str
    params: int
    train_mse: float
    test_id_rmse_mean: float
    test_ood_rmse_mean: float
    mono_violation_id: float
    mono_violation_ood: float
    seconds: float
    notes: str = ""


# -------------------------------
# Main
# -------------------------------

def main():
    p = argparse.ArgumentParser()
    # General
    p.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"])
    p.add_argument("--seed", type=int, default=45)
    p.add_argument("--epochs", type=int, default=4000)

    # Dataset
    p.add_argument("--dims", type=int, default=24)
    p.add_argument("--n_heads", type=int, default=25)
    p.add_argument("--eta", type=float, default=0.08)
    p.add_argument("--beta", type=float, default=1.2)
    p.add_argument("--test_size_id", type=int, default=50000)
    p.add_argument("--test_size_ood", type=int, default=50000)
    p.add_argument("--ood_margin", type=float, default=0.2)

    # SN (mirror your bench flags)
    p.add_argument("--sn_arch", type=str, default="16,16")
    p.add_argument("--sn_phi_knots", type=int, default=80)
    p.add_argument("--sn_Phi_knots", type=int, default=80)
    p.add_argument("--sn_norm_type", type=str, default="batch", choices=["none", "batch", "layer"])
    p.add_argument("--sn_norm_position", type=str, default="after", choices=["before", "after"])
    p.add_argument("--sn_norm_skip_first", action="store_true", default=True)
    p.add_argument("--sn_norm_first", action="store_true")
    p.add_argument("--sn_no_residual", action="store_true")
    p.add_argument("--sn_residual_style", type=str, default="linear",
                   choices=["node", "linear", "standard", "matrix"])
    p.add_argument("--sn_no_lateral", action="store_true", default=True)
    p.add_argument("--sn_freeze_domains_after", type=int, default=400,
                   help="Warm-up epochs with domain updates, then freeze (0=never)")
    p.add_argument("--sn_domain_margin", type=float, default=0.01)

    # KAN (mirror your bench flags)
    p.add_argument("--kan_arch", type=str, default="8,8")
    p.add_argument("--kan_degree", type=int, default=3, choices=[2, 3])
    p.add_argument("--kan_bn_type", type=str, default="batch", choices=["none", "batch"])
    p.add_argument("--kan_bn_position", type=str, default="after", choices=["before", "after"])
    p.add_argument("--kan_bn_skip_first", action="store_true", default=True)
    p.add_argument("--kan_outside", type=str, default="linear", choices=["linear", "clamp"])
    p.add_argument("--kan_residual_type", type=str, default="linear", choices=["silu", "linear", "none"])
    p.add_argument("--kan_lr", type=float, default=1e-3)
    p.add_argument("--kan_wd", type=float, default=1e-6)
    p.add_argument("--kan_impl", type=str, default="fast", choices=["fast", "slow"])

    # Parity + BN-eval mode
    p.add_argument("--equalize_params", action="store_true", default=True)
    p.add_argument("--prefer_leq", action="store_true", default=True)
    p.add_argument("--bn_eval_mode", type=str, default="batch_no_update",
                   choices=["batch_no_update", "recalc_eval", "off"])
    p.add_argument("--bn_recalc_passes", type=int, default=10)
    p.add_argument("--eval_batch_size", type=int, default=8192)

    # Output
    p.add_argument("--outdir", type=str, default="benchmarks/results")

    args = p.parse_args()
    device = ("cuda" if (args.device == "auto" and torch.cuda.is_available())
              else ("cpu" if args.device == "auto" else args.device))
    set_global_seeds(args.seed)

    # ------------------------------
    # Build dataset + fixed test sets
    # ------------------------------
    dataset = ShiftSharedPreActDataset(
        d_in=args.dims, n_heads=args.n_heads, seed=args.seed,
        eta=args.eta, beta=args.beta
    )

    torch.manual_seed(args.seed + 1001)
    with torch.no_grad():
        x_test_id,  y_test_id  = dataset.sample(args.test_size_id,  device=device)
        x_test_ood, y_test_ood = dataset.sample_ood(args.test_size_ood, device=device, margin=args.ood_margin)

    # ------------------------------
    # Configure SN fairness knobs
    # ------------------------------
    if args.sn_no_residual:
        CONFIG['use_residual_weights'] = False
    else:
        style = (args.sn_residual_style or "linear").lower()
        if style in ("standard", "matrix"):  # normalize synonyms
            style = "linear"
        CONFIG['residual_style'] = style

    if args.sn_no_lateral:
        CONFIG['use_lateral_mixing'] = False

    sn_norm_skip = False if args.sn_norm_first else args.sn_norm_skip_first
    CONFIG['domain_safety_margin'] = float(args.sn_domain_margin)

    # Parse SN arch
    sn_arch = [int(x) for x in args.sn_arch.split(",")] if args.sn_arch.strip() else []

    # ------------------------------
    # SN training: warm-up (domains update) then optional freeze
    # ------------------------------
    total_epochs = int(args.epochs)
    warmup_epochs = max(0, min(args.sn_freeze_domains_after, total_epochs))
    rest_epochs = total_epochs - warmup_epochs

    class _WrapDataset:
        """Adapter so sn_core.train_network can call .sample(n)."""
        def __init__(self, d): self.d = d
        @property
        def input_dim(self):  return self.d.input_dim
        @property
        def output_dim(self): return self.d.output_dim
        def sample(self, n, device="cpu"): return self.d.sample(n, device)

    t0_sn = time.time()
    residual_style_override = None if args.sn_no_residual else CONFIG.get('residual_style', 'linear')
    snapshot, _ = train_network(
        dataset=_WrapDataset(dataset),
        architecture=sn_arch,
        total_epochs=(warmup_epochs if warmup_epochs > 0 else total_epochs),
        print_every=max(1, (warmup_epochs if warmup_epochs > 0 else total_epochs)//10),
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

    sn_model = snapshot["model"].to(device)
    x_train  = snapshot["x_train"].to(device)  # reuse *exact* train set for KAN
    y_train  = snapshot["y_train"].to(device)

    if rest_epochs > 0:
        t1 = time.time()
        sn_model, sn_train_mse, _ = continue_train_sn_no_domain_updates(
            model=sn_model, x_train=x_train, y_train=y_train, epochs=rest_epochs,
            device=device, seed=args.seed
        )
        sn_secs += (time.time() - t1)
    else:
        with torch.no_grad():
            sn_train_mse = float(nn.MSELoss()(sn_model(x_train), y_train).cpu())

    sn_params = count_params(sn_model)

    # ------------------------------
    # Choose K for KAN to match SN parameter count
    # ------------------------------
    kan_arch = [int(x) for x in args.kan_arch.split(",")] if args.kan_arch.strip() else []
    if args.equalize_params:
        kan_K, est_cnt = choose_kan_basis_for_parity(
            target_params=sn_params, arch=kan_arch, input_dim=dataset.input_dim, final_dim=dataset.output_dim,
            degree=args.kan_degree, bn_type=args.kan_bn_type, bn_position=args.kan_bn_position,
            bn_skip_first=args.kan_bn_skip_first, residual_type=args.kan_residual_type,
            prefer_leq=args.prefer_leq
        )
        parity_note = f"[ParamMatch] SN params={sn_params}. KAN K={kan_K} (est. {est_cnt})."
        print(parity_note)
    else:
        kan_K = 60
        parity_note = ""

    # ------------------------------
    # Train KAN on the EXACT same (x_train, y_train)
    # ------------------------------
    t0_kan = time.time()
    kan_model, kan_train_mse, kan_secs = train_kan(
        x_train=x_train, y_train=y_train,
        input_dim=dataset.input_dim, final_dim=dataset.output_dim,
        arch=kan_arch, n_basis=kan_K, degree=args.kan_degree,
        device=device, epochs=args.epochs, seed=args.seed,
        lr=args.kan_lr, wd=args.kan_wd,
        bn_type=args.kan_bn_type, bn_position=args.kan_bn_position, bn_skip_first=args.kan_bn_skip_first,
        outside=args.kan_outside, residual_type=args.kan_residual_type, impl=args.kan_impl
    )
    kan_params = count_params(kan_model)

    # ------------------------------
    # Batched evaluation (BN-safe) — ID and OOD
    # ------------------------------
    def eval_both_sets(model, name):
        per_head_id,  rmse_id,  _, _ = evaluate_with_progress(
            model, x_test_id,  y_test_id,
            bn_mode=args.bn_eval_mode, bn_recalc_passes=args.bn_recalc_passes, x_train_for_bn=x_train,
            name=f"{name}-ID",  eval_batch_size=args.eval_batch_size
        )
        per_head_ood, rmse_ood, _, _ = evaluate_with_progress(
            model, x_test_ood, y_test_ood,
            bn_mode=args.bn_eval_mode, bn_recalc_passes=args.bn_recalc_passes, x_train_for_bn=x_train,
            name=f"{name}-OOD", eval_batch_size=args.eval_batch_size
        )
        # Compute monotonicity violation rate on predictions
        with torch.no_grad():
            # ID preds
            ypred_id  = []
            ypred_ood = []
            bs = max(1, int(args.eval_batch_size))
            for start in range(0, x_test_id.shape[0], bs):
                end = min(x_test_id.shape[0], start + bs)
                ypred_id.append(model(x_test_id[start:end].to(device)).cpu())
            for start in range(0, x_test_ood.shape[0], bs):
                end = min(x_test_ood.shape[0], start + bs)
                ypred_ood.append(model(x_test_ood[start:end].to(device)).cpu())
            ypred_id = torch.cat(ypred_id, dim=0)
            ypred_ood = torch.cat(ypred_ood, dim=0)
        mono_id = monotonicity_violation_rate(ypred_id)
        mono_ood = monotonicity_violation_rate(ypred_ood)
        return rmse_id, rmse_ood, mono_id, mono_ood

    sn_rmse_id,  sn_rmse_ood,  sn_mono_id,  sn_mono_ood  = eval_both_sets(sn_model,  "SN")
    kan_rmse_id, kan_rmse_ood, kan_mono_id, kan_mono_ood = eval_both_sets(kan_model, "KAN")

    # ------------------------------
    # Summaries
    # ------------------------------
    sn_title = (f"SN(arch={sn_arch}, phi_knots={args.sn_phi_knots}, Phi_knots={args.sn_Phi_knots}, "
                f"norm={args.sn_norm_type}/{args.sn_norm_position}/"
                f"{'skip_first' if sn_norm_skip else 'include_first'}, "
                f"residuals={'off' if args.sn_no_residual else 'on(style='+CONFIG.get('residual_style','node')+')'}, "
                f"lateral={'off' if args.sn_no_lateral else 'on'}, "
                f"domains={'warmup+freeze' if args.sn_freeze_domains_after>0 else 'updated'})")
    kan_title = (f"KAN[{args.kan_impl}](arch={kan_arch}, K={kan_K}, degree={args.kan_degree}, "
                 f"BN={args.kan_bn_type}/{args.kan_bn_position}/"
                 f"{'skip_first' if args.kan_bn_skip_first else 'include_first'}, "
                 f"outside={args.kan_outside}, residual={args.kan_residual_type})")

    out_sn = RunSummary(
        model=sn_title, params=int(sn_params), train_mse=float(sn_train_mse),
        test_id_rmse_mean=float(sn_rmse_id), test_ood_rmse_mean=float(sn_rmse_ood),
        mono_violation_id=float(sn_mono_id), mono_violation_ood=float(sn_mono_ood),
        seconds=float(sn_secs), notes=parity_note + " SN timing includes warm-up + optional freeze."
    ).__dict__
    out_kan = RunSummary(
        model=kan_title, params=int(kan_params), train_mse=float(kan_train_mse),
        test_id_rmse_mean=float(kan_rmse_id), test_ood_rmse_mean=float(kan_rmse_ood),
        mono_violation_id=float(kan_mono_id), mono_violation_ood=float(kan_mono_ood),
        seconds=float(kan_secs), notes=parity_note
    ).__dict__

    print("\n=== Head-to-Head Results (SSPAB) ===")
    print(json.dumps(out_sn, indent=2))
    print(json.dumps(out_kan, indent=2))

    os.makedirs(args.outdir, exist_ok=True)
    out_path = os.path.join(args.outdir, f"kan_sn_shiftshare_D{dataset.input_dim}_m{dataset.output_dim}_seed{args.seed}.json")
    with open(out_path, "w") as f:
        json.dump([out_sn, out_kan], f, indent=2)
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()