# benchmarks/kan_sn_densehead_shift_bench.py
"""
KAN vs SN — Dense-Head Shift Benchmark

Purpose
-------
Stress-test multi-head regression where each output head is a smoothly shifted
version of a shared latent index. This favors SN's "shared φ/Φ + per-head shift"
inductive bias but remains fair to KAN via parameter parity and aligned BN eval
semantics.

Key design points (aligned with your repo):
- Uses the same parity helpers and BN evaluation modes as in
  benchmarks/kan_sn_parity_bench.py (imported directly).
- CLI flags match the command you attempted to run:
    --dims, --heads, --test_size, --alpha, --beta, --q_bias,
    the full block of --sn_* flags, the full block of --kan_* flags,
    and parity/eval flags including --equalize_params, --prefer_leq,
    --bn_eval_mode, --eval_batch_size.

Dataset (DenseHeadShift)
------------------------
Let x ∈ [0,1]^D, heads j = 0..(m-1). We build:
  h_i(x_i) = sigmoid(a_i * (x_i - c_i))
  μ(x)     = Σ_i w_i h_i(x_i) + small positive pairwise terms
  σ(x)     = s0 + Σ_i v_i h_i(x_i)  (kept > 0)

Per-head index q_j ∈ [-1,1] (uniform grid) and z_j = Φ^{-1}(τ_j), τ_j ∈ (0,1).
Targets:
  y_j(x) = tanh( β * [ μ(x) + σ(x)*z_j + α*q_j + q_bias*q_j ] )

Where:
  - β  = tanh squashing strength (your --beta)
  - α  = head shift strength (your --alpha)
  - q_bias adds a simple linear bias across heads (your --q_bias)

Outputs are ordered in j and typically monotone in j; we also report a
monotonicity-violation rate like the MQSI bench.

Usage example
-------------
for s in 0; do
  python -m benchmarks.kan_sn_densehead_shift_bench \
    --dims 12 --heads 64 --test_size 50000 \
    --alpha 0.08 --beta 0.7 --q_bias 0.15 \
    --epochs 4000 --device cpu --seed $s \
    \
    --sn_arch 32,32 --sn_phi_knots 60 --sn_Phi_knots 60 \
    --sn_norm_type batch --sn_norm_position after --sn_norm_skip_first \
    --sn_residual_style linear --sn_no_lateral \
    --sn_freeze_domains_after 400 --sn_domain_margin 0.01 \
    \
    --kan_arch 12,12 --kan_degree 3 \
    --kan_bn_type batch --kan_bn_position after --kan_bn_skip_first \
    --kan_residual_type linear --kan_outside linear \
    --kan_impl fast \
    \
    --equalize_params --prefer_leq \
    --bn_eval_mode batch_no_update \
    --eval_batch_size 8192
done
"""

import argparse, math, os, json, time
import numpy as np
import torch
import torch.nn as nn

# SN core configuration + utilities
from sn_core import CONFIG  # noqa: F401  (CONFIG used via side-effects), has_batchnorm  # type: ignore
from sn_core import has_batchnorm  # imported for type checks used by eval helpers

# Parity benchmark helpers (reuse to keep training/eval identical)
# - set_global_seeds, choose_kan_basis_for_parity, train_kan, evaluate_with_progress, count_params
from benchmarks.kan_sn_parity_bench import (  # noqa: E402
    set_global_seeds,
    count_params,
    train_kan,
    choose_kan_basis_for_parity,
    evaluate_with_progress,
)

# SN training entrypoint (used for warmup; we feed the same (x_train, y_train) to KAN later)
from sn_core import train_network  # noqa: E402

# -------------------------------
# Dataset: DenseHeadShift
# -------------------------------

class DenseHeadShiftDataset:
    """
    Multi-head function with head-wise smooth shifts of a shared latent index.

    Inputs: x ∈ [0,1]^D  (uniform)
    Heads: m outputs, indexed by q_j ∈ [-1,1] and normal-quantiles z_j

    y_j(x) = tanh( β * [ μ(x) + σ(x)*z_j + α*q_j + q_bias*q_j ] )

    Parameters (set via CLI):
      D        : input dims (--dims)
      m        : #heads     (--heads)
      alpha    : head shift strength
      beta     : tanh squashing strength
      q_bias   : additional linear bias across heads
    """
    def __init__(self, d_in=12, n_heads=64, seed=0,
                 a_min=2.0, a_max=8.0, s0=0.15, v_scale=0.25,
                 pair_frac=0.15, pair_scale=0.08,
                 alpha=0.1, beta=0.7, q_bias=0.0):
        self._d = int(d_in)
        self._m = int(n_heads)
        self.alpha = float(alpha)
        self.beta = float(beta)
        self.q_bias = float(q_bias)

        rs = np.random.RandomState(seed)

        # per-dim monotone sigmoid transforms: h_i(x_i) = sigmoid(a_i (x_i - c_i))
        self.a = torch.tensor(rs.uniform(a_min, a_max, size=self._d), dtype=torch.float32)
        self.c = torch.tensor(rs.uniform(0.2, 0.8, size=self._d), dtype=torch.float32)

        # positive weights (normalize for stability)
        w_raw = rs.rand(self._d); w_raw = w_raw / (w_raw.sum() + 1e-12)
        v_raw = rs.rand(self._d); v_raw = v_raw / (v_raw.sum() + 1e-12)
        self.w = torch.tensor(w_raw, dtype=torch.float32)
        self.v = torch.tensor(v_raw * v_scale, dtype=torch.float32)

        # pairwise positive interactions
        n_pairs = int(round(pair_frac * self._d))
        self.pairs = []
        if n_pairs > 0:
            idx = rs.choice(self._d, size=(n_pairs, 2), replace=False)
            for i, j in idx:
                if i != j:
                    self.pairs.append((int(i), int(j)))
        self.pair_scale = float(pair_scale)

        # scale floor
        self.s0 = float(s0)

        # head grid q_j ∈ [-1,1]
        if self._m > 1:
            self.q = torch.linspace(-1.0, 1.0, self._m)
        else:
            self.q = torch.tensor([0.0])

        # normal-quantile z_j from evenly spaced τ_j ∈ (0.1,0.9) for stability
        taus = np.linspace(0.1, 0.9, self._m, dtype=np.float32)
        self.taus = torch.tensor(taus)
        self.z = torch.sqrt(torch.tensor(2.0)) * torch.erfinv(2 * self.taus - 1)

    @property
    def input_dim(self):
        return self._d

    @property
    def output_dim(self):
        return self._m

    def _h(self, x):
        return torch.sigmoid((x - self.c.to(x.device)) * self.a.to(x.device))

    def _mu(self, h):
        mu = torch.matmul(h, self.w.to(h.device))  # [N]
        if self.pairs:
            for (i, j) in self.pairs:
                mu = mu + self.pair_scale * (h[:, i] * h[:, j])
        return mu

    def _sigma(self, h):
        return (self.s0 + torch.matmul(h, self.v.to(h.device))).clamp_min(1e-5)

    def evaluate(self, x):
        # x: [N, D]  ->  y: [N, m]
        h = self._h(x)
        mu = self._mu(h)                      # [N]
        sig = self._sigma(h)                  # [N]

        z = self.z.to(x.device).view(1, -1)   # [1, m]
        q = self.q.to(x.device).view(1, -1)   # [1, m]

        # inside-tanh pre-activation:
        #   μ(x) + σ(x)*z_j + α*q_j + q_bias*q_j
        pre = mu.unsqueeze(1) + sig.unsqueeze(1) * z + self.alpha * q + self.q_bias * q
        return torch.tanh(self.beta * pre)

    def sample(self, n, device="cpu"):
        x = torch.rand(n, self._d, device=device)
        y = self.evaluate(x)
        return x, y


# -------------------------------
# Extra metrics
# -------------------------------

def monotonicity_violation_rate(y_pred):
    """
    Fraction of samples where predicted heads are NOT non-decreasing in j.
    """
    with torch.no_grad():
        diffs = y_pred[:, 1:] - y_pred[:, :-1]
        bad = (diffs < 0).any(dim=1).float().mean().item()
    return float(bad)


# -------------------------------
# SN warm-up + optional freeze (domains)
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
    for _ in range(max(0, int(epochs))):
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
# Main
# -------------------------------

def main():
    p = argparse.ArgumentParser()

    # Device/seed/epochs
    p.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"])
    p.add_argument("--seed", type=int, default=45)
    p.add_argument("--epochs", type=int, default=4000)

    # Dataset hyperparameters
    p.add_argument("--dims", type=int, default=12, help="Input dimension D")
    p.add_argument("--heads", type=int, default=64, help="Number of output heads m")
    p.add_argument("--test_size", type=int, default=50000, help="Test set size")
    p.add_argument("--alpha", type=float, default=0.1, help="Head shift strength")
    p.add_argument("--beta", type=float, default=0.7, help="Tanh squashing strength")
    p.add_argument("--q_bias", type=float, default=0.0, help="Linear bias across heads")

    # SN config (mirrors parity bench flags / semantics)
    p.add_argument("--sn_arch", type=str, default="32,32")
    p.add_argument("--sn_phi_knots", type=int, default=60)
    p.add_argument("--sn_Phi_knots", type=int, default=60)
    p.add_argument("--sn_norm_type", type=str, default="batch", choices=["none", "batch", "layer"])
    p.add_argument("--sn_norm_position", type=str, default="after", choices=["before", "after"])
    p.add_argument("--sn_norm_skip_first", action="store_true", default=True)
    p.add_argument("--sn_norm_first", action="store_true")
    p.add_argument("--sn_no_residual", action="store_true", default=False)
    p.add_argument("--sn_residual_style", type=str, default="linear",
                   choices=["node", "linear", "standard", "matrix"])
    p.add_argument("--sn_no_lateral", action="store_true", default=True)
    p.add_argument("--sn_freeze_domains_after", type=int, default=400,
                   help="Warm-up epochs with domain updates, then freeze (0=never)")
    p.add_argument("--sn_domain_margin", type=float, default=0.01)

    # KAN config (mirrors parity bench)
    p.add_argument("--kan_arch", type=str, default="12,12")
    p.add_argument("--kan_degree", type=int, default=3, choices=[2, 3])
    p.add_argument("--kan_bn_type", type=str, default="batch", choices=["none", "batch"])
    p.add_argument("--kan_bn_position", type=str, default="after", choices=["before", "after"])
    p.add_argument("--kan_bn_skip_first", action="store_true", default=True)
    p.add_argument("--kan_outside", type=str, default="linear", choices=["linear", "clamp"])
    p.add_argument("--kan_residual_type", type=str, default="linear", choices=["silu", "linear", "none"])
    p.add_argument("--kan_lr", type=float, default=1e-3)
    p.add_argument("--kan_wd", type=float, default=1e-6)
    p.add_argument("--kan_impl", type=str, default="fast", choices=["fast", "slow"])

    # Parity + BN eval semantics (reuse parity bench choices)
    p.add_argument("--equalize_params", action="store_true", default=False)
    p.add_argument("--prefer_leq", action="store_true", default=False)
    p.add_argument("--bn_eval_mode", type=str, default="batch_no_update",
                   choices=["batch_no_update", "recalc_eval", "off"])
    p.add_argument("--bn_recalc_passes", type=int, default=10)
    p.add_argument("--eval_batch_size", type=int, default=8192)

    # Output
    p.add_argument("--outdir", type=str, default="benchmarks/results")

    args = p.parse_args()
    device = ("cuda" if (args.device == "auto" and torch.cuda.is_available()) else
              ("cpu" if args.device == "auto" else args.device))
    set_global_seeds(args.seed)

    # Build dataset
    dataset = DenseHeadShiftDataset(
        d_in=args.dims, n_heads=args.heads, seed=args.seed,
        alpha=args.alpha, beta=args.beta, q_bias=args.q_bias
    )

    # Fixed test set
    torch.manual_seed(args.seed + 2031)
    with torch.no_grad():
        x_test, y_test = dataset.sample(args.test_size, device=device)

    # ------------------------------
    # Configure SN fairness knobs
    # ------------------------------
    if args.sn_no_residual:
        CONFIG['use_residual_weights'] = False
    else:
        style = (args.sn_residual_style or "linear").lower()
        if style in ("standard", "matrix"):
            style = "linear"
        CONFIG['residual_style'] = style

    if args.sn_no_lateral:
        CONFIG['use_lateral_mixing'] = False

    sn_norm_skip = False if args.sn_norm_first else args.sn_norm_skip_first
    CONFIG['domain_safety_margin'] = float(args.sn_domain_margin)
    sn_arch = [int(x) for x in args.sn_arch.split(",")] if args.sn_arch.strip() else []

    # ------------------------------
    # SN training: warm-up with domain updates, then optional freeze
    # ------------------------------
    total_epochs = int(args.epochs)
    warmup_epochs = max(0, min(args.sn_freeze_domains_after, total_epochs))
    rest_epochs = total_epochs - warmup_epochs

    # Phase 1: warm-up WITH domain updates (standard SN trainer)
    residual_style_override = None if args.sn_no_residual else CONFIG.get('residual_style', 'linear')
    t0_sn = time.time()
    # We wrap the custom dataset to match sn_core.train_network's expected interface
    class _WrapDataset:
        def __init__(self, base): self.base = base
        @property
        def input_dim(self): return self.base.input_dim
        @property
        def output_dim(self): return self.base.output_dim
        def sample(self, n, device="cpu"): return self.base.sample(n, device)

    plotting_snapshot, _ = train_network(
        dataset=_WrapDataset(dataset),
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
    # Choose K for KAN to match SN parameter count (if requested)
    # ------------------------------
    kan_arch = [int(x) for x in args.kan_arch.split(",")] if args.kan_arch.strip() else []
    if args.equalize_params:
        chosen_K, est_cnt = choose_kan_basis_for_parity(
            target_params=sn_params, arch=kan_arch,
            input_dim=dataset.input_dim, final_dim=dataset.output_dim,
            degree=args.kan_degree, bn_type=args.kan_bn_type, bn_position=args.kan_bn_position,
            bn_skip_first=args.kan_bn_skip_first, residual_type=args.kan_residual_type,
            prefer_leq=args.prefer_leq
        )
        kan_K = chosen_K
        parity_note = f"[ParamMatch] SN params = {sn_params}. KAN K = {kan_K} (est. {est_cnt})."
        print(parity_note)
    else:
        # sensible default if parity disabled
        kan_K = max(args.kan_degree + 1, 60)
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
    # Evaluation (aligned BN semantics)
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

    # Monotonicity across heads (lower = better)
    with torch.no_grad():
        sn_pred = []
        kan_pred = []
        bs = max(1, int(args.eval_batch_size))
        for start in range(0, x_test.shape[0], bs):
            end = min(x_test.shape[0], start + bs)
            sn_pred.append(sn_model(x_test[start:end]).cpu())
            kan_pred.append(kan_model(x_test[start:end]).cpu())
        sn_pred = torch.cat(sn_pred, dim=0)
        kan_pred = torch.cat(kan_pred, dim=0)
    sn_mono_bad = monotonicity_violation_rate(sn_pred)
    kan_mono_bad = monotonicity_violation_rate(kan_pred)

    # ------------------------------
    # Print & save results
    # ------------------------------
    def summarize(name, params, train_mse, mean_rmse, per_head, corrF, used, secs, mono_bad, notes=""):
        return {
            "model": name,
            "params": int(params),
            "train_mse": float(train_mse),
            "test_rmse_mean": float(mean_rmse),
            "test_rmse_per_head": [float(x) for x in per_head],
            "corr_Frob_error": (None if (corrF is None or not math.isfinite(corrF)) else float(corrF)),
            "corr_Frob_heads_used": int(used),
            "monotonicity_violation_rate": float(mono_bad),
            "train_seconds": float(secs),
            **({"notes": notes} if notes else {})
        }

    bn_note = {
        "batch_no_update": "BN uses batch stats at test without updating buffers (eval for non-BN).",
        "recalc_eval":     "BN running stats recomputed on train, then eval() used at test.",
        "off":             "BN eval() used at test with existing running stats."
    }[args.bn_eval_mode]

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

    sn_res = summarize(sn_title, sn_params, sn_train_mse, sn_rmse_mean, sn_per_head,
                       sn_corrF, sn_corr_used, sn_secs, sn_mono_bad,
                       notes=bn_note + " SN timing includes warm-up + optional freeze. " + parity_note)
    kan_res = summarize(kan_title, kan_params, kan_train_mse, kan_rmse_mean, kan_per_head,
                        kan_corrF, kan_corr_used, kan_secs, kan_mono_bad,
                        notes=bn_note + ("; " + parity_note if parity_note else ""))

    print(f"\n=== Head-to-Head Results (DenseHeadShift: D={dataset.input_dim}, m={dataset.output_dim}) ===")
    print(json.dumps(sn_res, indent=2))
    print(json.dumps(kan_res, indent=2))

    os.makedirs(args.outdir, exist_ok=True)
    out_path = os.path.join(
        args.outdir,
        f"kan_sn_densehead_shift_D{dataset.input_dim}_m{dataset.output_dim}_seed{args.seed}.json"
    )
    with open(out_path, "w") as f:
        json.dump([sn_res, kan_res], f, indent=2)
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()