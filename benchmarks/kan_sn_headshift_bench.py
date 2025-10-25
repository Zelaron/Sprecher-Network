# benchmarks/kan_sn_headshift_bench.py
"""
KAN vs SN — Head‑Shifted Family Benchmark (HSF)

Task
-----
Multi-output regression where all heads are *translated* versions of the same
latent function:
    y_j(x) = tanh( β * Σ_i w_i * sigmoid(a_i * (x_i + Δ*j - c_i)) ),   j=0..m-1.
That is, "same shape, shifted across head index j".

Why this favors SN (but remains fair)
-------------------------------------
• Structural match: SN blocks compute s_q = Σ_i λ_i φ(x_i + η q) + (const) and then apply Φ,
  i.e., they explicitly model *per-head translations* in their inner spline via η⋅q
  and add a global +q term (Q_VALUES_FACTOR). This is exactly the pattern in this dataset
  (shared function shifted along head index), so SN can share parameters across heads and
  fit efficiently,,.

• KANs do not share a head-translation mechanism: they use clamped-uniform B-splines
  with fixed knot placement; there is no native "shift-by-q" parameter tied across heads.
  With matched parameter counts, KANs must spend capacity to re-learn each head's shift,
  which is harder under tight K parity.

• Fairness preserved: identical training inputs/targets, same BN semantics at test
  (default 'batch_no_update'), same *linear* residuals and outside='linear' for KAN, matched
  parameter counts (we pick K to match SN params), equal epochs, and we train KAN on
  the *exact* (x_train, y_train) generated during SN warm-up:contentReference[oaicite:9]{index=9},:contentReference[oaicite:10]{index=10}.

Metrics
-------
  - Mean RMSE and per-head RMSE on a large i.i.d. test set (uniform over [0,1]^D)
  - Frobenius distance between target/pred correlation matrices (parity bench metric)
  - Monotonicity-violation rate across heads (lower is better)

Implementation notes
--------------------
• Uses SN training loop with domain updates enabled every iteration (as in your core),
  then (optionally) freezes domains and continues training identically, see CONFIG-driven
  update_all_domains and the two-phase warmup+freeze scheme,.
• KAN side uses your "fast" implementation and clamped-uniform knots.
• BN test semantics and parity/parameter matching logic are imported from your parity bench:contentReference[oaicite:14]{index=14},:contentReference[oaicite:15]{index=15}.
"""

import argparse, math, os, json, time
import numpy as np
import torch
import torch.nn as nn

from sn_core import CONFIG, train_network  # SN trainer (with domain updates each iter)
from benchmarks.kan_sn_parity_bench import (
    set_global_seeds,
    count_params,
    train_kan,
    choose_kan_basis_for_parity,
    evaluate_with_progress,  # returns (rmse_per_head, mean_rmse, corrF, heads_used):contentReference[oaicite:17]{index=17}
)

# -------------------------------
# Dataset: Head-Shifted Family (HSF)
# -------------------------------

class HeadShiftDataset:
    """
    y_j(x) = tanh( β * Σ_i w_i * sigmoid(a_i * (x_i + Δ*j - c_i)) )

    • x ∈ [0,1]^D (train/test i.i.d. uniform)
    • j ∈ {0,...,m-1} heads (monotone across j if Δ>0 and tanh monotone)
    • Parameters (w,a,c,Δ,β) are fixed at dataset creation.
    """
    def __init__(self, d_in=64, n_heads=16, seed=0, delta=0.05,
                 a_min=3.0, a_max=9.0, beta=1.2):
        self._d = int(d_in)
        self._m = int(n_heads)
        self.delta = float(delta)
        self.beta = float(beta)

        rs = np.random.RandomState(seed)
        # Positive slopes and centers (cover interior so the shift crosses informative region)
        self.a = torch.tensor(rs.uniform(a_min, a_max, size=self._d), dtype=torch.float32)
        self.c = torch.tensor(rs.uniform(0.15, 0.85, size=self._d), dtype=torch.float32)
        # Nonnegative weights that sum to 1 (stabilizes range)
        w_raw = rs.rand(self._d); w_raw = w_raw / (w_raw.sum() + 1e-12)
        self.w = torch.tensor(w_raw, dtype=torch.float32)

    @property
    def input_dim(self):  return self._d
    @property
    def output_dim(self): return self._m

    def _h(self, x_plus_shift):
        # x_plus_shift: [N, D, m]
        # sigmoid(a_i * (x_i + shift - c_i)) applied per-dim, broadcast over heads
        a = self.a.view(1, -1, 1)           # [1, D, 1]
        c = self.c.view(1, -1, 1)           # [1, D, 1]
        return torch.sigmoid(a * (x_plus_shift - c))

    def evaluate(self, x):
        # x: [N, D] in [0,1]
        N = x.shape[0]
        device = x.device
        j = torch.arange(self._m, device=device, dtype=x.dtype).view(1, 1, -1)  # [1,1,m]
        x_plus = x.unsqueeze(2) + self.delta * j                                # [N,D,m]
        h = self._h(x_plus)                                                     # [N,D,m]
        # Weighted sum over D → [N, m]
        y = torch.tensordot(h, self.w.to(device), dims=([1],[0]))               # [N,m]
        y = torch.tanh(self.beta * y)                                           # [N,m]
        return y

    def sample(self, n, device="cpu"):
        x = torch.rand(n, self._d, device=device)      # uniform in [0,1]^D
        y = self.evaluate(x)
        return x, y


# -------------------------------
# Extra metric: monotonicity across heads (lower is better)
# -------------------------------

def monotonicity_violation_rate(y_pred):
    """
    Fraction of samples where the predicted heads are NOT non-decreasing in j:
    any(y_pred[n, j] > y_pred[n, j+1] for some j).
    """
    with torch.no_grad():
        diffs = y_pred[:, 1:] - y_pred[:, :-1]
        bad = (diffs < 0).any(dim=1).float().mean().item()
    return float(bad)


# -------------------------------
# SN warm-up then freeze helper (identical hyper treatment as parity bench)
# -------------------------------

def continue_train_sn_no_domain_updates(model, x_train, y_train, epochs, device, seed,
                                        lr_other=3e-4, lr_codomain=1e-3, wd=1e-7, clip=1.0):
    # Mirrors the helper used in the parity & monoindex scripts
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
    p.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"])
    p.add_argument("--seed", type=int, default=45)
    p.add_argument("--epochs", type=int, default=4000)
    p.add_argument("--n_test", type=int, default=40000)

    # Dataset (HSF)
    p.add_argument("--dims", type=int, default=64)
    p.add_argument("--n_heads", type=int, default=16)
    p.add_argument("--delta", type=float, default=0.05)
    p.add_argument("--beta", type=float, default=1.2)

    # SN config
    p.add_argument("--sn_arch", type=str, default="24,24")
    p.add_argument("--sn_phi_knots", type=int, default=80)
    p.add_argument("--sn_Phi_knots", type=int, default=80)
    p.add_argument("--sn_norm_type", type=str, default="batch", choices=["none", "batch", "layer"])
    p.add_argument("--sn_norm_position", type=str, default="after", choices=["before", "after"])
    p.add_argument("--sn_norm_skip_first", action="store_true", default=True)
    p.add_argument("--sn_norm_first", action="store_true")
    p.add_argument("--sn_no_residual", action="store_true")
    p.add_argument("--sn_residual_style", type=str, default="linear", choices=["node","linear","standard","matrix"])
    p.add_argument("--sn_no_lateral", action="store_true")
    p.add_argument("--sn_freeze_domains_after", type=int, default=400)  # warmup with updates
    p.add_argument("--sn_domain_margin", type=float, default=0.01)

    # KAN config
    p.add_argument("--kan_arch", type=str, default="5,8")
    p.add_argument("--kan_degree", type=int, default=3, choices=[2,3])
    p.add_argument("--kan_bn_type", type=str, default="batch", choices=["none","batch"])
    p.add_argument("--kan_bn_position", type=str, default="after", choices=["before","after"])
    p.add_argument("--kan_bn_skip_first", action="store_true")
    p.add_argument("--kan_outside", type=str, default="linear", choices=["linear","clamp"])
    p.add_argument("--kan_residual_type", type=str, default="linear", choices=["silu","linear","none"])
    p.add_argument("--kan_lr", type=float, default=1e-3)
    p.add_argument("--kan_wd", type=float, default=1e-6)
    p.add_argument("--kan_impl", type=str, default="fast", choices=["fast","slow"])
    p.add_argument("--kan_K", type=int, default=60)

    # Parity / eval
    p.add_argument("--equalize_params", action="store_true")
    p.add_argument("--prefer_leq", action="store_true")
    p.add_argument("--bn_eval_mode", type=str, default="batch_no_update",
                   choices=["batch_no_update","recalc_eval","off"])
    p.add_argument("--bn_recalc_passes", type=int, default=10)
    p.add_argument("--eval_batch_size", type=int, default=8192)

    p.add_argument("--outdir", type=str, default="benchmarks/results")
    args = p.parse_args()

    # ------------------------------
    # Device
    # ------------------------------
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    # ------------------------------
    # Build dataset + fixed test set
    # ------------------------------
    set_global_seeds(args.seed)
    dataset = HeadShiftDataset(d_in=args.dims, n_heads=args.n_heads, seed=args.seed,
                               delta=args.delta, beta=args.beta)

    torch.manual_seed(args.seed + 2027)
    with torch.no_grad():
        x_test, y_test = dataset.sample(args.n_test, device=device)

    # ------------------------------
    # Configure SN fairness knobs
    # ------------------------------
    # Residuals: enable + use linear projection style unless explicitly disabled
    if args.sn_no_residual:
        CONFIG['use_residual_weights'] = False
    else:
        CONFIG['use_residual_weights'] = True
        CONFIG['residual_style'] = "linear"  # standard projection matrix residuals

    # Lateral mixing toggled off unless user wants it
    if args.sn_no_lateral:
        CONFIG['use_lateral_mixing'] = False

    # Normalization semantics
    CONFIG['use_normalization'] = (args.sn_norm_type != "none")
    CONFIG['norm_type'] = args.sn_norm_type
    CONFIG['norm_position'] = args.sn_norm_position
    CONFIG['norm_skip_first'] = (False if args.sn_norm_first else args.sn_norm_skip_first)
    CONFIG['domain_safety_margin'] = float(args.sn_domain_margin)

    sn_arch = [int(x) for x in args.sn_arch.split(",")] if args.sn_arch.strip() else []

    # ------------------------------
    # SN training: warm-up WITH domain updates, then (optionally) freeze
    # ------------------------------
    total_epochs = int(args.epochs)
    warmup_epochs = max(0, min(args.sn_freeze_domains_after, total_epochs))
    rest_epochs = total_epochs - warmup_epochs

    # Phase 1: warm-up with dynamic domain updates (SN trainer samples internally)
    t0_sn = time.time()
    residual_style_override = None if args.sn_no_residual else CONFIG.get('residual_style', 'linear')
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
        norm_skip_first=(False if args.sn_norm_first else args.sn_norm_skip_first),
        no_load_best=False,
        bn_recalc_on_load=False,
        residual_style=residual_style_override
    )
    sn_secs = time.time() - t0_sn

    sn_model = plotting_snapshot["model"].to(device)
    x_train  = plotting_snapshot["x_train"].to(device)
    y_train  = plotting_snapshot["y_train"].to(device)

    # Phase 2: continue training with domains frozen (identical loop as in parity bench)
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
    # Choose K for KAN to match SN parameter count
    # ------------------------------
    kan_arch = [int(x) for x in args.kan_arch.split(",")] if args.kan_arch.strip() else []
    if args.equalize_params:
        chosen_K, est_cnt = choose_kan_basis_for_parity(
            target_params=sn_params, arch=kan_arch, input_dim=dataset.input_dim, final_dim=dataset.output_dim,
            degree=args.kan_degree, bn_type=args.kan_bn_type, bn_position=args.kan_bn_position,
            bn_skip_first=args.kan_bn_skip_first, residual_type=args.kan_residual_type,
            prefer_leq=args.prefer_leq
        )
        kan_K = chosen_K
        parity_note = f"[ParamMatch] SN params = {sn_params}. KAN K = {kan_K} (est. {est_cnt})."
        print(parity_note)
    else:
        kan_K = args.kan_K
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
    # Evaluation (BN semantics aligned with training by default)
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

    sn_title = (
        f"SN(arch={sn_arch}, phi_knots={args.sn_phi_knots}, Phi_knots={args.sn_Phi_knots}, "
        f"norm={args.sn_norm_type}/{args.sn_norm_position}/"
        f"{'skip_first' if (not args.sn_norm_first and args.sn_norm_skip_first) else 'include_first'}, "
        f"residuals={'off' if args.sn_no_residual else 'on(style='+CONFIG.get('residual_style','linear')+')'}, "
        f"lateral={'off' if args.sn_no_lateral else 'on'}, "
        f"domains={'warmup+freeze' if args.sn_freeze_domains_after>0 else 'updated'})"
    )
    kan_title = (
        f"KAN[{args.kan_impl}](arch={kan_arch}, K={kan_K}, degree={args.kan_degree}, "
        f"BN={args.kan_bn_type}/{args.kan_bn_position}/"
        f"{'skip_first' if args.kan_bn_skip_first else 'include_first'}, "
        f"outside={args.kan_outside}, residual={args.kan_residual_type})"
    )

    bn_note = {
        "batch_no_update": "BN uses batch stats at test without updating buffers (eval for non-BN).",
        "recalc_eval":     "BN running stats recomputed on train, then eval() used at test.",
        "off":             "BN eval() used at test with existing running stats."
    }[args.bn_eval_mode]

    sn_res = summarize(sn_title, sn_params, sn_train_mse, sn_rmse_mean, sn_per_head,
                       sn_corrF, sn_corr_used, sn_secs, sn_mono_bad,
                       notes=bn_note + " SN timing includes warm-up + optional freeze. " + parity_note)
    kan_res = summarize(kan_title, kan_params, kan_train_mse, kan_rmse_mean, kan_per_head,
                        kan_corrF, kan_corr_used, kan_secs, kan_mono_bad,
                        notes=bn_note + ("; " + parity_note if parity_note else ""))

    print(f"\n=== Head-Shifted Family Results (D={dataset.input_dim}, m={dataset.output_dim}) ===")
    for r in (sn_res, kan_res):
        print(json.dumps(r, indent=2))

    os.makedirs(args.outdir, exist_ok=True)
    out_path = os.path.join(args.outdir, f"kan_sn_headshift_D{dataset.input_dim}_m{dataset.output_dim}_seed{args.seed}.json")
    with open(out_path, "w") as f:
        json.dump([sn_res, kan_res], f, indent=2)
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()