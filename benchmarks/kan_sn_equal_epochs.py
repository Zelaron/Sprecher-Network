# benchmarks/kan_sn_equal_epochs.py
"""
KAN vs SN — Equal-Epoch, Param-Parity Benchmark (train-loop time only)

Key fairness choices:
  • Same epochs (optimizer steps) for both models.
  • KAN basis count chosen to match (<=) SN's parameter count.
  • KAN trained on EXACT same (x_train, y_train) produced by SN.
  • BN test semantics default to 'batch_no_update' (matches training).
  • SN uses no_load_best=True to avoid SN-only debug/restore overhead in timing.
  • Optional SN warm-up with domain updates, then freeze, but total steps == --epochs.

Datasets:
  • Any dataset registered in sn_core.data via --dataset
  • Or pass --dataset mqsi to use the monotone quantile single-index task from your MQSI bench.
"""

import argparse, os, json, time, math
import numpy as np
import torch
import torch.nn as nn

from sn_core import (
    get_dataset, CONFIG, train_network, has_batchnorm
)

# Reuse parity-bench utilities to keep training/eval identical
from benchmarks.kan_sn_parity_bench import (
    set_global_seeds,
    count_params,
    train_kan,
    choose_kan_basis_for_parity,
    evaluate_with_progress,
)

# MQSI dataset & monotonicity metric (optional when --dataset mqsi)
try:
    from benchmarks.kan_sn_monoindex_bench import MQSIDataset, monotonicity_violation_rate
    HAS_MQSI = True
except Exception:
    HAS_MQSI = False

def build_dataset(name: str, args):
    if name.lower() != "mqsi":
        return get_dataset(name)
    if not HAS_MQSI:
        raise RuntimeError("MQSI requested but benchmarks.kan_sn_monoindex_bench not available.")
    return MQSIDataset(
        d_in=args.dims,
        n_heads=args.n_quantiles,
        seed=args.seed,
        beta=args.beta
    )

def main():
    p = argparse.ArgumentParser()
    # General
    p.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"])
    p.add_argument("--seed", type=int, default=45)
    p.add_argument("--epochs", type=int, default=4000)
    p.add_argument("--dataset", type=str, default="toy_4d_to_5d")
    p.add_argument("--n_test", type=int, default=20000)
    p.add_argument("--eval_batch_size", type=int, default=8192)
    p.add_argument("--outdir", type=str, default="benchmarks/results")

    # MQSI-specific (only used if --dataset mqsi)
    p.add_argument("--dims", type=int, default=20)
    p.add_argument("--n_quantiles", type=int, default=9)
    p.add_argument("--beta", type=float, default=0.8)

    # SN config
    p.add_argument("--sn_arch", type=str, default="15,15")
    p.add_argument("--sn_phi_knots", type=int, default=60)
    p.add_argument("--sn_Phi_knots", type=int, default=60)
    p.add_argument("--sn_norm_type", type=str, default="batch", choices=["none", "batch", "layer"])
    p.add_argument("--sn_norm_position", type=str, default="after", choices=["before", "after"])
    p.add_argument("--sn_norm_first", action="store_true")
    p.add_argument("--sn_norm_skip_first", action="store_true", default=True)
    p.add_argument("--sn_no_residual", action="store_true")
    p.add_argument("--sn_residual_style", type=str, default="linear",
                   choices=["node", "linear", "standard", "matrix"])
    p.add_argument("--sn_no_lateral", action="store_true")
    p.add_argument("--sn_freeze_domains_after", type=int, default=0,
                   help="Warm-up epochs with domain updates, then freeze (0 = disabled).")
    p.add_argument("--sn_domain_margin", type=float, default=0.0)

    # KAN config
    p.add_argument("--kan_arch", type=str, default="4,4")
    p.add_argument("--kan_degree", type=int, default=3, choices=[2, 3])
    p.add_argument("--kan_bn_type", type=str, default="batch", choices=["none", "batch"])
    p.add_argument("--kan_bn_position", type=str, default="after", choices=["before", "after"])
    p.add_argument("--kan_bn_skip_first", action="store_true", default=True)
    p.add_argument("--kan_outside", type=str, default="linear", choices=["linear", "clamp"])
    p.add_argument("--kan_residual_type", type=str, default="linear", choices=["silu", "linear", "none"])
    p.add_argument("--kan_lr", type=float, default=1e-3)
    p.add_argument("--kan_wd", type=float, default=1e-6)
    p.add_argument("--kan_impl", type=str, default="fast", choices=["fast", "slow"])

    # Parity + BN-eval
    p.add_argument("--equalize_params", action="store_true", default=True)
    p.add_argument("--prefer_leq", action="store_true", default=True)
    p.add_argument("--bn_eval_mode", type=str, default="batch_no_update",
                   choices=["batch_no_update", "recalc_eval", "off"])
    p.add_argument("--bn_recalc_passes", type=int, default=10)

    args = p.parse_args()
    device = ("cuda" if (args.device == "auto" and torch.cuda.is_available())
              else ("cpu" if args.device == "auto" else args.device))
    set_global_seeds(args.seed)

    # Build dataset
    dataset = build_dataset(args.dataset, args)
    input_dim = dataset.input_dim
    final_dim = dataset.output_dim

    # Fixed test set
    torch.manual_seed(args.seed + 2042)
    with torch.no_grad():
        x_test, y_test = dataset.sample(args.n_test, device=device)

    # ---------------- SN knobs ----------------
    if args.sn_no_residual:
        CONFIG['use_residual_weights'] = False
    else:
        style = (args.sn_residual_style or "linear").lower()
        if style in ("standard", "matrix"):
            style = "linear"
        CONFIG['residual_style'] = style

    CONFIG['use_lateral_mixing'] = not args.sn_no_lateral
    CONFIG['domain_safety_margin'] = float(args.sn_domain_margin)
    sn_norm_skip = False if args.sn_norm_first else args.sn_norm_skip_first
    sn_arch = [int(x) for x in args.sn_arch.split(",")] if args.sn_arch.strip() else []

    # ---------------- SN training ----------------
    total_epochs = int(args.epochs)
    warmup_epochs = max(0, min(args.sn_freeze_domains_after, total_epochs))
    rest_epochs   = total_epochs - warmup_epochs

    # Phase A: warm-up WITH domain updates OR full train if warm-up is 0.
    # IMPORTANT: we pass no_load_best=True so timing excludes SN-only checkpoint/restore debug.
    t0_sn = time.time()
    plotting_snapshot, _ = train_network(
        dataset=dataset,
        architecture=sn_arch,
        total_epochs=(warmup_epochs if warmup_epochs > 0 else total_epochs),
        print_every=max(1, (warmup_epochs if warmup_epochs > 0 else total_epochs) // 10),
        device=device,
        phi_knots=args.sn_phi_knots,
        Phi_knots=args.sn_Phi_knots,
        seed=args.seed,
        norm_type=args.sn_norm_type,
        norm_position=args.sn_norm_position,
        norm_skip_first=sn_norm_skip,
        no_load_best=True,          # fairness: use final model like KAN; avoid debug overhead
        bn_recalc_on_load=False,
        residual_style=(None if args.sn_no_residual else CONFIG.get('residual_style', 'linear'))
    )
    sn_secs = time.time() - t0_sn

    sn_model = plotting_snapshot["model"].to(device)
    x_train  = plotting_snapshot["x_train"].to(device)
    y_train  = plotting_snapshot["y_train"].to(device)

    # Phase B: continue WITHOUT domain updates for the remaining steps (if any)
    if warmup_epochs > 0 and rest_epochs > 0:
        from benchmarks.kan_sn_parity_bench import continue_train_sn_no_domain_updates
        t1 = time.time()
        sn_model, sn_train_mse, _ = continue_train_sn_no_domain_updates(
            model=sn_model, x_train=x_train, y_train=y_train, epochs=rest_epochs,
            device=device, seed=args.seed
        )
        sn_secs += (time.time() - t1)
    else:
        with torch.no_grad():
            sn_train_mse = float(torch.mean((sn_model(x_train) - y_train) ** 2).cpu())

    sn_params = count_params(sn_model)

    # ---------------- Choose K for KAN ----------------
    kan_arch = [int(x) for x in args.kan_arch.split(",")] if args.kan_arch.strip() else []
    if args.equalize_params:
        kan_K, est_cnt = choose_kan_basis_for_parity(
            target_params=sn_params, arch=kan_arch, input_dim=input_dim, final_dim=final_dim,
            degree=args.kan_degree, bn_type=args.kan_bn_type, bn_position=args.kan_bn_position,
            bn_skip_first=args.kan_bn_skip_first, residual_type=args.kan_residual_type, prefer_leq=args.prefer_leq
        )
        parity_note = f"[ParamMatch] SN params = {sn_params}. KAN K = {kan_K} (est. {est_cnt})."
        print(parity_note)
    else:
        kan_K = max(args.kan_degree + 1, 60)
        parity_note = ""

    # ---------------- Train KAN on EXACT same train set ----------------
    t0_kan = time.time()
    kan_model, kan_train_mse, kan_secs = train_kan(
        x_train=x_train, y_train=y_train,
        input_dim=input_dim, final_dim=final_dim,
        arch=kan_arch, n_basis=kan_K, degree=args.kan_degree,
        device=device, epochs=total_epochs, seed=args.seed,
        lr=args.kan_lr, wd=args.kan_wd,
        bn_type=args.kan_bn_type, bn_position=args.kan_bn_position, bn_skip_first=args.kan_bn_skip_first,
        outside=args.kan_outside, residual_type=args.kan_residual_type, impl=args.kan_impl
    )
    # kan_secs already measured inside train_kan

    kan_params = count_params(kan_model)

    # ---------------- Evaluation (BN mode selectable) ----------------
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

    # Optional monotonicity metric (meaningful for mqsi or any ordered-head target)
    mono_sn = mono_kan = None
    if final_dim > 1 and HAS_MQSI and args.dataset.lower() == "mqsi":
        with torch.no_grad():
            bs = max(1, int(args.eval_batch_size))
            pred_sn, pred_kan = [], []
            for s in range(0, x_test.shape[0], bs):
                e = min(x_test.shape[0], s + bs)
                pred_sn.append(sn_model(x_test[s:e]).cpu())
                pred_kan.append(kan_model(x_test[s:e]).cpu())
            mono_sn  = float(monotonicity_violation_rate(torch.cat(pred_sn,  dim=0)))
            mono_kan = float(monotonicity_violation_rate(torch.cat(pred_kan, dim=0)))

    # ---------------- Summaries ----------------
    def pack(name, params, train_mse, mean_rmse, per_head, corrF, used, secs, notes="", mono=None):
        out = {
            "model": name,
            "params": int(params),
            "train_mse": float(train_mse),
            "test_rmse_mean": float(mean_rmse),
            "test_rmse_per_head": [float(x) for x in per_head],
            "corr_Frob_error": (None if (corrF is None or not math.isfinite(corrF)) else float(corrF)),
            "corr_Frob_heads_used": int(used),
            "train_seconds": float(secs),
        }
        if mono is not None:
            out["monotonicity_violation_rate"] = float(mono)
        if notes:
            out["notes"] = notes
        return out

    bn_note = {
        "batch_no_update": "BN uses batch stats at test without updating buffers (eval for non-BN).",
        "recalc_eval":     "BN running stats recomputed on train, then eval() used at test.",
        "off":             "BN eval() used at test with existing running stats."
    }[args.bn_eval_mode]

    sn_title = (
        f"SN(arch={sn_arch}, phi_knots={args.sn_phi_knots}, Phi_knots={args.sn_Phi_knots}, "
        f"norm={args.sn_norm_type}/{args.sn_norm_position}/"
        f"{'include_first' if not sn_norm_skip else 'skip_first'}, "
        f"residuals={'off' if args.sn_no_residual else 'on(style='+CONFIG.get('residual_style','node')+')'}, "
        f"lateral={'off' if args.sn_no_lateral else 'on'}, "
        f"domains={'warmup+freeze' if warmup_epochs>0 else 'updated_all_epochs'}, "
        f"no_load_best=True)"
    )
    kan_title = (
        f"KAN[{args.kan_impl}](arch={kan_arch}, K={kan_K}, degree={args.kan_degree}, "
        f"BN={args.kan_bn_type}/{args.kan_bn_position}/"
        f"{'skip_first' if args.kan_bn_skip_first else 'include_first'}, "
        f"outside={args.kan_outside}, residual={args.kan_residual_type})"
    )

    sn_res  = pack(sn_title,  sn_params, sn_train_mse, sn_rmse_mean,  sn_per_head,  sn_corrF,  sn_corr_used,  sn_secs,
                   notes=bn_note + " SN timing excludes SN-only checkpoint/restore work. " + parity_note,
                   mono=mono_sn)
    kan_res = pack(kan_title, kan_params, kan_train_mse, kan_rmse_mean, kan_per_head, kan_corrF, kan_corr_used, kan_secs,
                   notes=bn_note + ("; " + parity_note if parity_note else ""),
                   mono=mono_kan)

    print("\n=== Head-to-Head Results (Equal Epochs) ===")
    print(json.dumps(sn_res,  indent=2))
    print(json.dumps(kan_res, indent=2))

    os.makedirs(args.outdir, exist_ok=True)
    tag = (f"mqsi_D{input_dim}_m{final_dim}" if (HAS_MQSI and args.dataset.lower()=="mqsi")
           else f"{args.dataset}")
    out_path = os.path.join(args.outdir, f"kan_sn_equal_{tag}_seed{args.seed}.json")
    with open(out_path, "w") as f:
        json.dump([sn_res, kan_res], f, indent=2)
    print(f"\nSaved: {out_path}")

if __name__ == "__main__":
    main()