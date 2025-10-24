# benchmarks/kan_sn_ridgecomp_bench.py
"""
KAN vs SN — Deep Ridge Composition (Outer-of-Sum) Benchmark

Task
-----
We generate a smooth regression task whose ground truth is a *two-block*
Sprecher-style composition:

  y(x) = Phi2( W2^T · u1(x) + b2 ),           where
  u1(x) = Phi1( A1^T · h(x) ),                h applied elementwise and monotone.

Concretely,
  h_i(t)   = sigmoid(a_i · (t - c_i))                           (monotone ↑)
  Phi1(s)  = tanh(beta1 · s)  (applied channel-wise)
  Phi2(s)  = sin(beta2 · s)   (applied per output head)

Why SN should win (but fairly)
------------------------------
• Structural match: Each SN block computes ϕ (monotone per-dim) → sum → Φ (general)
  and supports residuals, domains, BN exactly as in your code. Two blocks reproduce
  Phi2(W2^T Phi1(A1^T h(x))). The inner ϕ is monotone in the SN implementation,
  matching our h_i; Φ is general, matching tanh/sin here.  [Sprecher block: monotone inner ϕ, general Φ].  # noqa
• KAN limitation at the second mix: The KAN layers apply univariate splines on edges
  and sum them, with optional linear/silu residuals; there is **no post-sum Φ** at
  the layer output. With two KAN layers, realizing Phi2(sum_h …) is structurally
  harder than for SN at fixed depth/params.  [FastKANLayer sums transformed inputs].  # noqa
• Fairness knobs preserved from the benches:
  - Parameter parity: choose KAN K to match the SN parameter count (<= preferred).
  - Same BN semantics at test (`--bn_eval_mode batch_no_update` by default).
  - Same optim, epochs, residual style (‘linear’), outside behavior, and eval batching.
  - Identical (x_train, y_train) for both models.

Metrics
-------
We report mean test RMSE, per-head RMSE, and correlation Frobenius distance (as in
the parity bench). Results are printed and saved as JSON.

Implementation notes
--------------------
• We reuse training/eval utilities directly from the parity bench to ensure identical
  procedures (param equalization, BN-safe evaluation with progress bars).  # noqa
"""

import argparse, os, json, time, math
import numpy as np
import torch
import torch.nn as nn

# SN core (trainer + config switches)
from sn_core import CONFIG
# Reuse parity-bench utilities (identical behaviors)
from benchmarks.kan_sn_parity_bench import (
    set_global_seeds,
    count_params,
    train_kan,
    choose_kan_basis_for_parity,
    evaluate_with_progress,
    continue_train_sn_no_domain_updates,  # keep identical warmup/freeze behavior
)
from sn_core import train_network  # SN trainer with domain updates etc.

# -------------------------------
# Dataset: Deep Ridge Composition
# -------------------------------

class DeepRidgeCompositionDataset:
    """
    Two-block Sprecher-style composition:
      x in [0,1]^D
      h_i(t) = sigmoid(a_i * (t - c_i))  (monotone)
      u1     = tanh(beta1 * (A1^T h))
      y      = sin(beta2 * (W2^T u1 + b2))
    """
    def __init__(self, d_in=40, d_hidden=16, d_out=8, seed=0,
                 a_min=2.0, a_max=8.0, beta1=1.0, beta2=1.5):
        rs = np.random.RandomState(seed)
        self._D = int(d_in)
        self._H = int(d_hidden)
        self._M = int(d_out)

        # Per-dim monotone transform params
        self.a = torch.tensor(rs.uniform(a_min, a_max, size=self._D), dtype=torch.float32)
        self.c = torch.tensor(rs.uniform(0.2, 0.8, size=self._D), dtype=torch.float32)

        # Mixing/heads (fan-in He style scaling)
        A1 = rs.randn(self._D, self._H) / np.sqrt(self._D)
        W2 = rs.randn(self._H, self._M) / np.sqrt(self._H)
        b2 = rs.randn(self._M) * 0.25
        self.A1 = torch.tensor(A1, dtype=torch.float32)
        self.W2 = torch.tensor(W2, dtype=torch.float32)
        self.b2 = torch.tensor(b2, dtype=torch.float32)

        self.beta1 = float(beta1)
        self.beta2 = float(beta2)

    @property
    def input_dim(self):  return self._D

    @property
    def output_dim(self): return self._M

    @torch.no_grad()
    def _h(self, x):
        # x: [N, D] in [0,1]
        return torch.sigmoid(self.a.view(1, -1) * (x - self.c.view(1, -1)))

    @torch.no_grad()
    def evaluate(self, x):
        # h -> u1 -> y
        h = self._h(x)                                  # [N, D]
        s1 = h @ self.A1                                # [N, H]
        u1 = torch.tanh(self.beta1 * s1)                # [N, H]
        s2 = u1 @ self.W2 + self.b2.view(1, -1)         # [N, M]
        y  = torch.sin(self.beta2 * s2)                 # [N, M]
        return y

    @torch.no_grad()
    def sample(self, n, device="cpu"):
        x = torch.rand(int(n), self._D, device=device)
        y = self.evaluate(x.to(torch.float32)).to(device)
        return x, y


# -------------------------------
# SN warm-up + optional freeze (identical to parity bench)
# -------------------------------

def train_sn_then_freeze(dataset, *, sn_arch, device, seed, epochs,
                         phi_knots, Phi_knots, norm_type, norm_position,
                         norm_skip_first, sn_freeze_after):
    total_epochs = int(epochs)
    warmup_epochs = max(0, min(sn_freeze_after, total_epochs))
    rest_epochs = total_epochs - warmup_epochs

    # Phase 1: warm-up WITH domain updates (SN trainer draws its own (x_train,y_train))
    residual_style_override = CONFIG.get('residual_style', 'linear')
    t0 = time.time()
    plotting_snapshot, _ = train_network(
        dataset=dataset,
        architecture=sn_arch,
        total_epochs=warmup_epochs if warmup_epochs > 0 else total_epochs,
        print_every=max(1, (warmup_epochs if warmup_epochs > 0 else total_epochs) // 10),
        device=device,
        phi_knots=phi_knots,
        Phi_knots=Phi_knots,
        seed=seed,
        norm_type=norm_type,
        norm_position=norm_position,
        norm_skip_first=norm_skip_first,
        no_load_best=False,
        bn_recalc_on_load=False,
        residual_style=residual_style_override
    )
    secs = time.time() - t0

    sn_model = plotting_snapshot["model"].to(device)
    x_train  = plotting_snapshot["x_train"].to(device)
    y_train  = plotting_snapshot["y_train"].to(device)

    if rest_epochs > 0:
        t1 = time.time()
        sn_model, sn_train_mse, secs2 = continue_train_sn_no_domain_updates(
            model=sn_model, x_train=x_train, y_train=y_train, epochs=rest_epochs,
            device=device, seed=seed
        )
        secs += (time.time() - t1)
    else:
        with torch.no_grad():
            sn_train_mse = float(torch.mean((sn_model(x_train) - y_train) ** 2).cpu())

    return sn_model, sn_train_mse, secs, x_train, y_train


# -------------------------------
# Main
# -------------------------------

def main():
    p = argparse.ArgumentParser()
    # Device/seed/epochs
    p.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"])
    p.add_argument("--seed", type=int, default=45)
    p.add_argument("--epochs", type=int, default=4000)

    # Dataset
    p.add_argument("--dims", type=int, default=40)
    p.add_argument("--hidden", type=int, default=16)
    p.add_argument("--outputs", type=int, default=8)
    p.add_argument("--test_size", type=int, default=50000)
    p.add_argument("--beta1", type=float, default=1.0)
    p.add_argument("--beta2", type=float, default=1.5)

    # SN config (mirrors parity bench)
    p.add_argument("--sn_arch", type=str, default="16")
    p.add_argument("--sn_phi_knots", type=int, default=80)
    p.add_argument("--sn_Phi_knots", type=int, default=80)
    p.add_argument("--sn_norm_type", type=str, default="batch", choices=["none", "batch", "layer"])
    p.add_argument("--sn_norm_position", type=str, default="after", choices=["before", "after"])
    p.add_argument("--sn_norm_skip_first", action="store_true", default=True)
    p.add_argument("--sn_norm_first", action="store_true")
    p.add_argument("--sn_residual_style", type=str, default="linear",
                   choices=["node", "linear", "standard", "matrix"])
    p.add_argument("--sn_no_lateral", action="store_true", default=True)
    p.add_argument("--sn_freeze_domains_after", type=int, default=300)
    p.add_argument("--sn_domain_margin", type=float, default=0.01)

    # KAN config (reuse parity bench knobs)
    p.add_argument("--kan_arch", type=str, default="16")
    p.add_argument("--kan_degree", type=int, default=3, choices=[2, 3])
    p.add_argument("--kan_bn_type", type=str, default="batch", choices=["none", "batch"])
    p.add_argument("--kan_bn_position", type=str, default="after", choices=["before", "after"])
    p.add_argument("--kan_bn_skip_first", action="store_true", default=True)
    p.add_argument("--kan_outside", type=str, default="linear", choices=["linear", "clamp"])
    p.add_argument("--kan_residual_type", type=str, default="linear", choices=["silu", "linear", "none"])
    p.add_argument("--kan_lr", type=float, default=1e-3)
    p.add_argument("--kan_wd", type=float, default=1e-6)
    p.add_argument("--kan_impl", type=str, default="fast", choices=["fast", "slow"])

    # Parity + BN-eval mode (same choices as parity bench)
    p.add_argument("--equalize_params", action="store_true", default=True)
    p.add_argument("--prefer_leq", action="store_true", default=True)
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

    # ------------------------------
    # Build dataset & fixed test set
    # ------------------------------
    data = DeepRidgeCompositionDataset(
        d_in=args.dims, d_hidden=args.hidden, d_out=args.outputs,
        seed=args.seed, beta1=args.beta1, beta2=args.beta2
    )
    torch.manual_seed(args.seed + 2031)
    with torch.no_grad():
        x_test, y_test = data.sample(args.test_size, device=device)

    # ------------------------------
    # Configure SN fairness knobs
    # ------------------------------
    # residual style ('standard'/'matrix' synonyms map to 'linear' in core)
    style = (args.sn_residual_style or "linear").lower()
    if style in ("standard", "matrix"):
        style = "linear"
    CONFIG['residual_style'] = style
    CONFIG['use_residual_weights'] = True
    if args.sn_no_lateral:
        CONFIG['use_lateral_mixing'] = False
    CONFIG['domain_safety_margin'] = float(args.sn_domain_margin)
    sn_norm_skip = False if args.sn_norm_first else args.sn_norm_skip_first

    # Architecture
    sn_arch = [int(x) for x in args.sn_arch.split(",")] if args.sn_arch.strip() else []

    # Wrap dataset so the SN trainer sees the expected interface (same trick as monoindex bench)
    class _WrapDataset:
        def __init__(self, base): self.base = base
        @property
        def input_dim(self):  return self.base.input_dim
        @property
        def output_dim(self): return self.base.output_dim
        def sample(self, n, device="cpu"): return self.base.sample(n, device)

    wrapped = _WrapDataset(data)

    # ------------------------------
    # Train SN: warm-up w/ domain updates, then (optionally) freeze
    # ------------------------------
    sn_model, sn_train_mse, sn_secs, x_train, y_train = train_sn_then_freeze(
        wrapped,
        sn_arch=sn_arch, device=device, seed=args.seed, epochs=args.epochs,
        phi_knots=args.sn_phi_knots, Phi_knots=args.sn_Phi_knots,
        norm_type=args.sn_norm_type, norm_position=args.sn_norm_position,
        norm_skip_first=sn_norm_skip,
        sn_freeze_after=args.sn_freeze_domains_after
    )
    sn_params = count_params(sn_model)

    # ------------------------------
    # Match KAN params by choosing K
    # ------------------------------
    kan_arch = [int(x) for x in args.kan_arch.split(",")] if args.kan_arch.strip() else []
    if args.equalize_params:
        chosen_K, est_cnt = choose_kan_basis_for_parity(
            target_params=sn_params, arch=kan_arch, input_dim=data.input_dim, final_dim=data.output_dim,
            degree=args.kan_degree, bn_type=args.kan_bn_type, bn_position=args.kan_bn_position,
            bn_skip_first=args.kan_bn_skip_first, residual_type=args.kan_residual_type,
            prefer_leq=args.prefer_leq
        )
        kan_K = chosen_K
        parity_note = f"[ParamMatch] SN params = {sn_params}. KAN K = {kan_K} (est. {est_cnt})."
        print(parity_note)
    else:
        kan_K = max(3, args.kan_degree + 1)
        parity_note = ""

    # ------------------------------
    # Train KAN on EXACT same (x_train, y_train)
    # ------------------------------
    kan_model, kan_train_mse, kan_secs = train_kan(
        x_train=x_train, y_train=y_train,
        input_dim=data.input_dim, final_dim=data.output_dim,
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
        if notes: out["notes"] = notes
        return out

    sn_title = (
        f"SN(arch={sn_arch}, phi_knots={args.sn_phi_knots}, Phi_knots={args.sn_Phi_knots}, "
        f"norm={args.sn_norm_type}/{args.sn_norm_position}/"
        f"{'skip_first' if sn_norm_skip else 'include_first'}, "
        f"residuals=on(style={CONFIG.get('residual_style','linear')}), "
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

    sn_result  = pretty(sn_title,  sn_params,  sn_train_mse,  sn_rmse_mean,  sn_per_head,  sn_corrF,  sn_corr_used,  sn_secs,
                        notes=bn_note + " SN timing includes warm-up + optional freeze. " + parity_note)
    kan_result = pretty(kan_title, kan_params, kan_train_mse, kan_rmse_mean, kan_per_head, kan_corrF, kan_corr_used, kan_secs,
                        notes=bn_note + ("; " + parity_note if parity_note else ""))

    print(f"\n=== Head-to-Head Results (DRC: D={data.input_dim}, H={args.hidden}, M={data.output_dim}) ===")
    print(json.dumps(sn_result,  indent=2))
    print(json.dumps(kan_result, indent=2))

    os.makedirs(args.outdir, exist_ok=True)
    out_path = os.path.join(args.outdir, f"kan_sn_ridgecomp_D{data.input_dim}_H{args.hidden}_M{data.output_dim}_seed{args.seed}.json")
    with open(out_path, "w") as f:
        json.dump([sn_result, kan_result], f, indent=2)
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()