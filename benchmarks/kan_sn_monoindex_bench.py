# benchmarks/kan_sn_monoindex_bench.py
"""
KAN vs SN — Monotone Quantile Single-Index Benchmark

Task
-----
Multi-output regression with quantile-indexed heads:
  y_j(x) = g( μ(x) + σ(x) * z_j ),  j=1..m
where:
  - μ(x) and σ(x)>0 are monotone in each input coordinate (sum of monotone transforms).
  - z_j are fixed quantile locations (e.g., Normal inverse CDF at τ_j).
  - g is a smooth, bounded, monotone squashing (tanh).

Why this favors SN (but remains fair)
-------------------------------------
• Structural match: SN blocks compute s_q = Σ_i λ_i φ(x_i + η q) + ... and pass through Φ.
  The learned q-shift (η·q) + monotone φ + outer Φ naturally represent families of outputs
  that are smooth shifts of a common latent index. KANs can approximate this, but do not
  share this inductive bias and use fixed clamped-uniform knots.

• Fairness preserved: parameter parity (KAN K picked to match SN params), same BN semantics
  at test (“batch_no_update” by default), same residual style (linear) and outside behavior,
  identical training inputs/targets, and equal epochs.

Metrics
-------
We report:
  - Mean RMSE and per-head RMSE on a large i.i.d. test set
  - Frobenius distance between target/pred correlation matrices (as in parity bench)
  - Monotonicity-violation rate across heads (fraction of samples where y_pred[:,j] > y_pred[:,j+1])

Implementation notes
--------------------
We import KAN machinery, parameter-equalization, and evaluation (BN-safe) directly from
`benchmarks.kan_sn_parity_bench.py` to keep training loops identical to your current parity code.
"""

import argparse, math, os, json, time
import numpy as np
import torch
import torch.nn as nn

# SN core (training + config)
from sn_core import (
    CONFIG, train_network, has_batchnorm
)

# Reuse parity bench utilities to keep procedures identical
from benchmarks.kan_sn_parity_bench import (
    set_global_seeds,
    count_params,
    train_kan,
    choose_kan_basis_for_parity,
    evaluate_with_progress,
)

# -------------------------------
# Dataset: Monotone Quantile Single-Index (MQSI)
# -------------------------------

class MQSIDataset:
    """
    Monotone Quantile Single-Index dataset.

    Inputs:  x ∈ [0,1]^D
    Heads:   m outputs indexed by fixed quantiles τ_j (strictly increasing)
    Target:  y_j(x) = tanh( β * ( μ(x) + σ(x) * z_j ) )

    μ(x)  = Σ_i w_i * h_i(x_i)  +  small positive pairwise terms
    σ(x)  = s0 + Σ_i v_i * h_i(x_i)   (kept > 0)
    h_i   = sigmoid(a_i * (x_i - c_i))  (monotone ↑)

    This guarantees monotonicity in each coordinate and across j (because z_j is increasing).
    """
    def __init__(self, d_in=20, n_heads=9, seed=0, quantiles=None,
                 a_min=2.0, a_max=8.0, s0=0.15, v_scale=0.25, pair_frac=0.15, pair_scale=0.08, beta=0.8):
        self._d = int(d_in)
        self._m = int(n_heads)
        rs = np.random.RandomState(seed)

        # Per-dim monotone transform parameters
        self.a = torch.tensor(rs.uniform(a_min, a_max, size=self._d), dtype=torch.float32)  # slopes
        self.c = torch.tensor(rs.uniform(0.2, 0.8, size=self._d), dtype=torch.float32)      # centers

        # Positive weights
        w_raw = rs.rand(self._d); w_raw = w_raw / (w_raw.sum() + 1e-12)
        v_raw = rs.rand(self._d); v_raw = v_raw / (v_raw.sum() + 1e-12)
        self.w = torch.tensor(w_raw, dtype=torch.float32)
        self.v = torch.tensor(v_raw * v_scale, dtype=torch.float32)

        # Pairwise positive interactions (still monotone)
        n_pairs = int(round(pair_frac * self._d))
        pairs = []
        if n_pairs > 0:
            idx = rs.choice(self._d, size=(n_pairs, 2), replace=False)
            for i, j in idx:
                if i != j:
                    pairs.append((int(i), int(j)))
        self.pairs = pairs
        self.pair_scale = float(pair_scale)

        # Scale floor
        self.s0 = float(s0)
        self.beta = float(beta)

        # Quantiles τ_j and z_j = Φ^{-1}(τ_j)
        if quantiles is None:
            taus = np.linspace(0.1, 0.9, self._m)
        else:
            taus = np.asarray(quantiles, dtype=float)
            assert len(taus) == self._m and np.all(np.diff(taus) > 0), "quantiles must be strictly increasing."
        self.taus = torch.tensor(taus, dtype=torch.float32)
        # approximate Φ^{-1} using erfinv
        self.z = torch.sqrt(torch.tensor(2.0)) * torch.erfinv(2 * self.taus - 1)

    @property
    def input_dim(self):
        return self._d

    @property
    def output_dim(self):
        return self._m

    def _h(self, x):
        # x: [N, D], per-dim monotone sigmoid
        # h_i(x_i) = sigmoid(a_i*(x_i - c_i))
        return torch.sigmoid((x - self.c) * self.a)

    def _mu(self, h):
        # h: [N,D]
        mu = torch.matmul(h, self.w)  # [N]
        if len(self.pairs) > 0:
            # add small positive pairwise terms (monotone)
            for (i, j) in self.pairs:
                mu = mu + self.pair_scale * (h[:, i] * h[:, j])
        return mu  # [N]

    def _sigma(self, h):
        # σ(x) = s0 + Σ_i v_i h_i(x_i)  (kept > 0)
        sig = self.s0 + torch.matmul(h, self.v).clamp_min(1e-5)
        return sig

    def evaluate(self, x):
        # x: [N, D], returns [N, m]
        h = self._h(x)
        mu = self._mu(h)                      # [N]
        sig = self._sigma(h)                  # [N]
        # broadcast over heads
        z = self.z.to(x.device).view(1, -1)   # [1,m]
        y = mu.unsqueeze(1) + sig.unsqueeze(1) * z  # [N,m]
        y = torch.tanh(self.beta * y)
        return y

    def sample(self, n, device="cpu"):
        x = torch.rand(n, self._d, device=device)  # uniform over [0,1]^D
        y = self.evaluate(x)
        return x, y


# -------------------------------
# Extra metrics
# -------------------------------

def monotonicity_violation_rate(y_pred):
    """
    Fraction of samples where the predicted heads are NOT non-decreasing in j:
       any(y_pred[n, j] > y_pred[n, j+1] for some j)
    """
    with torch.no_grad():
        diffs = y_pred[:, 1:] - y_pred[:, :-1]
        bad = (diffs < 0).any(dim=1).float().mean().item()
    return float(bad)


# -------------------------------
# SN warm-up + optional freeze (identical to parity bench behavior)
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

    # MQSI dataset hyperparameters
    p.add_argument("--dims", type=int, default=20, help="Input dimension D")
    p.add_argument("--n_quantiles", type=int, default=9, help="Number of output heads m (monotone in j)")
    p.add_argument("--test_size", type=int, default=50000, help="Test set size")
    p.add_argument("--beta", type=float, default=0.8, help="Strength of tanh squashing")

    # SN config (mirrors parity bench flags)
    p.add_argument("--sn_arch", type=str, default="24,24")
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

    # KAN config (mirrors parity bench)
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

    # Parity + BN-eval mode (reuse parity bench choices)
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

    # Build dataset
    dataset = MQSIDataset(d_in=args.dims, n_heads=args.n_quantiles, seed=args.seed, beta=args.beta)

    # Fixed test set
    torch.manual_seed(args.seed + 2026)
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
    # SN training: warm-up with domain updates, then freeze (optional)
    # ------------------------------
    total_epochs = int(args.epochs)
    warmup_epochs = max(0, min(args.sn_freeze_domains_after, total_epochs))
    rest_epochs = total_epochs - warmup_epochs

    # Phase 1: warm-up WITH domain updates (standard SN trainer uses dataset.sample internally)
    from sn_core.data import Dataset  # type hint only; not strictly required

    # Wrap our custom generator so SN's trainer sees the right interface
    class _WrapDataset:
        def __init__(self, base):
            self.base = base
        @property
        def input_dim(self): return self.base.input_dim
        @property
        def output_dim(self): return self.base.output_dim
        def sample(self, n, device="cpu"):
            return self.base.sample(n, device)

    wrapped = _WrapDataset(dataset)

    residual_style_override = None if args.sn_no_residual else CONFIG.get('residual_style', 'linear')
    t0_sn = time.time()
    plotting_snapshot, _ = train_network(
        dataset=wrapped,
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

    print(f"\n=== Head-to-Head Results (MQSI: D={dataset.input_dim}, m={dataset.output_dim}) ===")
    print(json.dumps(sn_res, indent=2))
    print(json.dumps(kan_res, indent=2))

    os.makedirs(args.outdir, exist_ok=True)
    out_path = os.path.join(
        args.outdir,
        f"kan_sn_monoindex_D{dataset.input_dim}_m{dataset.output_dim}_seed{args.seed}.json"
    )
    with open(out_path, "w") as f:
        json.dump([sn_res, kan_res], f, indent=2)
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()