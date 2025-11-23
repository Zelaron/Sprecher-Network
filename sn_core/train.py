# sn_core/train.py

"""Training utilities for Sprecher Networks.

Evaluation policy (automatic; no flags needed):
  - If either φ or Φ is 'cubic' → canonical eval uses **batch BN** (train-style statistics, no buffer mutation).
  - Else (pure PWL)             → canonical eval uses **running BN** (true eval()).

We still compute/save BOTH eval modes (running BN, batch BN) for visibility, but
verification and the displayed “eval loss” use the canonical one. Canonical verification
is performed using the saved **snapshot** model to avoid spurious mismatches.

This file also preserves all legacy behavior for PWL runs.
"""

import copy
from contextlib import contextmanager
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm

from .config import CONFIG
from .model import SprecherMultiLayerNetwork


# ---------------------------------------------------------------------
# Scheduler (optional, off by default)
# ---------------------------------------------------------------------
class PlateauAwareCosineAnnealingLR:
    """Custom scheduler that increases learning rate when stuck in plateau."""
    def __init__(self, optimizer, base_lr, max_lr, patience=1000, threshold=1e-4):
        self.optimizer = optimizer
        self.base_lr = base_lr
        self.max_lr = max_lr
        self.patience = patience
        self.threshold = threshold
        self.best_loss = float('inf')
        self.plateau_counter = 0
        self.cycle_length = 2000
        self.current_step = 0

    def step(self, loss):
        self.current_step += 1

        # Check for plateau
        if abs(self.best_loss - loss) < self.threshold:
            self.plateau_counter += 1
        else:
            self.plateau_counter = 0
            if loss < self.best_loss:
                self.best_loss = loss

        # If in plateau, use higher learning rate
        if self.plateau_counter > self.patience:
            lr = self.max_lr
            self.plateau_counter = 0
        else:
            # Cosine annealing
            progress = (self.current_step % self.cycle_length) / self.cycle_length
            lr = self.base_lr + 0.5 * (self.max_lr - self.base_lr) * (1 + np.cos(np.pi * progress))

        # Update learning rates
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr * param_group.get('lr_scale', 1.0)

        return lr


# ---------------------------------------------------------------------
# BatchNorm helpers
# ---------------------------------------------------------------------
def recalculate_bn_stats(model, x_train, num_passes=10):
    """Recalculate BatchNorm statistics using the training data."""
    was_training = model.training

    for module in model.modules():
        if isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
            module.reset_running_stats()
            module.momentum = 0.1

    model.train()

    with torch.no_grad():
        for pass_idx in range(num_passes):
            _ = model(x_train)
            if pass_idx >= num_passes - 3:
                for module in model.modules():
                    if isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
                        module.momentum = 0.01

    if not was_training:
        model.eval()


def has_batchnorm(model):
    """Check if model contains any BatchNorm layers (1D/2D/3D)."""
    for module in model.modules():
        if isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
            return True
    return False


def set_bn_eval(model: nn.Module):
    """Put only BatchNorm layers into eval() without flipping the entire model."""
    for m in model.modules():
        if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
            m.eval()


def set_bn_train(model: nn.Module):
    """Put only BatchNorm layers back into train()."""
    for m in model.modules():
        if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
            m.train()


@contextmanager
def freeze_bn_running_stats(model: nn.Module):
    """
    Preserve all BN running statistics across the block.
    Lets you run forward passes in TRAIN mode (batch stats) without mutating running buffers.
    """
    bn_modules = []
    saved = []
    for m in model.modules():
        if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
            bn_modules.append(m)
            saved.append({
                "mean": (None if getattr(m, "running_mean", None) is None else m.running_mean.clone()),
                "var": (None if getattr(m, "running_var", None) is None else m.running_var.clone()),
                "nbt": (None if getattr(m, "num_batches_tracked", None) is None else m.num_batches_tracked.clone()),
                "momentum": getattr(m, "momentum", None),
            })
    try:
        yield
    finally:
        for m, s in zip(bn_modules, saved):
            if s["mean"] is not None and m.running_mean is not None:
                m.running_mean.copy_(s["mean"])
            if s["var"] is not None and m.running_var is not None:
                m.running_var.copy_(s["var"])
            if s["nbt"] is not None and getattr(m, "num_batches_tracked", None) is not None:
                m.num_batches_tracked.copy_(s["nbt"])
            if s["momentum"] is not None:
                m.momentum = s["momentum"]


@contextmanager
def use_batch_stats_without_updating_bn(model: nn.Module):
    """
    Force BN layers to use *batch* statistics (train behavior) while temporarily
    disabling running stats updates. Side‑effect free.
    """
    bn_modules = []
    saved_flags = []
    for m in model.modules():
        if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
            bn_modules.append(m)
            saved_flags.append((m.training, m.track_running_stats))
    try:
        for m in bn_modules:
            m.train(True)                 # use batch stats
            m.track_running_stats = False # don't update buffers
        yield
    finally:
        for m, (was_train, was_track) in zip(bn_modules, saved_flags):
            m.track_running_stats = was_track
            m.train(was_train)


@contextmanager
def evaluating(model: nn.Module):
    """Temporarily switch a model to eval() and run under torch.no_grad()."""
    was_training = model.training
    try:
        model.eval()
        with torch.no_grad():
            yield
    finally:
        model.train(was_training)


def _compute_eval_losses_both_modes(model: nn.Module, x: torch.Tensor, y: torch.Tensor):
    """
    Return (loss_eval_running, loss_eval_batch, output_eval_running, output_eval_batch).
    - loss_eval_running: eval() using BN running stats
    - loss_eval_batch:   train-style BN batch stats (no mutation)
    """
    # Eval with running stats
    with evaluating(model):
        out_run = model(x)
        loss_run = torch.mean((out_run - y) ** 2).item()

    # Eval with batch stats (train-style BN, no mutation)
    with torch.no_grad():
        with use_batch_stats_without_updating_bn(model):
            out_bat = model(x)
            loss_bat = torch.mean((out_bat - y) ** 2).item()

    return loss_run, loss_bat, out_run.detach().clone(), out_bat.detach().clone()


# ---------------------------------------------------------------------
# Main training function
# ---------------------------------------------------------------------
def train_network(
    dataset,
    architecture,
    total_epochs=4000,
    print_every=400,
    device="cpu",
    phi_knots=100,
    Phi_knots=100,
    seed=None,
    norm_type="none",
    norm_position="after",
    norm_skip_first=True,
    no_load_best=False,
    bn_recalc_on_load=False,
    residual_style=None,
    # --- Spline controls (legacy defaults preserved) ---
    phi_spline_type: str = "linear",
    Phi_spline_type: str = "linear",
    phi_spline_order: Optional[int] = None,
    Phi_spline_order: Optional[int] = None,
):
    """
    Train a Sprecher network on the given dataset.
    Returns:
        plotting_snapshot: dict with deep-copied model and data for consistent plotting
        losses: List[float] of training losses across epochs
    """
    # Use seed from config if not provided
    if seed is None:
        seed = CONFIG["seed"]

    # Set seeds
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    # Optional: set residual style override & normalize synonyms
    if residual_style is not None:
        CONFIG["residual_style"] = str(residual_style).lower()
    if CONFIG.get("residual_style", "node") in ["standard", "matrix"]:
        CONFIG["residual_style"] = "linear"
    CONFIG.setdefault("residual_style", "node")

    # Canonical eval policy (automatic)
    is_cubic_mode = (str(phi_spline_type).lower() == "cubic") or (str(Phi_spline_type).lower() == "cubic")
    canonical_eval_mode = "batch" if is_cubic_mode else "running"

    # Generate training data (same batch each epoch, consistent with original)
    n_samples = 32 if dataset.input_dim == 1 else 32 * 32
    x_train, y_train = dataset.sample(n_samples, device)

    # Compute target statistics for better initialization
    y_mean = y_train.mean(dim=0)

    # Create model
    model = SprecherMultiLayerNetwork(
        input_dim=dataset.input_dim,
        architecture=architecture,
        final_dim=dataset.output_dim,
        phi_knots=phi_knots,
        Phi_knots=Phi_knots,
        norm_type=norm_type,
        norm_position=norm_position,
        norm_skip_first=norm_skip_first,
        phi_spline_type=phi_spline_type,
        Phi_spline_type=Phi_spline_type,
        phi_spline_order=phi_spline_order,
        Phi_spline_order=Phi_spline_order,
    ).to(device)

    # Initialize output bias/scale
    with torch.no_grad():
        model.output_bias.data = y_mean.mean()
        model.output_scale.data = torch.tensor(0.1)

    # Parameter counts (summary)
    lambda_params = 0
    eta_params = 0
    spline_params = 0
    residual_params = 0
    residual_scalar_params = 0
    residual_pooling_params = 0
    residual_broadcast_params = 0
    residual_projection_params = 0
    codomain_params = 0
    norm_params = 0
    lateral_params = 0

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if "lambdas" in name:
            lambda_params += param.numel()
        elif "eta" in name:
            eta_params += param.numel()
        elif "coeffs" in name or "log_increments" in name:
            spline_params += param.numel()
        elif "residual_weight" in name:
            if CONFIG.get("use_residual_weights", True):
                residual_scalar_params += param.numel()
                residual_params += param.numel()
        elif "residual_pooling_weights" in name:
            if CONFIG.get("use_residual_weights", True):
                residual_pooling_params += param.numel()
                residual_params += param.numel()
        elif "residual_broadcast_weights" in name:
            if CONFIG.get("use_residual_weights", True):
                residual_broadcast_params += param.numel()
                residual_params += param.numel()
        elif "residual_projection" in name:
            if CONFIG.get("use_residual_weights", True):
                residual_projection_params += param.numel()
                residual_params += param.numel()
        elif "phi_codomain_params" in name:
            if CONFIG.get("train_phi_codomain", False):
                codomain_params += param.numel()
        elif "norm_layers" in name:
            norm_params += param.numel()
        elif "lateral" in name:
            if CONFIG.get("use_lateral_mixing", True):
                lateral_params += param.numel()

    output_params = 2  # output_scale and output_bias

    if CONFIG.get("train_phi_codomain", False):
        total_params = (
            lambda_params
            + eta_params
            + spline_params
            + residual_params
            + output_params
            + codomain_params
            + norm_params
            + lateral_params
        )
    else:
        total_params = (
            lambda_params
            + eta_params
            + spline_params
            + residual_params
            + output_params
            + norm_params
            + lateral_params
        )

    # Summary
    print(
        f"Dataset: {dataset} (input_dim={dataset.input_dim}, output_dim={dataset.output_dim})"
    )
    print(f"Architecture: {architecture}")
    print(f"phi knots: {phi_knots}, Phi knots: {Phi_knots}")
    print(f"Theoretical domains: {CONFIG.get('use_theoretical_domains', True)}")
    print(f"Domain safety margin: {CONFIG.get('domain_safety_margin', 0.0)}")
    print(
        f"Residual connections: {'enabled' if CONFIG.get('use_residual_weights', True) else 'disabled'}"
    )
    if CONFIG.get("use_residual_weights", True):
        print(f"  Residual style: {CONFIG.get('residual_style', 'node')}")
    if CONFIG.get("use_lateral_mixing", True):
        print(f"Lateral mixing: {CONFIG.get('lateral_mixing_type', 'cyclic')}")
    print(
        f"Spline types: phi={phi_spline_type} (order={phi_spline_order}), "
        f"Phi={Phi_spline_type} (order={Phi_spline_order})"
    )
    if norm_type != "none":
        print(
            f"Normalization: {norm_type} (position: {norm_position}, skip_first: {norm_skip_first})"
        )
    else:
        print("Normalization: disabled")
    print(
        f"Scheduler: {'PlateauAwareCosineAnnealingLR' if CONFIG.get('use_advanced_scheduler', False) else 'Adam (fixed LR)'}"
    )
    print(f"Total number of trainable parameters: {total_params}")
    print(f"  - Lambda weight VECTORS: {lambda_params}")
    print(f"  - Eta shift parameters: {eta_params}")
    print(f"  - Spline parameters: {spline_params}")
    if CONFIG.get("use_residual_weights", True) and residual_params > 0:
        print(f"  - Residual connection weights: {residual_params}")
        if residual_scalar_params > 0:
            print(f"    * Scalar weights (same dims): {residual_scalar_params}")
        if residual_pooling_params > 0:
            print(f"    * Pooling weights (d_in > d_out): {residual_pooling_params}")
        if residual_broadcast_params > 0:
            print(f"    * Broadcast weights (d_in < d_out): {residual_broadcast_params}")
        if residual_projection_params > 0:
            print(f"    * Projection matrices (d_in != d_out): {residual_projection_params}")
    if CONFIG.get("use_lateral_mixing", True) and lateral_params > 0:
        print(f"  - Lateral mixing parameters: {lateral_params}")
    print(f"  - Output scale and bias: {output_params}")
    if CONFIG.get("train_phi_codomain", False) and codomain_params > 0:
        print(f"  - Phi codomain parameters (cc, cr per block): {codomain_params}")
    if norm_params > 0:
        print(f"  - Normalization parameters: {norm_params}")

    # Setup optimizer
    if CONFIG.get("use_advanced_scheduler", False):
        params = [
            {
                "params": [p for n, p in model.named_parameters() if "phi_codomain_params" in n],
                "lr": 0.01,
                "lr_scale": 1.0,
            },
            {
                "params": [p for n, p in model.named_parameters() if "output" in n],
                "lr": 0.001,
                "lr_scale": 0.5,
            },
            {
                "params": [p for n, p in model.named_parameters() if "lateral" in n],
                "lr": 0.005,
                "lr_scale": 0.8,
            },
            {
                "params": [
                    p
                    for n, p in model.named_parameters()
                    if "phi_codomain_params" not in n and "output" not in n and "lateral" not in n
                ],
                "lr": 0.001,
                "lr_scale": 0.3,
            },
        ]
        optimizer = torch.optim.AdamW(
            params, weight_decay=CONFIG.get("weight_decay", 1e-6), betas=(0.9, 0.999)
        )
        scheduler = PlateauAwareCosineAnnealingLR(
            optimizer,
            base_lr=CONFIG.get("scheduler_base_lr", 1e-4),
            max_lr=CONFIG.get("scheduler_max_lr", 1e-2),
            patience=CONFIG.get("scheduler_patience", 500),
            threshold=CONFIG.get("scheduler_threshold", 1e-5),
        )
    else:
        if CONFIG.get("train_phi_codomain", False) or CONFIG.get("use_lateral_mixing", True):
            params = []
            if CONFIG.get("train_phi_codomain", False):
                params.append(
                    {
                        "params": [p for n, p in model.named_parameters() if "phi_codomain_params" in n],
                        "lr": 0.001,
                    }
                )
            if CONFIG.get("use_lateral_mixing", True):
                params.append(
                    {
                        "params": [p for n, p in model.named_parameters() if "lateral" in n],
                        "lr": 0.0005,
                    }
                )
            excluded = []
            if CONFIG.get("train_phi_codomain", False):
                excluded.append("phi_codomain_params")
            if CONFIG.get("use_lateral_mixing", True):
                excluded.append("lateral")
            params.append(
                {
                    "params": [
                        p
                        for n, p in model.named_parameters()
                        if not any(exc in n for exc in excluded)
                    ],
                    "lr": 0.0003,
                }
            )
        else:
            params = model.parameters()
        optimizer = torch.optim.Adam(params, weight_decay=1e-7)
        scheduler = None

    losses = []
    best_loss_train = float("inf")
    best_checkpoint = None
    max_grad_norm = CONFIG.get("max_grad_norm", 1.0)

    if CONFIG.get("track_domain_violations", False):
        model.reset_domain_violation_stats()

    has_bn = has_batchnorm(model)
    if has_bn:
        print("Model contains BatchNorm layers - will handle train/eval modes appropriately")

    model.train()

    pbar = tqdm(range(total_epochs), desc="Training Network")
    for epoch in pbar:
        # Update all domains every iteration (tight bounds)
        if CONFIG.get("use_theoretical_domains", True):
            model.update_all_domains(allow_resampling=True)

        optimizer.zero_grad()
        output = model(x_train)
        loss = torch.mean((output - y_train) ** 2)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
        optimizer.step()

        # Extra domain refresh *after* step (important for cubic)
        if CONFIG.get("use_theoretical_domains", True):
            # Cheaper refresh: no resampling, just propagate bounds
            model.update_all_domains(allow_resampling=False)

        current_lr = scheduler.step(loss.item()) if scheduler is not None else optimizer.param_groups[0]["lr"]
        losses.append(loss.item())

        # Track best by TRAIN loss (legacy default); store eval metrics alongside
        if loss.item() < best_loss_train:
            best_loss_train = loss.item()

            # Compute both eval losses deterministically
            loss_eval_running, loss_eval_batch, out_eval_running, out_eval_batch = _compute_eval_losses_both_modes(
                model, x_train, y_train
            )

            # Canonical eval selection
            if canonical_eval_mode == "batch":
                loss_eval_canonical = loss_eval_batch
                out_eval_canonical = out_eval_batch
            else:
                loss_eval_canonical = loss_eval_running
                out_eval_canonical = out_eval_running

            # Snapshot for plotting (deep copy of exact best state)
            plotting_snapshot = {
                "model": copy.deepcopy(model),
                "x_train": x_train.clone(),
                "y_train": y_train.clone(),
                "output": output.detach().clone(),               # train-mode output (reference)
                "loss": loss.item(),                             # train-mode loss
                # Canonical
                "output_eval": out_eval_canonical.clone(),
                "loss_eval": loss_eval_canonical,
                # Both modes for visibility
                "output_eval_running": out_eval_running.clone(),
                "loss_eval_running": loss_eval_running,
                "output_eval_batch": out_eval_batch.clone(),
                "loss_eval_batch": loss_eval_batch,
                "eval_mode": canonical_eval_mode,
                "epoch": epoch,
                "device": device,
            }

            # Save BN stats (for fresh-model reconstruction only)
            bn_statistics = {}
            if has_bn:
                for name, module in model.named_modules():
                    if isinstance(module, nn.BatchNorm1d):
                        bn_statistics[name] = {
                            "running_mean": module.running_mean.clone().cpu(),
                            "running_var": module.running_var.clone().cpu(),
                            "num_batches_tracked": module.num_batches_tracked.clone().cpu(),
                            "momentum": module.momentum,
                            "eps": module.eps,
                            "affine": module.affine,
                            "track_running_stats": module.track_running_stats,
                            "weight": module.weight.clone().cpu() if module.affine else None,
                            "bias": module.bias.clone().cpu() if module.affine else None,
                        }

            best_checkpoint = {
                "epoch": epoch,
                "model_state_dict": model.state_dict().copy(),
                "optimizer_state_dict": optimizer.state_dict(),
                "loss": loss.item(),                    # train-mode loss
                # Canonical
                "loss_eval": loss_eval_canonical,
                "eval_mode": canonical_eval_mode,
                # Both modes
                "loss_eval_running": loss_eval_running,
                "loss_eval_batch": loss_eval_batch,
                "has_batchnorm": has_bn,
                "bn_statistics": bn_statistics,
                "domain_states": model.get_all_domain_states(),
                "domain_ranges": model.get_domain_ranges(),
                "training_mode": model.training,
                "x_train": x_train.cpu().clone(),
                "y_train": y_train.cpu().clone(),
                "output": output.detach().cpu().clone(),                 # train-mode output
                "output_eval": out_eval_canonical.detach().cpu().clone(),
                "output_eval_running": out_eval_running.detach().cpu().clone(),
                "output_eval_batch": out_eval_batch.detach().cpu().clone(),
                "model_params": {
                    "input_dim": dataset.input_dim,
                    "architecture": architecture,
                    "final_dim": dataset.output_dim,
                    "phi_knots": phi_knots,
                    "Phi_knots": Phi_knots,
                    "norm_type": norm_type,
                    "norm_position": norm_position,
                    "norm_skip_first": norm_skip_first,
                    "phi_spline_type": phi_spline_type,
                    "Phi_spline_type": Phi_spline_type,
                    "phi_spline_order": phi_spline_order,
                    "Phi_spline_order": Phi_spline_order,
                },
                "residual_style": CONFIG.get("residual_style", "node"),
                "plotting_snapshot": plotting_snapshot,
            }

        # Progress bar fields
        pbar_dict = {
            "loss": f"{loss.item():.2e}",
            "best": f"{best_loss_train:.2e}",
        }
        if CONFIG.get("use_advanced_scheduler", False):
            pbar_dict["lr"] = f"{current_lr:.2e}"
        pbar.set_postfix(pbar_dict)

        if (epoch + 1) % print_every == 0:
            if CONFIG.get("use_advanced_scheduler", False):
                print(
                    f"Epoch {epoch+1}: Loss = {loss.item():.4e}, Best = {best_loss_train:.4e}, LR = {current_lr:.4e}"
                )
            else:
                print(f"Epoch {epoch+1}: Loss = {loss.item():.4e}, Best = {best_loss_train:.4e}")

            if CONFIG.get("debug_domains", False):
                print("\nDomain ranges:")
                for idx, layer in enumerate(model.layers):
                    print(
                        f"  Layer {idx}: phi domain=[{layer.phi.in_min:.3f}, {layer.phi.in_max:.3f}], "
                        f"Phi domain=[{layer.Phi.in_min:.3f}, {layer.Phi.in_max:.3f}]"
                    )
                print()

            if CONFIG.get("track_domain_violations", False) and (epoch + 1) % (print_every * 5) == 0:
                model.print_domain_violation_report()

    # Load best checkpoint before returning
    if best_checkpoint is not None and not no_load_best:
        print(f"\n{'='*60}")
        print("CHECKPOINT LOADING DEBUG INFO:")
        print(f"Best checkpoint from epoch: {best_checkpoint['epoch']}")
        print(f"Best checkpoint (train) loss: {best_checkpoint['loss']:.4e}")

        # Show both evals for visibility + canonical mode
        le_run = best_checkpoint.get("loss_eval_running", np.nan)
        le_bat = best_checkpoint.get("loss_eval_batch", np.nan)
        eval_mode_saved = best_checkpoint.get("eval_mode", "running")
        print(f"Canonical eval mode at save time: {eval_mode_saved}")
        if np.isfinite(le_run):
            print(f"Best checkpoint (eval running BN) loss: {le_run:.4e}")
        if np.isfinite(le_bat):
            print(f"Best checkpoint (eval batch BN)   loss: {le_bat:.4e}")

        # Prefer the snapshot for downstream plotting/eval (avoids spurious drift)
        if "plotting_snapshot" in best_checkpoint:
            print("\nUsing plotting snapshot from checkpoint - this ensures perfect restoration!")
            plotting_snapshot = best_checkpoint["plotting_snapshot"]
            print(
                f"\nSnapshot loaded from epoch {plotting_snapshot['epoch']} with train loss {plotting_snapshot['loss']:.4e}"
            )
            x_train = plotting_snapshot["x_train"].to(device)
            y_train = plotting_snapshot["y_train"].to(device)
            print(f"\n{'='*60}\n")
        else:
            print("\nWARNING: Old checkpoint format without plotting snapshot")
            # Fallback in this very rare case
            plotting_snapshot = None

        # Build a "fresh" model instance as an informational cross-check (not canonical)
        fresh_model = None
        if "model_params" in best_checkpoint:
            domain_ranges = best_checkpoint.get("domain_ranges", None)
            fresh_model = SprecherMultiLayerNetwork(
                **best_checkpoint["model_params"],
                initialize_domains=False,
                domain_ranges=domain_ranges,
            ).to(device)

            # Prime params and load
            fresh_model.train()
            with evaluating(fresh_model):
                _ = fresh_model(x_train[:5])
            fresh_model.load_state_dict(best_checkpoint["model_state_dict"])

            # Restore BN stats if saved
            if best_checkpoint.get("has_batchnorm", False) and "bn_statistics" in best_checkpoint:
                for name, module in fresh_model.named_modules():
                    if isinstance(module, nn.BatchNorm1d) and name in best_checkpoint["bn_statistics"]:
                        saved_stats = best_checkpoint["bn_statistics"][name]
                        module.running_mean.copy_(saved_stats["running_mean"])
                        module.running_var.copy_(saved_stats["running_var"])
                        module.num_batches_tracked.copy_(saved_stats["num_batches_tracked"])
                        if module.affine and saved_stats.get("weight") is not None:
                            module.weight.data.copy_(saved_stats["weight"])
                            module.bias.data.copy_(saved_stats["bias"])

            # Restore domain states
            if "domain_states" in best_checkpoint:
                fresh_model.set_all_domain_states(best_checkpoint["domain_states"])

            # Optional BN recalc (off by default; usually unnecessary for canonical batch eval)
            if bn_recalc_on_load and best_checkpoint.get("has_batchnorm", False):
                _x_bn = best_checkpoint.get("x_train", x_train.detach().cpu()).to(device)
                recalculate_bn_stats(fresh_model, _x_bn, num_passes=CONFIG.get("bn_recalc_passes", 10))

        # Keep same train/eval mode as during best epoch for fresh model
        if fresh_model is not None:
            if best_checkpoint.get("training_mode", True):
                fresh_model.train()
            else:
                fresh_model.eval()

        # ---------------- Canonical verification (snapshot) ----------------
        # Use the exact saved snapshot model; this is the authoritative check.
        snap = best_checkpoint.get("plotting_snapshot", None)
        if snap is not None:
            snap_model = copy.deepcopy(snap["model"]).to(device)
            snap_x = snap["x_train"].to(device)
            snap_y = snap["y_train"].to(device)
            saved_canonical = snap.get("loss_eval", float("nan"))
            saved_mode = snap.get("eval_mode", eval_mode_saved)

            if saved_mode == "batch":
                with torch.no_grad():
                    with use_batch_stats_without_updating_bn(snap_model):
                        snap_out = snap_model(snap_x)
                        snap_loss_now = torch.mean((snap_out - snap_y) ** 2).item()
            else:
                with evaluating(snap_model):
                    snap_out = snap_model(snap_x)
                    snap_loss_now = torch.mean((snap_out - snap_y) ** 2).item()

            print("Snapshot canonical verification:")
            print(f"  Saved canonical eval loss: {saved_canonical:.4e}")
            print(f"  Curr  canonical eval loss: {snap_loss_now:.4e}")

            # Tight check (snapshot should match extremely closely)
            snap_ok = abs(snap_loss_now - saved_canonical) <= 1e-8
            if snap_ok:
                print("[OK] Snapshot canonical verification passed.")
            else:
                print("\n" + "=" * 60)
                print("WARNING: Snapshot canonical verification mismatch (unexpected).")
                print(f"  |Δ|: {abs(snap_loss_now - saved_canonical):.4e}  (tol=1e-8)")
                print("=" * 60 + "\n")

        # ---------------- Informational fresh-model metrics (non-canonical) ----------------
        if fresh_model is not None:
            with evaluating(fresh_model):
                current_out_run = fresh_model(x_train)
                current_loss_run = torch.mean((current_out_run - y_train) ** 2).item()
            with torch.no_grad():
                with use_batch_stats_without_updating_bn(fresh_model):
                    current_out_bat = fresh_model(x_train)
                    current_loss_bat = torch.mean((current_out_bat - y_train) ** 2).item()

            print("\nFresh-model (informational) metrics:")
            if np.isfinite(le_run):
                print(f"  Saved eval (running BN): {le_run:.4e}")
                print(f"  Curr  eval (running BN): {current_loss_run:.4e}")
            if np.isfinite(le_bat):
                print(f"  Saved eval (batch BN):   {le_bat:.4e}")
                print(f"  Curr  eval (batch BN):   {current_loss_bat:.4e}")
            print("[Note] Fresh-model values are informational only; canonical checks use the snapshot.")

        # Return the snapshot as authoritative for plotting & downstream use
        plotting_snapshot = best_checkpoint["plotting_snapshot"]

    elif no_load_best:
        print("\nSkipping best model loading (--no_load_best flag set)")
        print(f"Using final model state with loss: {losses[-1]:.4e}")
        print(f"Best train loss during training was: {best_loss_train:.4e}")
        model.train()

        loss_eval_running, loss_eval_batch, out_eval_running, out_eval_batch = _compute_eval_losses_both_modes(
            model, x_train, y_train
        )
        if canonical_eval_mode == "batch":
            loss_eval_canonical = loss_eval_batch
            out_eval_canonical = out_eval_batch
        else:
            loss_eval_canonical = loss_eval_running
            out_eval_canonical = out_eval_running

        plotting_snapshot = {
            "model": copy.deepcopy(model),
            "x_train": x_train.clone(),
            "y_train": y_train.clone(),
            "output": model(x_train).detach().clone(),
            "loss": torch.mean((model(x_train) - y_train) ** 2).item(),
            "output_eval": out_eval_canonical.clone(),
            "loss_eval": loss_eval_canonical,
            "output_eval_running": out_eval_running.clone(),
            "loss_eval_running": loss_eval_running,
            "output_eval_batch": out_eval_batch.clone(),
            "loss_eval_batch": loss_eval_batch,
            "eval_mode": canonical_eval_mode,
            "epoch": total_epochs - 1,
            "device": device,
        }

    # Final debug prints and snapshot fallback
    print(f"\nDEBUG: Model is in {'training' if model.training else 'eval'} mode for final operations")

    print("\nDEBUG: Final model output (first 5):")
    with evaluating(model):
        test_out = model(x_train[:5])
        print(f"Output: {test_out.cpu().numpy().flatten()[:5]}")

    if CONFIG.get("track_domain_violations", False):
        print("\nFinal domain violation report:")
        model.print_domain_violation_report()

    print("\nFinal parameters:")
    for idx, layer in enumerate(model.layers, start=1):
        print(f"Block {idx}: eta = {layer.eta.item():.6f}")
        print(f"Block {idx}: lambdas shape = {tuple(layer.lambdas.shape)}")
        print("Block {idx}: lambdas =")
        print(layer.lambdas.detach().cpu().numpy())
        if CONFIG.get("use_residual_weights", True):
            if hasattr(layer, "residual_weight") and layer.residual_weight is not None:
                print(f"Block {idx}: residual_weight = {layer.residual_weight.item():.6f}")
            elif hasattr(layer, "residual_projection") and layer.residual_projection is not None:
                W = layer.residual_projection.detach().cpu().numpy()
                print(f"Block {idx}: residual_projection shape = {W.shape}")
                r_preview = min(4, W.shape[0])
                c_preview = min(4, W.shape[1])
                print(f"Block {idx}: residual_projection preview (top-left):")
                print(W[:r_preview, :c_preview])
            elif hasattr(layer, "residual_pooling_weights") and layer.residual_pooling_weights is not None:
                print(
                    f"Block {idx}: residual_pooling_weights shape = "
                    f"{tuple(layer.residual_pooling_weights.shape)}"
                )
                print("Block {idx}: residual_pooling_weights =")
                print(layer.residual_pooling_weights.detach().cpu().numpy())
                print(f"Block {idx}: pooling assignment = {layer.pooling_assignment.cpu().numpy()}")
                print(f"Block {idx}: pooling counts = {layer.pooling_counts.cpu().numpy()}")
            elif hasattr(layer, "residual_broadcast_weights") and layer.residual_broadcast_weights is not None:
                print(
                    f"Block {idx}: residual_broadcast_weights shape = "
                    f"{tuple(layer.residual_broadcast_weights.shape)}"
                )
                print("Block {idx}: residual_broadcast_weights =")
                print(layer.residual_broadcast_weights.detach().cpu().numpy())
                print(f"Block {idx}: broadcast sources = {layer.broadcast_sources.cpu().numpy()}")

        if CONFIG.get("use_lateral_mixing", True):
            if hasattr(layer, "lateral_scale") and layer.lateral_scale is not None:
                print(f"Block {idx}: lateral_scale = {layer.lateral_scale.item():.6f}")
                if CONFIG.get("lateral_mixing_type", "cyclic") == "bidirectional":
                    print("Block {idx}: lateral_weights_forward =")
                    print(layer.lateral_weights_forward.detach().cpu().numpy())
                    print("Block {idx}: lateral_weights_backward =")
                    print(layer.lateral_weights_backward.detach().cpu().numpy())
                else:
                    print("Block {idx}: lateral_weights =")
                    print(layer.lateral_weights.detach().cpu().numpy())

        if CONFIG.get("train_phi_codomain", False):
            if hasattr(layer, "phi_codomain_params") and layer.phi_codomain_params is not None:
                print(
                    f"Block {idx}: Phi codomain center = {layer.phi_codomain_params.cc.item():.6f}"
                )
                print(
                    f"Block {idx}: Phi codomain radius = {layer.phi_codomain_params.cr.item():.6f}"
                )

        if hasattr(layer, "input_range") and layer.input_range is not None:
            print(f"Block {idx}: input_range = {layer.input_range}")
        if hasattr(layer, "output_range") and layer.output_range is not None:
            print(f"Block {idx}: output_range = {layer.output_range}")
        print()

    print(f"Final (best train) loss: {best_loss_train:.4e}")

    if "plotting_snapshot" not in locals():
        print("\nCreating plotting snapshot from current model state.")
        loss_eval_running, loss_eval_batch, out_eval_running, out_eval_batch = _compute_eval_losses_both_modes(
            model, x_train, y_train
        )
        if canonical_eval_mode == "batch":
            loss_eval_canonical = loss_eval_batch
            out_eval_canonical = out_eval_batch
        else:
            loss_eval_canonical = loss_eval_running
            out_eval_canonical = out_eval_running

        plotting_snapshot = {
            "model": copy.deepcopy(model),
            "x_train": x_train.clone(),
            "y_train": y_train.clone(),
            "output": model(x_train).detach().clone(),
            "loss": torch.mean((model(x_train) - y_train) ** 2).item(),
            "output_eval": out_eval_canonical.clone(),
            "loss_eval": loss_eval_canonical,
            "output_eval_running": out_eval_running.clone(),
            "loss_eval_running": loss_eval_running,
            "output_eval_batch": out_eval_batch.clone(),
            "loss_eval_batch": loss_eval_batch,
            "eval_mode": canonical_eval_mode,
            "epoch": total_epochs - 1,
            "device": device,
        }
        print(f"Snapshot created. Canonical eval ({canonical_eval_mode}) loss: {loss_eval_canonical:.4e}")

    return plotting_snapshot, losses