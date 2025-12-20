"""Command-line interface for Sprecher Network experiments."""

import os
import argparse
import torch
import numpy as np
import matplotlib

from sn_core import (
    train_network,
    get_dataset,
    plot_results,
    plot_loss_curve,
    export_parameters,
    parse_param_types,
)
# Import the BN helper to evaluate with batch stats without mutating BN buffers
from sn_core.train import use_batch_stats_without_updating_bn


# ----------------------------
# Argument parsing
# ----------------------------
def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Train Sprecher Networks")

    # Core experiment setup
    parser.add_argument(
        "--dataset",
        type=str,
        default="toy_1d_poly",
        help="Dataset name (default: toy_1d_poly)",
    )
    parser.add_argument(
        "--arch",
        type=str,
        default="15,15",
        help="Architecture as comma-separated values (default: 15,15)",
    )
    parser.add_argument(
        "--phi_knots",
        type=int,
        default=100,
        help="Number of knots for phi splines (default: 100)",
    )
    parser.add_argument(
        "--Phi_knots",
        type=int,
        default=100,
        help="Number of knots for Phi splines (default: 100)",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=4000,
        help="Number of training epochs (default: 4000)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=45,
        help="Random seed (default: 45)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cpu", "cuda"],
        help="Device: auto, cpu, or cuda (default: auto)",
    )

    # Plotting control
    parser.add_argument("--save_plots", action="store_true", help="Save plots to files")
    parser.add_argument(
        "--no_show",
        action="store_true",
        help="Don't show plots (useful for batch runs)",
    )

    # Domain debugging / violation tracking
    parser.add_argument(
        "--debug_domains",
        action="store_true",
        help="Enable domain debugging output",
    )
    parser.add_argument(
        "--track_violations",
        action="store_true",
        help="Track domain violations during training",
    )

    # Normalization arguments
    parser.add_argument(
        "--norm_type",
        type=str,
        choices=["none", "batch", "layer"],
        help="Type of normalization to use (default: from CONFIG)",
    )
    parser.add_argument(
        "--norm_position",
        type=str,
        default="after",
        choices=["before", "after"],
        help="Position of normalization relative to blocks (default: after)",
    )
    parser.add_argument(
        "--norm_skip_first",
        action="store_true",
        default=True,
        help="Skip normalization for the first block (default: True)",
    )
    parser.add_argument(
        "--norm_first",
        action="store_true",
        help="Enable normalization for the first block (overrides norm_skip_first)",
    )

    # Checkpoint / BN statistics behavior
    parser.add_argument(
        "--no_load_best",
        action="store_true",
        help="Don't load best checkpoint at end of training (for debugging)",
    )
    parser.add_argument(
        "--bn_recalc_on_load",
        action="store_true",
        help="Recalculate BatchNorm statistics when loading best checkpoint "
             "(you shouldn't need this for cubic; canonical eval is handled automatically).",
    )

    # Feature control
    parser.add_argument(
        "--no_residual",
        action="store_true",
        help="Disable residual connections (default: enabled)",
    )
    parser.add_argument(
        "--residual_style",
        type=str,
        default=None,
        choices=["node", "linear", "standard", "matrix"],
        help=(
            "Residual style: 'node' (original) or 'linear' (standard). "
            "Aliases: 'standard'/'matrix' -> 'linear'."
        ),
    )
    parser.add_argument(
        "--no_norm",
        action="store_true",
        help="Disable normalization (default: enabled with batch norm)",
    )
    parser.add_argument(
        "--use_advanced_scheduler",
        action="store_true",
        help="Use PlateauAwareCosineAnnealingLR scheduler (default: disabled)",
    )

    # Lateral mixing
    parser.add_argument(
        "--no_lateral",
        action="store_true",
        help="Disable lateral mixing connections (default: enabled)",
    )
    parser.add_argument(
        "--lateral_type",
        type=str,
        default=None,
        choices=["cyclic", "bidirectional"],
        help="Type of lateral mixing (default: from CONFIG)",
    )

    # Parameter export
    parser.add_argument(
        "--export_params",
        nargs="?",
        const="all",
        default=None,
        help=(
            "Export parameters to file. Options: all, or comma-separated: "
            "lambda,eta,spline,residual,codomain,norm,output,lateral"
        ),
    )

    # Memory optimization
    parser.add_argument(
        "--low_memory_mode",
        action="store_true",
        help="Use memory-efficient computation (O(B × max(d_in, d_out)) memory).",
    )
    parser.add_argument(
        "--memory_debug",
        action="store_true",
        help="Print CUDA memory usage statistics during forward pass.",
    )

    # -------- NEW: spline types / orders --------
    parser.add_argument(
        "--spline_type",
        type=str,
        default=None,
        choices=["pwl", "linear", "cubic"],
        help="Convenience switch to set both φ and Φ spline types. "
             "Use --phi_spline_type/--Phi_spline_type to override individually.",
    )
    parser.add_argument(
        "--phi_spline_type",
        type=str,
        default=None,
        choices=["pwl", "linear", "cubic"],
        help="Spline type for φ (default: from --spline_type or project default).",
    )
    parser.add_argument(
        "--Phi_spline_type",
        type=str,
        default=None,
        choices=["pwl", "linear", "cubic"],
        help="Spline type for Φ (default: from --spline_type or project default).",
    )
    parser.add_argument(
        "--phi_spline_order",
        type=int,
        default=None,
        help="Optional polynomial order for φ for higher-order splines (ignored for cubic Hermite).",
    )
    parser.add_argument(
        "--Phi_spline_order",
        type=int,
        default=None,
        help="Optional polynomial order for Φ for higher-order splines (ignored for cubic Hermite).",
    )

    return parser.parse_args()


# ----------------------------
# Helpers
# ----------------------------
def _parse_architecture(arch_str: str):
    """Parse architecture string into a list of ints (robust to spaces/empties)."""
    if not arch_str:
        return []
    parts = [p.strip() for p in arch_str.split(",")]
    parts = [p for p in parts if p]
    try:
        return [int(p) for p in parts]
    except ValueError:
        raise ValueError(
            f"Invalid --arch value '{arch_str}'. Use comma-separated integers, e.g. '15,15'."
        )


def get_config_suffix(args, CONFIG):
    """Build filename suffix for non-default configurations (compact tags)."""
    parts = []

    # Normalization tags
    if args.no_norm or (hasattr(args, "norm_type") and args.norm_type == "none"):
        parts.append("NoNorm")
    elif hasattr(args, "norm_type") and args.norm_type and args.norm_type not in ["none", "batch"]:
        parts.append(f"Norm{args.norm_type.capitalize()}")

    # Residuals
    if not CONFIG.get("use_residual_weights", True):
        parts.append("NoResidual")
    else:
        style = CONFIG.get("residual_style", "node")
        if style not in (None, "node"):
            parts.append("ResLinear")

    # Lateral
    if not CONFIG.get("use_lateral_mixing", True):
        parts.append("NoLateral")
    elif CONFIG.get("lateral_mixing_type", "cyclic") != "cyclic":
        parts.append(f"Lateral{CONFIG['lateral_mixing_type'].capitalize()}")

    # Scheduler
    if CONFIG.get("use_advanced_scheduler", False):
        parts.append("AdvScheduler")

    # Memory
    if CONFIG.get("low_memory_mode", False):
        parts.append("LowMem")

    # Spline (use effective resolved values; see main())
    phi_t = getattr(args, "phi_spline_type_effective", None)
    Phi_t = getattr(args, "Phi_spline_type_effective", None)
    if phi_t is not None and Phi_t is not None:
        if phi_t == Phi_t:
            parts.append(f"Spline{phi_t.capitalize()}")
        else:
            parts.append(f"SplinePhi{phi_t.capitalize()}-Phi{Phi_t.capitalize()}")

    return "-" + "-".join(parts) if parts else ""


def profile_memory_usage(model, x_sample):
    """Profile memory usage of forward pass (CUDA only)."""
    import torch.cuda

    if x_sample.is_cuda:
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
        start_memory = torch.cuda.memory_allocated()

        with torch.no_grad():
            _ = model(x_sample)

        torch.cuda.synchronize()
        peak_memory = torch.cuda.max_memory_allocated()
        end_memory = torch.cuda.memory_allocated()

        print(f"Memory usage:")
        print(f"  Start: {start_memory / 1024**2:.2f} MB")
        print(f"  Peak:  {peak_memory / 1024**2:.2f} MB")
        print(f"  End:   {end_memory / 1024**2:.2f} MB")
        print(f"  Delta: {(peak_memory - start_memory) / 1024**2:.2f} MB")
    else:
        print("Memory profiling only available for CUDA tensors")


def verify_memory_efficient_mode(device="cpu"):
    """Verify that memory-efficient mode produces identical results."""
    from sn_core.model import SprecherLayerBlock
    from sn_core.config import CONFIG

    print("\nVerifying memory-efficient mode...")
    torch.manual_seed(42)

    layer = SprecherLayerBlock(d_in=100, d_out=200, layer_num=0).to(device)
    x = torch.randn(32, 100, device=device)

    CONFIG["low_memory_mode"] = False
    with torch.no_grad():
        output_original = layer._forward_original(x, None)

    CONFIG["low_memory_mode"] = True
    with torch.no_grad():
        output_efficient = layer._forward_memory_efficient(x, None)

    max_diff = torch.abs(output_original - output_efficient).max().item()
    mean_diff = torch.abs(output_original - output_efficient).mean().item()

    print(f"Maximum difference: {max_diff:.2e}")
    print(f"Mean difference: {mean_diff:.2e}")

    if max_diff < 1e-6:
        print("✓ Memory-efficient mode produces mathematically identical results")
    else:
        print(f"✗ WARNING: Outputs differ by {max_diff:.2e}")

    if device == "cuda" and torch.cuda.is_available():
        print("\nMemory comparison:")
        CONFIG["low_memory_mode"] = False
        print("Original mode:")
        profile_memory_usage(layer, x)

        CONFIG["low_memory_mode"] = True
        print("\nMemory-efficient mode:")
        profile_memory_usage(layer, x)

    CONFIG["low_memory_mode"] = False
    return max_diff < 1e-6


# ----------------------------
# Main
# ----------------------------
def main():
    """Main training script."""
    args = parse_args()

    # If --no_show is used (e.g., in batch runs), force the 'Agg' backend upfront.
    if args.no_show:
        matplotlib.use("Agg")
        print("Using non-interactive 'Agg' backend for plotting (as requested by --no_show).")

    import matplotlib.pyplot as plt

    # Parse architecture & dataset
    architecture = _parse_architecture(args.arch)
    dataset = get_dataset(args.dataset)

    # Device
    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    elif args.device == "cuda" and not torch.cuda.is_available():
        print("WARNING: --device cuda was requested but CUDA is not available; falling back to CPU.")
        device = "cpu"
    else:
        device = args.device

    # CONFIG adjustments
    from sn_core.config import CONFIG

    if args.debug_domains:
        CONFIG["debug_domains"] = True
    if args.track_violations:
        CONFIG["track_domain_violations"] = True
        print("Domain violation tracking enabled.")

    # Memory flags
    if args.low_memory_mode:
        CONFIG["low_memory_mode"] = True
        print("Low memory mode: ENABLED (sequential output computation)")

        if args.memory_debug:
            verify_success = verify_memory_efficient_mode(device)
            if not verify_success:
                print("WARNING: Memory-efficient mode verification failed! Continuing anyway...")

    if args.memory_debug:
        CONFIG["memory_debug"] = True
        print("Memory debugging: ENABLED")

    # Residuals
    if args.no_residual:
        CONFIG["use_residual_weights"] = False
        if args.residual_style is not None:
            print("NOTE: --residual_style ignored because residuals are disabled via --no_residual.")
    else:
        if args.residual_style is not None:
            style = args.residual_style.lower()
            if style in ("standard", "matrix"):
                style = "linear"
            CONFIG["residual_style"] = style

    # Norm and scheduler toggles
    if args.no_norm:
        CONFIG["use_normalization"] = False
    if args.use_advanced_scheduler:
        CONFIG["use_advanced_scheduler"] = True

    # Lateral mixing
    if args.no_lateral:
        CONFIG["use_lateral_mixing"] = False
    if args.lateral_type is not None:
        CONFIG["lateral_mixing_type"] = args.lateral_type

    # Parameter export config
    if args.export_params is not None:
        CONFIG["export_params"] = args.export_params

    # Effective norm choice
    if CONFIG.get("use_normalization", True) and not args.no_norm:
        if args.norm_type == "none":
            effective_norm_type = "none"
        elif args.norm_type is not None:
            effective_norm_type = args.norm_type
        else:
            effective_norm_type = CONFIG.get("norm_type", "batch")
    else:
        effective_norm_type = "none"

    # --------- Resolve spline types/orders (effective) ---------
    # If user provided --spline_type, use it for both; can override per-spline.
    phi_spline_type = args.phi_spline_type if args.phi_spline_type is not None else args.spline_type
    Phi_spline_type = args.Phi_spline_type if args.Phi_spline_type is not None else args.spline_type

    # Map synonyms: 'pwl'/'linear' -> 'linear' for the training API
    def _normalize_spline_name(name):
        if name is None:
            return None
        name = name.lower()
        return "linear" if name in ("pwl", "linear") else name  # keep 'cubic' as-is

    phi_spline_type = _normalize_spline_name(phi_spline_type) or "linear"
    Phi_spline_type = _normalize_spline_name(Phi_spline_type) or "linear"

    # Store the resolved values on args so get_config_suffix can see them (use user-facing tags)
    args.phi_spline_type_effective = "cubic" if phi_spline_type == "cubic" else "pwl"
    args.Phi_spline_type_effective = "cubic" if Phi_spline_type == "cubic" else "pwl"

    # --------- Pretty run header ---------
    print("\n========== RUN CONFIG ==========")
    print(f"Dataset: {args.dataset} (dim: {dataset.input_dim}→{dataset.output_dim})")
    print(f"Architecture: {architecture}")
    print(f"Epochs: {args.epochs} | Device: {device}")
    print(f"phi_knots: {args.phi_knots} | Phi_knots: {args.Phi_knots}")
    if CONFIG.get("use_normalization", True) and effective_norm_type != "none":
        pos = args.norm_position if args.norm_position else "after"
        skip_first = not args.norm_first if hasattr(args, "norm_first") else True
        print(f"Normalization: type={effective_norm_type}, pos={pos}, skip_first={skip_first}")
    else:
        print("Normalization: disabled")
    print(f"Residuals: {'enabled' if CONFIG.get('use_residual_weights', True) else 'disabled'} "
          f"| style={CONFIG.get('residual_style', 'node') if CONFIG.get('use_residual_weights', True) else 'n/a'}")
    print(f"Lateral mixing: {'enabled' if CONFIG.get('use_lateral_mixing', True) else 'disabled'} "
          f"({CONFIG.get('lateral_mixing_type','cyclic') if CONFIG.get('use_lateral_mixing', True) else 'n/a'})")
    print(f"Spline types: φ={args.phi_spline_type_effective}, Φ={args.Phi_spline_type_effective}")
    print(f"Domain debug: {CONFIG.get('debug_domains', False)} | Track violations: {CONFIG.get('track_domain_violations', False)}")
    print("================================\n")

    # Also print a concise summary (matches training printout)
    if effective_norm_type != "none":
        norm_position = args.norm_position if args.norm_position else "after"
        if args.norm_first:
            norm_skip_first = False
        else:
            norm_skip_first = args.norm_skip_first if hasattr(args, "norm_skip_first") else True
        print(f"Normalization: {effective_norm_type} (position: {norm_position}, skip_first: {norm_skip_first})")
    else:
        print("Normalization: disabled")
    print(
        f"Scheduler: {'PlateauAwareCosineAnnealingLR' if CONFIG.get('use_advanced_scheduler', False) else 'Adam (fixed LR)'}"
    )
    # Print internal names ('linear'/'cubic') so training log matches
    print(f"Spline types: phi={phi_spline_type}, Phi={Phi_spline_type}")

    if CONFIG.get("export_params"):
        if CONFIG["export_params"] == "all" or CONFIG["export_params"] is True:
            print("Parameter export: all parameters")
        else:
            param_types = parse_param_types(CONFIG["export_params"])
            print(f"Parameter export: {', '.join(param_types)}")

    if args.no_load_best:
        print("WARNING: Best model loading disabled (--no_load_best)")
    if args.bn_recalc_on_load:
        print("BatchNorm stats will be recalculated when loading best checkpoint")
    print()

    # Final normalization params to pass into training
    if effective_norm_type != "none":
        final_norm_position = args.norm_position if args.norm_position else "after"
        final_norm_skip_first = not args.norm_first if hasattr(args, "norm_first") and args.norm_first else (
            args.norm_skip_first if hasattr(args, "norm_skip_first") else True
        )
    else:
        final_norm_position = args.norm_position
        final_norm_skip_first = args.norm_skip_first

    # Residual style override (only meaningful if residuals enabled)
    residual_style_override = None
    if CONFIG.get("use_residual_weights", True):
        residual_style_override = CONFIG.get("residual_style", "node")

    # ------------------------
    # Train
    # ------------------------
    plotting_snapshot, losses = train_network(
        dataset=dataset,
        architecture=architecture,
        total_epochs=args.epochs,
        print_every=max(1, args.epochs // 10),
        device=device,
        phi_knots=args.phi_knots,
        Phi_knots=args.Phi_knots,
        seed=args.seed,
        norm_type=effective_norm_type,
        norm_position=final_norm_position,
        norm_skip_first=final_norm_skip_first,
        no_load_best=args.no_load_best,
        bn_recalc_on_load=args.bn_recalc_on_load,
        residual_style=residual_style_override,
        # spline config forwarded to the model (internal names)
        phi_spline_type=phi_spline_type,
        Phi_spline_type=Phi_spline_type,
        phi_spline_order=args.phi_spline_order,
        Phi_spline_order=args.Phi_spline_order,
    )

    # Unpack snapshot
    model = plotting_snapshot["model"]
    x_train = plotting_snapshot["x_train"]
    y_train = plotting_snapshot["y_train"]
    layers = model.layers

    # Determine canonical eval mode saved by train_network
    eval_mode = plotting_snapshot.get("eval_mode", "running")
    is_canonical_batch = (eval_mode == "batch")

    # Verify snapshot determinism using the same BN semantics as saved (3 runs)
    if is_canonical_batch:
        print("\nSwitched model to EVAL mode for verification and plotting (BN uses batch stats; no gradients).")
        with torch.no_grad():
            eval_losses = []
            # Force batch stats evaluation without updating BN buffers
            with use_batch_stats_without_updating_bn(model):
                for _ in range(3):
                    y_hat = model(x_train)
                    eval_losses.append(torch.mean((y_hat - y_train) ** 2).item())
        saved_eval = plotting_snapshot.get("loss_eval", float("nan"))
        print(f"Saved eval loss: {saved_eval:.4e}")
        print(f"Recomputed eval loss mean ± std over 3 runs: {np.mean(eval_losses):.4e} ± {np.std(eval_losses):.4e}")
    else:
        # FIX: Actually put the model in eval() mode for running BN statistics!
        print("\nSwitched model to EVAL mode for verification and plotting (BN uses running stats; no gradients).")
        model.eval()  # <-- THIS WAS MISSING! Required for BN to use running stats
        with torch.no_grad():
            eval_losses = []
            for _ in range(3):
                y_hat = model(x_train)
                eval_losses.append(torch.mean((y_hat - y_train) ** 2).item())
        saved_eval = plotting_snapshot.get("loss_eval", float("nan"))
        print(f"Saved eval loss: {saved_eval:.4e}")
        print(f"Recomputed eval loss mean ± std over 3 runs: {np.mean(eval_losses):.4e} ± {np.std(eval_losses):.4e}")

    # Print domain info
    print("\nFinal domain ranges:")
    for idx, layer in enumerate(layers):
        try:
            print(f"Layer {idx}:")
            print(f"  phi domain: [{layer.phi.in_min:.3f}, {layer.phi.in_max:.3f}]")
            print(f"  Phi domain: [{layer.Phi.in_min:.3f}, {layer.Phi.in_max:.3f}]")
            if hasattr(layer, "input_range") and layer.input_range is not None:
                print(f"  Input range: {layer.input_range}")
            if hasattr(layer, "output_range") and layer.output_range is not None:
                print(f"  Output range: {layer.output_range}")
        except Exception:
            pass

    # Export parameters if requested
    if CONFIG.get("export_params"):
        os.makedirs(CONFIG.get("export_params_dir", "params"), exist_ok=True)
        arch_str = "-".join(map(str, architecture)) if len(architecture) > 0 else "None"
        config_suffix = get_config_suffix(args, CONFIG)
        params_path = os.path.join(CONFIG.get("export_params_dir", "params"), f"params-{args.dataset}-{arch_str}-{args.epochs}-epochs{config_suffix}.txt")

        dataset_info = {
            "name": args.dataset,
            "architecture": architecture,
            "input_dim": dataset.input_dim,
            "output_dim": dataset.output_dim,
            "epochs": args.epochs,
        }
        checkpoint_info = {
            "epoch": plotting_snapshot.get("epoch", "Unknown"),
            "loss": plotting_snapshot.get("loss_eval", plotting_snapshot.get("loss", "Unknown")),
        }

        export_parameters(
            model,
            CONFIG["export_params"],
            params_path,
            dataset_info=dataset_info,
            checkpoint_info=checkpoint_info,
        )

    # ------------------------
    # Plotting (robust)
    # ------------------------
    try:
        if args.save_plots:
            os.makedirs("plots", exist_ok=True)

        prefix = "OneVar" if dataset.input_dim == 1 else f"{dataset.input_dim}Vars"
        arch_str = "-".join(map(str, architecture)) if len(architecture) > 0 else "None"
        config_suffix = get_config_suffix(args, CONFIG)
        filename = (
            f"{prefix}-{args.dataset}-{arch_str}-{args.epochs}-epochs-"
            f"outdim{dataset.output_dim}{config_suffix}.png"
        )

        save_path = os.path.join("plots", filename) if args.save_plots else None
        # IMPORTANT: use plotting API's correct signature
        _ = plot_results(
            model, layers, dataset, save_path, x_train=x_train, y_train=y_train
        )

        # Loss curve
        loss_filename = f"loss-{args.dataset}-{arch_str}-{args.epochs}-epochs{config_suffix}.png"
        loss_save_path = os.path.join("plots", loss_filename) if args.save_plots else None
        plot_loss_curve(losses, loss_save_path)

        if not args.no_show:
            print("Displaying plots. Close the plot windows to exit.")
            plt.show()
        else:
            plt.close("all")

    except Exception as e:
        # Typical on headless systems
        print("\n" + "=" * 60)
        print("WARNING: A plotting error occurred.")
        print(f"Error type: {type(e).__name__}")
        print("The default plotting backend on your system has failed.")
        print("This is common on systems without a configured GUI toolkit.")
        print("\nSwitching to the reliable 'Agg' backend to proceed.")
        print("=" * 60 + "\n")

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt  # re-import after backend switch

        if not args.save_plots:
            print("Plots could not be shown interactively. Enabling file saving automatically.")
            args.save_plots = True

        os.makedirs("plots", exist_ok=True)

        prefix = "OneVar" if dataset.input_dim == 1 else f"{dataset.input_dim}Vars"
        arch_str = "-".join(map(str, architecture)) if len(architecture) > 0 else "None"
        config_suffix = get_config_suffix(args, CONFIG)
        filename = (
            f"{prefix}-{args.dataset}-{arch_str}-{args.epochs}-epochs-"
            f"outdim{dataset.output_dim}{config_suffix}.png"
        )
        save_path = os.path.join("plots", filename)

        _ = plot_results(model, layers, dataset, save_path, x_train=x_train, y_train=y_train)

        loss_filename = f"loss-{args.dataset}-{arch_str}-{args.epochs}-epochs{config_suffix}.png"
        loss_save_path = os.path.join("plots", loss_filename)
        plot_loss_curve(losses, loss_save_path)

        plt.close("all")
        print("Plots were successfully saved to the 'plots' directory.")


if __name__ == "__main__":
    main()
