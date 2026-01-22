#!/usr/bin/env python3
"""
Ablation Benchmark for Sprecher Networks (SN)
=============================================

This script runs the ablation experiments for Table 1 in the SN paper,
testing the effect of progressively adding architectural components.

Configurations tested (as per the table):
1. Base SN (no residuals/mixing/domain tracking)
2. + cyclic residuals
3. + bidirectional mixing  
4. + domain tracking
5. + resampling + domain tracking (full SN)

NOTE: The order of ablations might not be ideal from a stability perspective.
      Disabling domain tracking in particular can cause issues. See comments below.
      
      An alternative "removal" ablation (commented out below) starts from the 
      full SN and progressively removes features, which may be more stable.

USAGE:
    1. Copy this script to your Sprecher directory (where sn_core/ is located)
    2. Run: python run_ablation_benchmark.py
    
    Or run from any directory with the Sprecher directory in your PYTHONPATH:
        PYTHONPATH=/path/to/Sprecher python run_ablation_benchmark.py

Results are saved to 'ablation_results.txt' and printed to console.

CONFIGURATION:
    Edit the "USER CONFIGURABLE SETTINGS" section below to change:
    - Dataset name (default: toy_2d_vector)
    - Architecture (default: [10, 10])
    - Number of epochs (default: 5000)
    - Number of seeds (default: 10)
    - And more...
"""

import sys
import os
import copy
import numpy as np
import torch
from datetime import datetime

# ============================================================================
# USER CONFIGURABLE SETTINGS
# ============================================================================

# Dataset to use
DATASET_NAME = "toy_2d_vector"  # Options: "toy_2d", "toy_2d_vector", etc.

# Architecture 
ARCHITECTURE = [10, 10]  # List of hidden layer widths

# Training settings
EPOCHS = 5000
DEVICE = "auto"  # "auto", "cpu", or "cuda"

# Spline settings
PHI_KNOTS = 100
PHI_SPLINE_KNOTS = 100

# Number of random seeds to run
NUM_SEEDS = 10
SEED_START = 0  # Seeds will be SEED_START, SEED_START+1, ..., SEED_START+NUM_SEEDS-1

# Test set size (use non-perfect-square to ensure random sampling, not grid)
N_TEST_SAMPLES = 1000

# Output file
OUTPUT_FILE = "ablation_results.txt"

# Normalization settings (kept consistent across all ablations)
NORM_TYPE = "batch"
NORM_POSITION = "after"
NORM_SKIP_FIRST = True

# ============================================================================
# ABLATION CONFIGURATIONS
# ============================================================================
# Each config is a dict that will be used to override the default CONFIG values.
# The 'name' field is used for identification in the results.
#
# IMPORTANT NOTES on the ablation:
# 
# 1. "cyclic residuals" refers to the default 'node' residual style, which uses
#    cyclic/pooling/broadcast patterns for dimension changes.
#
# 2. "domain tracking" refers to `use_theoretical_domains` which computes 
#    input domains for each layer's splines theoretically based on data flow.
#
# 3. "resampling" happens automatically when domains are updated with 
#    `allow_resampling=True`. When domains change, the spline's learned function
#    is resampled onto new knot positions to preserve its shape.
#
# 4. Steps 4 and 5 in the original table are almost identical because resampling
#    is enabled by default when domain tracking is on. To create a meaningful
#    distinction, step 4 disables resampling via a special flag.
#
# POTENTIAL ISSUES:
# - Disabling domain tracking may cause the network to behave poorly because
#   spline domains won't be updated to match actual input ranges.
# - Disabling resampling while keeping domain tracking can cause learned
#   functions to be "stretched" onto new domains, losing information.
#
# ALTERNATIVE: See the commented-out "removal" ablation below which may be 
#              more stable and interpretable.
# ============================================================================

# Flag to control resampling behavior (used in training loop modification)
# This is a workaround since there's no direct CONFIG flag for resampling
_DISABLE_RESAMPLING = False

ABLATION_CONFIGS = [
    {
        "name": "Base SN (no residuals/mixing/domain tracking)",
        "use_residual_weights": False,
        "use_lateral_mixing": False,
        "use_theoretical_domains": False,
        "_note": "Bare minimum SN - may have poor performance due to fixed domains",
    },
    {
        "name": "+ cyclic residuals",
        "use_residual_weights": True,
        "residual_style": "node",  # 'node' is the cyclic/original style
        "use_lateral_mixing": False,
        "use_theoretical_domains": False,
        "_note": "Added residual connections (node/cyclic style)",
    },
    {
        "name": "+ bidirectional mixing",
        "use_residual_weights": True,
        "residual_style": "node",
        "use_lateral_mixing": True,
        "lateral_mixing_type": "bidirectional",
        "use_theoretical_domains": False,
        "_note": "Added bidirectional lateral mixing between outputs within blocks",
    },
    {
        "name": "+ domain tracking",
        "use_residual_weights": True,
        "residual_style": "node",
        "use_lateral_mixing": True,
        "lateral_mixing_type": "bidirectional",
        "use_theoretical_domains": True,
        "_disable_resampling": True,  # Custom flag to disable resampling
        "_note": "Added theoretical domain computation (but no resampling)",
    },
    {
        "name": "+ resampling + domain tracking (full SN)",
        "use_residual_weights": True,
        "residual_style": "node",
        "use_lateral_mixing": True,
        "lateral_mixing_type": "bidirectional",
        "use_theoretical_domains": True,
        "_disable_resampling": False,  # Full resampling enabled
        "_note": "Full SN with all features - domains tracked and functions resampled",
    },
]

# ============================================================================
# ALTERNATIVE: More sensible ablation order (uncomment to use)
# This starts with full SN and progressively removes features,
# which is more stable since the full SN is known to work well.
# ============================================================================

ALTERNATIVE_ABLATION_CONFIGS = [
    {
        "name": "Full SN (all features)",
        "use_residual_weights": True,
        "residual_style": "node",
        "use_lateral_mixing": True,
        "lateral_mixing_type": "bidirectional",
        "use_theoretical_domains": True,
    },
    {
        "name": "SN without bidirectional (cyclic lateral only)",
        "use_residual_weights": True,
        "residual_style": "node",
        "use_lateral_mixing": True,
        "lateral_mixing_type": "cyclic",
        "use_theoretical_domains": True,
    },
    {
        "name": "SN without lateral mixing",
        "use_residual_weights": True,
        "residual_style": "node",
        "use_lateral_mixing": False,
        "use_theoretical_domains": True,
    },
    {
        "name": "SN without residuals or lateral",
        "use_residual_weights": False,
        "use_lateral_mixing": False,
        "use_theoretical_domains": True,
    },
]

# ============================================================================
# CHOOSE WHICH ABLATION TO RUN
# Set this to True to use the alternative (removal-based) ablation
# ============================================================================
USE_ALTERNATIVE_ABLATION = False

# Select which configs to use
if USE_ALTERNATIVE_ABLATION:
    ACTIVE_CONFIGS = ALTERNATIVE_ABLATION_CONFIGS
else:
    ACTIVE_CONFIGS = ABLATION_CONFIGS

# ============================================================================
# MAIN BENCHMARK CODE
# ============================================================================

def run_single_experiment(dataset, architecture, config_overrides, seed, epochs, device,
                          phi_knots, Phi_knots, norm_type, norm_position, norm_skip_first):
    """
    Run a single training experiment with the given configuration.
    
    Returns:
        tuple: (train_rmse, test_rmse) or (None, None) on failure
    """
    # Import here to avoid issues with CONFIG being modified
    from sn_core.config import CONFIG
    from sn_core.train import train_network
    from sn_core.data import get_dataset
    
    # Store original CONFIG values
    original_config = copy.deepcopy(dict(CONFIG))
    
    # Check if we need to disable resampling
    disable_resampling = config_overrides.get("_disable_resampling", False)
    
    try:
        # Apply configuration overrides
        for key, value in config_overrides.items():
            if not key.startswith("_"):  # Skip internal keys like _note, _disable_resampling
                CONFIG[key] = value
        
        # Set seed in CONFIG
        CONFIG["seed"] = seed
        
        # Get dataset
        ds = get_dataset(dataset)
        
        # If we need to disable resampling, we need to monkey-patch the model's update method
        if disable_resampling:
            # We'll patch the model after creation in the training function
            # This is a bit hacky but necessary since train_network doesn't expose this
            # For now, we'll use a context manager to patch the SprecherMultiLayerNetwork class
            from sn_core.model import SprecherMultiLayerNetwork
            original_update = SprecherMultiLayerNetwork.update_all_domains
            
            def patched_update(self, allow_resampling=True, force_resample=False):
                # Always call with allow_resampling=False
                return original_update(self, allow_resampling=False, force_resample=False)
            
            SprecherMultiLayerNetwork.update_all_domains = patched_update
        
        # Suppress most output for cleaner logs
        import io
        import contextlib
        
        # Run training (capture output to reduce noise)
        with contextlib.redirect_stdout(io.StringIO()):
            plotting_snapshot, losses = train_network(
                dataset=ds,
                architecture=architecture,
                total_epochs=epochs,
                print_every=max(1, epochs),  # Only print at end
                device=device,
                phi_knots=phi_knots,
                Phi_knots=Phi_knots,
                seed=seed,
                norm_type=norm_type,
                norm_position=norm_position,
                norm_skip_first=norm_skip_first,
            )
        
        # Restore original update method if we patched it
        if disable_resampling:
            SprecherMultiLayerNetwork.update_all_domains = original_update
        
        # Extract model and compute RMSE
        model = plotting_snapshot["model"]
        x_train = plotting_snapshot["x_train"]
        y_train = plotting_snapshot["y_train"]
        
        # Use the BATCH-mode eval loss from the snapshot (computed with proper BN handling)
        # The snapshot stores both running-stats and batch-stats eval losses.
        # We want batch-stats to match how the progress bar displays loss and to be
        # consistent with how we evaluate test data below.
        # Fallback chain: loss_eval_batch -> loss_eval -> loss
        train_loss_eval = plotting_snapshot.get(
            "loss_eval_batch", 
            plotting_snapshot.get("loss_eval", plotting_snapshot.get("loss"))
        )
        train_rmse = np.sqrt(train_loss_eval)
        
        # For test data, we need to evaluate with the same BN mode (batch stats)
        # Import the helper for batch stats evaluation
        from sn_core.train import use_batch_stats_without_updating_bn
        
        with torch.no_grad():
            # Generate random test data
            x_test = torch.rand(N_TEST_SAMPLES, ds.input_dim, device=x_train.device)
            y_test = ds.evaluate(x_test)
            
            # Evaluate using batch stats (consistent with how training loss is computed)
            # This avoids issues with potentially uninitialized running stats
            model.eval()
            with use_batch_stats_without_updating_bn(model):
                y_pred_test = model(x_test)
            test_mse = torch.mean((y_pred_test - y_test) ** 2).item()
            test_rmse = np.sqrt(test_mse)
        
        return train_rmse, test_rmse
        
    except Exception as e:
        print(f"    ERROR: {e}")
        import traceback
        traceback.print_exc()
        return None, None
        
    finally:
        # Restore original CONFIG
        CONFIG.clear()
        CONFIG.update(original_config)
        
        # Make sure we restore the update method if something went wrong
        if disable_resampling:
            try:
                from sn_core.model import SprecherMultiLayerNetwork
                SprecherMultiLayerNetwork.update_all_domains = original_update
            except:
                pass


def run_ablation_benchmark():
    """Run the full ablation benchmark across all configurations and seeds."""
    
    print("=" * 70)
    print("SPRECHER NETWORK ABLATION BENCHMARK")
    print("=" * 70)
    print(f"Dataset: {DATASET_NAME}")
    print(f"Architecture: {ARCHITECTURE}")
    print(f"Epochs: {EPOCHS}")
    print(f"Seeds: {NUM_SEEDS} (from {SEED_START} to {SEED_START + NUM_SEEDS - 1})")
    print(f"Test samples: {N_TEST_SAMPLES} (random, not grid)")
    print(f"Device: {DEVICE}")
    print(f"Normalization: {NORM_TYPE} (position={NORM_POSITION}, skip_first={NORM_SKIP_FIRST})")
    print(f"Spline knots: phi={PHI_KNOTS}, Phi={PHI_SPLINE_KNOTS}")
    print(f"Using {'ALTERNATIVE (removal)' if USE_ALTERNATIVE_ABLATION else 'STANDARD (addition)'} ablation order")
    print("=" * 70)
    print()
    
    # Determine actual device
    if DEVICE == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = DEVICE
    print(f"Using device: {device}")
    print()
    
    # Store all results
    all_results = {}
    
    for config in ACTIVE_CONFIGS:
        config_name = config["name"]
        print(f"\n{'='*70}")
        print(f"Configuration: {config_name}")
        print(f"{'='*70}")
        
        # Print config details
        for key, value in config.items():
            if not key.startswith("_") and key != "name":  # Skip internal keys
                print(f"  {key}: {value}")
        if "_note" in config:
            print(f"  Note: {config['_note']}")
        print()
        
        train_rmses = []
        test_rmses = []
        
        for seed_idx in range(NUM_SEEDS):
            seed = SEED_START + seed_idx
            print(f"  Seed {seed} ({seed_idx + 1}/{NUM_SEEDS})... ", end="", flush=True)
            
            train_rmse, test_rmse = run_single_experiment(
                dataset=DATASET_NAME,
                architecture=ARCHITECTURE,
                config_overrides=config,
                seed=seed,
                epochs=EPOCHS,
                device=device,
                phi_knots=PHI_KNOTS,
                Phi_knots=PHI_SPLINE_KNOTS,
                norm_type=NORM_TYPE,
                norm_position=NORM_POSITION,
                norm_skip_first=NORM_SKIP_FIRST,
            )
            
            if train_rmse is not None:
                train_rmses.append(train_rmse)
                test_rmses.append(test_rmse)
                print(f"Train RMSE: {train_rmse:.4f}, Test RMSE: {test_rmse:.4f}")
            else:
                print("FAILED")
        
        # Compute statistics
        if test_rmses:
            mean_train = np.mean(train_rmses)
            std_train = np.std(train_rmses)
            mean_test = np.mean(test_rmses)
            std_test = np.std(test_rmses)
            
            all_results[config_name] = {
                "train_rmses": train_rmses,
                "test_rmses": test_rmses,
                "mean_train": mean_train,
                "std_train": std_train,
                "mean_test": mean_test,
                "std_test": std_test,
                "n_success": len(test_rmses),
            }
            
            print(f"\n  Results for '{config_name}':")
            print(f"    Train RMSE: {mean_train:.3f} ± {std_train:.3f}")
            print(f"    Test RMSE:  {mean_test:.3f} ± {std_test:.3f}")
            print(f"    Successful runs: {len(test_rmses)}/{NUM_SEEDS}")
        else:
            all_results[config_name] = None
            print(f"\n  All runs failed for '{config_name}'")
    
    # Print summary table
    print("\n")
    print("=" * 70)
    print("SUMMARY TABLE (for LaTeX)")
    print("=" * 70)
    print()
    print("\\begin{table}[t]")
    print("\\centering")
    print(f"\\caption{{Ablation on {DATASET_NAME.replace('_', '-').title()}. Mean test RMSE $\\pm$ std over {NUM_SEEDS} seeds. Lower is better.}}")
    print("\\label{tab:toy2d_addons_ablation}")
    print("\\small")
    print("\\begin{tabular}{lc}")
    print("\\toprule")
    print("Setting & Test RMSE \\\\")
    print("\\midrule")
    
    # Find best result for bolding
    best_mean = float('inf')
    for config_name, result in all_results.items():
        if result is not None and result["mean_test"] < best_mean:
            best_mean = result["mean_test"]
    
    for config in ACTIVE_CONFIGS:
        config_name = config["name"]
        result = all_results.get(config_name)
        
        if result is not None:
            mean = result["mean_test"]
            std = result["std_test"]
            
            # Format with appropriate precision
            if mean < 0.01:
                formatted = f"${mean:.4f} \\pm {std:.4f}$"
            elif mean < 0.1:
                formatted = f"${mean:.3f} \\pm {std:.3f}$"
            else:
                formatted = f"${mean:.3f} \\pm {std:.3f}$"
            
            # Bold the best result
            if abs(mean - best_mean) < 1e-6:
                formatted = f"$\\mathbf{{{mean:.3f} \\pm {std:.3f}}}$"
            
            print(f"{config_name} & {formatted} \\\\")
        else:
            print(f"{config_name} & FAILED \\\\")
    
    print("\\bottomrule")
    print("\\end{tabular}")
    print("\\end{table}")
    print()
    
    # Save results to file
    with open(OUTPUT_FILE, 'w') as f:
        f.write("SPRECHER NETWORK ABLATION BENCHMARK RESULTS\n")
        f.write("=" * 70 + "\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Dataset: {DATASET_NAME}\n")
        f.write(f"Architecture: {ARCHITECTURE}\n")
        f.write(f"Epochs: {EPOCHS}\n")
        f.write(f"Seeds: {NUM_SEEDS} (from {SEED_START} to {SEED_START + NUM_SEEDS - 1})\n")
        f.write(f"Device: {device}\n")
        f.write(f"Ablation type: {'ALTERNATIVE (removal)' if USE_ALTERNATIVE_ABLATION else 'STANDARD (addition)'}\n")
        f.write("=" * 70 + "\n\n")
        
        for config in ACTIVE_CONFIGS:
            config_name = config["name"]
            result = all_results.get(config_name)
            
            f.write(f"Configuration: {config_name}\n")
            f.write("-" * 50 + "\n")
            
            if result is not None:
                f.write(f"  Train RMSE: {result['mean_train']:.4f} ± {result['std_train']:.4f}\n")
                f.write(f"  Test RMSE:  {result['mean_test']:.4f} ± {result['std_test']:.4f}\n")
                f.write(f"  Successful runs: {result['n_success']}/{NUM_SEEDS}\n")
                f.write(f"  Individual test RMSEs: {[f'{r:.4f}' for r in result['test_rmses']]}\n")
            else:
                f.write("  All runs FAILED\n")
            f.write("\n")
        
        # Also write the LaTeX table
        f.write("\n" + "=" * 70 + "\n")
        f.write("LATEX TABLE\n")
        f.write("=" * 70 + "\n\n")
        
        f.write("\\begin{table}[t]\n")
        f.write("\\centering\n")
        f.write(f"\\caption{{Ablation on {DATASET_NAME.replace('_', '-').title()}. Mean test RMSE $\\pm$ std over {NUM_SEEDS} seeds. Lower is better.}}\n")
        f.write("\\label{tab:toy2d_addons_ablation}\n")
        f.write("\\small\n")
        f.write("\\begin{tabular}{lc}\n")
        f.write("\\toprule\n")
        f.write("Setting & Test RMSE \\\\\n")
        f.write("\\midrule\n")
        
        for config in ACTIVE_CONFIGS:
            config_name = config["name"]
            result = all_results.get(config_name)
            
            if result is not None:
                mean = result["mean_test"]
                std = result["std_test"]
                
                if abs(mean - best_mean) < 1e-6:
                    formatted = f"$\\mathbf{{{mean:.3f} \\pm {std:.3f}}}$"
                else:
                    formatted = f"${mean:.3f} \\pm {std:.3f}$"
                
                f.write(f"{config_name} & {formatted} \\\\\n")
            else:
                f.write(f"{config_name} & FAILED \\\\\n")
        
        f.write("\\bottomrule\n")
        f.write("\\end{tabular}\n")
        f.write("\\end{table}\n")
    
    print(f"\nResults saved to: {OUTPUT_FILE}")
    print("\nDone!")
    
    return all_results


if __name__ == "__main__":
    run_ablation_benchmark()
