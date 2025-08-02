"""Command-line interface for Sprecher Network experiments."""

import os
import argparse
import torch
import numpy as np
import matplotlib
from sn_core import train_network, get_dataset, plot_results, plot_loss_curve, export_parameters, parse_param_types


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Train Sprecher Networks")
    
    parser.add_argument("--dataset", type=str, default="toy_1d_poly",
                      help="Dataset name (default: toy_1d_poly)")
    parser.add_argument("--arch", type=str, default="15,15",
                      help="Architecture as comma-separated values (default: 15,15)")
    parser.add_argument("--phi_knots", type=int, default=100,
                      help="Number of knots for phi splines (default: 100)")
    parser.add_argument("--Phi_knots", type=int, default=100,
                      help="Number of knots for Phi splines (default: 100)")
    parser.add_argument("--epochs", type=int, default=4000,
                      help="Number of training epochs (default: 4000)")
    parser.add_argument("--seed", type=int, default=45,
                      help="Random seed (default: 45)")
    parser.add_argument("--device", type=str, default="auto",
                      help="Device: auto, cpu, or cuda (default: auto)")
    parser.add_argument("--save_plots", action="store_true",
                      help="Save plots to files")
    parser.add_argument("--no_show", action="store_true",
                      help="Don't show plots (useful for batch runs)")
    parser.add_argument("--debug_domains", action="store_true",
                      help="Enable domain debugging output")
    parser.add_argument("--track_violations", action="store_true",
                      help="Track domain violations during training")
    
    # Normalization arguments
    parser.add_argument("--norm_type", type=str,
                      choices=["none", "batch", "layer"],
                      help="Type of normalization to use (default: from CONFIG)")
    parser.add_argument("--norm_position", type=str, default="after",
                      choices=["before", "after"],
                      help="Position of normalization relative to blocks (default: after)")
    parser.add_argument("--norm_skip_first", action="store_true", default=True,
                      help="Skip normalization for the first block (default: True)")
    parser.add_argument("--norm_first", action="store_true",
                      help="Enable normalization for the first block (overrides norm_skip_first)")
    
    # Debugging/testing arguments
    parser.add_argument("--no_load_best", action="store_true",
                      help="Don't load best checkpoint at end of training (for debugging)")
    parser.add_argument("--bn_recalc_on_load", action="store_true",
                      help="Recalculate BatchNorm statistics when loading best checkpoint (default: use saved stats)")
    
    # Feature control arguments
    parser.add_argument("--no_residual", action="store_true",
                      help="Disable residual connections (default: enabled)")
    parser.add_argument("--no_norm", action="store_true",
                      help="Disable normalization (default: enabled with batch norm)")
    parser.add_argument("--use_advanced_scheduler", action="store_true",
                      help="Use PlateauAwareCosineAnnealingLR scheduler (default: disabled)")
    
    # Learnable routing arguments
    parser.add_argument("--routing_init", type=str, default=None,
                      choices=["uniform", "random", "identity"],
                      help="Initialization for residual routing phases (default: from CONFIG)")
    parser.add_argument("--routing_temperature", type=float, default=None,
                      help="Temperature for softmax in residual routing (default: from CONFIG)")
    
    # Parameter export argument
    parser.add_argument("--export_params", nargs='?', const='all', default=None,
                      help="Export parameters to text file. Options: all, or comma-separated: "
                           "lambda,eta,spline,residual,codomain,norm,output")
    parser.add_argument("--export_routing_matrices", action="store_true",
                      help="Export computed routing matrices when exporting residual params")
    
    return parser.parse_args()


def get_config_suffix(args, CONFIG):
    """Build filename suffix for non-default configurations."""
    parts = []
    
    # Check normalization (default is enabled with batch)
    if args.no_norm or (hasattr(args, 'norm_type') and args.norm_type == 'none'):
        parts.append("NoNorm")
    elif hasattr(args, 'norm_type') and args.norm_type and args.norm_type not in ['none', 'batch']:
        parts.append(f"Norm{args.norm_type.capitalize()}")
    
    # Check residuals (default is enabled)
    if not CONFIG.get('use_residual_weights', True):
        parts.append("NoResidual")
    
    # Check scheduler (default is disabled)
    if CONFIG.get('use_advanced_scheduler', False):
        parts.append("AdvScheduler")
    
    # Check routing configuration (only add if non-default)
    if CONFIG.get('routing_init', 'uniform') != 'uniform':
        parts.append(f"Route{CONFIG['routing_init'].capitalize()}")
    if CONFIG.get('routing_temperature', 1.0) != 1.0:
        parts.append(f"Temp{CONFIG['routing_temperature']:.1f}")
    
    # Join with dashes
    return "-" + "-".join(parts) if parts else ""


def main():
    """Main training script."""
    args = parse_args()

    # If --no_show is used (e.g., in batch runs), force the 'Agg' backend upfront.
    if args.no_show:
        matplotlib.use('Agg')
        print("Using non-interactive 'Agg' backend for plotting (as requested by --no_show).")

    # Now, import pyplot. We will wrap the first plotting call in a try-except
    # block to handle systems without a working GUI backend.
    import matplotlib.pyplot as plt
    
    # Parse architecture
    if args.arch:
        architecture = [int(x) for x in args.arch.split(",")]
    else:
        architecture = []
    
    # Get dataset
    dataset = get_dataset(args.dataset)
    
    # Determine device
    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device
    
    # Enable domain debugging if requested
    from sn_core.config import CONFIG
    if args.debug_domains:
        CONFIG['debug_domains'] = True
    if args.track_violations:
        CONFIG['track_domain_violations'] = True
        print("Domain violation tracking enabled.")
    
    # Handle new feature control flags
    if args.no_residual:
        CONFIG['use_residual_weights'] = False
    if args.no_norm:
        CONFIG['use_normalization'] = False
    if args.use_advanced_scheduler:
        CONFIG['use_advanced_scheduler'] = True
    
    # Handle learnable routing configuration
    if args.routing_init is not None:
        CONFIG['routing_init'] = args.routing_init
    if args.routing_temperature is not None:
        CONFIG['routing_temperature'] = args.routing_temperature
    
    # Handle parameter export configuration
    if args.export_params is not None:
        CONFIG['export_params'] = args.export_params
    if args.export_routing_matrices:
        CONFIG['export_routing_matrices'] = True
    
    # Determine effective normalization settings
    if CONFIG.get('use_normalization', True) and not args.no_norm:
        # Use CONFIG defaults or argparser overrides
        if args.norm_type == "none":
            # If user explicitly set norm_type to "none", disable normalization
            effective_norm_type = "none"
        elif args.norm_type is not None:
            # Use the provided norm_type
            effective_norm_type = args.norm_type
        else:
            # Use CONFIG default
            effective_norm_type = CONFIG.get('norm_type', 'batch')
    else:
        # Normalization is disabled
        effective_norm_type = "none"
    
    print(f"Using device: {device}")
    print(f"Dataset: {args.dataset}")
    print(f"Architecture: {architecture}")
    print(f"phi knots: {args.phi_knots}, Phi knots: {args.Phi_knots}")
    print(f"Epochs: {args.epochs}")
    print(f"Seed: {args.seed}")
    print(f"Theoretical domains: {CONFIG.get('use_theoretical_domains', True)}")
    print(f"Domain safety margin: {CONFIG.get('domain_safety_margin', 0.0)}")
    print(f"Residual connections: {'enabled' if CONFIG.get('use_residual_weights', True) else 'disabled'}")
    if CONFIG.get('use_residual_weights', True):
        print(f"  Routing initialization: {CONFIG.get('routing_init', 'uniform')}")
        print(f"  Routing temperature: {CONFIG.get('routing_temperature', 1.0)}")
    if effective_norm_type != "none":
        norm_position = args.norm_position if args.norm_position else CONFIG.get('norm_position', 'after')
        # Handle the --norm_first flag
        if args.norm_first:
            norm_skip_first = False
        else:
            norm_skip_first = args.norm_skip_first if hasattr(args, 'norm_skip_first') else CONFIG.get('norm_skip_first', True)
        print(f"Normalization: {effective_norm_type} (position: {norm_position}, skip_first: {norm_skip_first})")
    else:
        print("Normalization: disabled")
    print(f"Scheduler: {'PlateauAwareCosineAnnealingLR' if CONFIG.get('use_advanced_scheduler', False) else 'Adam (fixed LR)'}")
    
    # Print parameter export configuration
    if CONFIG.get('export_params'):
        if CONFIG['export_params'] == 'all' or CONFIG['export_params'] is True:
            print("Parameter export: all parameters")
        else:
            param_types = parse_param_types(CONFIG['export_params'])
            print(f"Parameter export: {', '.join(param_types)}")
        if CONFIG.get('export_routing_matrices', False):
            print("  Including routing matrices in export")
    
    if args.no_load_best:
        print("WARNING: Best model loading disabled (--no_load_best)")
    if args.bn_recalc_on_load:
        print("BatchNorm stats will be recalculated when loading best checkpoint")
    print()
    
    # Determine final normalization parameters
    if effective_norm_type != "none":
        final_norm_position = args.norm_position if args.norm_position else CONFIG.get('norm_position', 'after')
        # Handle the --norm_first flag for final parameters
        if args.norm_first:
            final_norm_skip_first = False
        else:
            final_norm_skip_first = args.norm_skip_first if hasattr(args, 'norm_skip_first') else CONFIG.get('norm_skip_first', True)
    else:
        final_norm_position = args.norm_position  # Keep for backward compatibility
        final_norm_skip_first = args.norm_skip_first
    
    # Train network - now returns a snapshot and losses
    plotting_snapshot, losses = train_network(
        dataset=dataset,
        architecture=architecture,
        total_epochs=args.epochs,
        print_every=args.epochs // 10,
        device=device,
        phi_knots=args.phi_knots,
        Phi_knots=args.Phi_knots,
        seed=args.seed,
        norm_type=effective_norm_type,
        norm_position=final_norm_position,
        norm_skip_first=final_norm_skip_first,
        no_load_best=args.no_load_best,
        bn_recalc_on_load=args.bn_recalc_on_load
    )
    
    # Extract components from the snapshot
    model = plotting_snapshot['model']
    x_train = plotting_snapshot['x_train']
    y_train = plotting_snapshot['y_train']
    layers = model.layers
    
    # Verify the snapshot works correctly with multiple computations
    print("\nVerifying plotting snapshot consistency...")
    print("Computing loss multiple times to ensure perfect reproducibility:")
    
    losses_verification = []
    for i in range(5):
        with torch.no_grad():
            snapshot_output = model(x_train)
            snapshot_loss = torch.mean((snapshot_output - y_train) ** 2).item()
            losses_verification.append(snapshot_loss)
            print(f"  Computation {i+1}: loss = {snapshot_loss:.4e}")
    
    print(f"\nSaved loss from snapshot: {plotting_snapshot['loss']:.4e}")
    print(f"Mean of verifications: {np.mean(losses_verification):.4e}")
    print(f"Std of verifications: {np.std(losses_verification):.4e}")
    
    max_diff = max(abs(l - plotting_snapshot['loss']) for l in losses_verification)
    print(f"Max difference from saved: {max_diff:.4e}")
    
    if max_diff < 1e-8:
        print("[OK] Perfect consistency achieved! The snapshot is completely isolated.")
    
    # Keep model in training mode for consistency with checkpoint
    # model.eval()  # Commented out - we maintain the mode from training
    
    # Print final domain information
    print("\nFinal domain ranges:")
    for idx, layer in enumerate(layers):
        print(f"Layer {idx}:")
        print(f"  phi domain: [{layer.phi.in_min:.3f}, {layer.phi.in_max:.3f}]")
        print(f"  Phi domain: [{layer.Phi.in_min:.3f}, {layer.Phi.in_max:.3f}]")
        if hasattr(layer, 'input_range') and layer.input_range is not None:
            print(f"  Input range: {layer.input_range}")
        if hasattr(layer, 'output_range') and layer.output_range is not None:
            print(f"  Output range: {layer.output_range}")
    
    # DEBUG: Model structure check before plotting
    print("\nDEBUG: Model structure check before plotting:")
    print(f"Model type: {type(model)}")
    print(f"Model has {len(model.layers)} Sprecher layers")
    print(f"Model has {len(model.norm_layers) if hasattr(model, 'norm_layers') else 0} norm layers")
    
    # Test the full model vs just the layers
    # Create test input with correct dimensions for the dataset
    if dataset.input_dim == 1:
        test_input = torch.linspace(0, 1, 5).unsqueeze(1).to(device)
    else:
        # For multi-dimensional inputs, create random test points
        test_input = torch.rand(5, dataset.input_dim).to(device)
    
    with torch.no_grad():
        full_model_output = model(test_input)
        print(f"Full model output shape: {full_model_output.shape}")
        print(f"Full model output: {full_model_output.flatten().cpu().numpy()}")
    
    # Also check if we're in eval mode
    print(f"Model training mode: {model.training}")
    
    # DEBUG: Testing with same points as checkpoint loading
    print("\nDEBUG: Testing with same points as checkpoint loading:")
    # Generate same training data
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    n_samples = 32 if dataset.input_dim == 1 else 32 * 32
    x_train_debug, _ = dataset.sample(n_samples, device)
    with torch.no_grad():
        output_debug = model(x_train_debug[:5])
        print(f"Output on training points: {output_debug.cpu().numpy().flatten()[:5]}")
    
    print("=" * 60)
    
    # Export parameters if requested
    if CONFIG.get('export_params'):
        # Create params directory if saving
        os.makedirs(CONFIG.get('export_params_dir', 'params'), exist_ok=True)
        
        # Build filename similar to plots
        prefix = "OneVar" if dataset.input_dim == 1 else f"{dataset.input_dim}Vars"
        arch_str = "-".join(map(str, architecture)) if len(architecture) > 0 else "None"
        config_suffix = get_config_suffix(args, CONFIG)
        params_filename = f"{prefix}-{args.dataset}-{arch_str}-{args.epochs}-epochs-outdim{dataset.output_dim}{config_suffix}-params.txt"
        params_path = os.path.join(CONFIG.get('export_params_dir', 'params'), params_filename)
        
        # Prepare dataset info
        dataset_info = {
            'name': args.dataset,
            'architecture': architecture,
            'input_dim': dataset.input_dim,
            'output_dim': dataset.output_dim,
            'epochs': args.epochs
        }
        
        # Prepare checkpoint info
        checkpoint_info = {
            'epoch': plotting_snapshot.get('epoch', 'Unknown'),
            'loss': plotting_snapshot.get('loss', 'Unknown')
        }
        
        # Export parameters
        export_parameters(model, CONFIG['export_params'], params_path, 
                         dataset_info=dataset_info, checkpoint_info=checkpoint_info)
    
    # --- Graceful Fallback Plotting Logic ---
    try:
        # Create plots directory if saving
        if args.save_plots:
            os.makedirs("plots", exist_ok=True)
        
        # Plot results
        prefix = "OneVar" if dataset.input_dim == 1 else f"{dataset.input_dim}Vars"
        arch_str = "-".join(map(str, architecture)) if len(architecture) > 0 else "None"
        config_suffix = get_config_suffix(args, CONFIG)
        filename = f"{prefix}-{args.dataset}-{arch_str}-{args.epochs}-epochs-outdim{dataset.output_dim}{config_suffix}.png"
        
        save_path = os.path.join("plots", filename) if args.save_plots else None
        fig_results = plot_results(model, layers, dataset, save_path, x_train=x_train, y_train=y_train)
        
        # Plot loss curve
        loss_filename = f"loss-{args.dataset}-{arch_str}-{args.epochs}-epochs{config_suffix}.png"
        loss_save_path = os.path.join("plots", loss_filename) if args.save_plots else None
        plot_loss_curve(losses, loss_save_path)
        
        # Show plots if requested and not in Agg mode
        if not args.no_show:
            print("Displaying plots. Close the plot windows to exit.")
            plt.show()
        else:
            plt.close('all') # Free up memory in non-interactive mode
            
    except Exception as e:
        # This block will catch the TclError on Windows or other GUI-related errors
        print("\n" + "="*60)
        print("WARNING: A plotting error occurred.")
        print(f"Error type: {type(e).__name__}")
        print("The default plotting backend on your system has failed.")
        print("This is common on systems without a configured GUI toolkit.")
        print("\nSwitching to the reliable 'Agg' backend to proceed.")
        print("="*60 + "\n")

        # Set the backend and re-run the plotting logic
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt

        # Ensure we save plots if this fallback is triggered
        if not args.save_plots:
            print("Plots could not be shown interactively. Enabling file saving automatically.")
            args.save_plots = True

        # Re-run plotting logic with the safe backend
        os.makedirs("plots", exist_ok=True)
        
        prefix = "OneVar" if dataset.input_dim == 1 else f"{dataset.input_dim}Vars"
        arch_str = "-".join(map(str, architecture)) if len(architecture) > 0 else "None"
        config_suffix = get_config_suffix(args, CONFIG)
        filename = f"{prefix}-{args.dataset}-{arch_str}-{args.epochs}-epochs-outdim{dataset.output_dim}{config_suffix}.png"
        save_path = os.path.join("plots", filename)
        
        plot_results(model, layers, dataset, save_path, x_train=x_train, y_train=y_train)
        
        loss_filename = f"loss-{args.dataset}-{arch_str}-{args.epochs}-epochs{config_suffix}.png"
        loss_save_path = os.path.join("plots", loss_filename)
        plot_loss_curve(losses, loss_save_path)
        
        plt.close('all')
        print("Plots were successfully saved to the 'plots' directory.")


if __name__ == "__main__":
    main()