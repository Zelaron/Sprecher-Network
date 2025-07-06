"""Command-line interface for Sprecher Network experiments."""

import os
import argparse
import torch
import matplotlib
from sn_core import train_network, get_dataset, plot_results, plot_loss_curve
from sn_core.model import NormalizationType


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
    parser.add_argument("--normalization", type=str, default="none",
                      choices=["none", "batchnorm", "layernorm", "domain_aware"],
                      help="Normalization type (default: none)")
    
    return parser.parse_args()


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
    
    # Parse normalization type
    normalization_type = NormalizationType(args.normalization)
    
    print(f"Using device: {device}")
    print(f"Dataset: {args.dataset}")
    print(f"Architecture: {architecture}")
    print(f"phi knots: {args.phi_knots}, Phi knots: {args.Phi_knots}")
    print(f"Epochs: {args.epochs}")
    print(f"Seed: {args.seed}")
    print(f"Normalization: {normalization_type.value}")
    print(f"Theoretical domains: {CONFIG.get('use_theoretical_domains', True)}")
    print(f"Domain safety margin: {CONFIG.get('domain_safety_margin', 0.0)}")
    print()
    
    # Train network
    model, losses, layers = train_network(
        dataset=dataset,
        architecture=architecture,
        total_epochs=args.epochs,
        print_every=args.epochs // 10,
        device=device,
        phi_knots=args.phi_knots,
        Phi_knots=args.Phi_knots,
        seed=args.seed,
        normalization_type=normalization_type
    )
    
    # Ensure model is in eval mode for plotting
    model.eval()
    
    # Print final domain information
    print("\nFinal domain ranges:")
    for idx, layer in enumerate(layers):
        print(f"Layer {idx}:")
        print(f"  φ domain: [{layer.phi.in_min:.3f}, {layer.phi.in_max:.3f}]")
        print(f"  Φ domain: [{layer.Phi.in_min:.3f}, {layer.Phi.in_max:.3f}]")
        if hasattr(layer, 'input_range') and layer.input_range is not None:
            print(f"  Input range: {layer.input_range}")
        if hasattr(layer, 'output_range') and layer.output_range is not None:
            print(f"  Output range: {layer.output_range}")
    
    # --- Graceful Fallback Plotting Logic ---
    try:
        # Create plots directory if saving
        if args.save_plots:
            os.makedirs("plots", exist_ok=True)
        
        # Plot results
        prefix = "OneVar" if dataset.input_dim == 1 else f"{dataset.input_dim}Vars"
        arch_str = "-".join(map(str, architecture)) if len(architecture) > 0 else "None"
        norm_str = f"-{args.normalization}" if args.normalization != "none" else ""
        filename = f"{prefix}-{args.dataset}-{arch_str}-{args.epochs}-epochs{norm_str}-outdim{dataset.output_dim}.png"
        
        save_path = os.path.join("plots", filename) if args.save_plots else None
        fig_results = plot_results(model, layers, dataset, save_path)
        
        # Plot loss curve
        loss_filename = f"loss-{args.dataset}-{arch_str}-{args.epochs}-epochs{norm_str}.png"
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
        norm_str = f"-{args.normalization}" if args.normalization != "none" else ""
        filename = f"{prefix}-{args.dataset}-{arch_str}-{args.epochs}-epochs{norm_str}-outdim{dataset.output_dim}.png"
        save_path = os.path.join("plots", filename)
        
        plot_results(model, layers, dataset, save_path)
        
        loss_filename = f"loss-{args.dataset}-{arch_str}-{args.epochs}-epochs{norm_str}.png"
        loss_save_path = os.path.join("plots", loss_filename)
        plot_loss_curve(losses, loss_save_path)
        
        plt.close('all')
        print("Plots were successfully saved to the 'plots' directory.")


if __name__ == "__main__":
    main()