"""Command-line interface for Sprecher Network experiments."""

import os
import argparse
import torch
import matplotlib.pyplot as plt
from sn_core import train_network, get_dataset, plot_results, plot_loss_curve


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
    
    return parser.parse_args()


def main():
    """Main training script."""
    args = parse_args()
    
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
    
    print(f"Using device: {device}")
    print(f"Dataset: {args.dataset}")
    print(f"Architecture: {architecture}")
    print(f"phi knots: {args.phi_knots}, Phi knots: {args.Phi_knots}")
    print(f"Epochs: {args.epochs}")
    print(f"Seed: {args.seed}")
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
        seed=args.seed
    )
    
    # Create plots directory if saving
    if args.save_plots:
        os.makedirs("plots", exist_ok=True)
    
    # Plot results
    prefix = "OneVar" if dataset.input_dim == 1 else f"{dataset.input_dim}Vars"
    arch_str = "-".join(map(str, architecture)) if len(architecture) > 0 else "None"
    filename = f"{prefix}-{args.dataset}-{arch_str}-{args.epochs}-epochs-outdim{dataset.output_dim}.png"
    
    save_path = os.path.join("plots", filename) if args.save_plots else None
    fig_results = plot_results(model, layers, dataset, save_path)
    
    # Plot loss curve
    loss_filename = f"loss-{args.dataset}-{arch_str}-{args.epochs}-epochs.png"
    loss_save_path = os.path.join("plots", loss_filename) if args.save_plots else None
    plot_loss_curve(losses, loss_save_path)
    
    # Show plots if requested
    if not args.no_show:
        plt.show()
    else:
        plt.close('all')


if __name__ == "__main__":
    main()
