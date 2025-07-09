"""Test script to investigate BatchNorm checkpoint loading issue with long vs short runs."""

import torch
import torch.nn as nn
import numpy as np
import argparse
from sn_core import train_network, get_dataset
from sn_core.model import SprecherMultiLayerNetwork
from sn_core.config import CONFIG

def test_checkpoint_reproducibility(epochs, norm_type='batch', seed=45):
    """Test if checkpoint loading produces same results for different epoch counts."""
    
    print(f"\n{'='*60}")
    print(f"Testing checkpoint reproducibility with {epochs} epochs")
    print(f"Normalization: {norm_type}")
    print(f"Seed: {seed}")
    print(f"{'='*60}\n")
    
    # Get dataset
    dataset = get_dataset("toy_1d_poly")
    architecture = [15, 15]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Enable checkpoint debugging
    CONFIG['debug_checkpoint_loading'] = True
    
    # Train network
    model, losses, layers, x_train, y_train = train_network(
        dataset=dataset,
        architecture=architecture,
        total_epochs=epochs,
        print_every=max(epochs // 10, 100),
        device=device,
        phi_knots=100,
        Phi_knots=100,
        seed=seed,
        norm_type=norm_type,
        norm_position='after',
        norm_skip_first=True,
        no_load_best=False,  # Allow checkpoint loading
        bn_recalc_on_load=False  # Use saved stats
    )
    
    # Get final loss
    final_loss = losses[-1]
    
    # Test model on training data
    model.eval()
    with torch.no_grad():
        output = model(x_train)
        test_loss = torch.mean((output - y_train) ** 2).item()
    
    print(f"\nFinal training loss: {final_loss:.4e}")
    print(f"Test loss after loading: {test_loss:.4e}")
    print(f"Loss ratio: {test_loss/final_loss:.2f}x")
    
    # Check if model has BatchNorm
    has_bn = False
    for module in model.modules():
        if isinstance(module, nn.BatchNorm1d):
            has_bn = True
            print(f"\nBatchNorm stats:")
            print(f"  num_batches_tracked: {module.num_batches_tracked.item()}")
            print(f"  momentum: {module.momentum}")
            print(f"  running_mean[:5]: {module.running_mean[:5].cpu().numpy()}")
            print(f"  running_var[:5]: {module.running_var[:5].cpu().numpy()}")
            break
    
    # Additional debug info
    if has_bn and epochs > 10000:
        print(f"\nPotential issues for long runs:")
        print(f"1. num_batches_tracked = {module.num_batches_tracked.item()}")
        print(f"   - This counter increments each forward pass")
        print(f"   - For {epochs} epochs, this is a large number")
        print(f"2. Momentum-based running stats update")
        print(f"   - With momentum={module.momentum}, stats converge over time")
        print(f"   - Very long runs may have extremely stable stats")
        print(f"3. Numerical precision")
        print(f"   - Loss values: {final_loss:.4e}")
        print(f"   - Small losses may have precision issues")
    
    return final_loss, test_loss, has_bn


def compare_checkpoint_behavior():
    """Compare checkpoint loading behavior between short and long runs."""
    
    # Test configurations
    test_configs = [
        {"epochs": 1000, "norm_type": "batch"},
        {"epochs": 10000, "norm_type": "batch"},
        {"epochs": 100000, "norm_type": "batch"},
        {"epochs": 1000, "norm_type": "none"},
        {"epochs": 100000, "norm_type": "none"},
    ]
    
    results = []
    
    for config in test_configs:
        final_loss, test_loss, has_bn = test_checkpoint_reproducibility(**config)
        results.append({
            **config,
            "final_loss": final_loss,
            "test_loss": test_loss,
            "loss_ratio": test_loss / final_loss if final_loss > 0 else float('inf'),
            "has_bn": has_bn
        })
    
    # Print summary
    print(f"\n{'='*80}")
    print("SUMMARY: Checkpoint Loading Behavior")
    print(f"{'='*80}")
    print(f"{'Epochs':<10} {'Norm':<10} {'Final Loss':<15} {'Test Loss':<15} {'Ratio':<10} {'BN':<5}")
    print(f"{'-'*80}")
    
    for result in results:
        print(f"{result['epochs']:<10} {result['norm_type']:<10} "
              f"{result['final_loss']:<15.4e} {result['test_loss']:<15.4e} "
              f"{result['loss_ratio']:<10.2f} {str(result['has_bn']):<5}")
    
    # Analysis
    print(f"\n{'='*80}")
    print("ANALYSIS:")
    print(f"{'='*80}")
    
    # Check if BatchNorm models have issues with long runs
    bn_short = next((r for r in results if r['norm_type'] == 'batch' and r['epochs'] == 1000), None)
    bn_long = next((r for r in results if r['norm_type'] == 'batch' and r['epochs'] == 100000), None)
    
    if bn_short and bn_long:
        print(f"\nBatchNorm comparison:")
        print(f"  Short run (1000 epochs): ratio = {bn_short['loss_ratio']:.2f}")
        print(f"  Long run (100000 epochs): ratio = {bn_long['loss_ratio']:.2f}")
        
        if bn_long['loss_ratio'] > 1.5 and bn_short['loss_ratio'] < 1.1:
            print("\n  ⚠️  ISSUE DETECTED: Long runs with BatchNorm show checkpoint loading problems!")
            print("  Possible causes:")
            print("  1. num_batches_tracked overflow or numerical issues")
            print("  2. Running statistics convergence to extreme values")
            print("  3. Momentum calculation differences after many updates")
            print("  4. Training/eval mode inconsistencies")
    
    # Check models without BatchNorm
    no_bn_short = next((r for r in results if r['norm_type'] == 'none' and r['epochs'] == 1000), None)
    no_bn_long = next((r for r in results if r['norm_type'] == 'none' and r['epochs'] == 100000), None)
    
    if no_bn_short and no_bn_long:
        print(f"\nNo BatchNorm comparison:")
        print(f"  Short run (1000 epochs): ratio = {no_bn_short['loss_ratio']:.2f}")
        print(f"  Long run (100000 epochs): ratio = {no_bn_long['loss_ratio']:.2f}")
        
        if no_bn_long['loss_ratio'] < 1.1 and no_bn_short['loss_ratio'] < 1.1:
            print("\n  ✓ Models without BatchNorm show consistent checkpoint loading!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test BatchNorm checkpoint loading issue")
    parser.add_argument("--mode", type=str, default="compare", 
                        choices=["compare", "single"],
                        help="Test mode: compare multiple configs or single test")
    parser.add_argument("--epochs", type=int, default=1000,
                        help="Number of epochs for single test")
    parser.add_argument("--norm_type", type=str, default="batch",
                        choices=["none", "batch", "layer"],
                        help="Normalization type for single test")
    
    args = parser.parse_args()
    
    if args.mode == "compare":
        compare_checkpoint_behavior()
    else:
        test_checkpoint_reproducibility(args.epochs, args.norm_type)