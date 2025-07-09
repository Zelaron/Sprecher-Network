"""Diagnostic script to investigate BatchNorm numerical issues in long training runs."""

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from sn_core import train_network, get_dataset
from sn_core.model import SprecherMultiLayerNetwork
from sn_core.config import CONFIG

def analyze_batchnorm_evolution(epochs_list=[100, 1000, 10000, 50000], 
                                norm_type='batch', seed=45):
    """Analyze how BatchNorm statistics evolve with training length."""
    
    print(f"\n{'='*60}")
    print(f"Analyzing BatchNorm evolution across different epoch counts")
    print(f"{'='*60}\n")
    
    dataset = get_dataset("toy_1d_poly")
    architecture = [15, 15]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    results = {}
    
    for epochs in epochs_list:
        print(f"\nTraining for {epochs} epochs...")
        
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
            no_load_best=True,  # Don't load checkpoint to see final state
            bn_recalc_on_load=False
        )
        
        # Collect BatchNorm statistics
        bn_stats = {}
        for name, module in model.named_modules():
            if isinstance(module, nn.BatchNorm1d):
                bn_stats[name] = {
                    'num_batches_tracked': module.num_batches_tracked.item(),
                    'momentum': module.momentum,
                    'running_mean': module.running_mean.cpu().numpy().copy(),
                    'running_var': module.running_var.cpu().numpy().copy(),
                    'weight': module.weight.data.cpu().numpy().copy(),
                    'bias': module.bias.data.cpu().numpy().copy(),
                }
        
        # Test model accuracy
        model.eval()
        with torch.no_grad():
            output = model(x_train)
            test_loss = torch.mean((output - y_train) ** 2).item()
        
        results[epochs] = {
            'final_loss': losses[-1],
            'test_loss': test_loss,
            'bn_stats': bn_stats,
            'losses': losses
        }
    
    # Analyze results
    print(f"\n{'='*60}")
    print("ANALYSIS RESULTS:")
    print(f"{'='*60}")
    
    # Print summary table
    print(f"\n{'Epochs':<10} {'Final Loss':<15} {'Test Loss':<15} {'num_batches':<15}")
    print(f"{'-'*55}")
    
    for epochs, data in results.items():
        if data['bn_stats']:
            first_bn = list(data['bn_stats'].values())[0]
            num_batches = first_bn['num_batches_tracked']
        else:
            num_batches = 0
        
        print(f"{epochs:<10} {data['final_loss']:<15.4e} "
              f"{data['test_loss']:<15.4e} {num_batches:<15}")
    
    # Detailed analysis
    if norm_type == 'batch' and results:
        print("\nBatchNorm Statistics Evolution:")
        
        # Get first BatchNorm layer for detailed analysis
        first_bn_name = None
        for name, module in model.named_modules():
            if isinstance(module, nn.BatchNorm1d):
                first_bn_name = name
                break
        
        if first_bn_name:
            print(f"\nAnalyzing layer: {first_bn_name}")
            
            # Compare running statistics
            epochs_sorted = sorted(results.keys())
            for i in range(len(epochs_sorted) - 1):
                e1, e2 = epochs_sorted[i], epochs_sorted[i+1]
                stats1 = results[e1]['bn_stats'][first_bn_name]
                stats2 = results[e2]['bn_stats'][first_bn_name]
                
                mean_change = np.mean(np.abs(stats2['running_mean'] - stats1['running_mean']))
                var_change = np.mean(np.abs(stats2['running_var'] - stats1['running_var']))
                
                print(f"\n{e1} → {e2} epochs:")
                print(f"  Mean change: {mean_change:.6e}")
                print(f"  Var change: {var_change:.6e}")
                print(f"  num_batches ratio: {stats2['num_batches_tracked'] / stats1['num_batches_tracked']:.1f}")
            
            # Check for numerical issues
            longest_run = results[epochs_sorted[-1]]['bn_stats'][first_bn_name]
            print(f"\nLongest run ({epochs_sorted[-1]} epochs) statistics:")
            print(f"  Running mean range: [{longest_run['running_mean'].min():.6e}, "
                  f"{longest_run['running_mean'].max():.6e}]")
            print(f"  Running var range: [{longest_run['running_var'].min():.6e}, "
                  f"{longest_run['running_var'].max():.6e}]")
            print(f"  Weight range: [{longest_run['weight'].min():.6e}, "
                  f"{longest_run['weight'].max():.6e}]")
            
            # Check for potential overflow
            if longest_run['num_batches_tracked'] > 1e6:
                print(f"\n⚠️  WARNING: num_batches_tracked is very large: {longest_run['num_batches_tracked']}")
                print("  This could lead to numerical precision issues in momentum calculations")
            
            # Check for extreme values
            if longest_run['running_var'].min() < 1e-10:
                print("\n⚠️  WARNING: Very small variance values detected!")
                print("  This could cause numerical instability during normalization")
    
    return results


def test_checkpoint_with_different_modes():
    """Test checkpoint loading with different train/eval modes."""
    
    print(f"\n{'='*60}")
    print("Testing checkpoint loading with different modes")
    print(f"{'='*60}\n")
    
    dataset = get_dataset("toy_1d_poly")
    architecture = [15, 15]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    epochs = 10000
    
    # Train network
    model, losses, layers, x_train, y_train = train_network(
        dataset=dataset,
        architecture=architecture,
        total_epochs=epochs,
        print_every=1000,
        device=device,
        phi_knots=100,
        Phi_knots=100,
        seed=45,
        norm_type='batch',
        norm_position='after',
        norm_skip_first=True,
        no_load_best=False,  # Load best checkpoint
        bn_recalc_on_load=False
    )
    
    # Test in different modes
    print("\nTesting model in different modes:")
    
    test_modes = [
        ("Training mode", True),
        ("Eval mode", False)
    ]
    
    for mode_name, training in test_modes:
        if training:
            model.train()
        else:
            model.eval()
        
        with torch.no_grad():
            output = model(x_train)
            loss = torch.mean((output - y_train) ** 2).item()
        
        print(f"\n{mode_name}:")
        print(f"  Loss: {loss:.4e}")
        print(f"  Model.training: {model.training}")
        
        # Check BatchNorm mode
        for name, module in model.named_modules():
            if isinstance(module, nn.BatchNorm1d):
                print(f"  BatchNorm.training: {module.training}")
                print(f"  BatchNorm.track_running_stats: {module.track_running_stats}")
                break


def investigate_numerical_precision():
    """Investigate numerical precision issues with very small losses."""
    
    print(f"\n{'='*60}")
    print("Investigating numerical precision with small losses")
    print(f"{'='*60}\n")
    
    # Test with different precision settings
    test_values = [1e-4, 1e-6, 1e-8, 1e-10, 1e-12, 1e-14]
    
    print("Testing float32 precision limits:")
    for val in test_values:
        tensor_val = torch.tensor(val, dtype=torch.float32)
        recovered = tensor_val.item()
        relative_error = abs(recovered - val) / val if val != 0 else 0
        
        print(f"  Value: {val:.2e}, Recovered: {recovered:.2e}, "
              f"Relative Error: {relative_error:.2e}")
    
    # Test momentum calculation precision
    print("\nTesting momentum calculation precision:")
    momentum = 0.1
    running_mean = torch.tensor(1.0, dtype=torch.float32)
    
    for i, batch_mean_offset in enumerate([1e-4, 1e-6, 1e-8, 1e-10]):
        batch_mean = torch.tensor(1.0 + batch_mean_offset, dtype=torch.float32)
        
        # Simulate momentum update
        new_running_mean = (1 - momentum) * running_mean + momentum * batch_mean
        change = (new_running_mean - running_mean).item()
        
        print(f"  Batch mean offset: {batch_mean_offset:.2e}, "
              f"Running mean change: {change:.2e}")
    
    # Test with very large num_batches_tracked
    print("\nTesting large num_batches_tracked effects:")
    for num_batches in [1000, 10000, 100000, 1000000]:
        # In some PyTorch versions, momentum might be adjusted based on num_batches
        # Check if there are precision issues
        tensor_num = torch.tensor(num_batches, dtype=torch.long)
        print(f"  num_batches: {num_batches}, as tensor: {tensor_num.item()}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Diagnose BatchNorm checkpoint issues")
    parser.add_argument("--test", type=str, default="all",
                        choices=["all", "evolution", "modes", "precision"],
                        help="Which test to run")
    
    args = parser.parse_args()
    
    if args.test == "all" or args.test == "evolution":
        analyze_batchnorm_evolution()
    
    if args.test == "all" or args.test == "modes":
        test_checkpoint_with_different_modes()
    
    if args.test == "all" or args.test == "precision":
        investigate_numerical_precision()