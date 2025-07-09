"""Focused test to identify the exact BatchNorm checkpoint issue."""

import torch
import torch.nn as nn
import numpy as np
from sn_core.train import recalculate_bn_stats
from sn_core.model import SprecherMultiLayerNetwork
from sn_core.config import CONFIG

def create_test_checkpoint(epochs=1000):
    """Create a checkpoint after training for specified epochs."""
    from sn_core import train_network, get_dataset
    
    dataset = get_dataset("toy_1d_poly")
    architecture = [15, 15]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Disable checkpoint loading to get raw final state
    model, losses, layers, x_train, y_train = train_network(
        dataset=dataset,
        architecture=architecture,
        total_epochs=epochs,
        print_every=max(epochs // 10, 100),
        device=device,
        phi_knots=100,
        Phi_knots=100,
        seed=45,
        norm_type='batch',
        norm_position='after',
        norm_skip_first=True,
        no_load_best=True,  # Don't load checkpoint
        bn_recalc_on_load=False
    )
    
    # Get final state
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'model_params': {
            'input_dim': dataset.input_dim,
            'architecture': architecture,
            'final_dim': dataset.output_dim,
            'phi_knots': 100,
            'Phi_knots': 100,
            'norm_type': 'batch',
            'norm_position': 'after',
            'norm_skip_first': True
        },
        'x_train': x_train.cpu(),
        'y_train': y_train.cpu(),
        'final_loss': losses[-1],
        'epochs': epochs
    }
    
    # Test current model
    model.eval()
    with torch.no_grad():
        output = model(x_train)
        test_loss = torch.mean((output - y_train) ** 2).item()
    
    checkpoint['test_loss_original'] = test_loss
    
    return checkpoint, model


def test_checkpoint_loading(checkpoint):
    """Test loading a checkpoint in different ways."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    x_train = checkpoint['x_train'].to(device)
    y_train = checkpoint['y_train'].to(device)
    
    print(f"\n{'='*60}")
    print(f"Testing checkpoint from {checkpoint['epochs']} epochs")
    print(f"Original final loss: {checkpoint['final_loss']:.4e}")
    print(f"Original test loss: {checkpoint['test_loss_original']:.4e}")
    print(f"{'='*60}\n")
    
    results = {}
    
    # Test 1: Fresh model with standard loading
    print("Test 1: Fresh model with standard state_dict loading")
    model1 = SprecherMultiLayerNetwork(**checkpoint['model_params']).to(device)
    model1.load_state_dict(checkpoint['model_state_dict'])
    model1.eval()
    
    with torch.no_grad():
        output1 = model1(x_train)
        loss1 = torch.mean((output1 - y_train) ** 2).item()
    
    print(f"  Loss: {loss1:.4e}")
    print(f"  Loss ratio: {loss1 / checkpoint['final_loss']:.2f}x")
    results['standard_loading'] = loss1
    
    # Check BatchNorm stats
    for name, module in model1.named_modules():
        if isinstance(module, nn.BatchNorm1d):
            print(f"  BN stats - mean[:5]: {module.running_mean[:5].cpu().numpy()}")
            print(f"  BN stats - var[:5]: {module.running_var[:5].cpu().numpy()}")
            print(f"  BN num_batches_tracked: {module.num_batches_tracked.item()}")
            break
    
    # Test 2: Model in training mode
    print("\nTest 2: Model in training mode")
    model2 = SprecherMultiLayerNetwork(**checkpoint['model_params']).to(device)
    model2.load_state_dict(checkpoint['model_state_dict'])
    model2.train()  # Keep in training mode
    
    with torch.no_grad():
        output2 = model2(x_train)
        loss2 = torch.mean((output2 - y_train) ** 2).item()
    
    print(f"  Loss: {loss2:.4e}")
    print(f"  Loss ratio: {loss2 / checkpoint['final_loss']:.2f}x")
    results['training_mode'] = loss2
    
    # Test 3: Recalculate BatchNorm stats
    print("\nTest 3: Recalculating BatchNorm statistics")
    model3 = SprecherMultiLayerNetwork(**checkpoint['model_params']).to(device)
    model3.load_state_dict(checkpoint['model_state_dict'])
    
    # Recalculate stats
    recalculate_bn_stats(model3, x_train, num_passes=10)
    model3.eval()
    
    with torch.no_grad():
        output3 = model3(x_train)
        loss3 = torch.mean((output3 - y_train) ** 2).item()
    
    print(f"  Loss: {loss3:.4e}")
    print(f"  Loss ratio: {loss3 / checkpoint['final_loss']:.2f}x")
    results['recalc_stats'] = loss3
    
    # Check new BatchNorm stats
    for name, module in model3.named_modules():
        if isinstance(module, nn.BatchNorm1d):
            print(f"  BN stats after recalc - mean[:5]: {module.running_mean[:5].cpu().numpy()}")
            print(f"  BN stats after recalc - var[:5]: {module.running_var[:5].cpu().numpy()}")
            break
    
    # Test 4: Model without domains initialized
    print("\nTest 4: Model with domain initialization disabled")
    model4 = SprecherMultiLayerNetwork(
        **checkpoint['model_params'],
        initialize_domains=False
    ).to(device)
    model4.load_state_dict(checkpoint['model_state_dict'])
    
    # Now initialize domains
    model4.update_all_domains(allow_resampling=False)
    model4.eval()
    
    with torch.no_grad():
        output4 = model4(x_train)
        loss4 = torch.mean((output4 - y_train) ** 2).item()
    
    print(f"  Loss: {loss4:.4e}")
    print(f"  Loss ratio: {loss4 / checkpoint['final_loss']:.2f}x")
    results['delayed_domain_init'] = loss4
    
    # Test 5: Check if the issue is batch-size dependent
    print("\nTest 5: Testing with different batch sizes")
    model5 = SprecherMultiLayerNetwork(**checkpoint['model_params']).to(device)
    model5.load_state_dict(checkpoint['model_state_dict'])
    model5.eval()
    
    # Test with single sample
    with torch.no_grad():
        single_output = model5(x_train[:1])
        single_loss = torch.mean((single_output - y_train[:1]) ** 2).item()
    
    print(f"  Single sample loss: {single_loss:.4e}")
    
    # Test with half batch
    half_size = len(x_train) // 2
    with torch.no_grad():
        half_output = model5(x_train[:half_size])
        half_loss = torch.mean((half_output - y_train[:half_size]) ** 2).item()
    
    print(f"  Half batch loss: {half_loss:.4e}")
    results['batch_size_test'] = {'single': single_loss, 'half': half_loss, 'full': loss1}
    
    return results


def compare_short_vs_long_runs():
    """Compare checkpoint behavior between short and long training runs."""
    
    print("\n" + "="*80)
    print("COMPARING SHORT VS LONG TRAINING RUNS")
    print("="*80)
    
    # Test with different epoch counts
    epoch_counts = [1000, 10000, 100000]
    all_results = {}
    
    for epochs in epoch_counts:
        print(f"\n\nCreating checkpoint after {epochs} epochs...")
        checkpoint, original_model = create_test_checkpoint(epochs)
        
        # Save original model info
        original_bn_stats = {}
        for name, module in original_model.named_modules():
            if isinstance(module, nn.BatchNorm1d):
                original_bn_stats[name] = {
                    'running_mean': module.running_mean.cpu().numpy().copy(),
                    'running_var': module.running_var.cpu().numpy().copy(),
                    'num_batches_tracked': module.num_batches_tracked.item()
                }
        
        checkpoint['original_bn_stats'] = original_bn_stats
        
        # Test checkpoint loading
        results = test_checkpoint_loading(checkpoint)
        all_results[epochs] = results
    
    # Summary analysis
    print("\n" + "="*80)
    print("SUMMARY ANALYSIS")
    print("="*80)
    
    print("\nLoss ratios (loaded loss / original loss):")
    print(f"{'Epochs':<10} {'Standard':<12} {'Train Mode':<12} {'Recalc BN':<12} {'No Domain':<12}")
    print("-" * 58)
    
    for epochs in epoch_counts:
        results = all_results[epochs]
        checkpoint_loss = create_test_checkpoint(epochs)[0]['final_loss']
        
        ratios = {
            'standard': results['standard_loading'] / checkpoint_loss,
            'train': results['training_mode'] / checkpoint_loss,
            'recalc': results['recalc_stats'] / checkpoint_loss,
            'no_domain': results['delayed_domain_init'] / checkpoint_loss
        }
        
        print(f"{epochs:<10} {ratios['standard']:<12.2f} {ratios['train']:<12.2f} "
              f"{ratios['recalc']:<12.2f} {ratios['no_domain']:<12.2f}")
    
    # Identify the issue
    print("\n" + "="*60)
    print("DIAGNOSIS:")
    print("="*60)
    
    # Check if issue is specific to long runs with BatchNorm
    short_run = all_results[1000]
    long_run = all_results[100000]
    
    short_ratio = short_run['standard_loading'] / create_test_checkpoint(1000)[0]['final_loss']
    long_ratio = long_run['standard_loading'] / create_test_checkpoint(100000)[0]['final_loss']
    
    if long_ratio > 1.5 and short_ratio < 1.1:
        print("\n⚠️  ISSUE CONFIRMED: Long training runs have checkpoint loading problems!")
        print(f"  - Short run (1K epochs): {short_ratio:.2f}x loss increase")
        print(f"  - Long run (100K epochs): {long_ratio:.2f}x loss increase")
        
        # Check which method helps
        if long_run['recalc_stats'] / create_test_checkpoint(100000)[0]['final_loss'] < 1.1:
            print("\n✓ SOLUTION: Recalculating BatchNorm statistics fixes the issue!")
            print("  This suggests the saved BatchNorm statistics are the problem.")
        
        if long_run['training_mode'] / create_test_checkpoint(100000)[0]['final_loss'] < long_ratio:
            print("\n📊 OBSERVATION: Training mode gives different results than eval mode.")
            print("  This is expected with BatchNorm but the difference is significant.")
    else:
        print("\n✓ No significant checkpoint loading issues detected.")


if __name__ == "__main__":
    # Run the comparison
    compare_short_vs_long_runs()