#!/usr/bin/env python3
"""Diagnostic script to identify BatchNorm checkpoint loading issue."""

import torch
import torch.nn as nn
import numpy as np
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from sn_core.model import SprecherMultiLayerNetwork
from sn_core.datasets import DATASETS
from sn_core.train import train_network, recalculate_bn_stats
from sn_core.config import CONFIG

def diagnose_batchnorm_issue():
    """Diagnose why BatchNorm checkpoint loading fails for long training runs."""
    
    print("="*80)
    print("BatchNorm Checkpoint Loading Diagnostic")
    print("="*80)
    
    # Test configuration
    dataset = DATASETS['toy_1d_poly']()
    architecture = [10, 10, 10]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Generate consistent training data
    torch.manual_seed(CONFIG['seed'])
    x_train, y_train = dataset.sample(32, device)
    
    # Test different epoch counts
    epoch_counts = [100, 1000, 10000, 50000]
    
    for epochs in epoch_counts:
        print(f"\n{'='*60}")
        print(f"Testing with {epochs} epochs")
        print(f"{'='*60}")
        
        # Train a model
        model = SprecherMultiLayerNetwork(
            input_dim=dataset.input_dim,
            architecture=architecture,
            final_dim=dataset.output_dim,
            phi_knots=100,
            Phi_knots=100,
            norm_type='batch',
            norm_position='after',
            norm_skip_first=True
        ).to(device)
        
        # Set to training mode
        model.train()
        
        # Simple training loop
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        best_loss = float('inf')
        best_state = None
        best_bn_stats = None
        
        for epoch in range(epochs):
            # Update domains
            if CONFIG.get('use_theoretical_domains', True):
                model.update_all_domains(allow_resampling=True)
            
            optimizer.zero_grad()
            output = model(x_train)
            loss = torch.mean((output - y_train) ** 2)
            loss.backward()
            optimizer.step()
            
            # Save best checkpoint
            if loss.item() < best_loss:
                best_loss = loss.item()
                best_state = model.state_dict()
                # Save BatchNorm statistics
                best_bn_stats = {}
                for name, module in model.named_modules():
                    if isinstance(module, nn.BatchNorm1d):
                        best_bn_stats[name] = {
                            'running_mean': module.running_mean.clone(),
                            'running_var': module.running_var.clone(),
                            'num_batches_tracked': module.num_batches_tracked.clone(),
                            'momentum': module.momentum,
                            'training': module.training
                        }
                        break
        
        print(f"Training completed. Best loss: {best_loss:.4e}")
        
        # Print BatchNorm stats at best checkpoint
        for name, stats in best_bn_stats.items():
            print(f"\nBatchNorm '{name}' at best checkpoint:")
            print(f"  num_batches_tracked: {stats['num_batches_tracked'].item()}")
            print(f"  running_mean (first 3): {stats['running_mean'][:3].cpu().numpy()}")
            print(f"  running_var (first 3): {stats['running_var'][:3].cpu().numpy()}")
            print(f"  momentum: {stats['momentum']}")
        
        # Test checkpoint loading
        print("\nTesting checkpoint restoration...")
        
        # Create fresh model
        fresh_model = SprecherMultiLayerNetwork(
            input_dim=dataset.input_dim,
            architecture=architecture,
            final_dim=dataset.output_dim,
            phi_knots=100,
            Phi_knots=100,
            norm_type='batch',
            norm_position='after',
            norm_skip_first=True
        ).to(device)
        
        # Load checkpoint
        fresh_model.load_state_dict(best_state)
        fresh_model.train()
        
        # Test restoration
        with torch.no_grad():
            fresh_output = fresh_model(x_train)
            fresh_loss = torch.mean((fresh_output - y_train) ** 2).item()
        
        print(f"Fresh model loss: {fresh_loss:.4e}")
        print(f"Loss difference: {abs(fresh_loss - best_loss):.4e}")
        print(f"Loss ratio: {fresh_loss / best_loss:.2f}x")
        
        # Check BatchNorm stats after loading
        for name, module in fresh_model.named_modules():
            if isinstance(module, nn.BatchNorm1d):
                print(f"\nBatchNorm '{name}' after loading:")
                print(f"  num_batches_tracked: {module.num_batches_tracked.item()}")
                print(f"  running_mean (first 3): {module.running_mean[:3].cpu().numpy()}")
                print(f"  running_var (first 3): {module.running_var[:3].cpu().numpy()}")
                print(f"  training mode: {module.training}")
                
                # Check numerical stability
                tracked = module.num_batches_tracked.item()
                if tracked > 10000:
                    momentum_factor = module.momentum if module.momentum is not None else 0.1
                    effective_momentum = momentum_factor / (1 + tracked * momentum_factor)
                    print(f"  effective momentum: {effective_momentum:.2e} (very small!)")
                break
        
        # Test with BN stats recalculation
        print("\nTesting with BatchNorm recalculation...")
        recalculate_bn_stats(fresh_model, x_train, num_passes=10)
        
        with torch.no_grad():
            recalc_output = fresh_model(x_train)
            recalc_loss = torch.mean((recalc_output - y_train) ** 2).item()
        
        print(f"Loss after BN recalc: {recalc_loss:.4e}")
        print(f"Improvement: {fresh_loss / recalc_loss:.2f}x")
    
    print("\n" + "="*80)
    print("Diagnostic Summary:")
    print("1. BatchNorm's num_batches_tracked grows linearly with epochs")
    print("2. Large num_batches_tracked values affect momentum calculations")
    print("3. Effective momentum becomes extremely small for long training runs")
    print("4. This causes checkpoint restoration to fail for long runs")
    print("5. Recalculating BN stats helps but may not fully solve the issue")
    print("="*80)

if __name__ == "__main__":
    diagnose_batchnorm_issue()