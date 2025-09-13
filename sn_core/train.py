"""Training utilities for Sprecher Networks.

Adds optional standard residuals support via CONFIG['residual_style'] ∈ {'node','linear'}.
- 'node'   (default): existing node-centric residuals (scalar/pooling/broadcast)
- 'linear': standard residuals (α·x when d_in==d_out; x @ W projection when d_in!=d_out)
"""

import torch
import torch.nn as nn
import numpy as np
import copy
from tqdm import tqdm
from .model import SprecherMultiLayerNetwork
from .config import CONFIG


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
            self.plateau_counter = 0  # Reset counter
        else:
            # Cosine annealing
            progress = (self.current_step % self.cycle_length) / self.cycle_length
            lr = self.base_lr + 0.5 * (self.max_lr - self.base_lr) * (1 + np.cos(np.pi * progress))
        
        # Update learning rates
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr * param_group.get('lr_scale', 1.0)
        
        return lr


def recalculate_bn_stats(model, x_train, num_passes=10):
    """Recalculate BatchNorm statistics using the training data."""
    was_training = model.training
    
    # First, reset all BatchNorm running statistics
    for module in model.modules():
        if isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
            module.reset_running_stats()
            # Also reset momentum to accumulate stats faster
            module.momentum = 0.1
    
    # Set to train mode to update running stats
    model.train()
    
    # Run through the data multiple times to accumulate statistics
    with torch.no_grad():
        for pass_idx in range(num_passes):
            _ = model(x_train)
            # For the last few passes, reduce momentum to stabilize
            if pass_idx >= num_passes - 3:
                for module in model.modules():
                    if isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
                        module.momentum = 0.01
    
    # Restore original mode
    if not was_training:
        model.eval()


def has_batchnorm(model):
    """Check if model contains any BatchNorm layers."""
    for module in model.modules():
        if isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
            return True
    return False


# Removed set_bn_eval and set_bn_train - we should never use eval mode


def train_network(dataset, architecture, total_epochs=4000, print_every=400, 
                  device="cpu", phi_knots=100, Phi_knots=100, seed=None,
                  norm_type='none', norm_position='after', norm_skip_first=True,
                  no_load_best=False, bn_recalc_on_load=False,
                  residual_style=None):
    """
    Train a Sprecher network on the given dataset.
    
    Args:
        dataset: Dataset instance with sample() method
        architecture: List of hidden layer sizes
        total_epochs: Number of training epochs
        print_every: Print frequency
        device: Training device
        phi_knots: Number of knots for phi splines
        Phi_knots: Number of knots for Phi splines
        seed: Random seed (uses config default if None)
        norm_type: Type of normalization ('none', 'batch', 'layer')
        norm_position: Position of normalization ('before', 'after')
        norm_skip_first: Whether to skip normalization for first block
        no_load_best: If True, don't load best checkpoint at end
        bn_recalc_on_load: If True, recalculate BatchNorm stats when loading checkpoint
        residual_style: Optional override for CONFIG['residual_style'] ∈ {'node','linear'}.
                        Aliases 'standard'/'matrix' → 'linear'. If None, uses CONFIG as-is.
    
    Returns:
        plotting_snapshot: dict with a deep-copied model and data for consistent plotting
        losses: List of training losses across epochs
    """
    # Use seed from config if not provided
    if seed is None:
        seed = CONFIG['seed']
    
    # Set seeds
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    # Optional: set residual style override & normalize synonyms
    if residual_style is not None:
        CONFIG['residual_style'] = str(residual_style).lower()
    if CONFIG.get('residual_style', 'node') in ['standard', 'matrix']:
        CONFIG['residual_style'] = 'linear'
    CONFIG.setdefault('residual_style', 'node')
    
    # Generate training data
    n_samples = 32 if dataset.input_dim == 1 else 32 * 32
    x_train, y_train = dataset.sample(n_samples, device)
    
    # Compute target statistics for better initialization
    y_mean = y_train.mean(dim=0)
    y_std = y_train.std(dim=0)
    
    # Create model
    model = SprecherMultiLayerNetwork(
        input_dim=dataset.input_dim,
        architecture=architecture,
        final_dim=dataset.output_dim,
        phi_knots=phi_knots,
        Phi_knots=Phi_knots,
        norm_type=norm_type,
        norm_position=norm_position,
        norm_skip_first=norm_skip_first
    ).to(device)
    
    # Initialize output bias to target mean for faster convergence
    with torch.no_grad():
        model.output_bias.data = y_mean.mean()
        model.output_scale.data = torch.tensor(0.1)  # Start with small scale
    
    # Compute parameter counts (now includes residual_projection for 'linear' style)
    lambda_params = 0
    eta_params = 0
    spline_params = 0
    residual_params = 0
    residual_scalar_params = 0
    residual_pooling_params = 0
    residual_broadcast_params = 0
    residual_projection_params = 0  # NEW
    codomain_params = 0
    norm_params = 0
    lateral_params = 0
    
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
            
        if 'lambdas' in name:
            lambda_params += param.numel()
        elif 'eta' in name:
            eta_params += param.numel()
        elif 'coeffs' in name or 'log_increments' in name:
            spline_params += param.numel()
        elif 'residual_weight' in name:
            if CONFIG.get('use_residual_weights', True):
                residual_scalar_params += param.numel()
                residual_params += param.numel()
        elif 'residual_pooling_weights' in name:
            if CONFIG.get('use_residual_weights', True):
                residual_pooling_params += param.numel()
                residual_params += param.numel()
        elif 'residual_broadcast_weights' in name:
            if CONFIG.get('use_residual_weights', True):
                residual_broadcast_params += param.numel()
                residual_params += param.numel()
        elif 'residual_projection' in name:  # NEW
            if CONFIG.get('use_residual_weights', True):
                residual_projection_params += param.numel()
                residual_params += param.numel()
        elif 'phi_codomain_params' in name:
            if CONFIG.get('train_phi_codomain', False):
                codomain_params += param.numel()
        elif 'norm_layers' in name:
            norm_params += param.numel()
        elif 'lateral' in name:
            if CONFIG.get('use_lateral_mixing', True):
                lateral_params += param.numel()
    
    # Core parameters excluding residual and output params
    output_params = 2  # output_scale and output_bias
    
    if CONFIG.get('train_phi_codomain', False):
        total_params = (lambda_params + eta_params + spline_params + residual_params +
                        output_params + codomain_params + norm_params + lateral_params)
    else:
        total_params = (lambda_params + eta_params + spline_params + residual_params +
                        output_params + norm_params + lateral_params)
    
    print(f"Dataset: {dataset} (input_dim={dataset.input_dim}, output_dim={dataset.output_dim})")
    print(f"Architecture: {architecture}")
    print(f"Residual connections: {'enabled' if CONFIG.get('use_residual_weights', True) else 'disabled'}")
    if CONFIG.get('use_residual_weights', True):
        print(f"  Residual style: {CONFIG.get('residual_style','node')}")
    print(f"Normalization: {norm_type} (position: {norm_position}, skip_first: {norm_skip_first})")
    if CONFIG.get('use_lateral_mixing', True):
        print(f"Lateral mixing: {CONFIG.get('lateral_mixing_type', 'cyclic')}")
    print(f"Total number of trainable parameters: {total_params}")
    print(f"  - Lambda weight VECTORS: {lambda_params}")
    print(f"  - Eta shift parameters: {eta_params}")
    print(f"  - Spline parameters: {spline_params}")
    if CONFIG.get('use_residual_weights', True) and residual_params > 0:
        print(f"  - Residual connection weights: {residual_params}")
        if residual_scalar_params > 0:
            print(f"    * Scalar weights (same dims): {residual_scalar_params}")
        if residual_pooling_params > 0:
            print(f"    * Pooling weights (d_in > d_out): {residual_pooling_params}")
        if residual_broadcast_params > 0:
            print(f"    * Broadcast weights (d_in < d_out): {residual_broadcast_params}")
        if residual_projection_params > 0:
            print(f"    * Projection matrices (d_in != d_out): {residual_projection_params}")
    if CONFIG.get('use_lateral_mixing', True) and lateral_params > 0:
        print(f"  - Lateral mixing parameters: {lateral_params}")
    print(f"  - Output scale and bias: {output_params}")
    if CONFIG.get('train_phi_codomain', False) and codomain_params > 0:
        print(f"  - Phi codomain parameters (cc, cr per block): {codomain_params}")
    if norm_params > 0:
        print(f"  - Normalization parameters: {norm_params}")
    
    # Setup optimizer
    if CONFIG.get('use_advanced_scheduler', False):
        # Use AdamW optimizer with weight decay and advanced scheduler
        params = [
            {"params": [p for n, p in model.named_parameters() if "phi_codomain_params" in n], 
             "lr": 0.01, "lr_scale": 1.0},  # Higher base LR for codomain params
            {"params": [p for n, p in model.named_parameters() if "output" in n], 
             "lr": 0.001, "lr_scale": 0.5},  # Medium LR for output params
            {"params": [p for n, p in model.named_parameters() if "lateral" in n], 
             "lr": 0.005, "lr_scale": 0.8},  # Higher LR for lateral params to adapt quickly
            # Everything else (includes splines, lambdas, residuals including any residual_projection)
            {"params": [p for n, p in model.named_parameters() if "phi_codomain_params" not in n 
                        and "output" not in n and "lateral" not in n], 
             "lr": 0.001, "lr_scale": 0.3}
        ]
        optimizer = torch.optim.AdamW(params, weight_decay=CONFIG.get('weight_decay', 1e-6), betas=(0.9, 0.999))
        
        # Create custom scheduler
        scheduler = PlateauAwareCosineAnnealingLR(
            optimizer, 
            base_lr=CONFIG.get('scheduler_base_lr', 1e-4), 
            max_lr=CONFIG.get('scheduler_max_lr', 1e-2),
            patience=CONFIG.get('scheduler_patience', 500),
            threshold=CONFIG.get('scheduler_threshold', 1e-5)
        )
    else:
        # Use original simple Adam setup with optional param groups
        if CONFIG.get('train_phi_codomain', False) or CONFIG.get('use_lateral_mixing', True):
            params = []
            if CONFIG.get('train_phi_codomain', False):
                params.append({"params": [p for n, p in model.named_parameters() if "phi_codomain_params" in n], "lr": 0.001})
            if CONFIG.get('use_lateral_mixing', True):
                params.append({"params": [p for n, p in model.named_parameters() if "lateral" in n], "lr": 0.0005})
            excluded = []
            if CONFIG.get('train_phi_codomain', False):
                excluded.append("phi_codomain_params")
            if CONFIG.get('use_lateral_mixing', True):
                excluded.append("lateral")
            params.append({"params": [p for n, p in model.named_parameters() if not any(exc in n for exc in excluded)], "lr": 0.0003})
        else:
            params = model.parameters()
        optimizer = torch.optim.Adam(params, weight_decay=1e-7)
        scheduler = None
    
    losses = []
    best_loss = float("inf")
    best_checkpoint = None
    
    # Gradient clipping value
    max_grad_norm = CONFIG.get('max_grad_norm', 1.0)
    
    # Reset domain violation tracking if enabled
    if CONFIG.get('track_domain_violations', False):
        model.reset_domain_violation_stats()
    
    # Check if model has BatchNorm
    has_bn = has_batchnorm(model)
    if has_bn:
        print("Model contains BatchNorm layers - will handle train/eval modes appropriately")
    
    # Set model to training mode
    model.train()
    
    pbar = tqdm(range(total_epochs), desc="Training Network")
    for epoch in pbar:
        # Update all domains based on current parameters EVERY ITERATION
        if CONFIG.get('use_theoretical_domains', True):
            model.update_all_domains(allow_resampling=True)
        
        # Regular training
        optimizer.zero_grad()
        output = model(x_train)
        
        # Simple MSE loss only
        loss = torch.mean((output - y_train) ** 2)
        
        loss.backward()
        
        # Gradient clipping to prevent instabilities
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
        
        # Get some monitoring values
        if CONFIG.get('train_phi_codomain', False):
            # Monitor codomain params from first block (if present)
            first_block = model.layers[0]
            if hasattr(first_block, 'phi_codomain_params') and first_block.phi_codomain_params is not None:
                cc_val = first_block.phi_codomain_params.cc.item()
                cr_val = first_block.phi_codomain_params.cr.item()
                cc_grad = first_block.phi_codomain_params.cc.grad.item() if first_block.phi_codomain_params.cc.grad is not None else 0.0
                cr_grad = first_block.phi_codomain_params.cr.grad.item() if first_block.phi_codomain_params.cr.grad is not None else 0.0
            else:
                cc_val = cr_val = cc_grad = cr_grad = 0.0
        else:
            cc_val = cr_val = cc_grad = cr_grad = 0.0
        
        optimizer.step()
        
        # Update learning rate if using advanced scheduler
        if scheduler is not None:
            current_lr = scheduler.step(loss.item())
        else:
            current_lr = optimizer.param_groups[0]['lr']
        
        losses.append(loss.item())
        
        # Update best loss and save checkpoint
        if loss.item() < best_loss:
            if has_bn and CONFIG.get('debug_checkpoint_loading', False):
                print(f"\n[CHECKPOINT DEBUG] New best loss at epoch {epoch}: {loss.item():.4e}")
                for name, module in model.named_modules():
                    if isinstance(module, nn.BatchNorm1d):
                        print(f"  BN stats at best loss - {name}: running_mean[:3]={module.running_mean[:3].cpu().numpy()}, "
                              f"num_batches_tracked={module.num_batches_tracked.item()}")
                        break
            
            best_loss = loss.item()
            
            # Complete snapshot for plotting (exact state)
            if CONFIG.get('debug_checkpoint_loading', False):
                print(f"\n[CHECKPOINT DEBUG] Creating snapshot at epoch {epoch} with loss {loss.item():.4e}")
            plotting_snapshot = {
                'model': copy.deepcopy(model),
                'x_train': x_train.clone(),
                'y_train': y_train.clone(),
                'output': output.detach().clone(),
                'loss': loss.item(),
                'epoch': epoch,
                'device': device
            }
            
            # Save BN stats if any
            bn_statistics = {}
            if has_bn:
                for name, module in model.named_modules():
                    if isinstance(module, nn.BatchNorm1d):
                        bn_statistics[name] = {
                            'running_mean': module.running_mean.clone().cpu(),
                            'running_var': module.running_var.clone().cpu(),
                            'num_batches_tracked': module.num_batches_tracked.clone().cpu(),
                            'momentum': module.momentum,
                            'eps': module.eps,
                            'affine': module.affine,
                            'track_running_stats': module.track_running_stats,
                            'weight': module.weight.clone().cpu() if module.affine else None,
                            'bias': module.bias.clone().cpu() if module.affine else None
                        }
            
            # Create complete checkpoint with all state
            best_checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict().copy(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': loss.item(),
                'has_batchnorm': has_bn,
                'bn_statistics': bn_statistics,
                'domain_states': model.get_all_domain_states(),
                'domain_ranges': model.get_domain_ranges(),
                'training_mode': model.training,
                'x_train': x_train.cpu().clone(),
                'y_train': y_train.cpu().clone(),
                'output': output.detach().cpu().clone(),
                # Add model creation params (for exact reconstruction)
                'model_params': {
                    'input_dim': dataset.input_dim,
                    'architecture': architecture,
                    'final_dim': dataset.output_dim,
                    'phi_knots': phi_knots,
                    'Phi_knots': Phi_knots,
                    'norm_type': norm_type,
                    'norm_position': norm_position,
                    'norm_skip_first': norm_skip_first
                },
                # Persist residual style used at best epoch for deterministic reload
                'residual_style': CONFIG.get('residual_style', 'node'),
                # Plotting snapshot for perfect restoration
                'plotting_snapshot': plotting_snapshot
            }
        
        # Calculate output standard deviation for monitoring
        std_output = torch.std(output)
        std_target = torch.std(y_train)
        
        pbar_dict = {
            'loss': f'{loss.item():.2e}',
            'best': f'{best_loss:.2e}',
            'std_out': f'{std_output.item():.3f}',
            'std_tar': f'{std_target.item():.3f}'
        }
        
        if CONFIG.get('train_phi_codomain', False):
            pbar_dict['cc'] = f'{cc_val:.3f}'
            pbar_dict['cr'] = f'{cr_val:.3f}'
        
        if CONFIG.get('use_advanced_scheduler', False):
            pbar_dict['lr'] = f'{current_lr:.2e}'
            if CONFIG.get('train_phi_codomain', False):
                pbar_dict['g_cc'] = f'{cc_grad:.3e}'
                pbar_dict['g_cr'] = f'{cr_grad:.3e}'
        
        pbar.set_postfix(pbar_dict)
        
        if (epoch + 1) % print_every == 0:
            if CONFIG.get('use_advanced_scheduler', False):
                print(f"Epoch {epoch+1}: Loss = {loss.item():.4e}, Best = {best_loss:.4e}, LR = {current_lr:.4e}")
            else:
                print(f"Epoch {epoch+1}: Loss = {loss.item():.4e}, Best = {best_loss:.4e}")
            
            if CONFIG.get('debug_domains', False):
                print("\nDomain ranges:")
                for idx, layer in enumerate(model.layers):
                    print(f"  Layer {idx}: phi domain=[{layer.phi.in_min:.3f}, {layer.phi.in_max:.3f}], "
                          f"Phi domain=[{layer.Phi.in_min:.3f}, {layer.Phi.in_max:.3f}]")
                print()
            
            if CONFIG.get('track_domain_violations', False) and (epoch + 1) % (print_every * 5) == 0:
                model.print_domain_violation_report()
    
    # Load best checkpoint before returning
    if best_checkpoint is not None and not no_load_best:
        print(f"\n{'='*60}")
        print("CHECKPOINT LOADING DEBUG INFO:")
        print(f"Best checkpoint from epoch: {best_checkpoint['epoch']}")
        print(f"Best checkpoint loss: {best_checkpoint['loss']:.4e}")
        print(f"Current (final) loss: {losses[-1]:.4e}")
        print(f"Loss ratio (final/best): {losses[-1]/best_checkpoint['loss']:.2f}x")
        
        if 'plotting_snapshot' in best_checkpoint:
            print("\nUsing plotting snapshot from checkpoint - this ensures perfect restoration!")
            plotting_snapshot = best_checkpoint['plotting_snapshot']
            print(f"\nSnapshot loaded from epoch {plotting_snapshot['epoch']} with loss {plotting_snapshot['loss']:.4e}")
            x_train = plotting_snapshot['x_train'].to(device)
            y_train = plotting_snapshot['y_train'].to(device)
            print(f"\n{'='*60}\n")
        else:
            print("\nWARNING: Old checkpoint format without plotting snapshot")
        
        # Get a sample output before loading checkpoint
        with torch.no_grad():
            output_before = model(x_train[:5]).cpu().numpy()
        print(f"Sample outputs BEFORE loading checkpoint: {output_before.flatten()[:5]}")
        
        # Enable checkpoint debug logging
        CONFIG['debug_checkpoint_loading'] = True

        # Ensure same residual semantics as at best epoch
        if 'residual_style' in best_checkpoint:
            CONFIG['residual_style'] = best_checkpoint['residual_style']
        
        # Create a fresh model instance without domain initialization
        if 'model_params' in best_checkpoint:
            print("Creating fresh model instance for checkpoint loading...")
            domain_ranges = best_checkpoint.get('domain_ranges', None)
            fresh_model = SprecherMultiLayerNetwork(
                **best_checkpoint['model_params'],
                initialize_domains=False,
                domain_ranges=domain_ranges
            ).to(device)
            
            fresh_model.train()
            print(f"[CHECKPOINT DEBUG] Fresh model training mode: {fresh_model.training}")
            
            print("\n[CHECKPOINT DEBUG] Testing fresh model BEFORE loading state_dict:")
            with torch.no_grad():
                test_out = fresh_model(x_train[:5])
                print(f"  Fresh model output: {test_out.cpu().numpy().flatten()[:5]}")
            
            print("\n[CHECKPOINT DEBUG] About to load state_dict...")
            fresh_model.load_state_dict(best_checkpoint['model_state_dict'])
            print("\nCheckpoint loaded into fresh model successfully!")
            print(f"[CHECKPOINT DEBUG] Model training mode after state_dict: {fresh_model.training}")
            
            if best_checkpoint.get('has_batchnorm', False):
                print("\n[CHECKPOINT DEBUG] BatchNorm state after loading state_dict:")
                for name, module in fresh_model.named_modules():
                    if isinstance(module, nn.BatchNorm1d):
                        print(f"  {name}: num_batches_tracked={module.num_batches_tracked.item()}, "
                              f"momentum={module.momentum}, training={module.training}")
                        break
            
            if 'domain_states' in best_checkpoint:
                print("\n[CHECKPOINT DEBUG] Restoring domain states after state_dict load...")
                fresh_model.set_all_domain_states(best_checkpoint['domain_states'])
                print("Domain states restored after state_dict!")
                
                print("\n[CHECKPOINT DEBUG] Checking theoretical ranges after restoration:")
                for i, layer in enumerate(fresh_model.layers):
                    ranges = layer.get_theoretical_ranges()
                    print(f"  Layer {i}: {ranges}")
            
            model = fresh_model
        else:
            print("WARNING: Old checkpoint format, using existing model")
            model.load_state_dict(best_checkpoint['model_state_dict'])
            print("Checkpoint loaded successfully!")
        
        # Always restore domain states if present
        if 'domain_states' in best_checkpoint:
            print("Restoring spline domain states from checkpoint...")
            model.set_all_domain_states(best_checkpoint['domain_states'])
            print("Domain states restored successfully!")
        
        # Restore exact BN stats if saved
        if best_checkpoint.get('has_batchnorm', False) and 'bn_statistics' in best_checkpoint:
            print("\n[CHECKPOINT DEBUG] Restoring exact BatchNorm statistics from best epoch...")
            for name, module in model.named_modules():
                if isinstance(module, nn.BatchNorm1d) and name in best_checkpoint['bn_statistics']:
                    saved_stats = best_checkpoint['bn_statistics'][name]
                    module.running_mean.copy_(saved_stats['running_mean'])
                    module.running_var.copy_(saved_stats['running_var'])
                    module.num_batches_tracked.copy_(saved_stats['num_batches_tracked'])
                    if module.affine and saved_stats.get('weight') is not None:
                        module.weight.data.copy_(saved_stats['weight'])
                        module.bias.data.copy_(saved_stats['bias'])
                    print(f"  {name}: Restored exact BN stats from epoch {best_checkpoint['epoch']}")
        elif best_checkpoint.get('has_batchnorm', False):
            print("\n[CHECKPOINT DEBUG] WARNING: No saved BN statistics in checkpoint (old format)")
            print("  BatchNorm statistics will be from final training epoch, not best epoch!")
        
        # Keep training mode if saved that way
        if best_checkpoint.get('training_mode', True):
            model.train()
            print("Model kept in training mode (as it was during best checkpoint)")
        else:
            model.eval()
            print("Model set to eval mode")
        
        # Verify outputs after loading
        if has_bn and CONFIG.get('debug_checkpoint_loading', False):
            bn_stats_before_forward = {}
            for name, module in model.named_modules():
                if isinstance(module, nn.BatchNorm1d):
                    bn_stats_before_forward[name] = {
                        'mean': module.running_mean.clone(),
                        'var': module.running_var.clone(),
                        'tracked': module.num_batches_tracked.clone()
                    }
        
        with torch.no_grad():
            output_after = model(x_train[:5]).cpu().numpy()
        print(f"Sample outputs AFTER loading checkpoint: {output_after.flatten()[:5]}")
        print(f"Output change: {np.abs(output_after - output_before).mean():.4e}")
        
        if has_bn and CONFIG.get('debug_checkpoint_loading', False):
            print("\n[CHECKPOINT DEBUG] Verifying BN stats didn't change during forward pass...")
            for name, module in model.named_modules():
                if isinstance(module, nn.BatchNorm1d) and name in bn_stats_before_forward:
                    before = bn_stats_before_forward[name]
                    mean_changed = not torch.allclose(module.running_mean, before['mean'])
                    var_changed = not torch.allclose(module.running_var, before['var'])
                    tracked_changed = module.num_batches_tracked.item() != before['tracked'].item()
                    if mean_changed or var_changed or tracked_changed:
                        print(f"  WARNING: {name} stats changed during forward pass!")
                        print(f"    mean changed: {mean_changed}")
                        print(f"    var changed: {var_changed}")
                        print(f"    tracked changed: {tracked_changed} ({before['tracked'].item()} -> {module.num_batches_tracked.item()})")
                    else:
                        print(f"  [OK] {name} stats unchanged")
                    break
        
        # If we have saved training data, verify the checkpoint
        if 'x_train' in best_checkpoint:
            print("\nVerifying checkpoint with saved training data...")
            saved_x = best_checkpoint['x_train'].to(device)
            saved_y = best_checkpoint['y_train'].to(device)
            saved_output = best_checkpoint['output']
            
            with torch.no_grad():
                current_output = model(saved_x)
                current_loss = torch.mean((current_output - saved_y) ** 2).item()
            
            print(f"Saved checkpoint loss: {best_checkpoint['loss']:.4e}")
            print(f"Current loss on saved data: {current_loss:.4e}")
            print(f"Loss difference: {abs(current_loss - best_checkpoint['loss']):.4e}")
            
            output_diff = torch.abs(current_output.cpu() - saved_output).max().item()
            print(f"Max output difference: {output_diff:.4e}")
            
            tolerance = 1e-8 if best_checkpoint.get('has_batchnorm', False) else 1e-9
            if abs(current_loss - best_checkpoint['loss']) > tolerance:
                print("\n" + "="*60)
                print("WARNING: Checkpoint restoration failed!")
                print(f"Expected loss: {best_checkpoint['loss']:.4e}")
                print(f"Actual loss: {current_loss:.4e}")
                print(f"Difference: {abs(current_loss - best_checkpoint['loss']):.4e}")
                print("="*60 + "\n")
                
                print("Additional debugging information:")
                print(f"Model training mode: {model.training}")
                print(f"Has BatchNorm: {best_checkpoint.get('has_batchnorm', False)}")
                
                print("\nCurrent domain ranges:")
                for i, layer in enumerate(model.layers):
                    print(f"  Layer {i}: phi=[{layer.phi.in_min:.6f}, {layer.phi.in_max:.6f}], "
                          f"Phi=[{layer.Phi.in_min:.6f}, {layer.Phi.in_max:.6f}]")
                
                print("\nSpline coefficient ranges:")
                for i, layer in enumerate(model.layers):
                    phi_coeffs = layer.phi.get_coeffs()
                    Phi_coeffs = layer.Phi.get_coeffs()
                    print(f"  Layer {i}: phi coeffs=[{phi_coeffs.min():.6f}, {phi_coeffs.max():.6f}], "
                          f"Phi coeffs=[{Phi_coeffs.min():.6f}, {Phi_coeffs.max():.6f}]")
            else:
                print("[OK] Checkpoint verification passed - outputs match!")
            
            x_train = saved_x
            y_train = saved_y
        
        CONFIG['debug_checkpoint_loading'] = False
        print("\n[CHECKPOINT DEBUG] Debug logging disabled")
        
        if best_checkpoint.get('has_batchnorm', False):
            print("\nDEBUG: BatchNorm stats AFTER loading and domain update:")
            for name, module in model.named_modules():
                if isinstance(module, nn.BatchNorm1d):
                    print(f"  {name}: mean={module.running_mean.data[:3].cpu().numpy()}, var={module.running_var.data[:3].cpu().numpy()}")
        
        if best_checkpoint.get('has_batchnorm', False):
            for name, module in model.named_modules():
                if isinstance(module, nn.BatchNorm1d):
                    print(f"\nBatchNorm layer '{name}' stats BEFORE recalc:")
                    print(f"  Running mean: {module.running_mean.data[:3].cpu().numpy()}")
                    print(f"  Running var: {module.running_var.data[:3].cpu().numpy()}")
                    break
            
            if bn_recalc_on_load:
                print("\nRecalculating BatchNorm statistics on training data...")
                recalculate_bn_stats(model, x_train, num_passes=10)
                
                for name, module in model.named_modules():
                    if isinstance(module, nn.BatchNorm1d):
                        print(f"\nBatchNorm layer '{name}' stats AFTER recalc:")
                        print(f"  Running mean: {module.running_mean.data[:3].cpu().numpy()}")
                        print(f"  Running var: {module.running_var.data[:3].cpu().numpy()}")
                        break
                
                with torch.no_grad():
                    output_after_recalc = model(x_train[:5]).cpu().numpy()
                print(f"Sample outputs AFTER BN recalc: {output_after_recalc.flatten()[:5]}")
            else:
                print(f"Using saved BatchNorm statistics from best checkpoint (epoch {best_checkpoint['epoch']})")
        
        print(f"{'='*60}\n")
        
    elif no_load_best:
        print("\nSkipping best model loading (--no_load_best flag set)")
        print(f"Using final model state with loss: {losses[-1]:.4e}")
        print(f"Best loss during training was: {best_loss:.4e}")
        model.train()
        
        print("\nCreating snapshot of final model state for plotting...")
        with torch.no_grad():
            final_output = model(x_train)
            final_loss = torch.mean((final_output - y_train) ** 2).item()
        
        plotting_snapshot = {
            'model': copy.deepcopy(model),
            'x_train': x_train.clone(),
            'y_train': y_train.clone(),
            'output': final_output.detach().clone(),
            'loss': final_loss,
            'epoch': total_epochs - 1,
            'device': device
        }
        
        print(f"Snapshot created with final loss: {final_loss:.4e}")
    else:
        # No checkpoint was saved (shouldn't happen normally)
        model.train()
    
    print(f"\nDEBUG: Model is in {'training' if model.training else 'eval'} mode for final operations")
    
    print("\nDEBUG: Final model output:")
    with torch.no_grad():
        test_out = model(x_train[:5])
        print(f"Output: {test_out.cpu().numpy().flatten()[:5]}")
    
    print("\nDEBUG: Skipping final domain update (domains already set correctly)")
    
    if CONFIG.get('track_domain_violations', False):
        print("\nFinal domain violation report:")
        model.print_domain_violation_report()
    
    # Print final parameters (compact preview for projection matrices to avoid huge logs)
    print("\nFinal parameters:")
    for idx, layer in enumerate(model.layers, start=1):
        print(f"Block {idx}: eta = {layer.eta.item():.6f}")
        print(f"Block {idx}: lambdas shape = {tuple(layer.lambdas.shape)}")
        print(f"Block {idx}: lambdas =")
        print(layer.lambdas.detach().cpu().numpy())
        if CONFIG.get('use_residual_weights', True):
            if hasattr(layer, 'residual_weight') and layer.residual_weight is not None:
                print(f"Block {idx}: residual_weight = {layer.residual_weight.item():.6f}")
            elif hasattr(layer, 'residual_projection') and layer.residual_projection is not None:
                W = layer.residual_projection.detach().cpu().numpy()
                print(f"Block {idx}: residual_projection shape = {W.shape}")
                # Print a small preview (top-left 4x4) to keep console readable
                r_preview = min(4, W.shape[0]); c_preview = min(4, W.shape[1])
                print(f"Block {idx}: residual_projection preview (top-left):")
                print(W[:r_preview, :c_preview])
            elif hasattr(layer, 'residual_pooling_weights') and layer.residual_pooling_weights is not None:
                print(f"Block {idx}: residual_pooling_weights shape = {tuple(layer.residual_pooling_weights.shape)}")
                print(f"Block {idx}: residual_pooling_weights =")
                print(layer.residual_pooling_weights.detach().cpu().numpy())
                print(f"Block {idx}: pooling assignment = {layer.pooling_assignment.cpu().numpy()}")
                print(f"Block {idx}: pooling counts = {layer.pooling_counts.cpu().numpy()}")
            elif hasattr(layer, 'residual_broadcast_weights') and layer.residual_broadcast_weights is not None:
                print(f"Block {idx}: residual_broadcast_weights shape = {tuple(layer.residual_broadcast_weights.shape)}")
                print(f"Block {idx}: residual_broadcast_weights =")
                print(layer.residual_broadcast_weights.detach().cpu().numpy())
                print(f"Block {idx}: broadcast sources = {layer.broadcast_sources.cpu().numpy()}")
        
        if CONFIG.get('use_lateral_mixing', True):
            if hasattr(layer, 'lateral_scale') and layer.lateral_scale is not None:
                print(f"Block {idx}: lateral_scale = {layer.lateral_scale.item():.6f}")
                if CONFIG.get('lateral_mixing_type', 'cyclic') == 'bidirectional':
                    print(f"Block {idx}: lateral_weights_forward =")
                    print(layer.lateral_weights_forward.detach().cpu().numpy())
                    print(f"Block {idx}: lateral_weights_backward =")
                    print(layer.lateral_weights_backward.detach().cpu().numpy())
                else:
                    print(f"Block {idx}: lateral_weights =")
                    print(layer.lateral_weights.detach().cpu().numpy())
        
        if CONFIG.get('train_phi_codomain', False):
            if hasattr(layer, 'phi_codomain_params') and layer.phi_codomain_params is not None:
                print(f"Block {idx}: Phi codomain center = {layer.phi_codomain_params.cc.item():.6f}")
                print(f"Block {idx}: Phi codomain radius = {layer.phi_codomain_params.cr.item():.6f}")
        
        if hasattr(layer, 'input_range') and layer.input_range is not None:
            print(f"Block {idx}: input_range = {layer.input_range}")
        if hasattr(layer, 'output_range') and layer.output_range is not None:
            print(f"Block {idx}: output_range = {layer.output_range}")
        print()
    
    print(f"Final loss: {best_loss:.4e}")
    
    # Ensure plotting_snapshot is defined
    if 'plotting_snapshot' not in locals():
        print("\nCreating plotting snapshot from current model state...")
        with torch.no_grad():
            current_output = model(x_train)
            current_loss = torch.mean((current_output - y_train) ** 2).item()
        
        plotting_snapshot = {
            'model': copy.deepcopy(model),
            'x_train': x_train.clone(),
            'y_train': y_train.clone(),
            'output': current_output.detach().clone(),
            'loss': current_loss,
            'epoch': best_checkpoint['epoch'] if best_checkpoint else total_epochs - 1,
            'device': device
        }
    
    return plotting_snapshot, losses