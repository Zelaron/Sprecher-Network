"""Training utilities for Sprecher Networks."""

import torch
import torch.nn as nn
import numpy as np
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


def train_network(dataset, architecture, total_epochs=4000, print_every=400, 
                  device="cpu", phi_knots=100, Phi_knots=100, seed=None,
                  norm_type='none', norm_position='after', norm_skip_first=True,
                  no_load_best=False, bn_recalc_on_load=False):
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
    
    Returns:
        model: Trained model
        losses: List of training losses
        layers: Model layers
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
    
    # Compute parameter counts for TRUE SPRECHER NETWORK
    # Count lambda parameters (now vectors, not matrices)
    lambda_params = 0
    eta_params = 0
    spline_params = 0
    residual_params = 0
    codomain_params = 0
    norm_params = 0
    
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
            
        if 'lambdas' in name:
            # Lambda parameters are now VECTORS
            lambda_params += param.numel()
        elif 'eta' in name:
            eta_params += param.numel()
        elif 'coeffs' in name or 'log_increments' in name:
            # Spline parameters
            spline_params += param.numel()
        elif 'residual_weight' in name or 'residual_projection' in name:
            if CONFIG['use_residual_weights']:
                residual_params += param.numel()
        elif 'phi_codomain_params' in name:
            if CONFIG['train_phi_codomain']:
                codomain_params += param.numel()
        elif 'norm_layers' in name:
            norm_params += param.numel()
    
    # Core parameters excluding residual and output params
    output_params = 2  # output_scale and output_bias
    
    if CONFIG['train_phi_codomain']:
        total_params = lambda_params + eta_params + spline_params + residual_params + output_params + codomain_params + norm_params
    else:
        total_params = lambda_params + eta_params + spline_params + residual_params + output_params + norm_params
    
    print(f"Dataset: {dataset} (input_dim={dataset.input_dim}, output_dim={dataset.output_dim})")
    print(f"Architecture: {architecture}")
    print(f"Normalization: {norm_type} (position: {norm_position}, skip_first: {norm_skip_first})")
    print(f"Total number of trainable parameters: {total_params}")
    print(f"  - Lambda weight VECTORS: {lambda_params} (TRUE SPRECHER!)")
    print(f"  - Eta shift parameters: {eta_params}")
    print(f"  - Spline parameters: {spline_params}")
    if CONFIG['use_residual_weights'] and residual_params > 0:
        print(f"  - Residual connection weights: {residual_params}")
    print(f"  - Output scale and bias: {output_params}")
    if CONFIG['train_phi_codomain'] and codomain_params > 0:
        print(f"  - Phi codomain parameters (cc, cr per block): {codomain_params}")
    if norm_params > 0:
        print(f"  - Normalization parameters: {norm_params}")
    
    # Setup optimizer
    if CONFIG['use_advanced_scheduler']:
        # Use AdamW optimizer with weight decay and advanced scheduler
        params = [
            {"params": [p for n, p in model.named_parameters() if "phi_codomain_params" in n], 
             "lr": 0.01, "lr_scale": 1.0},  # Higher base LR for codomain params
            {"params": [p for n, p in model.named_parameters() if "output" in n], 
             "lr": 0.001, "lr_scale": 0.5},  # Medium LR for output params
            {"params": [p for n, p in model.named_parameters() if "phi_codomain_params" not in n and "output" not in n], 
             "lr": 0.001, "lr_scale": 0.3}  # Lower LR for spline params
        ]
        optimizer = torch.optim.AdamW(params, weight_decay=CONFIG['weight_decay'], betas=(0.9, 0.999))
        
        # Create custom scheduler
        scheduler = PlateauAwareCosineAnnealingLR(
            optimizer, 
            base_lr=CONFIG['scheduler_base_lr'], 
            max_lr=CONFIG['scheduler_max_lr'],
            patience=CONFIG['scheduler_patience'],
            threshold=CONFIG['scheduler_threshold']
        )
    else:
        # Use original simple Adam optimizer setup
        if CONFIG['train_phi_codomain']:
            params = [
                {"params": [p for n, p in model.named_parameters() if "phi_codomain_params" in n], "lr": 0.001},
                {"params": [p for n, p in model.named_parameters() if "phi_codomain_params" not in n], "lr": 0.0003}
            ]
        else:
            params = model.parameters()
        optimizer = torch.optim.Adam(params, weight_decay=1e-7)
        scheduler = None
    
    losses = []
    best_loss = float("inf")
    best_checkpoint = None
    
    # Gradient clipping value
    max_grad_norm = CONFIG['max_grad_norm']
    
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
        # During training, allow resampling (default behavior)
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
        if CONFIG['train_phi_codomain']:
            # Get codomain values from first block for monitoring
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
            best_loss = loss.item()
            best_checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': loss.item(),
                'has_batchnorm': has_bn
            }
        
        # Calculate output standard deviation for monitoring
        std_output = torch.std(output)
        std_target = torch.std(y_train)
        
        pbar_dict = {
            'loss': f'{loss.item():.2e}',
            'best': f'{best_loss:.2e}',  # Show best loss
            'std_out': f'{std_output.item():.3f}',
            'std_tar': f'{std_target.item():.3f}'
        }
        
        if CONFIG['train_phi_codomain']:
            pbar_dict['cc'] = f'{cc_val:.3f}'
            pbar_dict['cr'] = f'{cr_val:.3f}'
        
        if CONFIG['use_advanced_scheduler']:
            pbar_dict['lr'] = f'{current_lr:.2e}'
            if CONFIG['train_phi_codomain']:
                pbar_dict['g_cc'] = f'{cc_grad:.3e}'
                pbar_dict['g_cr'] = f'{cr_grad:.3e}'
        
        pbar.set_postfix(pbar_dict)
        
        if (epoch + 1) % print_every == 0:
            if CONFIG['use_advanced_scheduler']:
                print(f"Epoch {epoch+1}: Loss = {loss.item():.4e}, Best = {best_loss:.4e}, LR = {current_lr:.4e}")
            else:
                print(f"Epoch {epoch+1}: Loss = {loss.item():.4e}, Best = {best_loss:.4e}")
            
            # Print domain information if debugging
            if CONFIG.get('debug_domains', False):
                print("\nDomain ranges:")
                for idx, layer in enumerate(model.layers):
                    print(f"  Layer {idx}: φ domain=[{layer.phi.in_min:.3f}, {layer.phi.in_max:.3f}], "
                          f"Φ domain=[{layer.Phi.in_min:.3f}, {layer.Phi.in_max:.3f}]")
                print()
            
            # Print domain violation statistics if tracking
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
        
        # Get a sample output before loading checkpoint
        model.eval()
        with torch.no_grad():
            output_before = model(x_train[:5]).cpu().numpy()
        print(f"Sample outputs BEFORE loading checkpoint: {output_before.flatten()[:5]}")
        
        # Load the checkpoint
        model.load_state_dict(best_checkpoint['model_state_dict'])
        print("Checkpoint loaded successfully!")
        
        # IMMEDIATELY set to eval mode to prevent any training-mode behavior
        model.eval()
        print("Model set to eval mode immediately after loading checkpoint")
        
        # Get a sample output after loading checkpoint
        with torch.no_grad():
            output_after = model(x_train[:5]).cpu().numpy()
        print(f"Sample outputs AFTER loading checkpoint: {output_after.flatten()[:5]}")
        print(f"Output change: {np.abs(output_after - output_before).mean():.4e}")
        
        # Update domains with loaded parameters WITHOUT resampling
        # This is done AFTER setting to eval mode
        if CONFIG.get('use_theoretical_domains', True):
            print("\nDEBUG: About to update domains (without resampling)...")
            model.update_all_domains(allow_resampling=False)
            print("DEBUG: Domain update complete")
            
            # Verify output hasn't changed after domain update
            with torch.no_grad():
                output_after_domain = model(x_train[:5]).cpu().numpy()
            print(f"Sample outputs AFTER domain update: {output_after_domain.flatten()[:5]}")
            if np.abs(output_after_domain - output_after).max() > 1e-6:
                print("WARNING: Output changed after domain update!")
        
        # DEBUG: Check BatchNorm stats after loading and domain update
        if best_checkpoint['has_batchnorm']:
            print("\nDEBUG: BatchNorm stats AFTER loading and domain update:")
            for name, module in model.named_modules():
                if isinstance(module, nn.BatchNorm1d):
                    print(f"  {name}: mean={module.running_mean.data[:3].cpu().numpy()}, var={module.running_var.data[:3].cpu().numpy()}")
        
        # Handle BatchNorm statistics based on user preference
        if best_checkpoint['has_batchnorm']:
            # Check current BN stats
            for name, module in model.named_modules():
                if isinstance(module, nn.BatchNorm1d):
                    print(f"\nBatchNorm layer '{name}' stats BEFORE recalc:")
                    print(f"  Running mean: {module.running_mean.data[:3].cpu().numpy()}")
                    print(f"  Running var: {module.running_var.data[:3].cpu().numpy()}")
                    break
            
            if bn_recalc_on_load:
                print("\nRecalculating BatchNorm statistics on training data...")
                # recalc_bn_stats will handle train/eval modes internally
                recalculate_bn_stats(model, x_train, num_passes=10)
                
                # Check BN stats after recalc
                for name, module in model.named_modules():
                    if isinstance(module, nn.BatchNorm1d):
                        print(f"\nBatchNorm layer '{name}' stats AFTER recalc:")
                        print(f"  Running mean: {module.running_mean.data[:3].cpu().numpy()}")
                        print(f"  Running var: {module.running_var.data[:3].cpu().numpy()}")
                        break
                
                # Check output after recalc
                model.eval()
                with torch.no_grad():
                    output_after_recalc = model(x_train[:5]).cpu().numpy()
                print(f"Sample outputs AFTER BN recalc: {output_after_recalc.flatten()[:5]}")
            else:
                print(f"Using saved BatchNorm statistics from best checkpoint (epoch {best_checkpoint['epoch']})")
                print("Note: These statistics reflect the training state at that epoch")
        
        print(f"{'='*60}\n")
        
    elif no_load_best:
        print("\nSkipping best model loading (--no_load_best flag set)")
        print(f"Using final model state with loss: {losses[-1]:.4e}")
        print(f"Best loss during training was: {best_loss:.4e}")
        # Set to eval mode
        model.eval()
    else:
        # No checkpoint was saved (shouldn't happen normally)
        model.eval()
    
    # The model is already in eval mode from above
    print("\nDEBUG: Model is in eval mode for final operations")
    
    # DEBUG: Check model output before final domain update
    print("\nDEBUG: Model output BEFORE final domain update:")
    with torch.no_grad():
        test_out = model(x_train[:5])
        print(f"Output: {test_out.cpu().numpy().flatten()[:5]}")
    
    # Update domains one final time with best parameters WITHOUT resampling
    if CONFIG.get('use_theoretical_domains', True):
        model.update_all_domains(allow_resampling=False)
    
    print("\nDEBUG: Model output AFTER final domain update:")
    with torch.no_grad():
        test_out = model(x_train[:5])
        print(f"Output: {test_out.cpu().numpy().flatten()[:5]}")
    
    # Print final domain violation report if tracking
    if CONFIG.get('track_domain_violations', False):
        print("\nFinal domain violation report:")
        model.print_domain_violation_report()
    
    # Print final eta and lambda parameters for each block
    print("\nFinal parameters:")
    for idx, layer in enumerate(model.layers, start=1):
        print(f"Block {idx}: eta = {layer.eta.item():.6f}")
        print(f"Block {idx}: lambdas shape = {tuple(layer.lambdas.shape)} (VECTOR!)")
        print(f"Block {idx}: lambdas =")
        print(layer.lambdas.detach().cpu().numpy())
        if CONFIG['use_residual_weights']:
            if hasattr(layer, 'residual_weight') and layer.residual_weight is not None:
                print(f"Block {idx}: residual_weight = {layer.residual_weight.item():.6f}")
            elif hasattr(layer, 'residual_projection') and layer.residual_projection is not None:
                print(f"Block {idx}: residual_projection weight shape = {tuple(layer.residual_projection.weight.shape)}")
        if CONFIG['train_phi_codomain']:
            if hasattr(layer, 'phi_codomain_params') and layer.phi_codomain_params is not None:
                print(f"Block {idx}: Phi codomain center = {layer.phi_codomain_params.cc.item():.6f}")
                print(f"Block {idx}: Phi codomain radius = {layer.phi_codomain_params.cr.item():.6f}")
        
        # Print theoretical ranges
        if hasattr(layer, 'input_range') and layer.input_range is not None:
            print(f"Block {idx}: input_range = {layer.input_range}")
        if hasattr(layer, 'output_range') and layer.output_range is not None:
            print(f"Block {idx}: output_range = {layer.output_range}")
        print()
    
    print(f"Final loss: {best_loss:.4e}")
    
    return model, losses, model.layers