"""Training utilities for Sprecher Networks."""

import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm
from .model import SprecherMultiLayerNetwork, NormalizationType
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


def train_network(dataset, architecture, total_epochs=4000, print_every=400, 
                  device="cpu", phi_knots=100, Phi_knots=100, seed=None,
                  normalization_type=NormalizationType.NONE):
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
        normalization_type: Type of normalization to use
    
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
        normalization_type=normalization_type
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
    print(f"Normalization: {normalization_type.value}")
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
    best_state = None
    
    # Gradient clipping value
    max_grad_norm = CONFIG['max_grad_norm']
    
    # Reset domain violation tracking if enabled
    if CONFIG.get('track_domain_violations', False):
        model.reset_domain_violation_stats()
    
    # Set model to training mode
    model.train()
    
    pbar = tqdm(range(total_epochs), desc="Training Network")
    for epoch in pbar:
        # Update all domains based on current parameters EVERY ITERATION
        if CONFIG.get('use_theoretical_domains', True):
            model.update_all_domains()
        
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
        
        # Update best loss
        if loss.item() < best_loss:
            best_loss = loss.item()
            best_state = model.state_dict()
        
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
    
    # Load best state and set to eval mode
    if best_state is not None:
        model.load_state_dict(best_state)
        model.eval()  # CRITICAL for proper BatchNorm behavior!
        print(f"\nLoaded best model with loss: {best_loss:.4e}")
    
    # Final domain update in eval mode
    if CONFIG.get('use_theoretical_domains', True):
        model.update_all_domains()
    
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