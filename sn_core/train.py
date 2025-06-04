"""Training utilities for Sprecher Networks."""

import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm
from .model import SprecherMultiLayerNetwork, USE_RESIDUAL_WEIGHTS, TRAIN_PHI_RANGE


# Global training configuration
USE_ADVANCED_SCHEDULER = True


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
                  device="cpu", phi_knots=100, Phi_knots=100, seed=45):
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
        seed: Random seed
    
    Returns:
        model: Trained model
        losses: List of training losses
        layers: Model layers
    """
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
        Phi_knots=Phi_knots
    ).to(device)
    
    # Initialize output bias to target mean for faster convergence
    with torch.no_grad():
        model.output_bias.data = y_mean.mean()
        model.output_scale.data = torch.tensor(0.1)  # Start with small scale
    
    # Compute parameter counts
    core_params = sum(
        p.numel()
        for n, p in model.named_parameters()
        if p.requires_grad and "phi_range_params" not in n
    )
    phi_params = sum(p.numel() for p in model.phi_range_params.parameters())
    
    # Count additional architectural parameters
    residual_params = 0
    if USE_RESIDUAL_WEIGHTS:
        for layer in model.layers:
            if hasattr(layer, 'residual_weight') and layer.residual_weight is not None:
                residual_params += 1
            elif hasattr(layer, 'residual_projection') and layer.residual_projection is not None:
                residual_params += layer.residual_projection.weight.numel()
    output_params = 2  # output_scale and output_bias
    
    if TRAIN_PHI_RANGE:
        total_params = core_params + phi_params
    else:
        total_params = core_params
    
    print(f"Dataset: {dataset} (input_dim={dataset.input_dim}, output_dim={dataset.output_dim})")
    print(f"Architecture: {architecture}")
    print(f"Total number of trainable parameters: {total_params}")
    print(f"  - Spline, weight, and shift parameters: {core_params - residual_params - output_params}")
    if USE_RESIDUAL_WEIGHTS:
        print(f"  - Residual connection weights: {residual_params}")
    print(f"  - Output scale and bias: {output_params}")
    if TRAIN_PHI_RANGE:
        print(f"  - Global range parameters (dc, dr, cc, cr): {phi_params}")
    
    # Setup optimizer
    if USE_ADVANCED_SCHEDULER:
        # Use AdamW optimizer with weight decay and advanced scheduler
        params = [
            {"params": [p for n, p in model.named_parameters() if "phi_range_params" in n], 
             "lr": 0.01, "lr_scale": 1.0},  # Higher base LR for range params
            {"params": [p for n, p in model.named_parameters() if "output" in n], 
             "lr": 0.001, "lr_scale": 0.5},  # Medium LR for output params
            {"params": [p for n, p in model.named_parameters() if "phi_range_params" not in n and "output" not in n], 
             "lr": 0.001, "lr_scale": 0.3}  # Lower LR for spline params
        ]
        optimizer = torch.optim.AdamW(params, weight_decay=1e-6, betas=(0.9, 0.999))
        
        # Create custom scheduler
        scheduler = PlateauAwareCosineAnnealingLR(
            optimizer, 
            base_lr=0.0001, 
            max_lr=0.01,
            patience=500,
            threshold=1e-5
        )
    else:
        # Use original simple Adam optimizer setup
        params = [
            {"params": [p for n, p in model.named_parameters() if "phi_range_params" in n], "lr": 0.001},
            {"params": [p for n, p in model.named_parameters() if "phi_range_params" not in n], "lr": 0.0003}
        ]
        optimizer = torch.optim.Adam(params, weight_decay=1e-7)
        scheduler = None
    
    losses = []
    best_loss = float("inf")
    best_state = None
    
    # Gradient clipping value
    max_grad_norm = 1.0
    
    pbar = tqdm(range(total_epochs), desc="Training Network")
    for epoch in pbar:
        optimizer.zero_grad()
        output = model(x_train)
        
        # Simple MSE loss only
        loss = torch.mean((output - y_train) ** 2)
        
        loss.backward()
        
        # Gradient clipping to prevent instabilities
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
        
        # Capture gradients for the global range parameters for debugging
        dc_val = model.phi_range_params.dc.item()
        dr_val = model.phi_range_params.dr.item()
        cc_val = model.phi_range_params.cc.item()
        cr_val = model.phi_range_params.cr.item()
        
        dc_grad = model.phi_range_params.dc.grad.item() if model.phi_range_params.dc.grad is not None else 0.0
        dr_grad = model.phi_range_params.dr.grad.item() if model.phi_range_params.dr.grad is not None else 0.0
        cc_grad = model.phi_range_params.cc.grad.item() if model.phi_range_params.cc.grad is not None else 0.0
        cr_grad = model.phi_range_params.cr.grad.item() if model.phi_range_params.cr.grad is not None else 0.0
        
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
            'dc': f'{dc_val:.3f}',
            'dr': f'{dr_val:.3f}',
            'cc': f'{cc_val:.3f}',
            'cr': f'{cr_val:.3f}',
            'std_out': f'{std_output.item():.3f}',
            'std_tar': f'{std_target.item():.3f}'
        }
        
        if USE_ADVANCED_SCHEDULER:
            pbar_dict['lr'] = f'{current_lr:.2e}'
            pbar_dict['g_dc'] = f'{dc_grad:.3e}'
            pbar_dict['g_dr'] = f'{dr_grad:.3e}'
            pbar_dict['g_cc'] = f'{cc_grad:.3e}'
            pbar_dict['g_cr'] = f'{cr_grad:.3e}'
        
        pbar.set_postfix(pbar_dict)
        
        if (epoch + 1) % print_every == 0:
            if USE_ADVANCED_SCHEDULER:
                print(f"Epoch {epoch+1}: Loss = {loss.item():.4e}, Best = {best_loss:.4e}, LR = {current_lr:.4e}")
            else:
                print(f"Epoch {epoch+1}: Loss = {loss.item():.4e}, Best = {best_loss:.4e}")
    
    # Load best state before returning
    if best_state is not None:
        model.load_state_dict(best_state)
        print(f"\nLoaded best model with loss: {best_loss:.4e}")
    
    # Print final eta and lambda parameters for each block
    print("\nFinal parameters:")
    for idx, layer in enumerate(model.layers, start=1):
        print(f"Block {idx}: eta = {layer.eta.item():.6f}")
        print(f"Block {idx}: lambdas shape = {tuple(layer.lambdas.shape)}")
        print(f"Block {idx}: lambdas =")
        print(layer.lambdas.detach().cpu().numpy())
        if USE_RESIDUAL_WEIGHTS:
            if hasattr(layer, 'residual_weight') and layer.residual_weight is not None:
                print(f"Block {idx}: residual_weight = {layer.residual_weight.item():.6f}")
            elif hasattr(layer, 'residual_projection') and layer.residual_projection is not None:
                print(f"Block {idx}: residual_projection weight shape = {tuple(layer.residual_projection.weight.shape)}")
        print()
    
    print(f"Final loss: {best_loss:.4e}")
    
    return model, losses, model.layers