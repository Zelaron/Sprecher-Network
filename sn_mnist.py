"""
sn_mnist.py - MNIST classification using Sprecher Networks (Modernized)
"""

import os
import sys
import shutil
import argparse
import copy
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torchvision.datasets import MNIST
from torch.utils.data import DataLoader
from tqdm import tqdm
from PIL import Image
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

# Import Sprecher Network components
from sn_core import SprecherMultiLayerNetwork, plot_results, plot_loss_curve
from sn_core.config import CONFIG, MNIST_CONFIG


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Train Sprecher Networks on MNIST")
    
    # Mode selection
    parser.add_argument("--mode", type=str, required=True,
                        choices=["train", "test", "infer", "plot"],
                        help="Operation mode: train, test, infer, or plot")
    
    # Architecture and model settings
    parser.add_argument("--arch", type=str, default=None,
                        help="Architecture as comma-separated values (default: from MNIST_CONFIG)")
    parser.add_argument("--phi_knots", type=int, default=None,
                        help="Number of knots for phi splines (default: from MNIST_CONFIG)")
    parser.add_argument("--Phi_knots", type=int, default=None,
                        help="Number of knots for Phi splines (default: from MNIST_CONFIG)")
    
    # Training settings
    parser.add_argument("--epochs", type=int, default=None,
                        help="Number of training epochs (default: from MNIST_CONFIG)")
    parser.add_argument("--batch_size", type=int, default=None,
                        help="Batch size (default: from MNIST_CONFIG)")
    parser.add_argument("--lr", type=float, default=None,
                        help="Learning rate (default: from MNIST_CONFIG)")
    parser.add_argument("--seed", type=int, default=45,
                        help="Random seed (default: 45)")
    parser.add_argument("--device", type=str, default="auto",
                        help="Device: auto, cpu, or cuda (default: auto)")
    
    # File paths
    parser.add_argument("--model_file", type=str, default=None,
                        help="Model file path (default: from MNIST_CONFIG)")
    parser.add_argument("--data_dir", type=str, default=None,
                        help="Data directory path (default: from MNIST_CONFIG)")
    parser.add_argument("--image", type=str, default="digit.png",
                        help="Image file for inference mode (default: digit.png)")
    
    # Training options
    parser.add_argument("--retrain", action="store_true",
                        help="Delete existing model and retrain from scratch")
    
    # Plotting options
    parser.add_argument("--save_plots", action="store_true",
                        help="Save plots to files")
    parser.add_argument("--no_show", action="store_true",
                        help="Don't show plots (useful for batch runs)")
    
    # Normalization arguments
    parser.add_argument("--norm_type", type=str,
                        choices=["none", "batch", "layer"],
                        help="Type of normalization to use (default: from CONFIG)")
    parser.add_argument("--norm_position", type=str, default="after",
                        choices=["before", "after"],
                        help="Position of normalization relative to blocks (default: after)")
    parser.add_argument("--norm_skip_first", action="store_true", default=True,
                        help="Skip normalization for the first block (default: True)")
    parser.add_argument("--norm_first", action="store_true",
                        help="Enable normalization for the first block (overrides norm_skip_first)")
    
    # Feature control arguments
    parser.add_argument("--no_residual", action="store_true",
                        help="Disable residual connections (default: enabled)")
    parser.add_argument("--no_norm", action="store_true",
                        help="Disable normalization (default: enabled with batch norm)")
    parser.add_argument("--use_advanced_scheduler", action="store_true",
                        help="Use PlateauAwareCosineAnnealingLR scheduler (default: disabled)")
    
    # Debug options
    parser.add_argument("--debug_domains", action="store_true",
                        help="Enable domain debugging output")
    parser.add_argument("--track_violations", action="store_true",
                        help="Track domain violations during training")
    
    return parser.parse_args()


def get_config_suffix(args, CONFIG):
    """Build filename suffix for non-default configurations."""
    parts = []
    
    # Check normalization (default is enabled with batch)
    if args.no_norm or (hasattr(args, 'norm_type') and args.norm_type == 'none'):
        parts.append("NoNorm")
    elif hasattr(args, 'norm_type') and args.norm_type and args.norm_type not in ['none', 'batch']:
        parts.append(f"Norm{args.norm_type.capitalize()}")
    
    # Check residuals (default is enabled)
    if not CONFIG.get('use_residual_weights', True):
        parts.append("NoResidual")
    
    # Check scheduler (default is disabled)
    if CONFIG.get('use_advanced_scheduler', False):
        parts.append("AdvScheduler")
    
    # Join with dashes
    return "-" + "-".join(parts) if parts else ""


def discover_models(model_dir="."):
    """Discover trained MNIST models and extract their architectures.
    
    Returns:
        dict: A dictionary mapping architectures to model info:
              {
                  "40-40-40-40": {
                      "epochs": 3,
                      "files": ["sn_mnist_model-40-40-40-40-3epochs.pth", 
                               "sn_mnist_model-40-40-40-40-3epochs_best.pth"]
                  }
              }
    """
    import glob
    import re
    
    models = {}
    pattern = os.path.join(model_dir, "sn_mnist_model-*-*epochs*.pth")
    
    for filepath in glob.glob(pattern):
        filename = os.path.basename(filepath)
        # Extract architecture and epochs from filename
        # Pattern: sn_mnist_model-{arch}-{epochs}epochs[_best].pth
        match = re.match(r"sn_mnist_model-([\d-]+)-(\d+)epochs(_best)?\.pth", filename)
        if match:
            arch_str = match.group(1)
            epochs = int(match.group(2))
            
            if arch_str not in models:
                models[arch_str] = {
                    "epochs": epochs,
                    "files": []
                }
            models[arch_str]["files"].append(filename)
            # Update epochs if we find a different value (shouldn't happen normally)
            if models[arch_str]["epochs"] != epochs:
                print(f"Warning: Found models with same architecture but different epochs: {epochs} vs {models[arch_str]['epochs']}")
    
    return models


def auto_discover_architecture(args, mode_name):
    """Auto-discover model architecture if not specified.
    
    Args:
        args: Command line arguments
        mode_name: Name of the mode (test/infer/plot) for messages
        
    Returns:
        tuple: (architecture, epochs) or (None, None) if discovery fails
    """
    if args.arch is not None:
        # User specified architecture, use it
        architecture = [int(x) for x in args.arch.split(",")]
        epochs = args.epochs if args.epochs else MNIST_CONFIG['epochs']
        return architecture, epochs
    
    # Try to discover models
    print(f"No architecture specified, searching for trained models...")
    models = discover_models()
    
    if not models:
        print("No trained models found in current directory.")
        print(f"Will use default architecture: {MNIST_CONFIG['architecture']}")
        return None, None
    
    if len(models) == 1:
        # Only one architecture found, use it
        arch_str = list(models.keys())[0]
        model_info = models[arch_str]
        architecture = [int(x) for x in arch_str.split("-")]
        epochs = model_info["epochs"]
        
        print(f"Auto-detected model: architecture={architecture}, epochs={epochs}")
        print(f"Found files: {', '.join(model_info['files'])}")
        return architecture, epochs
    
    else:
        # Multiple architectures found
        print(f"\nFound {len(models)} different model architectures:")
        for arch_str, info in sorted(models.items()):
            arch_list = [int(x) for x in arch_str.split("-")]
            print(f"  - Architecture: 784 -> {arch_list} -> 10 (epochs={info['epochs']})")
            print(f"    Files: {', '.join(info['files'])}")
        
        print(f"\nPlease specify which architecture to use with --arch")
        print(f"Example: python sn_mnist.py --mode {mode_name} --arch {list(models.keys())[0]}")
        return None, None


def count_parameters_detailed(model):
    """Count parameters with detailed breakdown matching SN code style."""
    # Count different parameter types
    total_spline_knots = 0
    lambda_params = 0
    eta_params = 0
    residual_params = 0
    codomain_params = 0
    norm_params = 0
    
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
            
        if 'coeffs' in name or 'log_increments' in name:
            # Spline knots
            total_spline_knots += param.numel()
        elif 'lambdas' in name:
            # Lambda weight vectors
            lambda_params += param.numel()
        elif 'eta' in name:
            # Eta shift parameters
            eta_params += param.numel()
        elif 'residual_weight' in name or 'residual_projection' in name:
            # Residual connection parameters
            residual_params += param.numel()
        elif 'phi_codomain_params' in name:
            # Codomain parameters (cc, cr)
            codomain_params += param.numel()
        elif 'norm' in name.lower():
            # Normalization parameters
            norm_params += param.numel()
    
    # Core parameters (excluding residual and output params)
    core_params = total_spline_knots + lambda_params + eta_params
    
    # Total parameters
    if CONFIG['train_phi_codomain']:
        total_params = core_params + residual_params + codomain_params + norm_params
    else:
        total_params = core_params + residual_params + norm_params
    
    # Count blocks
    num_blocks = len(model.sprecher_net.layers)
    
    return {
        'total': total_params,
        'core': core_params,
        'spline_knots': total_spline_knots,
        'lambda': lambda_params,
        'eta': eta_params,
        'residual': residual_params,
        'codomain': codomain_params,
        'norm': norm_params,
        'num_blocks': num_blocks
    }


def process_image(file_path):
    """Process an image file for inference."""
    image = Image.open(file_path)
    image = image.convert('L')
    if image.size != (28, 28):
        image = image.resize((28, 28), Image.Resampling.LANCZOS)
    data = np.asarray(image)
    
    # Check if the image is likely to have a white background
    if data.sum() > 127.5 * 28 * 28:
        data = 255 - data  # Invert the pixel values
    
    # Normalize to [0, 1] range for Sprecher Networks
    normalized_data = data / 255.0
    return normalized_data.flatten()


class MNISTSprecherNet(nn.Module):
    """Wrapper around SprecherMultiLayerNetwork for MNIST classification."""
    
    def __init__(self, architecture, phi_knots=100, Phi_knots=100,
                 norm_type="none", norm_position="after", norm_skip_first=True):
        super().__init__()
        self.sprecher_net = SprecherMultiLayerNetwork(
            input_dim=784,  # 28x28 flattened
            architecture=architecture,
            final_dim=10,   # 10 digit classes
            phi_knots=phi_knots,
            Phi_knots=Phi_knots,
            norm_type=norm_type,
            norm_position=norm_position,
            norm_skip_first=norm_skip_first
        )
    
    def forward(self, x):
        # Flatten the input if needed
        if x.dim() > 2:
            x = x.view(x.size(0), -1)
        return self.sprecher_net(x)
    
    def update_all_domains(self):
        """Forward domain update call to the underlying Sprecher network."""
        self.sprecher_net.update_all_domains()
    
    def get_domain_violation_stats(self):
        """Forward domain violation stats call."""
        return self.sprecher_net.get_domain_violation_stats()
    
    def reset_domain_violation_stats(self):
        """Forward domain violation reset call."""
        self.sprecher_net.reset_domain_violation_stats()
    
    def print_domain_violation_report(self):
        """Forward domain violation report call."""
        self.sprecher_net.print_domain_violation_report()


def train_epoch(model, train_loader, optimizer, scheduler, loss_function, device):
    """Train for one epoch."""
    model.train()
    epoch_loss = 0
    correct = 0
    total = 0
    
    # Reset domain violation stats at start of epoch if tracking
    if CONFIG.get('track_domain_violations', False):
        model.reset_domain_violation_stats()
    
    with tqdm(train_loader, unit="batch") as tepoch:
        for batch_idx, (images, labels) in enumerate(tepoch):
            images, labels = images.to(device), labels.to(device)
            
            # Flatten and normalize to [0, 1]
            images = images.view(images.size(0), -1)
            images = (images + 1) / 2  # Convert from [-1, 1] to [0, 1]
            
            # Update domains before each forward pass
            if CONFIG.get('use_theoretical_domains', True):
                model.update_all_domains()
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = loss_function(outputs, labels)
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), CONFIG.get('max_grad_norm', 1.0))
            
            optimizer.step()
            
            epoch_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            accuracy = 100 * correct / total
            tepoch.set_postfix(loss=epoch_loss/(batch_idx+1), acc=f"{accuracy:.1f}%")
    
    # Step scheduler if using advanced scheduler
    if scheduler is not None:
        scheduler.step(epoch_loss / len(train_loader))
    
    # Print domain violation report at end of epoch if tracking
    if CONFIG.get('track_domain_violations', False):
        print("\nEpoch domain violations:")
        model.print_domain_violation_report()
    
    return epoch_loss / len(train_loader), accuracy


def test_model(model, test_loader, device):
    """Evaluate model on test set."""
    # Keep model in training mode for consistency with checkpoint
    # model.eval()  # Commented out - we maintain the mode from training
    # This ensures BatchNorm uses the same statistics as during training
    correct = 0
    total = 0
    
    # Reset domain violation stats if tracking
    if CONFIG.get('track_domain_violations', False):
        model.reset_domain_violation_stats()
    
    with torch.no_grad():
        for images, labels in tqdm(test_loader, desc="Testing"):
            images, labels = images.to(device), labels.to(device)
            
            # Flatten and normalize to [0, 1]
            images = images.view(images.size(0), -1)
            images = (images + 1) / 2
            
            # Update domains before forward pass
            if CONFIG.get('use_theoretical_domains', True):
                model.update_all_domains()
            
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    # Print domain violation report if tracking
    if CONFIG.get('track_domain_violations', False):
        print("\nTest set domain violations:")
        model.print_domain_violation_report()
    
    accuracy = 100 * correct / total
    return accuracy


def print_configuration(args, architecture, phi_knots, Phi_knots, epochs, batch_size, lr, final_norm_skip_first):
    """Print configuration summary."""
    print(f"Using device: {args.device if args.device != 'auto' else 'cuda' if torch.cuda.is_available() else 'cpu'}")
    print(f"Mode: {args.mode}")
    print(f"Architecture: 784 -> {architecture} -> 10")
    print(f"phi knots: {phi_knots}, Phi knots: {Phi_knots}")
    print(f"Epochs: {epochs}")
    print(f"Batch size: {batch_size}")
    print(f"Learning rate: {lr}")
    print(f"Seed: {args.seed}")
    print(f"Theoretical domains: {CONFIG.get('use_theoretical_domains', True)}")
    print(f"Domain safety margin: {CONFIG.get('domain_safety_margin', 0.0)}")
    print(f"Residual connections: {'enabled' if CONFIG.get('use_residual_weights', True) else 'disabled'}")
    
    # Determine effective normalization
    if args.no_norm:
        print("Normalization: disabled")
    elif args.norm_type:
        print(f"Normalization: {args.norm_type} (position: {args.norm_position}, skip_first: {final_norm_skip_first})")
    else:
        print(f"Normalization: {CONFIG.get('norm_type', 'batch')} (position: {args.norm_position}, skip_first: {final_norm_skip_first})")
    
    print(f"Scheduler: {'PlateauAwareCosineAnnealingLR' if CONFIG.get('use_advanced_scheduler', False) else 'Adam (fixed LR)'}")
    print()


def load_checkpoint_and_get_model_params(model_path, device, args):
    """Load checkpoint and extract model parameters.
    
    Returns:
        tuple: (checkpoint, model_params_dict)
    """
    if not os.path.exists(model_path):
        return None, None
    
    checkpoint = torch.load(model_path, map_location=device)
    
    # For new format checkpoints with model_params
    if isinstance(checkpoint, dict) and 'model_params' in checkpoint:
        model_params = checkpoint['model_params'].copy()
        
        # Override with command-line args if explicitly provided
        # But preserve critical architecture parameters from checkpoint
        if args.phi_knots is not None:
            model_params['phi_knots'] = args.phi_knots
        if args.Phi_knots is not None:
            model_params['Phi_knots'] = args.Phi_knots
            
        print("Using model configuration from checkpoint:")
        print(f"  Architecture: {model_params['architecture']}")
        print(f"  Normalization: {model_params['norm_type']} (skip_first: {model_params['norm_skip_first']})")
        
        return checkpoint, model_params
    else:
        # Old format - return None for model_params
        return checkpoint, None


def train_mnist(args):
    """Training mode."""
    # Get configuration values
    architecture = [int(x) for x in args.arch.split(",")] if args.arch else MNIST_CONFIG['architecture']
    phi_knots = args.phi_knots if args.phi_knots else MNIST_CONFIG['phi_knots']
    Phi_knots = args.Phi_knots if args.Phi_knots else MNIST_CONFIG['Phi_knots']
    epochs = args.epochs if args.epochs else MNIST_CONFIG['epochs']
    batch_size = args.batch_size if args.batch_size else MNIST_CONFIG['batch_size']
    lr = args.lr if args.lr else MNIST_CONFIG['learning_rate']
    model_file = args.model_file if args.model_file else MNIST_CONFIG['model_file']
    data_dir = args.data_dir if args.data_dir else MNIST_CONFIG['data_directory']
    
    # Determine device
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    
    # Set random seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    # Handle configuration overrides
    if args.debug_domains:
        CONFIG['debug_domains'] = True
    if args.track_violations:
        CONFIG['track_domain_violations'] = True
        print("Domain violation tracking enabled.")
    if args.no_residual:
        CONFIG['use_residual_weights'] = False
    if args.no_norm:
        CONFIG['use_normalization'] = False
    if args.use_advanced_scheduler:
        CONFIG['use_advanced_scheduler'] = True
    
    # Determine effective normalization settings
    if CONFIG.get('use_normalization', True) and not args.no_norm:
        if args.norm_type == "none":
            effective_norm_type = "none"
        elif args.norm_type is not None:
            effective_norm_type = args.norm_type
        else:
            effective_norm_type = CONFIG.get('norm_type', 'batch')
    else:
        effective_norm_type = "none"
    
    # Handle the --norm_first flag
    if args.norm_first:
        final_norm_skip_first = False
    else:
        final_norm_skip_first = args.norm_skip_first if hasattr(args, 'norm_skip_first') else CONFIG.get('norm_skip_first', True)
    
    # Print configuration
    print_configuration(args, architecture, phi_knots, Phi_knots, epochs, batch_size, lr, final_norm_skip_first)
    
    # Initialize model
    model = MNISTSprecherNet(
        architecture=architecture,
        phi_knots=phi_knots,
        Phi_knots=Phi_knots,
        norm_type=effective_norm_type,
        norm_position=args.norm_position,
        norm_skip_first=final_norm_skip_first
    ).to(device)
    
    # Count and display parameters
    param_counts = count_parameters_detailed(model)
    print(f"Total number of trainable parameters: {param_counts['total']:,}")
    print(f"  - Lambda weight VECTORS: {param_counts['lambda']:,} (TRUE SPRECHER!)")
    print(f"  - Eta shift parameters: {param_counts['eta']:,}")
    print(f"  - Spline parameters: {param_counts['spline_knots']:,}")
    if CONFIG['use_residual_weights'] and param_counts['residual'] > 0:
        print(f"  - Residual connection weights: {param_counts['residual']:,}")
    if param_counts['norm'] > 0:
        print(f"  - Normalization parameters: {param_counts['norm']:,}")
    if CONFIG['train_phi_codomain'] and param_counts['codomain'] > 0:
        print(f"  - Phi codomain parameters (cc, cr per block): {param_counts['codomain']:,}")
    print()
    
    # Check if model exists and handle retrain
    config_suffix = get_config_suffix(args, CONFIG)
    arch_str = "-".join(map(str, architecture))
    model_filename = f"sn_mnist_model-{arch_str}-{epochs}epochs{config_suffix}.pth"
    model_path = os.path.join(os.path.dirname(model_file), model_filename)
    
    if os.path.exists(model_path) and args.retrain:
        os.remove(model_path)
        print(f"Deleted existing model file: {model_path}")
        if os.path.exists(data_dir):
            shutil.rmtree(data_dir)
            print(f"Deleted existing data directory: {data_dir}")
    elif os.path.exists(model_path) and not args.retrain:
        model.load_state_dict(torch.load(model_path, map_location=device), strict=False)
        print(f"Loaded saved model from {model_path} for further training.")
    
    # Setup data
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))  # This gives [-1, 1], we'll convert to [0, 1] later
    ])
    train_dataset = MNIST(root=data_dir, train=True, download=True, transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    
    # Setup optimizer
    optimizer = optim.AdamW(
        model.parameters(), 
        lr=lr,
        weight_decay=MNIST_CONFIG.get('weight_decay', 1e-6)
    )
    
    # Setup scheduler if requested
    scheduler = None
    if CONFIG.get('use_advanced_scheduler', False):
        from sn_core.train import PlateauAwareCosineAnnealingLR
        scheduler = PlateauAwareCosineAnnealingLR(
            optimizer,
            base_lr=CONFIG['scheduler_base_lr'],
            max_lr=CONFIG['scheduler_max_lr'],
            patience=CONFIG['scheduler_patience'],
            threshold=CONFIG['scheduler_threshold']
        )
    
    loss_function = nn.CrossEntropyLoss()
    
    # Training loop
    print(f"Starting training for {epochs} epochs...")
    best_accuracy = 0
    losses = []
    accuracies = []
    best_plotting_snapshot = None  # Will store complete snapshot for plotting
    best_checkpoint = None  # Will store complete checkpoint
    
    # Get a sample batch for snapshot (we'll update this when we find best accuracy)
    sample_images, sample_labels = next(iter(train_loader))
    sample_images = sample_images.to(device)
    sample_labels = sample_labels.to(device)
    # Flatten and normalize sample images
    sample_images = sample_images.view(sample_images.shape[0], -1)
    sample_images = (sample_images + 1) / 2  # Convert from [-1, 1] to [0, 1]
    
    for epoch in range(epochs):
        print(f"\nEpoch {epoch + 1}/{epochs}")
        
        avg_loss, train_acc = train_epoch(model, train_loader, optimizer, scheduler, loss_function, device)
        losses.append(avg_loss)
        accuracies.append(train_acc)
        
        # Print epoch summary every 10%
        if (epoch + 1) % max(1, epochs // 10) == 0 or epoch == 0:
            print(f"Epoch {epoch + 1}: Loss = {avg_loss:.4e}, Accuracy = {train_acc:.2f}%")
        
        # Always save current model state
        torch.save(model.state_dict(), model_path)
        
        if train_acc > best_accuracy:
            best_accuracy = train_acc
            
            # Get current model output on sample batch
            with torch.no_grad():
                sample_output = model(sample_images)
                sample_loss = loss_function(sample_output, sample_labels).item()
            
            # Create complete plotting snapshot (similar to sn_core/train.py)
            best_plotting_snapshot = {
                'model': copy.deepcopy(model),  # Complete deep copy of the model
                'x_train': sample_images.clone(),  # Training inputs
                'y_train': sample_labels.clone(),  # Training targets
                'output': sample_output.detach().clone(),  # Model output at this moment
                'loss': sample_loss,  # Loss value
                'accuracy': train_acc,  # Accuracy value
                'epoch': epoch,  # Epoch number
                'device': device  # Device for later use
            }
            
            # Save complete BatchNorm statistics if normalization is used
            bn_statistics = {}
            has_bn = effective_norm_type in ['batch', 'layer']
            if has_bn:
                for name, module in model.named_modules():
                    if isinstance(module, nn.BatchNorm1d):
                        # Deep clone all BN state to isolate from future changes
                        bn_statistics[name] = {
                            'running_mean': module.running_mean.clone().cpu(),
                            'running_var': module.running_var.clone().cpu(),
                            'num_batches_tracked': module.num_batches_tracked.clone().cpu(),
                            'momentum': module.momentum,
                            'eps': module.eps,
                            'affine': module.affine,
                            'track_running_stats': module.track_running_stats,
                            # Also save the affine parameters if they exist
                            'weight': module.weight.clone().cpu() if module.affine else None,
                            'bias': module.bias.clone().cpu() if module.affine else None
                        }
            
            # Create complete checkpoint with all state
            best_checkpoint = {
                'epoch': epoch,
                'model_state_dict': copy.deepcopy(model.state_dict()),  # Deep copy the state dict
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': sample_loss,
                'accuracy': train_acc,
                'has_batchnorm': has_bn,
                'bn_statistics': bn_statistics,  # Save exact BN stats at best epoch
                'training_mode': model.training,  # Save training state
                'x_train': sample_images.cpu().clone(),  # Save exact training batch
                'y_train': sample_labels.cpu().clone(),  # Save exact target values
                'output': sample_output.detach().cpu().clone(),  # Save exact model output
                # Save model creation parameters for exact reconstruction
                'model_params': {
                    'architecture': architecture,
                    'phi_knots': phi_knots,
                    'Phi_knots': Phi_knots,
                    'norm_type': effective_norm_type,
                    'norm_position': args.norm_position,
                    'norm_skip_first': final_norm_skip_first
                }
            }
            
            # Save to disk immediately
            best_model_path = model_path.replace('.pth', '_best.pth')
            torch.save(best_checkpoint, best_model_path)
    
    print(f"\nTraining complete! Best accuracy: {best_accuracy:.2f}%")
    print(f"Model saved to: {model_path}")
    
    # IMPORTANT: We keep the model in training mode throughout
    # This ensures BatchNorm statistics remain consistent with how the model was saved
    # Using model.eval() would change BatchNorm behavior and make results inconsistent
    
    # Verify the snapshot works correctly if we have one
    if best_plotting_snapshot is not None:
        print("\nVerifying plotting snapshot consistency...")
        print("Computing accuracy multiple times to ensure perfect reproducibility:")
        
        snapshot_model = best_plotting_snapshot['model']
        snapshot_images = best_plotting_snapshot['x_train']
        snapshot_labels = best_plotting_snapshot['y_train']
        
        accuracies_verification = []
        for i in range(3):
            with torch.no_grad():
                snapshot_output = snapshot_model(snapshot_images)
                _, predicted = torch.max(snapshot_output, 1)
                correct = (predicted == snapshot_labels).sum().item()
                accuracy = 100 * correct / snapshot_labels.size(0)
                accuracies_verification.append(accuracy)
                print(f"  Computation {i+1}: accuracy = {accuracy:.2f}%")
        
        print(f"\nSaved accuracy from snapshot: {best_plotting_snapshot['accuracy']:.2f}%")
        print(f"Mean of verifications: {np.mean(accuracies_verification):.2f}%")
        print(f"Std of verifications: {np.std(accuracies_verification):.4f}%")
        
        if np.std(accuracies_verification) < 1e-6:
            print("✓ Perfect consistency achieved! The snapshot is completely isolated.")
    
    # Plot training curves if requested
    if args.save_plots or not args.no_show:
        # Graceful fallback plotting logic (similar to sn_experiments.py)
        try:
            os.makedirs("plots", exist_ok=True)
            
            # Plot loss curve
            loss_filename = f"mnist_loss-{arch_str}-{epochs}epochs{config_suffix}.png"
            loss_save_path = os.path.join("plots", loss_filename) if args.save_plots else None
            plot_loss_curve(losses, loss_save_path)
            
            # Handle show/close for loss curve plot
            if not args.no_show:
                plt.show()
            else:
                plt.close()
            
            # Plot accuracy curve
            plt.figure(figsize=(10, 6))
            plt.plot(accuracies)
            plt.title("MNIST Training Accuracy")
            plt.xlabel("Epoch")
            plt.ylabel("Accuracy (%)")
            plt.grid(True)
            
            if args.save_plots:
                acc_filename = f"mnist_accuracy-{arch_str}-{epochs}epochs{config_suffix}.png"
                acc_save_path = os.path.join("plots", acc_filename)
                plt.savefig(acc_save_path, dpi=150, bbox_inches='tight')
                print(f"Accuracy curve saved to {acc_save_path}")
            
            if not args.no_show:
                plt.show()
            else:
                plt.close()
                
        except Exception as e:
            # This block will catch TclError on Windows or other GUI-related errors
            print("\n" + "="*60)
            print("WARNING: A plotting error occurred.")
            print(f"Error type: {type(e).__name__}")
            print("The default plotting backend on your system has failed.")
            print("This is common on systems without a configured GUI toolkit.")
            print("\nSwitching to the reliable 'Agg' backend to proceed.")
            print("="*60 + "\n")
            
            # Set the backend and re-import pyplot
            matplotlib.use('Agg')
            import matplotlib.pyplot as plt
            
            # Ensure we save plots if this fallback is triggered
            if not args.save_plots:
                print("Plots could not be shown interactively. Enabling file saving automatically.")
                args.save_plots = True
            
            # Re-run plotting logic with the safe backend
            os.makedirs("plots", exist_ok=True)
            
            # Plot loss curve
            loss_filename = f"mnist_loss-{arch_str}-{epochs}epochs{config_suffix}.png"
            loss_save_path = os.path.join("plots", loss_filename)
            plot_loss_curve(losses, loss_save_path)
            plt.close()
            
            # Plot accuracy curve
            plt.figure(figsize=(10, 6))
            plt.plot(accuracies)
            plt.title("MNIST Training Accuracy")
            plt.xlabel("Epoch")
            plt.ylabel("Accuracy (%)")
            plt.grid(True)
            
            acc_filename = f"mnist_accuracy-{arch_str}-{epochs}epochs{config_suffix}.png"
            acc_save_path = os.path.join("plots", acc_filename)
            plt.savefig(acc_save_path, dpi=150, bbox_inches='tight')
            plt.close()
            
            print(f"Loss curve saved to {loss_save_path}")
            print(f"Accuracy curve saved to {acc_save_path}")
            print(f"\nPlots have been saved to the 'plots/' directory.")


def test_mnist(args):
    """Testing mode."""
    # Try to auto-discover architecture if not specified
    discovered_arch, discovered_epochs = auto_discover_architecture(args, "test")
    
    # Get configuration values
    if discovered_arch is not None:
        architecture = discovered_arch
        epochs = discovered_epochs
    else:
        # Use defaults or exit if discovery failed with multiple architectures
        if args.arch is None and discovered_arch is None:
            # Either no models found or multiple architectures found
            architecture = MNIST_CONFIG['architecture']
            epochs = args.epochs if args.epochs else MNIST_CONFIG['epochs']
            # If we're using defaults because of multiple architectures, exit
            models = discover_models()
            if len(models) > 1:
                return
        else:
            architecture = [int(x) for x in args.arch.split(",")]
            epochs = args.epochs if args.epochs else MNIST_CONFIG['epochs']
    
    phi_knots = args.phi_knots if args.phi_knots else MNIST_CONFIG['phi_knots']
    Phi_knots = args.Phi_knots if args.Phi_knots else MNIST_CONFIG['Phi_knots']
    batch_size = args.batch_size if args.batch_size else MNIST_CONFIG['batch_size']
    model_file = args.model_file if args.model_file else MNIST_CONFIG['model_file']
    data_dir = args.data_dir if args.data_dir else MNIST_CONFIG['data_directory']
    
    # Determine device
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    
    # Handle configuration
    if args.debug_domains:
        CONFIG['debug_domains'] = True
    if args.track_violations:
        CONFIG['track_domain_violations'] = True
    if args.no_residual:
        CONFIG['use_residual_weights'] = False
    if args.no_norm:
        CONFIG['use_normalization'] = False
    
    # Construct model filename to match training convention
    config_suffix = get_config_suffix(args, CONFIG)
    arch_str = "-".join(map(str, architecture))
    epochs = args.epochs if args.epochs else MNIST_CONFIG['epochs']
    model_filename = f"sn_mnist_model-{arch_str}-{epochs}epochs{config_suffix}.pth"
    model_path = os.path.join(os.path.dirname(model_file), model_filename)
    
    # Try to find the model file (check for _best variant first)
    best_model_path = model_path.replace('.pth', '_best.pth')
    if os.path.exists(best_model_path):
        model_path = best_model_path
        print(f"Loading best model from {model_path}")
    elif os.path.exists(model_path):
        print(f"Loading model from {model_path}")
    else:
        print(f"Error: Model file {model_path} not found. Please train first.")
        return
    
    # Load checkpoint and get model parameters
    checkpoint, model_params = load_checkpoint_and_get_model_params(model_path, device, args)
    if checkpoint is None:
        print(f"Error: Could not load model from {model_path}")
        return
    
    # Use checkpoint parameters if available, otherwise fall back to command-line args
    if model_params is not None:
        # Use saved parameters
        effective_norm_type = model_params['norm_type']
        final_norm_skip_first = model_params['norm_skip_first']
        norm_position = model_params['norm_position']
        architecture = model_params['architecture']
        phi_knots = model_params['phi_knots']
        Phi_knots = model_params['Phi_knots']
    else:
        # Fall back to command-line arguments for old checkpoints
        print("WARNING: Old checkpoint format. Using command-line arguments for model configuration.")
        print("Make sure to use the same flags as during training!")
        
        # Determine effective normalization
        if CONFIG.get('use_normalization', True) and not args.no_norm:
            if args.norm_type == "none":
                effective_norm_type = "none"
            elif args.norm_type is not None:
                effective_norm_type = args.norm_type
            else:
                effective_norm_type = CONFIG.get('norm_type', 'batch')
        else:
            effective_norm_type = "none"
        
        # Handle the --norm_first flag
        if args.norm_first:
            final_norm_skip_first = False
        else:
            final_norm_skip_first = args.norm_skip_first if hasattr(args, 'norm_skip_first') else CONFIG.get('norm_skip_first', True)
        
        norm_position = args.norm_position
    
    print(f"Using device: {device}")
    print(f"Architecture: 784 -> {architecture} -> 10")
    print(f"Normalization: {effective_norm_type} (skip_first: {final_norm_skip_first})")
    print()
    
    # Initialize model with the correct configuration
    model = MNISTSprecherNet(
        architecture=architecture,
        phi_knots=phi_knots,
        Phi_knots=Phi_knots,
        norm_type=effective_norm_type,
        norm_position=norm_position,
        norm_skip_first=final_norm_skip_first
    ).to(device)
    
    # Load the model state
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        # New format with complete checkpoint
        model.load_state_dict(checkpoint['model_state_dict'], strict=False)
        print(f"Loaded checkpoint from epoch {checkpoint.get('epoch', 'unknown')} with accuracy {checkpoint.get('accuracy', 'unknown'):.2f}%")
        
        # Restore BatchNorm statistics if available
        if checkpoint.get('has_batchnorm', False) and 'bn_statistics' in checkpoint:
            for name, module in model.named_modules():
                if isinstance(module, nn.BatchNorm1d) and name in checkpoint['bn_statistics']:
                    bn_stats = checkpoint['bn_statistics'][name]
                    module.running_mean.copy_(bn_stats['running_mean'].to(device))
                    module.running_var.copy_(bn_stats['running_var'].to(device))
                    if 'num_batches_tracked' in bn_stats:
                        module.num_batches_tracked.copy_(bn_stats['num_batches_tracked'].to(device))
    else:
        # Old format - just state dict
        model.load_state_dict(checkpoint, strict=False)
    
    # Keep model in training mode - do not use model.eval()
    # This maintains consistency with how the model was saved
    
    # Count parameters
    param_counts = count_parameters_detailed(model)
    print(f"Model has {param_counts['total']:,} parameters")
    
    # Setup test data
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    test_dataset = MNIST(root=data_dir, train=False, download=True, transform=transform)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    # Test
    accuracy = test_model(model, test_loader, device)
    print(f"\nTest accuracy: {accuracy:.2f}%")


def infer_mnist(args):
    """Inference mode."""
    # Try to auto-discover architecture if not specified
    discovered_arch, discovered_epochs = auto_discover_architecture(args, "infer")
    
    # Get configuration values
    if discovered_arch is not None:
        architecture = discovered_arch
        epochs = discovered_epochs
    else:
        # Use defaults or exit if discovery failed with multiple architectures
        if args.arch is None and discovered_arch is None:
            # Either no models found or multiple architectures found
            architecture = MNIST_CONFIG['architecture']
            epochs = args.epochs if args.epochs else MNIST_CONFIG['epochs']
            # If we're using defaults because of multiple architectures, exit
            models = discover_models()
            if len(models) > 1:
                return
        else:
            architecture = [int(x) for x in args.arch.split(",")]
            epochs = args.epochs if args.epochs else MNIST_CONFIG['epochs']
    
    phi_knots = args.phi_knots if args.phi_knots else MNIST_CONFIG['phi_knots']
    Phi_knots = args.Phi_knots if args.Phi_knots else MNIST_CONFIG['Phi_knots']
    model_file = args.model_file if args.model_file else MNIST_CONFIG['model_file']
    
    # Determine device
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    
    # Handle configuration
    if args.no_residual:
        CONFIG['use_residual_weights'] = False
    if args.no_norm:
        CONFIG['use_normalization'] = False
    
    # Construct model filename to match training convention
    config_suffix = get_config_suffix(args, CONFIG)
    arch_str = "-".join(map(str, architecture))
    epochs = args.epochs if args.epochs else MNIST_CONFIG['epochs']
    model_filename = f"sn_mnist_model-{arch_str}-{epochs}epochs{config_suffix}.pth"
    model_path = os.path.join(os.path.dirname(model_file), model_filename)
    
    # Try to find the model file (check for _best variant first)
    best_model_path = model_path.replace('.pth', '_best.pth')
    if os.path.exists(best_model_path):
        model_path = best_model_path
        print(f"Loading best model from {model_path}")
    elif os.path.exists(model_path):
        print(f"Loading model from {model_path}")
    else:
        print(f"Error: Model file {model_path} not found. Please train first.")
        return
    
    # Load checkpoint and get model parameters
    checkpoint, model_params = load_checkpoint_and_get_model_params(model_path, device, args)
    if checkpoint is None:
        print(f"Error: Could not load model from {model_path}")
        return
    
    # Use checkpoint parameters if available, otherwise fall back to command-line args
    if model_params is not None:
        # Use saved parameters
        effective_norm_type = model_params['norm_type']
        final_norm_skip_first = model_params['norm_skip_first']
        norm_position = model_params['norm_position']
        architecture = model_params['architecture']
        phi_knots = model_params['phi_knots']
        Phi_knots = model_params['Phi_knots']
    else:
        # Fall back to command-line arguments for old checkpoints
        print("WARNING: Old checkpoint format. Using command-line arguments for model configuration.")
        print("Make sure to use the same flags as during training!")
        
        # Determine effective normalization
        if CONFIG.get('use_normalization', True) and not args.no_norm:
            if args.norm_type == "none":
                effective_norm_type = "none"
            elif args.norm_type is not None:
                effective_norm_type = args.norm_type
            else:
                effective_norm_type = CONFIG.get('norm_type', 'batch')
        else:
            effective_norm_type = "none"
        
        # Handle the --norm_first flag
        if args.norm_first:
            final_norm_skip_first = False
        else:
            final_norm_skip_first = args.norm_skip_first if hasattr(args, 'norm_skip_first') else CONFIG.get('norm_skip_first', True)
        
        norm_position = args.norm_position
    
    print(f"Using device: {device}")
    print(f"Image file: {args.image}")
    print()
    
    # Initialize model with the correct configuration
    model = MNISTSprecherNet(
        architecture=architecture,
        phi_knots=phi_knots,
        Phi_knots=Phi_knots,
        norm_type=effective_norm_type,
        norm_position=norm_position,
        norm_skip_first=final_norm_skip_first
    ).to(device)
    
    # Load the model state
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        # New format with complete checkpoint
        model.load_state_dict(checkpoint['model_state_dict'], strict=False)
        print(f"Loaded checkpoint from epoch {checkpoint.get('epoch', 'unknown')} with accuracy {checkpoint.get('accuracy', 'unknown'):.2f}%")
        
        # Restore BatchNorm statistics if available
        if checkpoint.get('has_batchnorm', False) and 'bn_statistics' in checkpoint:
            for name, module in model.named_modules():
                if isinstance(module, nn.BatchNorm1d) and name in checkpoint['bn_statistics']:
                    bn_stats = checkpoint['bn_statistics'][name]
                    module.running_mean.copy_(bn_stats['running_mean'].to(device))
                    module.running_var.copy_(bn_stats['running_var'].to(device))
                    if 'num_batches_tracked' in bn_stats:
                        module.num_batches_tracked.copy_(bn_stats['num_batches_tracked'].to(device))
    else:
        # Old format - just state dict
        model.load_state_dict(checkpoint, strict=False)
        checkpoint = None  # Mark as old format
    
    # Keep model in training mode - do not use model.eval()
    # This ensures consistent behavior with the saved model state
    
    # Update domains
    if CONFIG.get('use_theoretical_domains', True):
        model.update_all_domains()
    
    # Process image
    if not os.path.exists(args.image):
        print(f"Error: '{args.image}' not found. Please provide an image file.")
        return
    
    image_data = process_image(args.image)
    image_tensor = torch.from_numpy(image_data).float().unsqueeze(0).to(device)
    
    # For BatchNorm in training mode, we need a diverse batch
    if effective_norm_type in ['batch', 'layer']:
        # Try to use saved batch from checkpoint for diversity
        if isinstance(checkpoint, dict) and 'x_train' in checkpoint:
            # Use saved batch from checkpoint
            saved_batch = checkpoint['x_train'].to(device)
            # Replace first sample with our inference image
            batch_tensor = saved_batch.clone()
            batch_tensor[0] = image_tensor.squeeze(0)
            image_tensor = batch_tensor
            print(f"Using saved batch of {batch_tensor.shape[0]} samples for BatchNorm diversity")
        else:
            # Fallback: Create diverse batch using noise
            print("No saved batch found, creating diverse batch with noise")
            batch_size = 16
            noise_levels = torch.linspace(0.0, 0.1, batch_size).to(device)
            batch_list = []
            
            for i in range(batch_size):
                if i == 0:
                    # First sample is the original image
                    batch_list.append(image_tensor.squeeze(0))
                else:
                    # Add noise to create diversity
                    noise = torch.randn_like(image_tensor.squeeze(0)) * noise_levels[i]
                    noisy_image = image_tensor.squeeze(0) + noise
                    # Clamp to valid range [0, 1]
                    noisy_image = torch.clamp(noisy_image, 0, 1)
                    batch_list.append(noisy_image)
            
            image_tensor = torch.stack(batch_list)
        
    # Inference
    with torch.no_grad():
        output = model(image_tensor)
        
        # Take only the first result (our target image)
        if effective_norm_type in ['batch', 'layer']:
            output = output[0:1]  # Take first sample
            
        probabilities = torch.softmax(output, dim=1)
        probs_list = probabilities.cpu().numpy().flatten()
        
        print("Probabilities:")
        for i, prob in enumerate(probs_list):
            print(f"  Digit {i}: {prob:.6f}")
        
        predicted = torch.argmax(probabilities, dim=1).item()
        confidence = probs_list[predicted] * 100
        print(f"\nPredicted digit: {predicted} (confidence: {confidence:.1f}%)")


def plot_mnist_splines(args):
    """Plot splines mode."""
    # Try to auto-discover architecture if not specified
    discovered_arch, discovered_epochs = auto_discover_architecture(args, "plot")
    
    # Get configuration values
    if discovered_arch is not None:
        architecture = discovered_arch
        epochs = discovered_epochs
    else:
        # Use defaults or exit if discovery failed with multiple architectures
        if args.arch is None and discovered_arch is None:
            # Either no models found or multiple architectures found
            architecture = MNIST_CONFIG['architecture']
            epochs = args.epochs if args.epochs else MNIST_CONFIG['epochs']
            # If we're using defaults because of multiple architectures, exit
            models = discover_models()
            if len(models) > 1:
                return
        else:
            architecture = [int(x) for x in args.arch.split(",")]
            epochs = args.epochs if args.epochs else MNIST_CONFIG['epochs']
    
    phi_knots = args.phi_knots if args.phi_knots else MNIST_CONFIG['phi_knots']
    Phi_knots = args.Phi_knots if args.Phi_knots else MNIST_CONFIG['Phi_knots']
    model_file = args.model_file if args.model_file else MNIST_CONFIG['model_file']
    
    # Determine device
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    
    # Handle configuration
    if args.no_residual:
        CONFIG['use_residual_weights'] = False
    if args.no_norm:
        CONFIG['use_normalization'] = False
    
    # Construct model filename to match training convention
    config_suffix = get_config_suffix(args, CONFIG)
    arch_str = "-".join(map(str, architecture))
    epochs = args.epochs if args.epochs else MNIST_CONFIG['epochs']
    model_filename = f"sn_mnist_model-{arch_str}-{epochs}epochs{config_suffix}.pth"
    model_path = os.path.join(os.path.dirname(model_file), model_filename)
    
    # Try to find the model file (check for _best variant first)
    best_model_path = model_path.replace('.pth', '_best.pth')
    if os.path.exists(best_model_path):
        model_path = best_model_path
        print(f"Loading best model from {model_path}")
    elif os.path.exists(model_path):
        print(f"Loading model from {model_path}")
    else:
        print(f"Error: Model file {model_path} not found. Please train first.")
        return
    
    # Load checkpoint and get model parameters
    checkpoint, model_params = load_checkpoint_and_get_model_params(model_path, device, args)
    if checkpoint is None:
        print(f"Error: Could not load model from {model_path}")
        return
    
    # Use checkpoint parameters if available, otherwise fall back to command-line args
    if model_params is not None:
        # Use saved parameters
        effective_norm_type = model_params['norm_type']
        final_norm_skip_first = model_params['norm_skip_first']
        norm_position = model_params['norm_position']
        architecture = model_params['architecture']
        phi_knots = model_params['phi_knots']
        Phi_knots = model_params['Phi_knots']
    else:
        # Fall back to command-line arguments for old checkpoints
        print("WARNING: Old checkpoint format. Using command-line arguments for model configuration.")
        print("Make sure to use the same flags as during training!")
        
        # Determine effective normalization
        if CONFIG.get('use_normalization', True) and not args.no_norm:
            if args.norm_type == "none":
                effective_norm_type = "none"
            elif args.norm_type is not None:
                effective_norm_type = args.norm_type
            else:
                effective_norm_type = CONFIG.get('norm_type', 'batch')
        else:
            effective_norm_type = "none"
        
        # Handle the --norm_first flag
        if args.norm_first:
            final_norm_skip_first = False
        else:
            final_norm_skip_first = args.norm_skip_first if hasattr(args, 'norm_skip_first') else CONFIG.get('norm_skip_first', True)
        
        norm_position = args.norm_position
    
    print(f"Using device: {device}")
    print(f"Architecture: 784 -> {architecture} -> 10")
    print(f"phi knots: {phi_knots}, Phi knots: {Phi_knots}")
    print()
    
    # Initialize model with the correct configuration
    model = MNISTSprecherNet(
        architecture=architecture,
        phi_knots=phi_knots,
        Phi_knots=Phi_knots,
        norm_type=effective_norm_type,
        norm_position=norm_position,
        norm_skip_first=final_norm_skip_first
    ).to(device)
    
    # Load the model state
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        # New format with complete checkpoint
        model.load_state_dict(checkpoint['model_state_dict'], strict=False)
        print(f"Loaded checkpoint from epoch {checkpoint.get('epoch', 'unknown')} with accuracy {checkpoint.get('accuracy', 'unknown'):.2f}%")
        
        # Restore BatchNorm statistics if available
        if checkpoint.get('has_batchnorm', False) and 'bn_statistics' in checkpoint:
            for name, module in model.named_modules():
                if isinstance(module, nn.BatchNorm1d) and name in checkpoint['bn_statistics']:
                    bn_stats = checkpoint['bn_statistics'][name]
                    module.running_mean.copy_(bn_stats['running_mean'].to(device))
                    module.running_var.copy_(bn_stats['running_var'].to(device))
                    if 'num_batches_tracked' in bn_stats:
                        module.num_batches_tracked.copy_(bn_stats['num_batches_tracked'].to(device))
    else:
        # Old format - just state dict
        model.load_state_dict(checkpoint, strict=False)
    
    # Keep model in training mode for consistency with checkpoint
    # model.eval()  # Commented out - we maintain the mode from training
    
    # Update domains
    if CONFIG.get('use_theoretical_domains', True):
        print("Updating spline domains based on loaded parameters...")
        model.update_all_domains()
    
    # Print domain information
    print("\nSpline domains after theoretical update:")
    for idx, layer in enumerate(model.sprecher_net.layers):
        print(f"Block {idx+1}:")
        print(f"  phi domain: [{layer.phi.in_min:.3f}, {layer.phi.in_max:.3f}]")
        print(f"  Phi domain: [{layer.Phi.in_min:.3f}, {layer.Phi.in_max:.3f}]")
    
    # Create save path
    if args.save_plots:
        os.makedirs("plots", exist_ok=True)
        config_suffix = get_config_suffix(args, CONFIG)
        arch_str = "-".join(map(str, architecture))
        save_path = f"plots/mnist_splines-{arch_str}{config_suffix}.png"
    else:
        save_path = None
    
    # Plot splines
    print("\nPlotting learned splines...")
    try:
        fig = plot_results(
            model.sprecher_net, 
            model.sprecher_net.layers,
            dataset=None,
            save_path=save_path,
            plot_network=False,
            plot_function=False,
            plot_splines=True,
            title_suffix="MNIST Sprecher Network - Learned Splines"
        )
        
        if not args.no_show and fig is not None:
            print("Displaying plots. Close the plot windows to exit.")
            plt.show()
        elif fig is not None:
            plt.close(fig)
            
    except Exception as e:
        # This block will catch TclError on Windows or other GUI-related errors
        print("\n" + "="*60)
        print("WARNING: A plotting error occurred.")
        print(f"Error type: {type(e).__name__}")
        print("The default plotting backend on your system has failed.")
        print("This is common on systems without a configured GUI toolkit.")
        print("\nSwitching to the reliable 'Agg' backend to proceed.")
        print("="*60 + "\n")
        
        # Set the backend and re-import pyplot
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        
        # Ensure we save plots if this fallback is triggered
        if not args.save_plots:
            print("Plots could not be shown interactively. Enabling file saving automatically.")
            args.save_plots = True
            
        # Re-create save path since we're now saving
        os.makedirs("plots", exist_ok=True)
        config_suffix = get_config_suffix(args, CONFIG)
        arch_str = "-".join(map(str, architecture))
        save_path = f"plots/mnist_splines-{arch_str}{config_suffix}.png"
        
        # Re-run plotting logic with the safe backend
        fig = plot_results(
            model.sprecher_net, 
            model.sprecher_net.layers,
            dataset=None,
            save_path=save_path,
            plot_network=False,
            plot_function=False,
            plot_splines=True,
            title_suffix="MNIST Sprecher Network - Learned Splines"
        )
        
        if fig is not None:
            plt.close(fig)
            
        print(f"Plots saved to {save_path}")
        print(f"\nPlots have been saved to the 'plots/' directory.")
        print("Tip: Use --no_show flag to avoid GUI dependencies in the future.")


def main():
    """Main function."""
    args = parse_args()
    
    # If --no_show is used, force the 'Agg' backend upfront
    if args.no_show:
        matplotlib.use('Agg')
        print("Using non-interactive 'Agg' backend for plotting (as requested by --no_show).")
    
    # Enable domain debugging if requested
    from sn_core.config import CONFIG
    if args.debug_domains:
        CONFIG['debug_domains'] = True
    if args.track_violations:
        CONFIG['track_domain_violations'] = True
    
    # Handle different modes
    if args.mode == "train":
        train_mnist(args)
    elif args.mode == "test":
        test_mnist(args)
    elif args.mode == "infer":
        infer_mnist(args)
    elif args.mode == "plot":
        plot_mnist_splines(args)


if __name__ == "__main__":
    main()