"""
sn_mnist.py - MNIST classification using Sprecher Networks (Modernized)
"""

import os
import sys
import shutil
import argparse
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


def count_parameters_detailed(model):
    """Count parameters with detailed breakdown matching SN code style."""
    # Count different parameter types
    total_spline_knots = 0
    lambda_params = 0
    eta_params = 0
    residual_params = 0
    output_params = 2  # output_scale and output_bias
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
        elif 'output_scale' in name or 'output_bias' in name:
            # Already counted above
            pass
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
        total_params = core_params + residual_params + output_params + codomain_params + norm_params
    else:
        total_params = core_params + residual_params + output_params + norm_params
    
    # Count blocks
    num_blocks = len(model.sprecher_net.layers)
    
    return {
        'total': total_params,
        'core': core_params,
        'spline_knots': total_spline_knots,
        'lambda': lambda_params,
        'eta': eta_params,
        'residual': residual_params,
        'output': output_params,
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
    model.eval()
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


def print_configuration(args, architecture, phi_knots, Phi_knots, epochs, batch_size, lr):
    """Print configuration summary."""
    print(f"Using device: {args.device if args.device != 'auto' else 'cuda' if torch.cuda.is_available() else 'cpu'}")
    print(f"Mode: {args.mode}")
    print(f"Architecture: 784 → {architecture} → 10")
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
        print(f"Normalization: {args.norm_type} (position: {args.norm_position}, skip_first: {args.norm_skip_first})")
    else:
        print(f"Normalization: {CONFIG.get('norm_type', 'batch')} (position: {args.norm_position}, skip_first: {args.norm_skip_first})")
    
    print(f"Scheduler: {'PlateauAwareCosineAnnealingLR' if CONFIG.get('use_advanced_scheduler', False) else 'Adam (fixed LR)'}")
    print()


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
    
    # Print configuration
    print_configuration(args, architecture, phi_knots, Phi_knots, epochs, batch_size, lr)
    
    # Initialize model
    model = MNISTSprecherNet(
        architecture=architecture,
        phi_knots=phi_knots,
        Phi_knots=Phi_knots,
        norm_type=effective_norm_type,
        norm_position=args.norm_position,
        norm_skip_first=args.norm_skip_first
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
    print(f"  - Output scale and bias: {param_counts['output']:,}")
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
    
    for epoch in range(epochs):
        print(f"\nEpoch {epoch + 1}/{epochs}")
        
        avg_loss, train_acc = train_epoch(model, train_loader, optimizer, scheduler, loss_function, device)
        losses.append(avg_loss)
        accuracies.append(train_acc)
        
        # Print epoch summary every 10%
        if (epoch + 1) % max(1, epochs // 10) == 0 or epoch == 0:
            print(f"Epoch {epoch + 1}: Loss = {avg_loss:.4e}, Accuracy = {train_acc:.2f}%")
        
        # Save model
        torch.save(model.state_dict(), model_path)
        
        if train_acc > best_accuracy:
            best_accuracy = train_acc
            best_model_path = model_path.replace('.pth', '_best.pth')
            torch.save(model.state_dict(), best_model_path)
    
    print(f"\nTraining complete! Best accuracy: {best_accuracy:.2f}%")
    print(f"Model saved to: {model_path}")
    
    # Plot training curves if requested
    if args.save_plots or not args.no_show:
        os.makedirs("plots", exist_ok=True)
        
        # Plot loss curve
        loss_filename = f"mnist_loss-{arch_str}-{epochs}epochs{config_suffix}.png"
        loss_save_path = os.path.join("plots", loss_filename) if args.save_plots else None
        plot_loss_curve(losses, loss_save_path)
        
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


def test_mnist(args):
    """Testing mode."""
    # Get configuration values
    architecture = [int(x) for x in args.arch.split(",")] if args.arch else MNIST_CONFIG['architecture']
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
    
    print(f"Using device: {device}")
    print(f"Architecture: 784 → {architecture} → 10")
    print(f"Normalization: {effective_norm_type}")
    print()
    
    # Initialize model
    model = MNISTSprecherNet(
        architecture=architecture,
        phi_knots=phi_knots,
        Phi_knots=Phi_knots,
        norm_type=effective_norm_type,
        norm_position=args.norm_position,
        norm_skip_first=args.norm_skip_first
    ).to(device)
    
    # Load model
    if not os.path.exists(model_file):
        print(f"Error: Model file {model_file} not found. Please train first.")
        return
    
    model.load_state_dict(torch.load(model_file, map_location=device), strict=False)
    model.eval()
    
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
    # Get configuration values
    architecture = [int(x) for x in args.arch.split(",")] if args.arch else MNIST_CONFIG['architecture']
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
    
    print(f"Using device: {device}")
    print(f"Image file: {args.image}")
    print()
    
    # Initialize model
    model = MNISTSprecherNet(
        architecture=architecture,
        phi_knots=phi_knots,
        Phi_knots=Phi_knots,
        norm_type=effective_norm_type,
        norm_position=args.norm_position,
        norm_skip_first=args.norm_skip_first
    ).to(device)
    
    # Load model
    if not os.path.exists(model_file):
        print(f"Error: Model file {model_file} not found. Please train first.")
        return
    
    model.load_state_dict(torch.load(model_file, map_location=device), strict=False)
    model.eval()
    
    # Update domains
    if CONFIG.get('use_theoretical_domains', True):
        model.update_all_domains()
    
    # Process image
    if not os.path.exists(args.image):
        print(f"Error: '{args.image}' not found. Please provide an image file.")
        return
    
    image_data = process_image(args.image)
    image_tensor = torch.from_numpy(image_data).float().unsqueeze(0).to(device)
    
    # Inference
    with torch.no_grad():
        output = model(image_tensor)
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
    # Get configuration values
    architecture = [int(x) for x in args.arch.split(",")] if args.arch else MNIST_CONFIG['architecture']
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
    
    print(f"Using device: {device}")
    print(f"Architecture: 784 → {architecture} → 10")
    print(f"phi knots: {phi_knots}, Phi knots: {Phi_knots}")
    print()
    
    # Initialize model
    model = MNISTSprecherNet(
        architecture=architecture,
        phi_knots=phi_knots,
        Phi_knots=Phi_knots,
        norm_type=effective_norm_type,
        norm_position=args.norm_position,
        norm_skip_first=args.norm_skip_first
    ).to(device)
    
    # Load model
    if not os.path.exists(model_file):
        print(f"Error: Model file {model_file} not found. Please train first.")
        return
    
    model.load_state_dict(torch.load(model_file, map_location=device), strict=False)
    model.eval()
    
    # Update domains
    if CONFIG.get('use_theoretical_domains', True):
        print("Updating spline domains based on loaded parameters...")
        model.update_all_domains()
    
    # Print domain information
    print("\nSpline domains after theoretical update:")
    for idx, layer in enumerate(model.sprecher_net.layers):
        print(f"Block {idx+1}:")
        print(f"  φ domain: [{layer.phi.in_min:.3f}, {layer.phi.in_max:.3f}]")
        print(f"  Φ domain: [{layer.Phi.in_min:.3f}, {layer.Phi.in_max:.3f}]")
    
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
        print(f"Plotting error: {type(e).__name__}")
        if save_path:
            print(f"Attempting to save with Agg backend...")
            matplotlib.use('Agg')
            import matplotlib.pyplot as plt
            
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


def main():
    """Main function."""
    args = parse_args()
    
    # If --no_show is used, force the 'Agg' backend upfront
    if args.no_show:
        matplotlib.use('Agg')
        print("Using non-interactive 'Agg' backend for plotting (as requested by --no_show).")
    
    # Import pyplot after backend selection
    import matplotlib.pyplot as plt
    
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