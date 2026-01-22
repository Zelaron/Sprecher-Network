
"""
sn_fashion_mnist.py - Fashion-MNIST classification using Sprecher Networks

This script provides a benchmark for comparing Sprecher Networks against
GS-KAN and other KAN architectures on the Fashion-MNIST dataset.

GS-KAN reference results:
  - Architecture: [784, 15, 15, 10]
  - Parameters: 12,346
  - Best Test Accuracy: 87.03 ± 0.32%
"""

import os
import sys
import shutil
import argparse
import copy
import time
import json
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torchvision.datasets import FashionMNIST
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
import matplotlib

if '--save_plots' in sys.argv or '--no_show' in sys.argv:
    matplotlib.use('Agg')

import matplotlib.pyplot as plt
from PIL import Image

# Import Sprecher Network components
from sn_core import (
    SprecherMultiLayerNetwork,
    plot_results,
    plot_loss_curve,
    has_batchnorm,
    use_batch_stats_without_updating_bn,
    evaluation_mode,
)
from sn_core.config import CONFIG, MNIST_CONFIG

# Fashion-MNIST class labels
FASHION_CLASSES = [
    'T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
    'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot'
]

# Fashion-MNIST specific defaults
FASHION_CONFIG = {
    'architecture': [15, 15],  # Match GS-KAN architecture for fair comparison
    'phi_knots': 60,           # Slightly higher resolution for Fashion
    'Phi_knots': 60,
    'learning_rate': 0.001,
    'weight_decay': 1e-6,
    'batch_size': 64,
    'epochs': 20,              # GS-KAN used 20 epochs
    'data_directory': './data',
    'model_file': 'sn_fashion_model.pth',
    'plot_directory': './plots',
    'use_batch_norm': True,
}


# =============================================================================
# DEBUGGING UTILITIES - Added to diagnose NaN instability
# =============================================================================

def safe_update_all_domains(model):
    try:
        model.update_all_domains()
    except AssertionError as e:
        if "Invalid interval" in str(e) and "nan" in str(e).lower():
            pass
        else:
            raise


class FashionSprecherNet(nn.Module):
    """Wrapper for SprecherMultiLayerNetwork adapted for Fashion-MNIST."""
    def __init__(self, architecture, phi_knots=60, Phi_knots=60, norm_type='batch',
                 norm_position='after', norm_skip_first=True,
                 phi_spline_type="linear", Phi_spline_type="linear",
                 initialize_domains=True):
        super().__init__()
        self.net = SprecherMultiLayerNetwork(
            input_dim=784,
            architecture=architecture,
            final_dim=10,
            phi_knots=phi_knots,
            Phi_knots=Phi_knots,
            norm_type=norm_type,
            norm_position=norm_position,
            norm_skip_first=norm_skip_first,
            phi_spline_type=phi_spline_type,
            Phi_spline_type=Phi_spline_type,
            initialize_domains=initialize_domains
        )

    def forward(self, x):
        return self.net(x)

    def update_all_domains(self, *args, **kwargs):
        return self.net.update_all_domains(*args, **kwargs)

    def reset_domain_violation_stats(self):
        if hasattr(self.net, "reset_domain_violation_stats"):
            return self.net.reset_domain_violation_stats()

    def print_domain_violation_report(self):
        if hasattr(self.net, "print_domain_violation_report"):
            return self.net.print_domain_violation_report()


def count_parameters(model):
    """Count trainable parameters in model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def create_parser():
    """Create command line argument parser."""
    parser = argparse.ArgumentParser(description="Sprecher Networks on Fashion-MNIST")

    parser.add_argument("--mode", type=str, default="train",
                        choices=["train", "test", "infer", "plot", "benchmark"],
                        help="Mode: train, test, infer, plot, benchmark (default: train)")
    parser.add_argument("--arch", type=str, default=None,
                        help="Hidden layer sizes as comma-separated values (e.g., '15,15')")
    parser.add_argument("--phi_knots", type=int, default=None,
                        help=f"Number of knots for phi splines (default: {FASHION_CONFIG['phi_knots']})")
    parser.add_argument("--Phi_knots", type=int, default=None,
                        help=f"Number of knots for Phi splines (default: {FASHION_CONFIG['Phi_knots']})")
    parser.add_argument("--spline_type", type=str, default=None,
                        choices=["pwl", "linear", "cubic"],
                        help="Convenience switch to set both phi and Phi spline types")
    parser.add_argument("--phi_spline_type", type=str, default=None,
                        choices=["pwl", "linear", "cubic"],
                        help="Spline type for phi (default: 'linear')")
    parser.add_argument("--Phi_spline_type", type=str, default=None,
                        choices=["pwl", "linear", "cubic"],
                        help="Spline type for Phi (default: 'linear')")
    parser.add_argument("--epochs", type=int, default=None,
                        help=f"Number of training epochs (default: {FASHION_CONFIG['epochs']})")
    parser.add_argument("--batch_size", type=int, default=None,
                        help=f"Batch size (default: {FASHION_CONFIG['batch_size']})")
    parser.add_argument("--lr", type=float, default=None,
                        help=f"Learning rate (default: {FASHION_CONFIG['learning_rate']})")
    parser.add_argument("--seed", type=int, default=45,
                        help="Random seed (default: 45)")
    parser.add_argument("--device", type=str, default="auto",
                        choices=["auto", "cpu", "cuda"],
                        help="Device: auto, cpu, or cuda (default: auto)")
    parser.add_argument("--data_dir", type=str, default=None,
                        help="Data directory path (default: ./data)")
    parser.add_argument("--model_file", type=str, default=None,
                        help="Model file path (default: sn_fashion_model.pth)")
    parser.add_argument("--image", type=str, default="clothing.png",
                        help="Image file for inference mode (default: clothing.png)")
    parser.add_argument("--retrain", action="store_true",
                        help="Delete existing model and retrain from scratch")
    parser.add_argument("--save_plots", action="store_true",
                        help="Save plots to files")
    parser.add_argument("--no_show", action="store_true",
                        help="Don't show plots (useful for batch runs)")
    parser.add_argument("--norm_type", type=str,
                        choices=["none", "batch", "layer"],
                        help="Type of normalization to use (default: batch)")
    parser.add_argument("--norm_position", type=str, default="after",
                        choices=["before", "after"],
                        help="Position of normalization relative to blocks (default: after)")
    parser.add_argument("--norm_skip_first", action="store_true", default=True,
                        help="Skip normalization for the first block (default: True)")
    parser.add_argument("--norm_first", action="store_true",
                        help="Enable normalization for the first block (overrides norm_skip_first)")
    parser.add_argument("--no_residual", action="store_true",
                        help="Disable residual connections (default: enabled)")
    parser.add_argument("--residual_style", type=str, default=None,
                        choices=["node", "linear", "standard", "matrix"],
                        help="Residual style: 'node' (original) or 'linear' (standard)")
    parser.add_argument("--no_norm", action="store_true",
                        help="Disable normalization (default: enabled with batch norm)")
    parser.add_argument("--use_advanced_scheduler", action="store_true",
                        help="Use PlateauAwareCosineAnnealingLR scheduler (default: disabled)")
    parser.add_argument("--no_lateral", action="store_true",
                        help="Disable lateral mixing connections (default: enabled)")
    parser.add_argument("--lateral_type", type=str, default=None,
                        choices=["cyclic", "bidirectional"],
                        help="Type of lateral mixing (default: cyclic)")
    parser.add_argument("--low_memory", action="store_true",
                        help="Enable low memory mode for large models")
    parser.add_argument("--debug_domains", action="store_true",
                        help="Enable domain debugging output")
    parser.add_argument("--track_violations", action="store_true",
                        help="Track domain violations statistics")
    parser.add_argument("--seeds", type=int, default=3,
                        help="Number of random seeds for benchmark mode (default: 3)")

    return parser


def get_config_suffix(args, config):
    """Generate suffix string for model filename based on configuration."""
    parts = []
    if args.no_residual:
        parts.append("NoRes")
    else:
        residual_style = config.get('residual_style', 'node')
        if residual_style != 'node':
            parts.append(f"Res{residual_style.capitalize()}")
    if args.no_lateral:
        parts.append("NoLat")
    else:
        lateral_type = config.get('lateral_mixing_type', 'cyclic')
        if lateral_type != 'cyclic':
            parts.append(f"Lat{lateral_type.capitalize()}")
    if args.no_norm:
        parts.append("NoNorm")
    else:
        norm_type = args.norm_type if args.norm_type else "batch"
        if norm_type != "batch":
            parts.append(f"Norm{norm_type.capitalize()}")
    if args.spline_type:
        parts.append(f"Spline{args.spline_type.capitalize()}")
    else:
        if args.phi_spline_type and args.phi_spline_type != 'linear':
            parts.append(f"Phi{args.phi_spline_type.capitalize()}")
        if args.Phi_spline_type and args.Phi_spline_type != 'linear':
            parts.append(f"PHI{args.Phi_spline_type.capitalize()}")
    if config.get('use_advanced_scheduler', False):
        parts.append("AdvScheduler")
    if config.get('low_memory_mode', False):
        parts.append("LowMem")
    if hasattr(args, 'norm_position') and args.norm_position != 'after':
        parts.append(f"Norm{args.norm_position.capitalize()}")
    if hasattr(args, 'norm_first') and args.norm_first:
        parts.append("NormFirst")
    elif hasattr(args, 'norm_skip_first') and not args.norm_skip_first:
        parts.append("NormAll")
    if parts:
        return "-" + "-".join(parts)
    return ""


def train_epoch(model, train_loader, optimizer, scheduler, loss_function, device):
    """Train for one epoch.

    This function contains safety guards against numerical issues:
      * Skip batches with non-finite outputs/loss.
      * Skip optimizer steps when any gradient is non-finite.
      * Avoid the Inf*0 -> NaN failure mode in gradient clipping.
    """
    model.train()

    loss_sum = 0.0
    loss_count = 0
    correct = 0
    total = 0

    skipped_batches = 0
    first_skip_info = None  # (batch_idx, reason)

    if CONFIG.get('track_domain_violations', False):
        model.reset_domain_violation_stats()

    with tqdm(train_loader, unit="batch", leave=False) as tepoch:
        for batch_idx, (images, labels) in enumerate(tepoch):
            images, labels = images.to(device), labels.to(device)
            images = images.view(images.size(0), -1)
            images = (images + 1) / 2

            if CONFIG.get('use_theoretical_domains', True):
                safe_update_all_domains(model)

            # Zero grads (set_to_none is a bit faster/more memory-friendly)
            try:
                optimizer.zero_grad(set_to_none=True)
            except TypeError:
                optimizer.zero_grad()

            outputs = model(images)

            if not torch.isfinite(outputs).all():
                skipped_batches += 1
                if first_skip_info is None:
                    first_skip_info = (batch_idx, "non-finite model outputs")
                continue

            loss = loss_function(outputs, labels)

            if not torch.isfinite(loss):
                skipped_batches += 1
                if first_skip_info is None:
                    first_skip_info = (batch_idx, "non-finite loss")
                continue

            loss.backward()

            grads_finite = True
            for p in model.parameters():
                if p.grad is None:
                    continue
                if not torch.isfinite(p.grad).all():
                    grads_finite = False
                    break

            if not grads_finite:
                skipped_batches += 1
                if first_skip_info is None:
                    first_skip_info = (batch_idx, "non-finite gradients")
                try:
                    optimizer.zero_grad(set_to_none=True)
                except TypeError:
                    optimizer.zero_grad()
                continue

            # Safe gradient clipping
            max_norm = float(CONFIG.get('max_grad_norm', 1.0))
            try:
                grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm, error_if_nonfinite=True)
            except TypeError:
                grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
            except RuntimeError:
                # Non-finite norm detected
                skipped_batches += 1
                if first_skip_info is None:
                    first_skip_info = (batch_idx, "non-finite grad norm")
                try:
                    optimizer.zero_grad(set_to_none=True)
                except TypeError:
                    optimizer.zero_grad()
                continue

            if not torch.isfinite(torch.as_tensor(grad_norm)):
                skipped_batches += 1
                if first_skip_info is None:
                    first_skip_info = (batch_idx, "non-finite grad norm")
                try:
                    optimizer.zero_grad(set_to_none=True)
                except TypeError:
                    optimizer.zero_grad()
                continue

            optimizer.step()

            # Stats
            loss_sum += float(loss.item())
            loss_count += 1

            _, predicted = torch.max(outputs.detach(), 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            # tqdm progress display (doesn't spam logs because leave=False)
            avg_loss_so_far = loss_sum / max(1, loss_count)
            acc_so_far = 100.0 * correct / max(1, total)
            tepoch.set_postfix(loss=f"{avg_loss_so_far:.4f}", acc=f"{acc_so_far:.1f}%")

    avg_loss = loss_sum / max(1, loss_count)
    train_accuracy = 100.0 * correct / max(1, total)

    # Step scheduler on a finite scalar
    if scheduler is not None:
        scheduler.step(avg_loss)

    if skipped_batches > 0:
        if first_skip_info is not None:
            bidx, reason = first_skip_info
            print(f"  [WARN] Skipped {skipped_batches} batches in this epoch (first at batch {bidx}: {reason}).")
        else:
            print(f"  [WARN] Skipped {skipped_batches} batches in this epoch due to numerical issues.")

    if CONFIG.get('track_domain_violations', False):
        print("\nTraining set domain violations:")
        model.print_domain_violation_report()

    return avg_loss, train_accuracy

def test_model(model, test_loader, device, use_batch_stats=True):
    """Evaluate model on test set."""
    correct = 0
    total = 0

    if CONFIG.get('track_domain_violations', False):
        model.reset_domain_violation_stats()

    was_training = model.training

    if use_batch_stats and has_batchnorm(model):
        model.train(True)
        with torch.no_grad():
            with use_batch_stats_without_updating_bn(model):
                for images, labels in tqdm(test_loader, desc="Testing", leave=False):
                    images, labels = images.to(device), labels.to(device)
                    images = images.view(images.size(0), -1)
                    images = (images + 1) / 2

                    if CONFIG.get('use_theoretical_domains', True):
                        safe_update_all_domains(model)

                    outputs = model(images)
                    _, predicted = torch.max(outputs.data, 1)
                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()
    else:
        with evaluation_mode(model):
            for images, labels in tqdm(test_loader, desc="Testing", leave=False):
                images, labels = images.to(device), labels.to(device)
                images = images.view(images.size(0), -1)
                images = (images + 1) / 2

                if CONFIG.get('use_theoretical_domains', True):
                    safe_update_all_domains(model)

                outputs = model(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

    model.train(was_training)

    if CONFIG.get('track_domain_violations', False):
        print("\nTest set domain violations:")
        model.print_domain_violation_report()

    accuracy = 100 * correct / total
    return accuracy


def print_comparison_header():
    """Print header with GS-KAN reference results."""
    print("\n" + "="*70)
    print("FASHION-MNIST BENCHMARK: Sprecher Networks vs GS-KAN")
    print("="*70)
    print("\nGS-KAN Reference Baseline:")
    print("  Architecture: [784, 15, 15, 10]")
    print("  Parameters: 12,346")
    print("  Test Accuracy: 87.03 ± 0.32%")
    print("-"*70)


def print_configuration(args, architecture, phi_knots, Phi_knots, epochs, batch_size, lr,
                        effective_norm_type, norm_position, final_norm_skip_first, use_residual,
                        residual_style, use_lateral, lateral_type, phi_spline_type, Phi_spline_type):
    """Print configuration summary."""
    device_str = args.device if args.device != 'auto' else ('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nSprecher Network Configuration:")
    print(f"  Device: {device_str}")
    print(f"  Architecture: 784 -> {architecture} -> 10")
    print(f"  Depth: {len(architecture)} hidden layers ({len(architecture) + 1} blocks)")
    print(f"  phi knots: {phi_knots}, Phi knots: {Phi_knots}")
    print(f"  Effective norm type: {effective_norm_type} (position: {norm_position}, skip_first: {final_norm_skip_first})")
    print(f"  Residual: {'OFF' if not use_residual else 'ON'} (style: {residual_style})")
    print(f"  Lateral mixing: {'OFF' if not use_lateral else 'ON'} (type: {lateral_type})")
    print(f"  Spline types: phi={phi_spline_type}, Phi={Phi_spline_type}")
    print(f"  Epochs: {epochs}, Batch size: {batch_size}, LR: {lr}")
    print("-"*70)


def train_fashion(args):
    """Training mode."""
    print_comparison_header()

    architecture = [int(x) for x in args.arch.split(",")] if args.arch else FASHION_CONFIG['architecture']
    phi_knots = args.phi_knots if args.phi_knots else FASHION_CONFIG['phi_knots']
    Phi_knots = args.Phi_knots if args.Phi_knots else FASHION_CONFIG['Phi_knots']
    epochs = args.epochs if args.epochs else FASHION_CONFIG['epochs']
    batch_size = args.batch_size if args.batch_size else FASHION_CONFIG['batch_size']
    lr = args.lr if args.lr else FASHION_CONFIG['learning_rate']
    model_file = args.model_file if args.model_file else FASHION_CONFIG['model_file']
    data_dir = args.data_dir if args.data_dir else FASHION_CONFIG['data_directory']

    if args.phi_spline_type:
        phi_spline_type = args.phi_spline_type
    elif args.spline_type:
        phi_spline_type = args.spline_type
    else:
        phi_spline_type = 'linear'

    if args.Phi_spline_type:
        Phi_spline_type = args.Phi_spline_type
    elif args.spline_type:
        Phi_spline_type = args.spline_type
    else:
        Phi_spline_type = 'linear'

    if args.device == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(args.device)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    CONFIG['seed'] = args.seed

    if args.debug_domains:
        CONFIG['debug_domains'] = True
    if args.track_violations:
        CONFIG['track_domain_violations'] = True

    if args.no_residual:
        CONFIG['use_residual_connections'] = False
    else:
        CONFIG['use_residual_connections'] = True
        if args.residual_style:
            CONFIG['residual_style'] = args.residual_style

    if args.no_lateral:
        CONFIG['use_lateral_mixing'] = False
    else:
        CONFIG['use_lateral_mixing'] = True
        if args.lateral_type:
            CONFIG['lateral_mixing_type'] = args.lateral_type

    if args.low_memory:
        CONFIG['low_memory_mode'] = True

    if args.use_advanced_scheduler:
        CONFIG['use_advanced_scheduler'] = True

    if args.no_norm:
        effective_norm_type = 'none'
    elif args.norm_type:
        effective_norm_type = args.norm_type
    else:
        effective_norm_type = 'batch'

    norm_position = args.norm_position if hasattr(args, 'norm_position') else 'after'

    final_norm_skip_first = True
    if args.norm_first:
        final_norm_skip_first = False
    else:
        if hasattr(args, 'norm_skip_first'):
            final_norm_skip_first = args.norm_skip_first

    print_configuration(
        args, architecture, phi_knots, Phi_knots, epochs, batch_size, lr,
        effective_norm_type, norm_position, final_norm_skip_first,
        not args.no_residual, CONFIG.get('residual_style', 'node'),
        not args.no_lateral, CONFIG.get('lateral_mixing_type', 'cyclic'),
        phi_spline_type, Phi_spline_type
    )

    model = FashionSprecherNet(
        architecture=architecture,
        phi_knots=phi_knots,
        Phi_knots=Phi_knots,
        norm_type=effective_norm_type,
        norm_position=norm_position,
        norm_skip_first=final_norm_skip_first,
        phi_spline_type=phi_spline_type,
        Phi_spline_type=Phi_spline_type
    ).to(device)

    sn_params = count_parameters(model)
    gskan_params = 12346
    print(f"\nParameter Count Comparison:")
    print(f"GS-KAN: {gskan_params:,} parameters")
    print(f"SN:     {sn_params:,} parameters ({gskan_params/sn_params:.1f}x FEWER than GS-KAN)")

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    train_dataset = FashionMNIST(root=data_dir, train=True, download=True, transform=transform)
    test_dataset = FashionMNIST(root=data_dir, train=False, download=True, transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    config_suffix = get_config_suffix(args, CONFIG)
    arch_str = "-".join(map(str, architecture))
    model_filename = f"sn_fashion_model-{arch_str}-{epochs}epochs{config_suffix}.pth"
    model_path = os.path.join(os.path.dirname(model_file) if os.path.dirname(model_file) else ".", model_filename)

    if os.path.exists(model_path) and args.retrain:
        os.remove(model_path)
        print(f"Deleted existing model file: {model_path}")
        best_path = model_path.replace('.pth', '_best.pth')
        if os.path.exists(best_path):
            os.remove(best_path)
    elif os.path.exists(model_path) and not args.retrain:
        model.load_state_dict(torch.load(model_path, map_location=device, weights_only=False), strict=False)
        print(f"Loaded saved model from {model_path} for further training.")

    optimizer = optim.AdamW(
        model.parameters(),
        lr=lr,
        weight_decay=FASHION_CONFIG.get('weight_decay', 1e-6)
    )

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

    print(f"Starting training for {epochs} epochs...")
    print("-"*70)
    best_test_accuracy = 0
    best_train_accuracy = 0
    losses = []
    train_accuracies = []
    test_accuracies = []
    best_checkpoint = None
    
    sample_batch = next(iter(train_loader))

    start_time = time.time()

    for epoch in range(epochs):
        epoch_start = time.time()

        avg_loss, train_acc = train_epoch(model, train_loader, optimizer, scheduler, loss_function, device)
        test_acc = test_model(model, test_loader, device)

        epoch_time = time.time() - epoch_start

        losses.append(avg_loss)
        train_accuracies.append(train_acc)
        test_accuracies.append(test_acc)

        print(f"Epoch {epoch + 1:3d}/{epochs}: Loss={avg_loss:.4f}, Train={train_acc:.2f}%, Test={test_acc:.2f}% ({epoch_time:.1f}s)")


        if test_acc > best_test_accuracy:
            best_test_accuracy = test_acc
            best_train_accuracy = train_acc

            bn_statistics = {}
            has_bn = effective_norm_type in ['batch', 'layer']
            if has_bn:
                for name, module in model.named_modules():
                    if isinstance(module, nn.BatchNorm1d):
                        bn_statistics[name] = {
                            'running_mean': module.running_mean.clone().cpu(),
                            'running_var': module.running_var.clone().cpu(),
                            'num_batches_tracked': module.num_batches_tracked.clone().cpu(),
                        }

            best_checkpoint = {
                'epoch': epoch,
                'model_state_dict': copy.deepcopy(model.state_dict()),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': avg_loss,
                'train_accuracy': train_acc,
                'test_accuracy': test_acc,
                'has_batchnorm': has_bn,
                'bn_statistics': bn_statistics,
                'model_params': {
                    'architecture': architecture,
                    'phi_knots': phi_knots,
                    'Phi_knots': Phi_knots,
                    'norm_type': effective_norm_type,
                    'norm_position': args.norm_position,
                    'norm_skip_first': final_norm_skip_first,
                    'phi_spline_type': phi_spline_type,
                    'Phi_spline_type': Phi_spline_type,
                },
                'training_args': {
                    'no_residual': args.no_residual,
                    'residual_style': CONFIG.get('residual_style', 'node'),
                    'no_lateral': args.no_lateral if hasattr(args, 'no_lateral') else False,
                    'lateral_type': CONFIG.get('lateral_mixing_type', 'cyclic'),
                    'no_norm': args.no_norm,
                    'norm_type': args.norm_type,
                    'norm_position': args.norm_position,
                    'norm_first': args.norm_first,
                    'phi_knots': phi_knots,
                    'Phi_knots': Phi_knots,
                    'phi_spline_type': phi_spline_type,
                    'Phi_spline_type': Phi_spline_type,
                    'seed': args.seed,
                }
            }

            best_path = model_path.replace('.pth', '_best.pth')
            torch.save(best_checkpoint, best_path)
            torch.save(model.state_dict(), model_path)

    total_time = time.time() - start_time

    print("-"*70)
    print(f"Training completed in {total_time:.1f}s")
    print(f"Best Test Accuracy: {best_test_accuracy:.2f}% (Train: {best_train_accuracy:.2f}%)")

    torch.save(model.state_dict(), model_path)
    print(f"Model saved to {model_path}")

    if args.save_plots or not args.no_show:
        os.makedirs(FASHION_CONFIG['plot_directory'], exist_ok=True)
        plot_dir = FASHION_CONFIG['plot_directory']
        config_suffix = get_config_suffix(args, CONFIG)
        plot_prefix = f"fashion_{arch_str}_{epochs}{config_suffix}"

        plt.figure(figsize=(10, 5))
        plt.plot(train_accuracies, label='Train')
        plt.plot(test_accuracies, label='Test')
        plt.axhline(y=87.03, color='r', linestyle='--', label='GS-KAN (87.03%)')
        plt.title('Fashion-MNIST Accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy (%)')
        plt.legend()
        plt.grid(True)

        if args.save_plots:
            acc_path = os.path.join(plot_dir, f"{plot_prefix}_accuracy.png")
            plt.savefig(acc_path)
            print(f"Saved accuracy plot to {acc_path}")

        if not args.no_show:
            plt.show()
        else:
            plt.close()

        plt.figure(figsize=(10, 5))
        plt.plot(losses)
        plt.title('Training Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.grid(True)

        if args.save_plots:
            loss_path = os.path.join(plot_dir, f"{plot_prefix}_loss.png")
            plt.savefig(loss_path)
            print(f"Saved loss plot to {loss_path}")

        if not args.no_show:
            plt.show()
        else:
            plt.close()


def test_fashion(args):
    """Test mode - evaluate saved model."""
    print_comparison_header()

    import glob
    model_files = glob.glob("sn_fashion_model-*_best.pth")
    if not model_files:
        model_files = glob.glob("sn_fashion_model-*.pth")
    if not model_files:
        print("ERROR: No saved Fashion-MNIST model found.")
        print("Train a model first with: python sn_fashion_mnist.py --mode train")
        return

    model_file = sorted(model_files)[-1]
    print(f"Loading model from: {model_file}")

    if args.device == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(args.device)

    checkpoint = torch.load(model_file, map_location=device, weights_only=False)

    if 'model_params' in checkpoint:
        params = checkpoint['model_params']
        architecture = params['architecture']
        phi_knots = params.get('phi_knots', FASHION_CONFIG['phi_knots'])
        Phi_knots = params.get('Phi_knots', FASHION_CONFIG['Phi_knots'])
        norm_type = params.get('norm_type', 'batch')
        norm_position = params.get('norm_position', 'after')
        norm_skip_first = params.get('norm_skip_first', True)
        phi_spline_type = params.get('phi_spline_type', 'linear')
        Phi_spline_type = params.get('Phi_spline_type', 'linear')
    else:
        print("Warning: Using default configuration")
        architecture = FASHION_CONFIG['architecture']
        phi_knots = FASHION_CONFIG['phi_knots']
        Phi_knots = FASHION_CONFIG['Phi_knots']
        norm_type = 'batch'
        norm_position = 'after'
        norm_skip_first = True
        phi_spline_type = 'linear'
        Phi_spline_type = 'linear'

    model = FashionSprecherNet(
        architecture=architecture,
        phi_knots=phi_knots,
        Phi_knots=Phi_knots,
        norm_type=norm_type,
        norm_position=norm_position,
        norm_skip_first=norm_skip_first,
        phi_spline_type=phi_spline_type,
        Phi_spline_type=Phi_spline_type,
        initialize_domains=False,
    ).to(device)

    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'], strict=False)
    else:
        model.load_state_dict(checkpoint, strict=False)

    if 'bn_statistics' in checkpoint:
        for name, stats in checkpoint['bn_statistics'].items():
            try:
                module = dict(model.named_modules())[name]
                if isinstance(module, nn.BatchNorm1d):
                    module.running_mean.copy_(stats['running_mean'].to(device))
                    module.running_var.copy_(stats['running_var'].to(device))
                    module.num_batches_tracked.copy_(stats['num_batches_tracked'].to(device))
            except KeyError:
                pass

    data_dir = args.data_dir if args.data_dir else FASHION_CONFIG['data_directory']
    batch_size = args.batch_size if args.batch_size else FASHION_CONFIG['batch_size']
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    test_dataset = FashionMNIST(root=data_dir, train=False, download=True, transform=transform)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    accuracy = test_model(model, test_loader, device)
    print(f"\nTest Accuracy: {accuracy:.2f}%")
    print(f"GS-KAN Reference: 87.03%")
    print(f"Difference: {87.03 - accuracy:.2f}%")


def infer_fashion(args):
    """Inference mode - classify a single image."""
    import glob

    model_files = glob.glob("sn_fashion_model-*_best.pth")
    if not model_files:
        model_files = glob.glob("sn_fashion_model-*.pth")
    if not model_files:
        print("ERROR: No saved Fashion-MNIST model found.")
        print("Train a model first with: python sn_fashion_mnist.py --mode train")
        return

    model_file = sorted(model_files)[-1]
    print(f"Loading model from: {model_file}")

    if args.device == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(args.device)

    checkpoint = torch.load(model_file, map_location=device, weights_only=False)

    if 'model_params' in checkpoint:
        params = checkpoint['model_params']
        architecture = params['architecture']
        phi_knots = params.get('phi_knots', FASHION_CONFIG['phi_knots'])
        Phi_knots = params.get('Phi_knots', FASHION_CONFIG['Phi_knots'])
        norm_type = params.get('norm_type', 'batch')
        norm_position = params.get('norm_position', 'after')
        norm_skip_first = params.get('norm_skip_first', True)
        phi_spline_type = params.get('phi_spline_type', 'linear')
        Phi_spline_type = params.get('Phi_spline_type', 'linear')
    else:
        architecture = FASHION_CONFIG['architecture']
        phi_knots = FASHION_CONFIG['phi_knots']
        Phi_knots = FASHION_CONFIG['Phi_knots']
        norm_type = 'batch'
        norm_position = 'after'
        norm_skip_first = True
        phi_spline_type = 'linear'
        Phi_spline_type = 'linear'

    model = FashionSprecherNet(
        architecture=architecture,
        phi_knots=phi_knots,
        Phi_knots=Phi_knots,
        norm_type=norm_type,
        norm_position=norm_position,
        norm_skip_first=norm_skip_first,
        phi_spline_type=phi_spline_type,
        Phi_spline_type=Phi_spline_type,
        initialize_domains=False,
    ).to(device)

    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'], strict=False)
    else:
        model.load_state_dict(checkpoint, strict=False)

    image = Image.open(args.image).convert('L')
    if image.size != (28, 28):
        image = image.resize((28, 28), Image.Resampling.LANCZOS)

    data = np.asarray(image)
    data = data.astype(np.float32) / 255.0
    data = (data - 0.5) / 0.5

    image_tensor = torch.tensor(data).view(1, -1).to(device)

    model.eval()
    with torch.no_grad():
        if has_batchnorm(model):
            with use_batch_stats_without_updating_bn(model):
                output = model(image_tensor)
        else:
            output = model(image_tensor)

        probabilities = torch.softmax(output, dim=1)
        probs_list = probabilities.cpu().numpy().flatten()

        print("\nProbabilities:")
        for i, (cls, prob) in enumerate(zip(FASHION_CLASSES, probs_list)):
            print(f"  {cls:12s}: {prob:.4f}")

        predicted = torch.argmax(probabilities, dim=1).item()
        confidence = probs_list[predicted] * 100
        print(f"\nPredicted: {FASHION_CLASSES[predicted]} ({confidence:.1f}%)")


def plot_splines(args):
    """Plot learned splines from a saved model."""
    print("\nPlotting learned splines...")
    print("-"*40)

    import glob
    model_files = glob.glob("sn_fashion_model-*_best.pth")
    if not model_files:
        model_files = glob.glob("sn_fashion_model-*.pth")
    if not model_files:
        print("ERROR: No saved Fashion-MNIST model found.")
        print("Train a model first with: python sn_fashion_mnist.py --mode train")
        return

    model_file = sorted(model_files)[-1]
    print(f"Loading model from: {model_file}")

    if args.device == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(args.device)

    checkpoint = torch.load(model_file, map_location=device, weights_only=False)

    if 'model_params' in checkpoint:
        params = checkpoint['model_params']
        architecture = params['architecture']
        phi_knots = params.get('phi_knots', FASHION_CONFIG['phi_knots'])
        Phi_knots = params.get('Phi_knots', FASHION_CONFIG['Phi_knots'])
        norm_type = params.get('norm_type', 'batch')
        norm_position = params.get('norm_position', 'after')
        norm_skip_first = params.get('norm_skip_first', True)
        phi_spline_type = params.get('phi_spline_type', 'linear')
        Phi_spline_type = params.get('Phi_spline_type', 'linear')
    else:
        print("Warning: Using default configuration")
        architecture = FASHION_CONFIG['architecture']
        phi_knots = FASHION_CONFIG['phi_knots']
        Phi_knots = FASHION_CONFIG['Phi_knots']
        norm_type = 'batch'
        norm_position = 'after'
        norm_skip_first = True
        phi_spline_type = 'linear'
        Phi_spline_type = 'linear'

    model = FashionSprecherNet(
        architecture=architecture,
        phi_knots=phi_knots,
        Phi_knots=Phi_knots,
        norm_type=norm_type,
        norm_position=norm_position,
        norm_skip_first=norm_skip_first,
        phi_spline_type=phi_spline_type,
        Phi_spline_type=Phi_spline_type,
        initialize_domains=False,
    ).to(device)

    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'], strict=False)
    else:
        model.load_state_dict(checkpoint, strict=False)

    if hasattr(model.net, 'layers'):
        first_block = model.net.layers[0]
        if hasattr(first_block, 'phi') and hasattr(first_block, 'Phi'):
            print("Plotting first layer phi and Phi splines...")

            os.makedirs(FASHION_CONFIG['plot_directory'], exist_ok=True)
            plot_dir = FASHION_CONFIG['plot_directory']

            x_phi = torch.linspace(first_block.phi.in_min, first_block.phi.in_max, 300).to(device)
            y_phi = first_block.phi(x_phi)

            plt.figure(figsize=(8, 4))
            plt.plot(x_phi.cpu().numpy(), y_phi.detach().cpu().numpy())
            plt.title("First Layer phi Spline")
            plt.xlabel("Input")
            plt.ylabel("Output")
            plt.grid(True)
            phi_path = os.path.join(plot_dir, "fashion_phi_spline.png")
            plt.savefig(phi_path)
            print(f"Saved phi spline plot to {phi_path}")
            if not args.no_show:
                plt.show()
            else:
                plt.close()

            x_Phi = torch.linspace(first_block.Phi.in_min, first_block.Phi.in_max, 300).to(device)
            y_Phi = first_block.Phi(x_Phi)

            plt.figure(figsize=(8, 4))
            plt.plot(x_Phi.cpu().numpy(), y_Phi.detach().cpu().numpy())
            plt.title("First Layer Phi Spline")
            plt.xlabel("Input")
            plt.ylabel("Output")
            plt.grid(True)
            Phi_path = os.path.join(plot_dir, "fashion_Phi_spline.png")
            plt.savefig(Phi_path)
            print(f"Saved Phi spline plot to {Phi_path}")
            if not args.no_show:
                plt.show()
            else:
                plt.close()
        else:
            print("Model does not expose phi/Phi splines for plotting.")
    else:
        print("Model does not expose layers for plotting.")


def run_benchmark(args):
    """Benchmark mode - run multiple seeds and compare to GS-KAN reference."""
    print_comparison_header()

    num_seeds = args.seeds
    print(f"\nRunning benchmark with {num_seeds} seeds...")

    results = []
    base_seed = args.seed
    
    for i in range(num_seeds):
        seed = base_seed + i * 100
        print(f"\n--- Seed {seed} ({i+1}/{num_seeds}) ---")

        seed_args = copy.deepcopy(args)
        seed_args.seed = seed
        seed_args.mode = 'train'
        seed_args.save_plots = False
        seed_args.no_show = True

        train_fashion(seed_args)

        import glob
        model_files = glob.glob("sn_fashion_model-*_best.pth")
        if model_files:
            model_file = sorted(model_files)[-1]
            checkpoint = torch.load(model_file, map_location='cpu', weights_only=False)
            test_acc = checkpoint.get('test_accuracy', None)
            if test_acc is not None:
                results.append(test_acc)
                print(f"Seed {seed}: Best Test Accuracy = {test_acc:.2f}%")

    if results:
        mean_acc = np.mean(results)
        std_acc = np.std(results)

        print("\n" + "="*50)
        print("Benchmark Results Summary:")
        print(f"  SN Mean Accuracy: {mean_acc:.2f}% +/- {std_acc:.2f}%")
        print(f"  GS-KAN Reference: 87.03% +/- 0.32%")
        print(f"  Difference: {87.03 - mean_acc:.2f}%")
        print("="*50)

        with open("fashion_benchmark_results.json", "w") as f:
            json.dump({
                'results': results,
                'mean': mean_acc,
                'std': std_acc,
                'gskan_mean': 87.03,
                'gskan_std': 0.32
            }, f, indent=2)
        print("Saved benchmark results to fashion_benchmark_results.json")


def main():
    parser = create_parser()
    args = parser.parse_args()

    if args.low_memory:
        CONFIG['low_memory_mode'] = True
    if args.use_advanced_scheduler:
        CONFIG['use_advanced_scheduler'] = True
    if args.debug_domains:
        CONFIG['debug_domains'] = True
    if args.track_violations:
        CONFIG['track_domain_violations'] = True

    if args.no_residual:
        CONFIG['use_residual_connections'] = False
    else:
        CONFIG['use_residual_connections'] = True
        if args.residual_style:
            CONFIG['residual_style'] = args.residual_style

    if args.no_lateral:
        CONFIG['use_lateral_mixing'] = False
    else:
        CONFIG['use_lateral_mixing'] = True
        if args.lateral_type:
            CONFIG['lateral_mixing_type'] = args.lateral_type

    if args.mode == "train":
        train_fashion(args)
    elif args.mode == "test":
        test_fashion(args)
    elif args.mode == "infer":
        infer_fashion(args)
    elif args.mode == "plot":
        plot_splines(args)
    elif args.mode == "benchmark":
        run_benchmark(args)


if __name__ == "__main__":
    main()
