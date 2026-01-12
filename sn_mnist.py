"""
sn_mnist.py - MNIST classification using Sprecher Networks
"""

import os
import sys
import shutil
import argparse
import copy
import torch
import torch.nn as nn
import torch.optim as optim
_TORCHVISION_IMPORT_ERROR = None
try:
    import torchvision.transforms as transforms
    from torchvision.datasets import MNIST
except Exception as _tv_e:
    transforms = None
    MNIST = None
    _TORCHVISION_IMPORT_ERROR = _tv_e
from torch.utils.data import DataLoader
from tqdm import tqdm
from PIL import Image
import numpy as np
import struct
import math
import matplotlib
import matplotlib.pyplot as plt

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


def _require_torchvision():
    """Raise a helpful error if torchvision failed to import."""
    if transforms is None or MNIST is None:
        raise ImportError(
            "torchvision is required for this mode, but it failed to import. "
            f"Original error: {_TORCHVISION_IMPORT_ERROR!r}"
        )


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Train Sprecher Networks on MNIST")
    
    # Mode selection
    parser.add_argument("--mode", type=str, required=True,
                        choices=["train", "test", "infer", "plot", "export_nds"],
                        help="Operation mode: train, test, infer, or plot")
    
    # Model selection for test/infer/plot modes
    parser.add_argument("--model-index", type=int, default=None,
                        help="Select model by index when multiple models are found (1-based)")
    
    # Architecture and model settings
    parser.add_argument("--arch", type=str, default=None,
                        help="Architecture as comma-separated values (default: from MNIST_CONFIG)")
    parser.add_argument("--phi_knots", type=int, default=None,
                        help="Number of knots for phi splines (default: from MNIST_CONFIG)")
    parser.add_argument("--Phi_knots", type=int, default=None,
                        help="Number of knots for Phi splines (default: from MNIST_CONFIG)")
    
    # Spline type arguments
    parser.add_argument("--spline_type", type=str, default=None,
                        choices=["pwl", "linear", "cubic"],
                        help="Convenience switch to set both phi and Phi spline types")
    parser.add_argument("--phi_spline_type", type=str, default=None,
                        choices=["pwl", "linear", "cubic"],
                        help="Spline type for phi (default: from --spline_type or 'linear')")
    parser.add_argument("--Phi_spline_type", type=str, default=None,
                        choices=["pwl", "linear", "cubic"],
                        help="Spline type for Phi (default: from --spline_type or 'linear')")
    
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

    # Export (Nintendo DS) options
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Checkpoint to load for export_nds mode (default: most recent in current directory)")
    parser.add_argument("--export_out", type=str, default="sprecher_ds/sn_weights.bin",
                        help="Output path for export_nds weights (default: sprecher_ds/sn_weights.bin)")
    
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
    parser.add_argument("--residual_style", type=str, default=None,
                        choices=["node", "linear", "standard", "matrix"],
                        help="Residual style: 'node' (original) or 'linear' (standard)")
    parser.add_argument("--no_norm", action="store_true",
                        help="Disable normalization (default: enabled with batch norm)")
    parser.add_argument("--use_advanced_scheduler", action="store_true",
                        help="Use PlateauAwareCosineAnnealingLR scheduler (default: disabled)")
    
    # Lateral mixing arguments
    parser.add_argument("--no_lateral", action="store_true",
                        help="Disable lateral mixing connections (default: enabled)")
    parser.add_argument("--lateral_type", type=str, default=None,
                        choices=["cyclic", "bidirectional"],
                        help="Type of lateral mixing (default: from CONFIG)")
    
    # Memory optimization
    parser.add_argument("--low_memory_mode", action="store_true",
                        help="Use memory-efficient computation")
    parser.add_argument("--memory_debug", action="store_true",
                        help="Print CUDA memory usage statistics during forward pass")
    
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
    else:
        style = CONFIG.get('residual_style', 'node')
        if style not in (None, 'node'):
            parts.append("ResLinear")
    
    # Lateral mixing
    if not CONFIG.get('use_lateral_mixing', True):
        parts.append("NoLateral")
    elif CONFIG.get('lateral_mixing_type', 'cyclic') != 'cyclic':
        parts.append(f"Lateral{CONFIG['lateral_mixing_type'].capitalize()}")
    
    # Check scheduler (default is disabled)
    if CONFIG.get('use_advanced_scheduler', False):
        parts.append("AdvScheduler")
    
    # Low memory mode
    if CONFIG.get('low_memory_mode', False):
        parts.append("LowMem")
    
    # Add norm position if not default
    if hasattr(args, 'norm_position') and args.norm_position != 'after':
        parts.append(f"Norm{args.norm_position.capitalize()}")
    
    # Add norm_skip_first info if not default
    if hasattr(args, 'norm_first') and args.norm_first:
        parts.append("NormFirst")
    elif hasattr(args, 'norm_skip_first') and not args.norm_skip_first:
        parts.append("NormAll")
    
    # Add knots info if not default
    phi_knots = args.phi_knots if hasattr(args, 'phi_knots') and args.phi_knots else MNIST_CONFIG['phi_knots']
    Phi_knots = args.Phi_knots if hasattr(args, 'Phi_knots') and args.Phi_knots else MNIST_CONFIG['Phi_knots']
    
    if phi_knots != MNIST_CONFIG['phi_knots']:
        parts.append(f"phi{phi_knots}")
    if Phi_knots != MNIST_CONFIG['Phi_knots']:
        parts.append(f"Phi{Phi_knots}")
    
    # Spline type (use effective resolved values)
    phi_t = getattr(args, 'phi_spline_type_effective', None)
    Phi_t = getattr(args, 'Phi_spline_type_effective', None)
    if phi_t is not None and Phi_t is not None:
        # Normalize pwl -> linear for comparison
        phi_t_norm = 'linear' if phi_t == 'pwl' else phi_t
        Phi_t_norm = 'linear' if Phi_t == 'pwl' else Phi_t
        # Only add suffix if not the default (linear)
        if phi_t_norm == Phi_t_norm and phi_t_norm != 'linear':
            parts.append(f"Spline{phi_t.capitalize()}")
        elif phi_t_norm != 'linear' or Phi_t_norm != 'linear':
            parts.append(f"SplinePhi{phi_t.capitalize()}-Phi{Phi_t.capitalize()}")
    
    # Join with dashes
    return "-" + "-".join(parts) if parts else ""


def parse_model_filename(filename):
    """Parse a model filename to extract architecture, epochs, and config flags.
    
    Returns:
        dict: Contains 'architecture', 'epochs', 'config_flags', etc. or None if parsing fails
    """
    import re
    
    # Pattern: sn_mnist_model-{arch}-{epochs}epochs{config_suffix}.pth
    # Where config_suffix might be -NoNorm-NoResidual etc.
    match = re.match(r"sn_mnist_model-([\d-]+)-(\d+)epochs(.*?)(?:_best)?\.pth", filename)
    if not match:
        return None
    
    arch_str = match.group(1)
    epochs = int(match.group(2))
    config_suffix = match.group(3)
    
    # Parse architecture
    architecture = [int(x) for x in arch_str.split("-")]
    
    # Parse config flags from suffix
    config_flags = {
        'no_norm': False,
        'no_residual': False,
        'use_advanced_scheduler': False,
        'norm_type': None,
        'norm_position': 'after',
        'norm_first': False,
        'norm_skip_first': True,
        'phi_knots': MNIST_CONFIG['phi_knots'],
        'Phi_knots': MNIST_CONFIG['Phi_knots'],
        'residual_style': 'node',
        'no_lateral': False,
        'lateral_type': 'cyclic',
        'phi_spline_type': 'linear',
        'Phi_spline_type': 'linear',
        'low_memory_mode': False,
    }
    
    if config_suffix:
        # Parse suffixes like -NoNorm-NoResidual-AdvScheduler-NormBefore-phi20-SplineCubic
        parts = config_suffix.strip('-').split('-')
        i = 0
        while i < len(parts):
            part = parts[i]
            if part == 'NoNorm':
                config_flags['no_norm'] = True
            elif part == 'NoResidual':
                config_flags['no_residual'] = True
            elif part == 'ResLinear':
                config_flags['residual_style'] = 'linear'
            elif part == 'NoLateral':
                config_flags['no_lateral'] = True
            elif part.startswith('Lateral'):
                config_flags['lateral_type'] = part[7:].lower()
            elif part == 'AdvScheduler':
                config_flags['use_advanced_scheduler'] = True
            elif part == 'LowMem':
                config_flags['low_memory_mode'] = True
            elif part.startswith('Norm') and part not in ['NoNorm', 'NormFirst', 'NormAll']:
                # Handle NormLayer, NormBefore, NormAfter
                if part == 'NormLayer':
                    config_flags['norm_type'] = 'layer'
                elif part == 'NormBefore':
                    config_flags['norm_position'] = 'before'
                elif part == 'NormAfter':
                    config_flags['norm_position'] = 'after'
            elif part == 'NormFirst':
                config_flags['norm_first'] = True
                config_flags['norm_skip_first'] = False
            elif part == 'NormAll':
                config_flags['norm_skip_first'] = False
            elif part.startswith('phi') and part[3:].isdigit():
                config_flags['phi_knots'] = int(part[3:])
            elif part.startswith('Phi') and part[3:].isdigit():
                config_flags['Phi_knots'] = int(part[3:])
            elif part.startswith('Spline'):
                # Handle SplineCubic or SplinePhiLinear-PhiCubic (two parts)
                spline_spec = part[6:]  # Remove 'Spline' prefix
                if spline_spec.startswith('Phi'):
                    # Format: SplinePhiX-PhiY - need next part too
                    phi_type = spline_spec[3:].lower()  # After 'Phi'
                    config_flags['phi_spline_type'] = phi_type
                    # Check next part for Phi type
                    if i + 1 < len(parts) and parts[i + 1].startswith('Phi'):
                        Phi_type = parts[i + 1][3:].lower()
                        config_flags['Phi_spline_type'] = Phi_type
                        i += 1
                else:
                    # Format: SplineCubic (both same type)
                    config_flags['phi_spline_type'] = spline_spec.lower()
                    config_flags['Phi_spline_type'] = spline_spec.lower()
            i += 1
    
    return {
        'architecture': architecture,
        'epochs': epochs,
        'config_flags': config_flags,
        'arch_str': arch_str,
        'config_suffix': config_suffix
    }


def discover_models(model_dir="."):
    """Discover trained MNIST models and extract their configurations.
    
    Returns:
        dict: A dictionary mapping full configurations to model info
    """
    import glob
    
    models = {}
    pattern = os.path.join(model_dir, "sn_mnist_model-*.pth")
    
    for filepath in glob.glob(pattern):
        filename = os.path.basename(filepath)
        parsed = parse_model_filename(filename)
        
        if parsed:
            # Create a unique key based on architecture and config
            config_key = f"{parsed['arch_str']}{parsed['config_suffix']}"
            
            if config_key not in models:
                models[config_key] = {
                    'architecture': parsed['architecture'],
                    'epochs': parsed['epochs'],
                    'config_flags': parsed['config_flags'],
                    'files': [],
                    'arch_str': parsed['arch_str'],
                    'config_suffix': parsed['config_suffix']
                }
            
            models[config_key]['files'].append(filename)
    
    return models


def load_checkpoint_and_extract_config(model_path, device):
    """Load checkpoint and extract both model parameters and configuration.
    
    Returns:
        tuple: (checkpoint, model_config_dict)
    """
    if not os.path.exists(model_path):
        return None, None
    
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    
    # Initialize config dict with defaults
    model_config = {
        'architecture': None,
        'phi_knots': MNIST_CONFIG['phi_knots'],
        'Phi_knots': MNIST_CONFIG['Phi_knots'],
        'norm_type': 'batch',
        'norm_position': 'after',
        'norm_skip_first': True,
        'use_residual': True,
        'residual_style': 'node',
        'use_lateral': True,
        'lateral_type': 'cyclic',
        'use_advanced_scheduler': False,
        'phi_spline_type': 'linear',
        'Phi_spline_type': 'linear',
        'low_memory_mode': False,
    }
    
    # First priority: Use saved training_args if available (most complete)
    if isinstance(checkpoint, dict) and 'training_args' in checkpoint:
        training_args = checkpoint['training_args']
        model_config.update({
            'norm_type': 'none' if training_args.get('no_norm', False) else training_args.get('norm_type', 'batch'),
            'norm_position': training_args.get('norm_position', 'after'),
            'norm_skip_first': not training_args.get('norm_first', False) if 'norm_first' in training_args else training_args.get('norm_skip_first', True),
            'use_residual': not training_args.get('no_residual', False),
            'residual_style': training_args.get('residual_style', 'node'),
            'use_lateral': not training_args.get('no_lateral', False),
            'lateral_type': training_args.get('lateral_type', 'cyclic'),
            'use_advanced_scheduler': training_args.get('use_advanced_scheduler', False),
            'phi_knots': training_args.get('phi_knots') or MNIST_CONFIG['phi_knots'],
            'Phi_knots': training_args.get('Phi_knots') or MNIST_CONFIG['Phi_knots'],
            'phi_spline_type': training_args.get('phi_spline_type', 'linear'),
            'Phi_spline_type': training_args.get('Phi_spline_type', 'linear'),
            'low_memory_mode': training_args.get('low_memory_mode', False),
        })
        print("Using complete training configuration from checkpoint")
    
    # Second priority: Use model_params if available
    if isinstance(checkpoint, dict) and 'model_params' in checkpoint:
        params = checkpoint['model_params']
        model_config.update({
            'architecture': params['architecture'],
            'phi_knots': params.get('phi_knots', model_config['phi_knots']),
            'Phi_knots': params.get('Phi_knots', model_config['Phi_knots']),
            'norm_type': params.get('norm_type', model_config['norm_type']),
            'norm_position': params.get('norm_position', model_config['norm_position']),
            'norm_skip_first': params.get('norm_skip_first', model_config['norm_skip_first']),
            'phi_spline_type': params.get('phi_spline_type', model_config['phi_spline_type']),
            'Phi_spline_type': params.get('Phi_spline_type', model_config['Phi_spline_type']),
        })
        
        # If training_args wasn't available, try to infer residual from state dict
        if 'training_args' not in checkpoint:
            state_dict = checkpoint.get('model_state_dict', {})
            has_residual = any('residual_weight' in k or 'residual_projection' in k for k in state_dict.keys())
            model_config['use_residual'] = has_residual
        
        if 'training_args' not in checkpoint:
            print("Using model parameters from checkpoint (partial configuration)")
        
    else:
        # Old format - try to infer from filename
        filename = os.path.basename(model_path)
        parsed = parse_model_filename(filename)
        
        if parsed:
            model_config['architecture'] = parsed['architecture']
            config_flags = parsed['config_flags']
            
            # Apply config flags
            if config_flags['no_norm']:
                model_config['norm_type'] = 'none'
            elif config_flags['norm_type']:
                model_config['norm_type'] = config_flags['norm_type']
                
            model_config['norm_position'] = config_flags['norm_position']
            model_config['norm_skip_first'] = config_flags['norm_skip_first']
            model_config['use_residual'] = not config_flags['no_residual']
            model_config['residual_style'] = config_flags.get('residual_style', 'node')
            model_config['use_lateral'] = not config_flags.get('no_lateral', False)
            model_config['lateral_type'] = config_flags.get('lateral_type', 'cyclic')
            model_config['use_advanced_scheduler'] = config_flags['use_advanced_scheduler']
            model_config['phi_knots'] = config_flags['phi_knots']
            model_config['Phi_knots'] = config_flags['Phi_knots']
            model_config['phi_spline_type'] = config_flags.get('phi_spline_type', 'linear')
            model_config['Phi_spline_type'] = config_flags.get('Phi_spline_type', 'linear')
            model_config['low_memory_mode'] = config_flags.get('low_memory_mode', False)
            
            print("WARNING: Old checkpoint format - configuration inferred from filename")
            print("Some settings may be incorrect. Consider retraining with latest code.")
        else:
            print("ERROR: Could not determine model configuration!")
            return None, None
    
    return checkpoint, model_config


def auto_discover_and_select_model(args, mode_name):
    """Auto-discover models and select the best one for the given mode.
    
    Returns:
        tuple: (model_path, model_config) or (None, None) if no suitable model found
    """
    print(f"Searching for trained models...")
    models = discover_models()
    
    if not models:
        print("No trained models found in current directory.")
        print(f"Please train a model first using: python sn_mnist.py --mode train")
        return None, None
    
    # If user specified architecture, filter to matching models
    if args.arch:
        arch_str = args.arch.replace(",", "-")
        matching_models = {k: v for k, v in models.items() if v['arch_str'] == arch_str}
        
        if not matching_models:
            print(f"No models found with architecture {arch_str}")
            print(f"Available architectures: {[v['arch_str'] for v in models.values()]}")
            return None, None
        
        models = matching_models
    
    # If only one model configuration exists, use it
    if len(models) == 1:
        config_key = list(models.keys())[0]
        model_info = models[config_key]
        
        print(f"\nAuto-detected model configuration:")
        print(f"  Architecture: {model_info['architecture']}")
        print(f"  Epochs: {model_info['epochs']}")
        if model_info['config_suffix']:
            print(f"  Config suffix: '{model_info['config_suffix']}'")
        else:
            print(f"  Config: default")
        print(f"  Files: {', '.join(model_info['files'])}")
        
        # Construct the filename
        arch_str = model_info['arch_str']
        epochs = model_info['epochs']
        config_suffix = model_info['config_suffix']
        
        # Try _best variant first
        best_filename = f"sn_mnist_model-{arch_str}-{epochs}epochs{config_suffix}_best.pth"
        regular_filename = f"sn_mnist_model-{arch_str}-{epochs}epochs{config_suffix}.pth"
        
        if os.path.exists(best_filename):
            model_path = best_filename
            print(f"Using best checkpoint: {best_filename}")
        elif os.path.exists(regular_filename):
            model_path = regular_filename
            print(f"Using checkpoint: {regular_filename}")
        else:
            print(f"ERROR: Could not find model file")
            return None, None
        
        # Determine device
        if args.device == "auto":
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            device = torch.device(args.device)
        
        checkpoint, model_config = load_checkpoint_and_extract_config(model_path, device)
        if model_config is None:
            return None, None
        
        # Update model_config with architecture if not found in checkpoint
        if model_config['architecture'] is None:
            model_config['architecture'] = model_info['architecture']
        
        return model_path, model_config
    
    # Multiple models found - let user choose
    print(f"\nMultiple model configurations found ({len(models)} total):")
    model_list = list(models.items())
    for i, (key, info) in enumerate(model_list, 1):
        print(f"  [{i}] Architecture: {info['architecture']}, Epochs: {info['epochs']}")
        if info['config_suffix']:
            print(f"      Config: {info['config_suffix']}")
    
    if args.model_index is not None:
        idx = args.model_index - 1
        if 0 <= idx < len(model_list):
            config_key, model_info = model_list[idx]
        else:
            print(f"ERROR: Invalid model index {args.model_index}. Must be 1-{len(model_list)}")
            return None, None
    else:
        print(f"\nUse --model-index N to select a specific model, or --arch to filter by architecture.")
        return None, None
    
    # Construct model path
    arch_str = model_info['arch_str']
    epochs = model_info['epochs']
    config_suffix = model_info['config_suffix']
    
    best_filename = f"sn_mnist_model-{arch_str}-{epochs}epochs{config_suffix}_best.pth"
    regular_filename = f"sn_mnist_model-{arch_str}-{epochs}epochs{config_suffix}.pth"
    
    if os.path.exists(best_filename):
        model_path = best_filename
        print(f"Using best checkpoint: {best_filename}")
    elif os.path.exists(regular_filename):
        model_path = regular_filename
        print(f"Using checkpoint: {regular_filename}")
    else:
        print(f"ERROR: Could not find model file")
        return None, None
    
    # Determine device
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    
    checkpoint, model_config = load_checkpoint_and_extract_config(model_path, device)
    if model_config is None:
        return None, None
    
    # Update model_config with architecture if not found in checkpoint
    if model_config['architecture'] is None:
        model_config['architecture'] = model_info['architecture']
    
    return model_path, model_config


def count_parameters_detailed(model):
    """Count parameters by category for detailed reporting."""
    total_spline_knots = 0
    lambda_params = 0
    eta_params = 0
    residual_params = 0
    codomain_params = 0
    norm_params = 0
    lateral_params = 0
    output_params = 0
    
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
        elif 'residual_weight' in name or 'residual_projection' in name or 'residual_pooling' in name or 'residual_broadcast' in name:
            # Residual connection parameters
            residual_params += param.numel()
        elif 'phi_codomain_params' in name:
            # Codomain parameters (cc, cr)
            codomain_params += param.numel()
        elif 'lateral' in name:
            # Lateral mixing parameters
            lateral_params += param.numel()
        elif 'output_scale' in name or 'output_bias' in name:
            # Output layer parameters
            output_params += param.numel()
        elif 'norm' in name.lower() or 'weight' in name or 'bias' in name:
            # Normalization parameters (catch remaining)
            norm_params += param.numel()
    
    # Core parameters (excluding residual and output params)
    core_params = total_spline_knots + lambda_params + eta_params
    
    # Total parameters
    total_params = core_params + residual_params + codomain_params + norm_params + lateral_params + output_params
    
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
        'lateral': lateral_params,
        'output': output_params,
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
                 norm_type="none", norm_position="after", norm_skip_first=True,
                 phi_spline_type="linear", Phi_spline_type="linear",
                 initialize_domains=True):
        super().__init__()
        self.sprecher_net = SprecherMultiLayerNetwork(
            input_dim=784,  # 28x28 flattened
            architecture=architecture,
            final_dim=10,   # 10 digit classes
            phi_knots=phi_knots,
            Phi_knots=Phi_knots,
            norm_type=norm_type,
            norm_position=norm_position,
            norm_skip_first=norm_skip_first,
            phi_spline_type=phi_spline_type,
            Phi_spline_type=Phi_spline_type,
            initialize_domains=initialize_domains,
        )
    
    def forward(self, x):
        # Flatten the input if needed
        if x.dim() > 2:
            x = x.view(x.size(0), -1)
        return self.sprecher_net(x)
    
    def update_all_domains(self, allow_resampling=True, force_resample=False):
        """Forward domain update call to the underlying Sprecher network."""
        self.sprecher_net.update_all_domains(allow_resampling, force_resample)
    
    def get_domain_violation_stats(self):
        """Forward domain violation stats call."""
        return self.sprecher_net.get_domain_violation_stats()
    
    def reset_domain_violation_stats(self):
        """Forward domain violation reset call."""
        self.sprecher_net.reset_domain_violation_stats()
    
    def print_domain_violation_report(self):
        """Forward domain violation report call."""
        self.sprecher_net.print_domain_violation_report()


def safe_update_all_domains(model):
    """Safely call update_all_domains, catching NaN-related interval errors.

    The Sprecher domain update can occasionally hit numerical edge cases (e.g., NaNs)
    when computing theoretical bounds. In those cases we skip the update rather than
    crashing evaluation/training, matching the behavior used in sn_fashion_mnist.py.
    """
    try:
        model.update_all_domains()
    except AssertionError as e:
        # Only suppress the specific NaN/invalid-interval assertion; re-raise everything else
        msg = str(e)
        if ("Invalid interval" in msg) and ("nan" in msg.lower()):
            pass
        else:
            raise



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
                safe_update_all_domains(model)
            
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



def test_model(model, test_loader, device, use_batch_stats=True):
    """Evaluate model on the test set.

    NOTE on BatchNorm:
    Sprecher Networks may call update_all_domains() during evaluation. That routine
    computes normalization-aware theoretical bounds and uses the model.training flag
    to decide whether BatchNorm is in "training" (batch stats) or "eval" (running stats)
    mode. Therefore, if we choose to evaluate with BatchNorm batch statistics, we must
    also keep the *whole model* in train(True) so that update_all_domains() stays
    consistent with the BN behavior.
    """
    correct = 0
    total = 0

    # Reset domain violation stats if tracking
    if CONFIG.get('track_domain_violations', False):
        model.reset_domain_violation_stats()

    was_training = model.training

    # If requested, evaluate with BN batch statistics (training behavior) but without
    # mutating BN running buffers.
    if use_batch_stats and has_batchnorm(model):
        model.train(True)
        with torch.no_grad():
            with use_batch_stats_without_updating_bn(model):
                for images, labels in tqdm(test_loader, desc="Testing"):
                    images, labels = images.to(device), labels.to(device)

                    # Flatten and convert from normalized [-1, 1] back to [0, 1]
                    images = images.view(images.size(0), -1)
                    images = (images + 1) / 2

                    # Update domains before forward pass
                    if CONFIG.get('use_theoretical_domains', True):
                        safe_update_all_domains(model)

                    outputs = model(images)
                    _, predicted = torch.max(outputs.data, 1)
                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()
    else:
        # Standard evaluation with BN running statistics
        with evaluation_mode(model):
            for images, labels in tqdm(test_loader, desc="Testing"):
                images, labels = images.to(device), labels.to(device)

                images = images.view(images.size(0), -1)
                images = (images + 1) / 2

                if CONFIG.get('use_theoretical_domains', True):
                    safe_update_all_domains(model)

                outputs = model(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

    # Restore original training/eval mode
    model.train(was_training)

    if CONFIG.get('track_domain_violations', False):
        print("\nTest set domain violations:")
        model.print_domain_violation_report()

    accuracy = 100 * correct / total
    return accuracy

def print_configuration(args, architecture, phi_knots, Phi_knots, epochs, batch_size, lr, 
                        effective_norm_type, norm_position, final_norm_skip_first, use_residual,
                        residual_style, use_lateral, lateral_type, phi_spline_type, Phi_spline_type):
    """Print configuration summary."""
    device_str = args.device if args.device != 'auto' else ('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device_str}")
    print(f"Mode: {args.mode}")
    print(f"Architecture: 784 -> {architecture} -> 10")
    print(f"phi knots: {phi_knots}, Phi knots: {Phi_knots}")
    print(f"Spline types: phi={phi_spline_type}, Phi={Phi_spline_type}")
    print(f"Epochs: {epochs}")
    print(f"Batch size: {batch_size}")
    print(f"Learning rate: {lr}")
    print(f"Seed: {args.seed}")
    print(f"Theoretical domains: {CONFIG.get('use_theoretical_domains', True)}")
    print(f"Domain safety margin: {CONFIG.get('domain_safety_margin', 0.0)}")
    
    if use_residual:
        print(f"Residual connections: enabled (style: {residual_style})")
    else:
        print("Residual connections: disabled")
    
    if use_lateral:
        print(f"Lateral mixing: enabled (type: {lateral_type})")
    else:
        print("Lateral mixing: disabled")
    
    if effective_norm_type == 'none':
        print("Normalization: disabled")
    else:
        print(f"Normalization: {effective_norm_type} (position: {norm_position}, skip_first: {final_norm_skip_first})")
    
    print(f"Scheduler: {'PlateauAwareCosineAnnealingLR' if CONFIG.get('use_advanced_scheduler', False) else 'Adam (fixed LR)'}")
    
    if CONFIG.get('low_memory_mode', False):
        print("Low memory mode: enabled")
    
    print()


def train_mnist(args):
    _require_torchvision()
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
    
    # Resolve spline types
    # Priority: specific args > --spline_type > default 'linear'
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
    
    # Normalize 'pwl' to 'linear' for internal use
    if phi_spline_type == 'pwl':
        phi_spline_type = 'linear'
    if Phi_spline_type == 'pwl':
        Phi_spline_type = 'linear'
    
    # Store effective values for config suffix
    args.phi_spline_type_effective = phi_spline_type
    args.Phi_spline_type_effective = Phi_spline_type
    
    # Determine device
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    
    # Set random seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    CONFIG['seed'] = args.seed
    
    # Handle configuration overrides
    if args.debug_domains:
        CONFIG['debug_domains'] = True
    if args.track_violations:
        CONFIG['track_domain_violations'] = True
        print("Domain violation tracking enabled.")
    if args.no_residual:
        CONFIG['use_residual_weights'] = False
    else:
        if args.residual_style:
            style = args.residual_style.lower()
            if style in ('standard', 'matrix'):
                style = 'linear'
            CONFIG['residual_style'] = style
    
    if args.no_norm:
        CONFIG['use_normalization'] = False
    if args.use_advanced_scheduler:
        CONFIG['use_advanced_scheduler'] = True
    
    # Lateral mixing
    if args.no_lateral:
        CONFIG['use_lateral_mixing'] = False
    if args.lateral_type:
        CONFIG['lateral_mixing_type'] = args.lateral_type
    
    # Memory optimization
    if args.low_memory_mode:
        CONFIG['low_memory_mode'] = True
        print("Low memory mode: ENABLED")
    if args.memory_debug:
        CONFIG['memory_debug'] = True
        print("Memory debugging: ENABLED")
    
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
    
    # Get residual and lateral settings for display
    use_residual = CONFIG.get('use_residual_weights', True)
    residual_style = CONFIG.get('residual_style', 'node')
    use_lateral = CONFIG.get('use_lateral_mixing', True)
    lateral_type = CONFIG.get('lateral_mixing_type', 'cyclic')
    
    # Print configuration
    print_configuration(args, architecture, phi_knots, Phi_knots, epochs, batch_size, lr,
                       effective_norm_type, args.norm_position, final_norm_skip_first,
                       use_residual, residual_style, use_lateral, lateral_type,
                       phi_spline_type, Phi_spline_type)
    
    # Initialize model
    model = MNISTSprecherNet(
        architecture=architecture,
        phi_knots=phi_knots,
        Phi_knots=Phi_knots,
        norm_type=effective_norm_type,
        norm_position=args.norm_position,
        norm_skip_first=final_norm_skip_first,
        phi_spline_type=phi_spline_type,
        Phi_spline_type=Phi_spline_type,
        initialize_domains=True,
    ).to(device)
    
    # Count and display parameters
    param_counts = count_parameters_detailed(model)
    print(f"Total number of trainable parameters: {param_counts['total']:,}")
    print(f"  - Lambda weight VECTORS: {param_counts['lambda']:,} (TRUE SPRECHER!)")
    print(f"  - Eta shift parameters: {param_counts['eta']:,}")
    print(f"  - Spline parameters: {param_counts['spline_knots']:,}")
    if use_residual and param_counts['residual'] > 0:
        print(f"  - Residual connection weights: {param_counts['residual']:,}")
    if use_lateral and param_counts['lateral'] > 0:
        print(f"  - Lateral mixing parameters: {param_counts['lateral']:,}")
    if param_counts['norm'] > 0:
        print(f"  - Normalization parameters: {param_counts['norm']:,}")
    if CONFIG.get('train_phi_codomain', True) and param_counts['codomain'] > 0:
        print(f"  - Phi codomain parameters (cc, cr per block): {param_counts['codomain']:,}")
    if param_counts['output'] > 0:
        print(f"  - Output parameters: {param_counts['output']:,}")
    print()
    
    # Check if model exists and handle retrain
    config_suffix = get_config_suffix(args, CONFIG)
    arch_str = "-".join(map(str, architecture))
    model_filename = f"sn_mnist_model-{arch_str}-{epochs}epochs{config_suffix}.pth"
    model_path = os.path.join(os.path.dirname(model_file), model_filename)
    
    if os.path.exists(model_path) and args.retrain:
        os.remove(model_path)
        print(f"Deleted existing model file: {model_path}")
        # Also remove _best variant if it exists
        best_path = model_path.replace('.pth', '_best.pth')
        if os.path.exists(best_path):
            os.remove(best_path)
            print(f"Deleted existing best model file: {best_path}")
        if os.path.exists(data_dir):
            shutil.rmtree(data_dir)
            print(f"Deleted existing data directory: {data_dir}")
    elif os.path.exists(model_path) and not args.retrain:
        model.load_state_dict(torch.load(model_path, map_location=device, weights_only=False), strict=False)
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
    best_plotting_snapshot = None
    best_checkpoint = None
    
    # Get a sample batch for snapshot
    sample_images, sample_labels = next(iter(train_loader))
    sample_images = sample_images.to(device)
    sample_labels = sample_labels.to(device)
    sample_images = sample_images.view(sample_images.shape[0], -1)
    sample_images = (sample_images + 1) / 2
    
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
            
            # Create complete plotting snapshot
            best_plotting_snapshot = {
                'model': copy.deepcopy(model),
                'x_train': sample_images.clone(),
                'y_train': sample_labels.clone(),
                'output': sample_output.detach().clone(),
                'loss': sample_loss,
                'accuracy': train_acc,
                'epoch': epoch,
                'device': device
            }
            
            # Save complete BatchNorm statistics if normalization is used
            bn_statistics = {}
            has_bn = effective_norm_type in ['batch', 'layer']
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
                'model_state_dict': copy.deepcopy(model.state_dict()),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': sample_loss,
                'accuracy': train_acc,
                'has_batchnorm': has_bn,
                'bn_statistics': bn_statistics,
                'training_mode': model.training,
                'x_train': sample_images.cpu().clone(),
                'y_train': sample_labels.cpu().clone(),
                'output': sample_output.detach().cpu().clone(),
                # Save model creation parameters for exact reconstruction
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
                # SAVE COMPLETE TRAINING ARGS FOR ROBUSTNESS
                'training_args': {
                    'no_residual': args.no_residual,
                    'residual_style': CONFIG.get('residual_style', 'node'),
                    'no_lateral': args.no_lateral if hasattr(args, 'no_lateral') else False,
                    'lateral_type': CONFIG.get('lateral_mixing_type', 'cyclic'),
                    'no_norm': args.no_norm,
                    'norm_type': args.norm_type,
                    'norm_position': args.norm_position,
                    'norm_first': args.norm_first if hasattr(args, 'norm_first') else False,
                    'norm_skip_first': args.norm_skip_first if hasattr(args, 'norm_skip_first') else True,
                    'use_advanced_scheduler': args.use_advanced_scheduler,
                    'phi_knots': args.phi_knots,
                    'Phi_knots': args.Phi_knots,
                    'phi_spline_type': phi_spline_type,
                    'Phi_spline_type': Phi_spline_type,
                    'low_memory_mode': args.low_memory_mode if hasattr(args, 'low_memory_mode') else False,
                    'epochs': epochs,
                    'seed': args.seed,
                    'batch_size': batch_size,
                    'lr': lr
                }
            }
            
            # Save to disk immediately
            best_model_path = model_path.replace('.pth', '_best.pth')
            torch.save(best_checkpoint, best_model_path)
    
    print(f"\nTraining complete! Best accuracy: {best_accuracy:.2f}%")
    print(f"Model saved to: {model_path}")
    
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
            print("Perfect consistency achieved! The snapshot is completely isolated.")
    
    # Plot training curves if requested
    if args.save_plots or not args.no_show:
        try:
            os.makedirs("plots", exist_ok=True)
            
            # Plot loss curve
            loss_filename = f"mnist_loss-{arch_str}-{epochs}epochs{config_suffix}.png"
            loss_save_path = os.path.join("plots", loss_filename) if args.save_plots else None
            plot_loss_curve(losses, loss_save_path)
            
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
            print("\n" + "="*60)
            print("WARNING: A plotting error occurred.")
            print(f"Error type: {type(e).__name__}")
            print("Switching to the reliable 'Agg' backend to proceed.")
            print("="*60 + "\n")
            
            matplotlib.use('Agg')
            import matplotlib.pyplot as plt
            
            if not args.save_plots:
                print("Plots could not be shown interactively. Enabling file saving automatically.")
                args.save_plots = True
            
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
    _require_torchvision()
    """Testing mode."""
    # Auto-discover and select model
    model_path, model_config = auto_discover_and_select_model(args, "test")
    
    if model_path is None or model_config is None:
        return
    
    # Extract configuration
    architecture = model_config['architecture']
    phi_knots = model_config.get('phi_knots', MNIST_CONFIG['phi_knots'])
    Phi_knots = model_config.get('Phi_knots', MNIST_CONFIG['Phi_knots'])
    norm_type = model_config.get('norm_type', 'batch')
    norm_position = model_config.get('norm_position', 'after')
    norm_skip_first = model_config.get('norm_skip_first', True)
    use_residual = model_config.get('use_residual', True)
    residual_style = model_config.get('residual_style', 'node')
    use_lateral = model_config.get('use_lateral', True)
    lateral_type = model_config.get('lateral_type', 'cyclic')
    phi_spline_type = model_config.get('phi_spline_type', 'linear')
    Phi_spline_type = model_config.get('Phi_spline_type', 'linear')
    batch_size = args.batch_size if args.batch_size else MNIST_CONFIG['batch_size']
    data_dir = args.data_dir if args.data_dir else MNIST_CONFIG['data_directory']
    
    # Update CONFIG based on discovered settings
    CONFIG['use_residual_weights'] = use_residual
    CONFIG['residual_style'] = residual_style
    CONFIG['use_lateral_mixing'] = use_lateral
    CONFIG['lateral_mixing_type'] = lateral_type
    if args.debug_domains:
        CONFIG['debug_domains'] = True
    if args.track_violations:
        CONFIG['track_domain_violations'] = True
    
    # Determine device
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    
    print(f"\nLoading model from: {model_path}")
    print(f"Using device: {device}")
    print(f"Architecture: 784 -> {architecture} -> 10")
    print(f"Spline types: phi={phi_spline_type}, Phi={Phi_spline_type}")
    print(f"Normalization: {norm_type} (position: {norm_position}, skip_first: {norm_skip_first})")
    print(f"Residual connections: {'enabled (' + residual_style + ')' if use_residual else 'disabled'}")
    print(f"Lateral mixing: {'enabled (' + lateral_type + ')' if use_lateral else 'disabled'}")
    print()
    
    # Load checkpoint
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    
    # Initialize model with the correct configuration
    model = MNISTSprecherNet(
        architecture=architecture,
        phi_knots=phi_knots,
        Phi_knots=Phi_knots,
        norm_type=norm_type,
        norm_position=norm_position,
        norm_skip_first=norm_skip_first,
        phi_spline_type=phi_spline_type,
        Phi_spline_type=Phi_spline_type,
        initialize_domains=False,  # Don't initialize domains until after loading
    ).to(device)
    
    # Load the model state
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
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
        model.load_state_dict(checkpoint, strict=False)
    
    # Update domains after loading
    if CONFIG.get('use_theoretical_domains', True):
        safe_update_all_domains(model)
    
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
    # Auto-discover and select model
    model_path, model_config = auto_discover_and_select_model(args, "infer")
    
    if model_path is None or model_config is None:
        return
    
    # Extract configuration
    architecture = model_config['architecture']
    phi_knots = model_config.get('phi_knots', MNIST_CONFIG['phi_knots'])
    Phi_knots = model_config.get('Phi_knots', MNIST_CONFIG['Phi_knots'])
    norm_type = model_config.get('norm_type', 'batch')
    norm_position = model_config.get('norm_position', 'after')
    norm_skip_first = model_config.get('norm_skip_first', True)
    use_residual = model_config.get('use_residual', True)
    residual_style = model_config.get('residual_style', 'node')
    use_lateral = model_config.get('use_lateral', True)
    lateral_type = model_config.get('lateral_type', 'cyclic')
    phi_spline_type = model_config.get('phi_spline_type', 'linear')
    Phi_spline_type = model_config.get('Phi_spline_type', 'linear')
    
    # Update CONFIG based on discovered settings
    CONFIG['use_residual_weights'] = use_residual
    CONFIG['residual_style'] = residual_style
    CONFIG['use_lateral_mixing'] = use_lateral
    CONFIG['lateral_mixing_type'] = lateral_type
    
    # Determine device
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    
    print(f"\nLoading model from: {model_path}")
    print(f"Using device: {device}")
    print(f"Image file: {args.image}")
    print()
    
    # Load checkpoint
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    
    # Initialize model with the correct configuration
    model = MNISTSprecherNet(
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
    
    # Load the model state
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
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
        model.load_state_dict(checkpoint, strict=False)
        checkpoint = None
    
    # Update domains
    if CONFIG.get('use_theoretical_domains', True):
        safe_update_all_domains(model)
    
    # Process image
    if not os.path.exists(args.image):
        print(f"Error: '{args.image}' not found. Please provide an image file.")
        return
    
    image_data = process_image(args.image)
    image_tensor = torch.from_numpy(image_data).float().unsqueeze(0).to(device)
    
    # For BatchNorm in training mode, we need a diverse batch
    if norm_type in ['batch', 'layer']:
        # Try to use saved batch from checkpoint for diversity
        if isinstance(checkpoint, dict) and 'x_train' in checkpoint:
            saved_batch = checkpoint['x_train'].to(device)
            batch_tensor = saved_batch.clone()
            batch_tensor[0] = image_tensor.squeeze(0)
            image_tensor = batch_tensor
            print(f"Using saved batch of {batch_tensor.shape[0]} samples for BatchNorm diversity")
        else:
            print("No saved batch found, creating diverse batch with noise")
            batch_size = 16
            noise_levels = torch.linspace(0.0, 0.1, batch_size).to(device)
            batch_list = []
            
            for i in range(batch_size):
                if i == 0:
                    batch_list.append(image_tensor.squeeze(0))
                else:
                    noise = torch.randn_like(image_tensor.squeeze(0)) * noise_levels[i]
                    noisy_image = image_tensor.squeeze(0) + noise
                    noisy_image = torch.clamp(noisy_image, 0, 1)
                    batch_list.append(noisy_image)
            
            image_tensor = torch.stack(batch_list)
    
    # Inference with proper eval mode handling
    model.eval()
    with torch.no_grad():
        if has_batchnorm(model):
            with use_batch_stats_without_updating_bn(model):
                output = model(image_tensor)
        else:
            output = model(image_tensor)
        
        # Take only the first result (our target image)
        if norm_type in ['batch', 'layer']:
            output = output[0:1]
            
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
    # Auto-discover and select model
    model_path, model_config = auto_discover_and_select_model(args, "plot")
    
    if model_path is None or model_config is None:
        return
    
    # Extract configuration
    architecture = model_config['architecture']
    phi_knots = model_config.get('phi_knots', MNIST_CONFIG['phi_knots'])
    Phi_knots = model_config.get('Phi_knots', MNIST_CONFIG['Phi_knots'])
    norm_type = model_config.get('norm_type', 'batch')
    norm_position = model_config.get('norm_position', 'after')
    norm_skip_first = model_config.get('norm_skip_first', True)
    use_residual = model_config.get('use_residual', True)
    residual_style = model_config.get('residual_style', 'node')
    use_lateral = model_config.get('use_lateral', True)
    lateral_type = model_config.get('lateral_type', 'cyclic')
    phi_spline_type = model_config.get('phi_spline_type', 'linear')
    Phi_spline_type = model_config.get('Phi_spline_type', 'linear')
    
    # Update CONFIG based on discovered settings
    CONFIG['use_residual_weights'] = use_residual
    CONFIG['residual_style'] = residual_style
    CONFIG['use_lateral_mixing'] = use_lateral
    CONFIG['lateral_mixing_type'] = lateral_type
    
    # Determine device
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    
    print(f"\nLoading model from: {model_path}")
    print(f"Using device: {device}")
    print(f"Architecture: 784 -> {architecture} -> 10")
    print(f"phi knots: {phi_knots}, Phi knots: {Phi_knots}")
    print()
    
    # Load checkpoint
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    
    # Initialize model with the correct configuration
    model = MNISTSprecherNet(
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
    
    # Load the model state
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
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
        model.load_state_dict(checkpoint, strict=False)
    
    # Update domains
    if CONFIG.get('use_theoretical_domains', True):
        print("Updating spline domains based on loaded parameters...")
        safe_update_all_domains(model)
    
    # Print domain information
    print("\nSpline domains after theoretical update:")
    for idx, layer in enumerate(model.sprecher_net.layers):
        print(f"Block {idx+1}:")
        print(f"  phi domain: [{layer.phi.in_min:.3f}, {layer.phi.in_max:.3f}]")
        print(f"  Phi domain: [{layer.Phi.in_min:.3f}, {layer.Phi.in_max:.3f}]")
    
    # Create save path
    if args.save_plots:
        os.makedirs("plots", exist_ok=True)
        filename = os.path.basename(model_path)
        if '_best.pth' in filename:
            base_filename = filename.replace('_best.pth', '')
        else:
            base_filename = filename.replace('.pth', '')
        parts = base_filename.split('-', 2)
        if len(parts) >= 3:
            arch_part = parts[1]
            config_part = parts[2].replace('epochs', '')
            if config_part.endswith(str(model_config.get('epochs', ''))):
                config_part = config_part[:-len(str(model_config.get('epochs', '')))]
        else:
            arch_part = "-".join(map(str, architecture))
            config_part = ""
        
        save_path = f"plots/mnist_splines-{arch_part}{config_part}.png"
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
        print("\n" + "="*60)
        print("WARNING: A plotting error occurred.")
        print(f"Error type: {type(e).__name__}")
        print("Switching to the reliable 'Agg' backend to proceed.")
        print("="*60 + "\n")
        
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        
        if not args.save_plots:
            print("Plots could not be shown interactively. Enabling file saving automatically.")
            args.save_plots = True
            
        os.makedirs("plots", exist_ok=True)
        if not save_path:
            filename = os.path.basename(model_path)
            if '_best.pth' in filename:
                base_filename = filename.replace('_best.pth', '')
            else:
                base_filename = filename.replace('.pth', '')
            parts = base_filename.split('-', 2)
            if len(parts) >= 3:
                arch_part = parts[1]
                config_part = parts[2].replace('epochs', '')
                if config_part.endswith(str(model_config.get('epochs', ''))):
                    config_part = config_part[:-len(str(model_config.get('epochs', '')))]
            else:
                arch_part = "-".join(map(str, architecture))
                config_part = ""
            save_path = f"plots/mnist_splines-{arch_part}{config_part}.png"
        
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




# ============================================================
# Nintendo DS export (fixed-point binary weights)
# ============================================================

def _q16_16_from_float(x: float) -> int:
    """Convert Python float to signed Q16.16 int32 with saturation."""
    v = int(round(float(x) * (1 << 16)))
    if v > 0x7FFFFFFF:
        return 0x7FFFFFFF
    if v < -0x80000000:
        return -0x80000000
    return v


def _tensor_to_q16_16_np(t: torch.Tensor) -> np.ndarray:
    """Convert a torch tensor to a little-endian int32 numpy array in Q16.16."""
    arr = t.detach().cpu().numpy().astype(np.float64)
    scaled = np.round(arr * (1 << 16))
    scaled = np.clip(scaled, -0x80000000, 0x7FFFFFFF).astype(np.int32)
    return scaled


def _write_u32(f, v: int):
    f.write(struct.pack('<I', int(v) & 0xFFFFFFFF))


def _write_i32(f, v: int):
    f.write(struct.pack('<i', int(v)))


def _write_i32_array(f, arr_i32: np.ndarray):
    """Write a numpy int32 array as little-endian bytes."""
    if arr_i32.dtype != np.int32:
        arr_i32 = arr_i32.astype(np.int32)
    # Ensure little-endian on disk
    if arr_i32.dtype.byteorder not in ('<', '='):
        arr_i32 = arr_i32.byteswap().newbyteorder('<')
    f.write(arr_i32.tobytes(order='C'))


def _get_effective_linear_spline_knot_values(spline: nn.Module) -> torch.Tensor:
    """
    Return the effective y-values at the spline knots for export.

    This handles the case where the spline uses a trainable codomain transform
    (train_codomain=True). Evaluating the spline at its own knot locations yields
    exactly the effective knot values used by the forward pass.
    """
    with torch.no_grad():
        knots = spline.knots  # 1D tensor
        y = spline(knots)
    return y


def export_nds(args):
    """
    Export the most recent (or selected) MNIST SprecherNet checkpoint to a
    Nintendo DS-friendly fixed-point binary format.

    Usage:
        python sn_mnist.py --mode export_nds [--checkpoint PATH] [--export_out sn_weights.bin]
    """
    device = torch.device("cpu")

    # ------------------------------------------------------------------
    # Select checkpoint
    # ------------------------------------------------------------------
    model_path = None
    model_config = None
    checkpoint = None

    if args.checkpoint:
        model_path = args.checkpoint
        checkpoint, model_config = load_checkpoint_and_extract_config(model_path, device)
        if checkpoint is None:
            raise FileNotFoundError(f"Checkpoint not found or could not be loaded: {model_path}")
    else:
        # Default to the most recently modified sn_mnist_model-*.pth in CWD.
        # If the user passed --arch or --model-index, defer to the interactive selector.
        if args.arch is not None or args.model_index is not None:
            model_path, model_config = auto_discover_and_select_model(args)
            if model_path is None:
                print("No model selected/found. Use --arch or --model-index, or pass --checkpoint.")
                return
            checkpoint, model_config = load_checkpoint_and_extract_config(model_path, device)
        else:
            import glob
            candidates = glob.glob("sn_mnist_model-*.pth")
            if not candidates:
                print("No checkpoints found (expected files matching sn_mnist_model-*.pth).")
                print("Train first with: python sn_mnist.py --mode train --arch 100,100 --epochs 5")
                return
            model_path = max(candidates, key=os.path.getmtime)
            checkpoint, model_config = load_checkpoint_and_extract_config(model_path, device)

    assert model_config is not None

    # ------------------------------------------------------------------
    # Configure global CONFIG to match the checkpoint
    # ------------------------------------------------------------------
    # Feature flags (these live in sn_core.config.CONFIG)
    CONFIG['use_residual_weights'] = bool(model_config.get('use_residual', True))
    CONFIG['residual_style'] = model_config.get('residual_style', 'node')
    CONFIG['use_lateral_mixing'] = bool(model_config.get('use_lateral', True))
    CONFIG['lateral_mixing_type'] = model_config.get('lateral_type', 'cyclic')

    # Spline kinds (MNIST export currently supports piecewise-linear splines)
    phi_spline_type = model_config.get('phi_spline_type', 'linear')
    Phi_spline_type = model_config.get('Phi_spline_type', 'linear')
    if phi_spline_type == 'pwl':
        phi_spline_type = 'linear'
    if Phi_spline_type == 'pwl':
        Phi_spline_type = 'linear'
    if phi_spline_type != 'linear' or Phi_spline_type != 'linear':
        raise ValueError(
            "export_nds currently supports only piecewise-linear ('linear' / 'pwl') splines. "
            f"Checkpoint uses phi={phi_spline_type}, Phi={Phi_spline_type}."
        )

    architecture = model_config.get('architecture')
    if architecture is None:
        raise ValueError("Could not determine architecture from checkpoint.")

    # ------------------------------------------------------------------
    # Build & load model
    # ------------------------------------------------------------------
    model = MNISTSprecherNet(
        architecture=architecture,
        phi_knots=model_config.get('phi_knots', MNIST_CONFIG['phi_knots']),
        Phi_knots=model_config.get('Phi_knots', MNIST_CONFIG['Phi_knots']),
        norm_type=model_config.get('norm_type', 'batch'),
        norm_position=model_config.get('norm_position', 'after'),
        norm_skip_first=model_config.get('norm_skip_first', True),
        phi_spline_type=phi_spline_type,
        Phi_spline_type=Phi_spline_type,
        initialize_domains=False,
    ).to(device)

    # Load weights (supports both dict-style checkpoints and raw state_dicts)
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)

    # Restore BN stats if present
    if isinstance(checkpoint, dict) and 'bn_statistics' in checkpoint:
        try:
            for name, stats in checkpoint['bn_statistics'].items():
                module = dict(model.named_modules()).get(name, None)
                if module is not None and hasattr(module, 'running_mean'):
                    module.running_mean = stats['running_mean'].to(device)
                    module.running_var = stats['running_var'].to(device)
        except Exception as e:
            print(f"Warning: Could not restore bn_statistics: {e}")

    model.eval()

    # Ensure spline domains/knots are up-to-date
    safe_update_all_domains(model)

    # ------------------------------------------------------------------
    # Validate DS-supported residual configuration
    # ------------------------------------------------------------------
    if CONFIG.get('use_residual_weights', True):
        if CONFIG.get('residual_style', 'node') == 'linear':
            # DS export could support linear residual projection but it breaks O(N) scaling.
            # If any projection exists, refuse to export with a clear message.
            for layer in model.sprecher_net.layers:
                if getattr(layer, 'residual_projection', None) is not None:
                    raise ValueError(
                        "This checkpoint uses residual_style='linear' with a projection matrix "
                        "(O(d_in*d_out) parameters). This is not DS-friendly and is not supported "
                        "by export_nds. Train with residual_style='node' instead."
                    )

    # ------------------------------------------------------------------
    # Write binary file (SNDS v3)
    # ------------------------------------------------------------------
    out_path = getattr(args, 'export_out', None) or "sprecher_ds/sn_weights.bin"

    # Ensure output directory exists
    out_dir = os.path.dirname(out_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    # Global header info
    num_blocks = len(model.sprecher_net.layers)
    input_dim = int(model.sprecher_net.layers[0].d_in)
    output_dim = int(model.sprecher_net.layers[-1].d_out)
    max_dim = max([max(int(l.d_in), int(l.d_out)) for l in model.sprecher_net.layers])

    # Compute param counts
    try:
        param_counts = count_parameters_detailed(model)
        total_params = int(param_counts.get('total', sum(p.numel() for p in model.parameters())))
    except Exception:
        total_params = int(sum(p.numel() for p in model.parameters()))
        param_counts = {'total': total_params}

    # Equivalent MLP param count for the same (784 -> arch -> 10) dims
    dims = [784] + list(architecture) + [10]
    mlp_params = 0
    for a, b in zip(dims[:-1], dims[1:]):
        mlp_params += a * b + b

    # Build global flags
    FLAG_USE_LATERAL = 1 << 0
    FLAG_LATERAL_BIDIR = 1 << 1
    FLAG_USE_RESIDUAL = 1 << 2
    FLAG_RESIDUAL_LINEAR = 1 << 3
    FLAG_NORM_NONE = 0 << 4
    FLAG_NORM_BATCH = 1 << 4
    FLAG_NORM_LAYER = 2 << 4
    FLAG_NORM_MASK = 3 << 4
    FLAG_NORM_BEFORE = 1 << 6
    FLAG_NORM_SKIP_FIRST = 1 << 7

    flags = 0
    if model_config.get('use_lateral', True):
        flags |= FLAG_USE_LATERAL
        if model_config.get('lateral_type', 'cyclic') == 'bidirectional':
            flags |= FLAG_LATERAL_BIDIR
    if model_config.get('use_residual', True):
        flags |= FLAG_USE_RESIDUAL
        if model_config.get('residual_style', 'node') == 'linear':
            flags |= FLAG_RESIDUAL_LINEAR

    norm_type = model_config.get('norm_type', 'batch')
    if norm_type == 'none':
        flags |= FLAG_NORM_NONE
    elif norm_type == 'layer':
        flags |= FLAG_NORM_LAYER
    else:
        flags |= FLAG_NORM_BATCH

    if model_config.get('norm_position', 'after') == 'before':
        flags |= FLAG_NORM_BEFORE
    if model_config.get('norm_skip_first', True):
        flags |= FLAG_NORM_SKIP_FIRST

    q_factor = float(CONFIG.get('q_values_factor', MNIST_CONFIG.get('q_values_factor', 1.0)))
    q_factor_fix = _q16_16_from_float(q_factor)

    # Write file
    with open(out_path, "wb") as f:
        # Header
        f.write(b"SNDS")
        _write_u32(f, 3)              # version
        _write_u32(f, 16)             # fix_shift
        _write_u32(f, flags)
        _write_u32(f, num_blocks)
        _write_u32(f, input_dim)
        _write_u32(f, output_dim)
        _write_u32(f, total_params)
        _write_u32(f, max_dim)
        _write_i32(f, q_factor_fix)

        # Blocks
        for i, layer in enumerate(model.sprecher_net.layers):
            d_in = int(layer.d_in)
            d_out = int(layer.d_out)

            phi = layer.phi
            Phi = layer.Phi

            phi_knots = _tensor_to_q16_16_np(phi.knots)
            phi_coeffs = _tensor_to_q16_16_np(phi.get_coeffs())

            Phi_knots = _tensor_to_q16_16_np(Phi.knots)
            # Effective knot values (handles train_codomain)
            Phi_coeffs = _tensor_to_q16_16_np(_get_effective_linear_spline_knot_values(Phi))

            lambdas = _tensor_to_q16_16_np(layer.lambdas)

            # Per-block features
            has_lateral = getattr(layer, 'lateral_scale', None) is not None
            lateral_scale = _q16_16_from_float(layer.lateral_scale.item()) if has_lateral else 0
            lateral_type = model_config.get('lateral_type', 'cyclic')

            # Residual
            residual_type = 0  # 0 none, 1 scalar, 2 pooling, 3 broadcast, 4 projection
            residual_scalar = 0
            residual_pool = None
            residual_bcast = None
            if CONFIG.get('use_residual_weights', True):
                if getattr(layer, 'residual_weight', None) is not None:
                    residual_type = 1
                    residual_scalar = _q16_16_from_float(layer.residual_weight.item())
                elif getattr(layer, 'residual_pooling_weights', None) is not None:
                    residual_type = 2
                    residual_pool = _tensor_to_q16_16_np(layer.residual_pooling_weights)
                elif getattr(layer, 'residual_broadcast_weights', None) is not None:
                    residual_type = 3
                    residual_bcast = _tensor_to_q16_16_np(layer.residual_broadcast_weights)
                elif getattr(layer, 'residual_projection', None) is not None:
                    residual_type = 4
                    # Not DS-friendly; should have been caught above, but keep a final guard.
                    raise ValueError("Residual projection is not supported by export_nds.")

            # Norm export (pre-folded affine). norm_dim==0 means no norm for this block.
            norm_dim = 0
            norm_scale = None
            norm_bias = None
            if norm_type != 'none':
                norm_layer = model.sprecher_net.norm_layers[i]
                if isinstance(norm_layer, nn.BatchNorm1d):
                    if not isinstance(norm_layer, nn.Identity):
                        # Identity BN layers are represented as nn.Identity
                        # Determine dim from the module itself
                        norm_dim = int(norm_layer.num_features)
                        w = norm_layer.weight.detach().cpu() if norm_layer.affine else torch.ones(norm_dim)
                        b = norm_layer.bias.detach().cpu() if norm_layer.affine else torch.zeros(norm_dim)
                        rm = norm_layer.running_mean.detach().cpu()
                        rv = norm_layer.running_var.detach().cpu()
                        eps = float(norm_layer.eps)
                        inv_std = 1.0 / torch.sqrt(rv + eps)
                        scale = w * inv_std
                        bias = b - rm * scale
                        norm_scale = _tensor_to_q16_16_np(scale)
                        norm_bias = _tensor_to_q16_16_np(bias)
                elif isinstance(norm_layer, nn.LayerNorm):
                    # LayerNorm on DS is not currently supported.
                    raise ValueError("LayerNorm export is not supported by export_nds (use --norm_type batch or none).")

            eta_fix = _q16_16_from_float(layer.eta.item())

            # Block header
            block_flags = 0
            if has_lateral:
                block_flags |= 1 << 0
            if norm_dim > 0:
                block_flags |= 1 << 1

            _write_u32(f, d_in)
            _write_u32(f, d_out)
            _write_u32(f, int(phi_knots.shape[0]))
            _write_u32(f, int(Phi_knots.shape[0]))
            _write_u32(f, block_flags)
            _write_u32(f, int(residual_type))
            _write_u32(f, int(norm_dim))
            _write_i32(f, eta_fix)

            # Spline data and core params
            _write_i32_array(f, phi_knots)
            _write_i32_array(f, phi_coeffs)
            _write_i32_array(f, Phi_knots)
            _write_i32_array(f, Phi_coeffs)
            _write_i32_array(f, lambdas)

            # Lateral mixing params
            if has_lateral:
                _write_i32(f, lateral_scale)
                if lateral_type == 'bidirectional':
                    lat_wf = _tensor_to_q16_16_np(layer.lateral_weights_forward)
                    lat_wb = _tensor_to_q16_16_np(layer.lateral_weights_backward)
                    _write_i32_array(f, lat_wf)
                    _write_i32_array(f, lat_wb)
                else:
                    lat_w = _tensor_to_q16_16_np(layer.lateral_weights)
                    _write_i32_array(f, lat_w)

            # Residual params
            if residual_type == 1:
                _write_i32(f, residual_scalar)
            elif residual_type == 2:
                _write_i32_array(f, residual_pool)
            elif residual_type == 3:
                _write_i32_array(f, residual_bcast)

            # Norm params
            if norm_dim > 0:
                _write_i32_array(f, norm_scale)
                _write_i32_array(f, norm_bias)

        # Footer
        out_scale_fix = _q16_16_from_float(model.sprecher_net.output_scale.item())
        out_bias_fix = _q16_16_from_float(model.sprecher_net.output_bias.item())
        _write_i32(f, out_scale_fix)
        _write_i32(f, out_bias_fix)

    # ------------------------------------------------------------------
    # Print summary
    # ------------------------------------------------------------------
    file_size = os.path.getsize(out_path)
    print("\n=== Nintendo DS Export Complete ===")
    print(f"Checkpoint: {model_path}")
    print(f"Output:     {out_path} ({file_size / 1024:.1f} KiB)")
    print(f"Arch:       784 -> {', '.join(str(x) for x in architecture)} -> 10  ({num_blocks} blocks)")
    print(f"Max dim:    {max_dim}")
    print(f"SN params:  {total_params}")
    print(f"MLP params: {mlp_params}")
    print("Features:")
    print(f"  Lateral mixing: {'ON' if model_config.get('use_lateral', True) else 'OFF'} ({model_config.get('lateral_type', 'cyclic')})")
    print(f"  Residuals:      {'ON' if model_config.get('use_residual', True) else 'OFF'} ({model_config.get('residual_style', 'node')})")
    print(f"  Norm:           {norm_type} (position={model_config.get('norm_position','after')}, skip_first={model_config.get('norm_skip_first', True)})")
    print(f"  Splines:        phi={phi_spline_type}, Phi={Phi_spline_type}")
    print("Param breakdown:")
    for k, v in param_counts.items():
        if k != 'total':
            print(f"  {k:22s}: {v}")
    print("  " + "-" * 30)
    print(f"  {'total':22s}: {param_counts.get('total', total_params)}")
    print("==================================\n")


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
    elif args.mode == "export_nds":
        export_nds(args)


if __name__ == "__main__":
    main()
