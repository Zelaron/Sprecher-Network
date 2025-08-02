"""Parameter export utilities for Sprecher Networks."""

import os
import torch
import numpy as np
from datetime import datetime
from .config import CONFIG, PARAM_CATEGORIES


def parse_param_types(param_string):
    """
    Parse parameter type string into a list of categories.
    
    Args:
        param_string: None, 'all', or comma-separated categories
        
    Returns:
        List of parameter categories to export
    """
    if param_string is None or param_string is False:
        return []
    elif param_string is True or param_string == 'all':
        return PARAM_CATEGORIES
    else:
        # Parse comma-separated categories
        requested = [p.strip().lower() for p in param_string.split(',')]
        # Validate categories
        valid_categories = []
        for cat in requested:
            if cat in PARAM_CATEGORIES:
                valid_categories.append(cat)
            else:
                print(f"Warning: Unknown parameter category '{cat}', skipping")
        return valid_categories


def format_tensor(tensor, name="", indent="  ", max_per_line=8):
    """
    Format a tensor for readable text output. Always prints ALL values.
    
    Args:
        tensor: PyTorch tensor to format
        name: Optional name for the tensor
        indent: Indentation string
        max_per_line: Maximum values per line for 1D tensors
        
    Returns:
        Formatted string representation
    """
    if tensor is None:
        return f"{indent}{name}: None\n" if name else f"{indent}None\n"
    
    # Convert to numpy for easier formatting
    arr = tensor.detach().cpu().numpy()
    
    lines = []
    if name:
        lines.append(f"{indent}{name}:")
        indent = indent + "  "
    
    # Format based on dimensions
    if arr.ndim == 0:
        # Scalar (0D)
        lines.append(f"{indent}{arr.item():.6f}")
    elif arr.ndim == 1:
        # Vector (1D)
        lines.append(f"{indent}Shape: {arr.shape}")
        if len(arr) <= max_per_line:
            values_str = ", ".join(f"{v:.6f}" for v in arr)
            lines.append(f"{indent}Values: [{values_str}]")
        else:
            lines.append(f"{indent}Values: [")
            for i in range(0, len(arr), max_per_line):
                chunk = arr[i:i+max_per_line]
                values_str = ", ".join(f"{v:.6f}" for v in chunk)
                suffix = "," if i + max_per_line < len(arr) else ""
                lines.append(f"{indent}  {values_str}{suffix}")
            lines.append(f"{indent}]")
    elif arr.ndim == 2:
        # Matrix (2D) - print the FULL matrix
        lines.append(f"{indent}Shape: {arr.shape}")
        lines.append(f"{indent}Values:")
        lines.append(f"{indent}[")
        for i in range(arr.shape[0]):
            row_str = ", ".join(f"{v:.6f}" for v in arr[i])
            suffix = "," if i < arr.shape[0] - 1 else ""
            lines.append(f"{indent}  [{row_str}]{suffix}")
        lines.append(f"{indent}]")
    else:
        # This should never happen in Sprecher Networks
        raise ValueError(f"Unexpected tensor dimension {arr.ndim} for parameter '{name}'. "
                         f"Shape: {arr.shape}. "
                         f"Sprecher Networks should only have 0D, 1D, or 2D parameters. "
                         f"This likely indicates a bug in the code.")
    
    return "\n".join(lines)


def format_integer_tensor(tensor, name="", indent="  ", max_per_line=16):
    """
    Format an integer tensor for readable text output.
    
    Args:
        tensor: PyTorch tensor with integer values
        name: Optional name for the tensor
        indent: Indentation string
        max_per_line: Maximum values per line
        
    Returns:
        Formatted string representation
    """
    if tensor is None:
        return f"{indent}{name}: None\n" if name else f"{indent}None\n"
    
    # Convert to numpy for easier formatting
    arr = tensor.detach().cpu().numpy()
    
    lines = []
    if name:
        lines.append(f"{indent}{name}:")
        indent = indent + "  "
    
    # Format 1D integer array
    lines.append(f"{indent}Shape: {arr.shape}")
    if len(arr) <= max_per_line:
        values_str = ", ".join(f"{int(v)}" for v in arr)
        lines.append(f"{indent}Values: [{values_str}]")
    else:
        lines.append(f"{indent}Values: [")
        for i in range(0, len(arr), max_per_line):
            chunk = arr[i:i+max_per_line]
            values_str = ", ".join(f"{int(v)}" for v in chunk)
            suffix = "," if i + max_per_line < len(arr) else ""
            lines.append(f"{indent}  {values_str}{suffix}")
        lines.append(f"{indent}]")
    
    return "\n".join(lines)


def export_parameters(model, param_types, save_path, dataset_info=None, checkpoint_info=None):
    """
    Export specified parameter types to a formatted text file.
    
    Args:
        model: Trained SprecherMultiLayerNetwork
        param_types: List of parameter types to export or 'all'
        save_path: Path to save the text file
        dataset_info: Optional dict with dataset metadata
        checkpoint_info: Optional dict with checkpoint information
    """
    # Parse parameter types
    categories = parse_param_types(param_types)
    if not categories:
        return
    
    # Create output directory if needed
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    # Start building the output
    lines = []
    lines.append("=" * 60)
    lines.append("Sprecher Network Parameters Export")
    lines.append("=" * 60)
    
    # Add metadata if provided
    if dataset_info:
        lines.append(f"Dataset: {dataset_info.get('name', 'Unknown')}")
        lines.append(f"Architecture: {dataset_info.get('architecture', [])}")
        lines.append(f"Input Dimension: {dataset_info.get('input_dim', 'Unknown')}")
        lines.append(f"Output Dimension: {dataset_info.get('output_dim', 'Unknown')}")
        lines.append(f"Training Epochs: {dataset_info.get('epochs', 'Unknown')}")
    
    if checkpoint_info:
        lines.append(f"Checkpoint Epoch: {checkpoint_info.get('epoch', 'Unknown')}")
        lines.append(f"Best Loss: {checkpoint_info.get('loss', 'Unknown'):.6e}")
    
    lines.append(f"Export Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append(f"Exported Parameters: {', '.join(categories)}")
    lines.append("=" * 60)
    lines.append("")
    
    # Add layer mapping information
    lines.append("LAYER MAPPING")
    lines.append("-" * 40)
    for i, layer in enumerate(model.layers):
        is_final_str = " (FINAL - summed)" if layer.is_final else ""
        lines.append(f"Layer {i}: {layer.d_in} → {layer.d_out}{is_final_str}")
    lines.append("")
    
    # Export Lambda parameters
    if 'lambda' in categories:
        lines.append("LAMBDA PARAMETERS (Weight Vectors)")
        lines.append("-" * 40)
        for i, layer in enumerate(model.layers):
            lines.append(f"Layer {i} (d_in={layer.d_in}):")
            lines.append(format_tensor(layer.lambdas, indent="  "))
        lines.append("")
    
    # Export Eta parameters
    if 'eta' in categories:
        lines.append("ETA PARAMETERS (Shift Values)")
        lines.append("-" * 40)
        for i, layer in enumerate(model.layers):
            lines.append(f"Layer {i}: {layer.eta.item():.6f}")
        lines.append("")
    
    # Export Spline parameters
    if 'spline' in categories:
        lines.append("SPLINE PARAMETERS")
        lines.append("-" * 40)
        for i, layer in enumerate(model.layers):
            lines.append(f"Layer {i} ({layer.d_in} → {layer.d_out}):")
            
            # Phi spline (monotonic)
            lines.append("  Phi spline (monotonic):")
            lines.append(f"    Domain: [{layer.phi.in_min:.6f}, {layer.phi.in_max:.6f}]")
            lines.append(f"    Codomain: [0.0, 1.0] (fixed)")
            lines.append(f"    Number of knots: {layer.phi.num_knots}")
            if hasattr(layer.phi, 'log_increments'):
                lines.append(format_tensor(layer.phi.log_increments, "Log increments", "    "))
            coeffs = layer.phi.get_coeffs()
            lines.append(format_tensor(coeffs, "Coefficients", "    "))
            
            # Phi spline (general)
            lines.append("  Phi spline (general):")
            lines.append(f"    Domain: [{layer.Phi.in_min:.6f}, {layer.Phi.in_max:.6f}]")
            if CONFIG['train_phi_codomain'] and hasattr(layer, 'phi_codomain_params'):
                cc = layer.phi_codomain_params.cc.item()
                cr = layer.phi_codomain_params.cr.item()
                lines.append(f"    Codomain: [{cc-cr:.6f}, {cc+cr:.6f}] (trainable)")
            lines.append(f"    Number of knots: {layer.Phi.num_knots}")
            lines.append(format_tensor(layer.Phi.coeffs, "Coefficients", "    "))
        lines.append("")
    
    # Export Residual parameters (updated for learnable soft routing)
    if 'residual' in categories and CONFIG['use_residual_weights']:
        lines.append("RESIDUAL CONNECTION PARAMETERS (Learnable Soft Routing)")
        lines.append("-" * 40)
        has_residual = False
        for i, layer in enumerate(model.layers):
            # Check for scalar residual weight (same dimensions)
            if hasattr(layer, 'residual_weight') and layer.residual_weight is not None:
                lines.append(f"Layer {i} ({layer.d_in} → {layer.d_out}): SCALAR residual")
                lines.append(f"  residual_weight = {layer.residual_weight.item():.6f}")
                has_residual = True
            
            # Check for learnable routing selectivity
            elif hasattr(layer, 'routing_selectivity') and layer.routing_selectivity is not None:
                routing_type = "POOLING" if layer.d_in > layer.d_out else "BROADCASTING"
                lines.append(f"Layer {i} ({layer.d_in} → {layer.d_out}): {routing_type} with learned routing")
                
                # Show raw selectivity values
                lines.append(format_tensor(layer.routing_selectivity, "Routing selectivity (raw)", "  "))
                
                # Show effective selectivity after sigmoid - FIXED: added .detach()
                selectivity_values = torch.sigmoid(layer.routing_selectivity).detach().cpu().numpy()
                lines.append(f"  Effective selectivity range: [{selectivity_values.min():.3f}, {selectivity_values.max():.3f}]")
                
                # Add interpretation
                avg_selectivity = selectivity_values.mean()
                if avg_selectivity < 0.3:
                    lines.append("  Interpretation: Highly distributed routing (low selectivity)")
                elif avg_selectivity > 0.7:
                    lines.append("  Interpretation: Focused routing (high selectivity)")
                else:
                    lines.append("  Interpretation: Moderate routing selectivity")
                
                has_residual = True
        
        if not has_residual:
            lines.append("  No residual parameters in this model")
        lines.append("")
    
    # Export Codomain parameters
    if 'codomain' in categories and CONFIG['train_phi_codomain']:
        lines.append("PHI CODOMAIN PARAMETERS")
        lines.append("-" * 40)
        for i, layer in enumerate(model.layers):
            if hasattr(layer, 'phi_codomain_params') and layer.phi_codomain_params is not None:
                cc = layer.phi_codomain_params.cc.item()
                cr = layer.phi_codomain_params.cr.item()
                lines.append(f"Layer {i}:")
                lines.append(f"  Center (cc): {cc:.6f}")
                lines.append(f"  Radius (cr): {cr:.6f}")
                lines.append(f"  Codomain: [{cc-cr:.6f}, {cc+cr:.6f}]")
        lines.append("")
    
    # Export Normalization parameters
    if 'norm' in categories and model.norm_type != 'none':
        lines.append("NORMALIZATION PARAMETERS")
        lines.append("-" * 40)
        for i, norm_layer in enumerate(model.norm_layers):
            if hasattr(norm_layer, 'weight'):  # Skip Identity layers
                lines.append(f"Layer {i} ({type(norm_layer).__name__}):")
                lines.append(format_tensor(norm_layer.weight, "Weight", "  "))
                lines.append(format_tensor(norm_layer.bias, "Bias", "  "))
                if hasattr(norm_layer, 'running_mean'):
                    lines.append(format_tensor(norm_layer.running_mean, "Running mean", "  "))
                    lines.append(format_tensor(norm_layer.running_var, "Running variance", "  "))
                    lines.append(f"  Num batches tracked: {norm_layer.num_batches_tracked.item()}")
        lines.append("")
    
    # Export Output parameters
    if 'output' in categories:
        lines.append("OUTPUT PARAMETERS")
        lines.append("-" * 40)
        lines.append(f"Output scale: {model.output_scale.item():.6f}")
        lines.append(f"Output bias: {model.output_bias.item():.6f}")
        lines.append("")
    
    # Write to file with UTF-8 encoding
    with open(save_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines))
    
    print(f"Parameters exported to: {save_path}")