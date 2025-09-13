"""Parameter export utilities for Sprecher Networks.

Fully supports both residual styles:
- 'node'   : original node-centric residuals (scalar / pooling / broadcast)
- 'linear' : standard residuals (α·x when d_in == d_out; projection W when d_in != d_out)
"""

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


def _format_loss_for_header(loss_value):
    """Safely format a loss value in scientific notation, or return as-is if not convertible."""
    try:
        return f"{float(loss_value):.6e}"
    except Exception:
        return str(loss_value)


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
        # Safer formatting for loss (avoids crash if 'Unknown')
        loss_val = _format_loss_for_header(checkpoint_info.get('loss', 'Unknown'))
        lines.append(f"Best Loss: {loss_val}")
    
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
            
            # phi spline (monotonic)
            lines.append("  phi spline (monotonic):")
            lines.append(f"    Domain: [{layer.phi.in_min:.6f}, {layer.phi.in_max:.6f}]")
            lines.append(f"    Codomain: [0.0, 1.0] (fixed)")
            lines.append(f"    Number of knots: {layer.phi.num_knots}")
            if hasattr(layer.phi, 'log_increments'):
                lines.append(format_tensor(layer.phi.log_increments, "Log increments", "    "))
            coeffs_phi = layer.phi.get_coeffs()
            lines.append(format_tensor(coeffs_phi, "Coefficients", "    "))
            
            # Phi spline (general)
            lines.append("  Phi spline (general):")
            lines.append(f"    Domain: [{layer.Phi.in_min:.6f}, {layer.Phi.in_max:.6f}]")
            if CONFIG.get('train_phi_codomain', False) and hasattr(layer, 'phi_codomain_params'):
                cc = layer.phi_codomain_params.cc.item()
                cr = layer.phi_codomain_params.cr.item()
                lines.append(f"    Codomain: [{cc-cr:.6f}, {cc+cr:.6f}] (trainable)")
            lines.append(f"    Number of knots: {layer.Phi.num_knots}")
            lines.append(format_tensor(layer.Phi.coeffs, "Coefficients", "    "))
        lines.append("")
    
    # Export Residual parameters (supports both styles)
    if 'residual' in categories and CONFIG.get('use_residual_weights', True):
        style = CONFIG.get('residual_style', 'node')
        lines.append(f"RESIDUAL CONNECTION PARAMETERS (style: {style})")
        lines.append("-" * 40)
        has_any_residual = False
        
        for i, layer in enumerate(model.layers):
            # Prefer explicit attributes; we don't rely only on the style flag
            # so this works regardless of how the model was constructed.
            
            # 1) Linear style projection matrix (d_in != d_out) — present only if implemented
            if hasattr(layer, 'residual_projection') and layer.residual_projection is not None:
                has_any_residual = True
                lines.append(f"Layer {i} ({layer.d_in} → {layer.d_out}): PROJECTION residual")
                lines.append(format_tensor(layer.residual_projection, "Projection matrix W", "  "))
                lines.append("  Note: residual = x @ W (added pre-sum / per-output)")
                continue  # Exclusive with other residual types in typical builds
            
            # 2) Scalar residual (same dims) — used in both 'node' and 'linear'
            if hasattr(layer, 'residual_weight') and layer.residual_weight is not None:
                has_any_residual = True
                lines.append(f"Layer {i} ({layer.d_in} → {layer.d_out}): SCALAR residual")
                lines.append(f"  residual_weight (alpha): {layer.residual_weight.item():.6f}")
                continue
            
            # 3) Pooling residual (node-centric; d_in > d_out)
            if hasattr(layer, 'residual_pooling_weights') and layer.residual_pooling_weights is not None:
                has_any_residual = True
                lines.append(f"Layer {i} ({layer.d_in} → {layer.d_out}): POOLING residual")
                lines.append(format_tensor(layer.residual_pooling_weights, "Pooling weights", "  "))
                if hasattr(layer, 'pooling_assignment'):
                    lines.append(format_integer_tensor(layer.pooling_assignment, "Pooling assignment", "  "))
                if hasattr(layer, 'pooling_counts'):
                    lines.append(format_tensor(layer.pooling_counts, "Inputs per output", "  "))
                lines.append("  Note: input j contributes to output pooling_assignment[j] with its weight")
                continue
            
            # 4) Broadcast residual (node-centric; d_in < d_out)
            if hasattr(layer, 'residual_broadcast_weights') and layer.residual_broadcast_weights is not None:
                has_any_residual = True
                lines.append(f"Layer {i} ({layer.d_in} → {layer.d_out}): BROADCAST residual")
                lines.append(format_tensor(layer.residual_broadcast_weights, "Broadcast weights", "  "))
                if hasattr(layer, 'broadcast_sources'):
                    lines.append(format_integer_tensor(layer.broadcast_sources, "Broadcast sources", "  "))
                lines.append("  Note: output k receives scaled copy of input broadcast_sources[k]")
                continue
        
        if not has_any_residual:
            lines.append("  No residual parameters in this model")
        lines.append("")
    
    # Export Codomain parameters
    if 'codomain' in categories and CONFIG.get('train_phi_codomain', False):
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
    if 'norm' in categories and getattr(model, 'norm_type', 'none') != 'none':
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
    
    # Export Lateral Mixing parameters
    if 'lateral' in categories and CONFIG.get('use_lateral_mixing', False):
        lines.append("LATERAL MIXING PARAMETERS (Intra-block Communication)")
        lines.append("-" * 40)
        has_lateral = False
        for i, layer in enumerate(model.layers):
            if hasattr(layer, 'lateral_scale') and layer.lateral_scale is not None:
                has_lateral = True
                lines.append(f"Layer {i} ({layer.d_in} → {layer.d_out}):")
                lines.append(f"  Lateral scale: {layer.lateral_scale.item():.6f}")
                
                ltype = CONFIG.get('lateral_mixing_type', 'cyclic')
                if ltype == 'bidirectional':
                    lines.append("  Type: Bidirectional mixing")
                    lines.append(format_tensor(layer.lateral_weights_forward, "Forward weights", "  "))
                    lines.append(format_tensor(layer.lateral_weights_backward, "Backward weights", "  "))
                    lines.append("  Note: Each output mixes with both cyclic neighbors")
                else:  # 'cyclic'
                    lines.append("  Type: Cyclic mixing")
                    lines.append(format_tensor(layer.lateral_weights, "Lateral weights", "  "))
                    lines.append("  Note: Output q receives contribution from output (q+1) % d_out")
        if not has_lateral:
            lines.append("  No lateral mixing parameters in this model")
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