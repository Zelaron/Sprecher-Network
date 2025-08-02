"""Plotting utilities for Sprecher Networks."""

import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from mpl_toolkits.mplot3d import Axes3D
from .config import CONFIG


def plot_network_structure_ax(ax, layers, input_dim, final_dim=1):
    """Plot network structure diagram."""
    num_blocks = len(layers)
    if num_blocks == 0:
        ax.text(0.5, 0.5, "No network", ha='center', va='center', fontsize=12)
        ax.axis('off')
        return
    
    # Check if we need an additional output node for summing
    need_sum_node = any(layer.is_final and final_dim == 1 for layer in layers)
    total_columns = num_blocks + 1 + (1 if need_sum_node else 0)
    
    # Create evenly spaced x-coordinates for all nodes including potential sum node
    layer_x = np.linspace(0.2, 0.8, total_columns)
    
    if input_dim is None and num_blocks > 0:
        input_dim = layers[0].d_in
    
    if input_dim == 1:
        input_y = [0.5]
    else:
        input_y = np.linspace(0.2, 0.8, input_dim)
    
    # Plot input nodes
    for i, y in enumerate(input_y):
        ax.scatter(layer_x[0], y, c='black', s=100)
        ax.text(layer_x[0] - 0.03, y, f'$x_{{{i+1}}}$', ha='right', fontsize=8)
    
    prev_y = input_y
    sum_node_placed = False
    
    for block_index, block in enumerate(layers):
        j = block_index + 1
        d_out = block.d_out
        new_x = layer_x[block_index + 1]
        color = 'red' if block.is_final else 'blue'
        
        if d_out == 1:
            new_y = [0.5]
        else:
            new_y = np.linspace(0.2, 0.8, d_out)
            
        # Plot this layer's nodes
        for ny in new_y:
            ax.scatter(new_x, ny, c=color, s=100)
            
        # Connect previous layer to this layer
        for py in prev_y:
            for ny in new_y:
                ax.plot([layer_x[block_index], new_x], [py, ny], 'k-', alpha=0.5)
                
        # Place φ and Φ labels
        mid_x = 0.5 * (layer_x[block_index] + new_x)
        ax.text(mid_x, 0.14, f"$\\phi^{{({j})}}$", ha='center', fontsize=9, color='green')
        ax.text(mid_x, 0.09, f"$\\Phi^{{({j})}}$", ha='center', fontsize=9, color='green')
        
        # Handle final output/sum node if needed
        if block.is_final and final_dim == 1 and not sum_node_placed:
            sum_x = layer_x[block_index + 2]  # Use next x position from our evenly spaced grid
            sum_y = 0.5
            ax.scatter(sum_x, sum_y, c='red', s=100)
            for ny in new_y:
                ax.plot([new_x, sum_x], [ny, sum_y], 'k-', alpha=0.5)
            prev_y = [sum_y]
            sum_node_placed = True
        else:
            prev_y = new_y
            
    ax.set_title("Network Structure", fontsize=11)
    ax.axis('off')


def plot_high_dim_function(ax, model, dataset, device):
    """Plot high-dimensional function using dimension slices or statistics."""
    print(f"DEBUG in plot_high_dim_function: model type = {type(model)}")
    print(f"DEBUG in plot_high_dim_function: model.training = {model.training}")
    
    # Keep model in its current mode (likely training mode)
    # was_training = model.training
    # model.eval()  # Removed to maintain consistent BatchNorm behavior
    
    input_dim = dataset.input_dim
    output_dim = dataset.output_dim
    
    if input_dim <= 10:
        # For moderately high dimensions, plot 1D slices
        num_slices = min(input_dim, 4)  # Limit to 4 for clarity
        
        # Create base input with all dimensions at 0.5
        x_base = torch.full((100, input_dim), 0.5, device=device)
        x_vals = torch.linspace(0, 1, 100, device=device)
        
        # Define base colors for each output dimension
        base_colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple']
        
        # Store lines for legend ordering
        pred_lines = []
        true_lines = []
        pred_labels = []
        true_labels = []
        
        # For single output, use different approach
        if output_dim == 1:
            colors = plt.cm.viridis(np.linspace(0, 1, num_slices))
            for i in range(num_slices):
                # Vary dimension i
                x_test = x_base.clone()
                for j in range(100):
                    x_test[j, i] = x_vals[j]
                
                # Get predictions and true values
                with torch.no_grad():
                    y_pred = model(x_test)
                y_true = dataset.evaluate(x_test)
                
                # Plot with distinct colors
                pred_line, = ax.plot(x_vals.cpu(), y_pred.cpu(), '-', 
                                    color=colors[i], label=f'Pred (vary dim {i})', 
                                    linewidth=2)
                true_line, = ax.plot(x_vals.cpu(), y_true.cpu(), '--', 
                                    color=colors[i], label=f'True (vary dim {i})', 
                                    linewidth=1.5, alpha=0.7)
                pred_lines.append(pred_line)
                true_lines.append(true_line)
        else:
            # Multiple outputs - use color families
            for out_idx in range(min(output_dim, 3)):  # Show first 3 outputs
                base_color = base_colors[out_idx % len(base_colors)]
                
                # Create color variations for this output dimension
                if base_color == 'tab:blue':
                    color_variations = plt.cm.Blues(np.linspace(0.4, 0.9, num_slices))
                elif base_color == 'tab:orange':
                    color_variations = plt.cm.Oranges(np.linspace(0.4, 0.9, num_slices))
                elif base_color == 'tab:green':
                    color_variations = plt.cm.Greens(np.linspace(0.4, 0.9, num_slices))
                
                # Plot all input dimension variations for this output
                for i in range(num_slices):
                    # Vary dimension i
                    x_test = x_base.clone()
                    for j in range(100):
                        x_test[j, i] = x_vals[j]
                    
                    # Get predictions and true values
                    with torch.no_grad():
                        y_pred = model(x_test)
                    y_true = dataset.evaluate(x_test)
                    
                    # Plot predictions with color variation
                    pred_line, = ax.plot(x_vals.cpu(), y_pred[:, out_idx].cpu(), '-', 
                                       color=color_variations[i], 
                                       linewidth=1.5,
                                       label=f'Pred out{out_idx} (vary in{i})')
                    pred_lines.append(pred_line)
                    pred_labels.append(f'Pred out{out_idx} (vary in{i})')
                    
                    # Plot true values with same color but dashed
                    true_line, = ax.plot(x_vals.cpu(), y_true[:, out_idx].cpu(), '--', 
                                       color=color_variations[i], 
                                       linewidth=1, 
                                       alpha=0.8,
                                       label=f'True out{out_idx} (vary in{i})')
                    true_lines.append(true_line)
                    true_labels.append(f'True out{out_idx} (vary in{i})')
            
            # Create custom legend with better organization
            # Group by output dimension
            handles = []
            labels = []
            for out_idx in range(min(output_dim, 3)):
                # Add predicted lines for this output
                for i in range(num_slices):
                    idx = out_idx * num_slices + i
                    if idx < len(pred_lines):
                        handles.append(pred_lines[idx])
                        labels.append(pred_labels[idx])
                # Add true lines for this output
                for i in range(num_slices):
                    idx = out_idx * num_slices + i
                    if idx < len(true_lines):
                        handles.append(true_lines[idx])
                        labels.append(true_labels[idx])
                # Add spacer if not last output
                if out_idx < min(output_dim, 3) - 1:
                    handles.append(plt.Line2D([0], [0], color='none'))
                    labels.append('')
                    
            ax.legend(handles, labels, fontsize=7, ncol=2, loc='center left', bbox_to_anchor=(1, 0.5))
        
        ax.set_xlabel('Variable value')
        ax.set_ylabel('Output')
        ax.set_title(f'1D Slices - Varying input dims 0-{num_slices-1}')
        ax.grid(True, alpha=0.3)
        
        # Adjust layout to prevent legend cutoff
        plt.tight_layout()
        
    else:
        # For very high dimensions, plot statistics
        n_samples = 1000
        x_test, y_true = dataset.sample(n_samples, device)
        
        with torch.no_grad():
            y_pred = model(x_test)
        
        # Convert to numpy
        y_true_np = y_true.cpu().numpy().flatten()
        y_pred_np = y_pred.cpu().numpy().flatten()
        
        # Calculate appropriate bin range
        y_min = min(y_true_np.min(), y_pred_np.min())
        y_max = max(y_true_np.max(), y_pred_np.max())
        
        # Add some padding
        y_range = y_max - y_min
        if y_range > 0:
            y_min -= 0.1 * y_range
            y_max += 0.1 * y_range
        else:
            y_min -= 0.1
            y_max += 0.1
        
        bins = np.linspace(y_min, y_max, 50)
        
        # Plot histograms
        ax.hist(y_true_np, bins=bins, alpha=0.5, label='True', density=True, color='blue')
        ax.hist(y_pred_np, bins=bins, alpha=0.5, label='Predicted', density=True, color='orange')
        
        # Add vertical lines for means
        ax.axvline(y_true_np.mean(), color='blue', linestyle='--', linewidth=2, alpha=0.8)
        ax.axvline(y_pred_np.mean(), color='orange', linestyle='--', linewidth=2, alpha=0.8)
        
        ax.set_xlabel('Output value')
        ax.set_ylabel('Density')
        ax.set_title(f'Output Distribution ({input_dim}D input)\nTrue mean: {y_true_np.mean():.3f}, Pred mean: {y_pred_np.mean():.3f}')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    # No need to restore mode since we didn't change it
    # if was_training:
    #     model.train()


def plot_routing_heatmap(ax, layer, layer_idx):
    """Plot the learned routing pattern as a heatmap."""
    if not hasattr(layer, 'routing_logits') or layer.routing_logits is None:
        ax.text(0.5, 0.5, 'No routing\n(d_in = d_out)', ha='center', va='center', fontsize=12)
        ax.axis('off')
        return
    
    with torch.no_grad():
        temp = CONFIG.get('routing_temperature', 1.0)
        
        if layer.d_in > layer.d_out:
            # Pooling case: show assignment probabilities
            routing_matrix = layer.routing_logits.view(-1, 1).expand(layer.d_in, layer.d_out)
            affinity = routing_matrix + torch.arange(layer.d_out, device=layer.routing_logits.device).float() * temp
            weights = F.softmax(affinity / temp, dim=1).cpu().numpy()
            
            # Create heatmap
            im = ax.imshow(weights, cmap='Blues', aspect='auto', vmin=0, vmax=1)
            ax.set_xlabel('Output dimension')
            ax.set_ylabel('Input dimension')
            ax.set_title(f'Layer {layer_idx}: Pooling ({layer.d_in}→{layer.d_out})')
            
            # Add text annotations for probabilities
            for i in range(layer.d_in):
                for j in range(layer.d_out):
                    text = ax.text(j, i, f'{weights[i, j]:.2f}',
                                 ha="center", va="center", color="black" if weights[i, j] < 0.5 else "white",
                                 fontsize=8)
            
            # Add importance weights on the side if used
            if CONFIG.get('routing_use_importance', True):
                importance = torch.sigmoid(layer.routing_logits).cpu().numpy()
                # Create a second axis for importance
                ax2 = ax.twinx()
                ax2.plot(importance, range(layer.d_in), 'ro-', linewidth=2, markersize=4)
                ax2.set_ylabel('Importance weight', color='red')
                ax2.tick_params(axis='y', labelcolor='red')
                ax2.set_ylim(layer.d_in - 0.5, -0.5)  # Align with heatmap
                
        else:
            # Broadcasting case: show attention weights
            routing_matrix = layer.routing_logits.view(-1, 1).expand(layer.d_out, layer.d_in)
            affinity = routing_matrix + torch.arange(layer.d_in, device=layer.routing_logits.device).float() * temp
            weights = F.softmax(affinity / temp, dim=1).cpu().numpy()
            
            # Create heatmap
            im = ax.imshow(weights, cmap='Oranges', aspect='auto', vmin=0, vmax=1)
            ax.set_xlabel('Input dimension')
            ax.set_ylabel('Output dimension')
            ax.set_title(f'Layer {layer_idx}: Broadcasting ({layer.d_in}→{layer.d_out})')
            
            # Add text annotations for probabilities
            for i in range(layer.d_out):
                for j in range(layer.d_in):
                    text = ax.text(j, i, f'{weights[i, j]:.2f}',
                                 ha="center", va="center", color="black" if weights[i, j] < 0.5 else "white",
                                 fontsize=8)
        
        # Add colorbar
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)


def plot_results(model, layers, dataset=None, save_path=None, 
                 plot_network=True, plot_function=True, plot_splines=True,
                 title_suffix="", x_train=None, y_train=None):
    """
    Plot network structure, splines, function approximation, and routing patterns.
    
    Args:
        model: Trained Sprecher network
        layers: Model layers
        dataset: Dataset instance (optional, for function comparison)
        save_path: Path to save figure (optional)
        plot_network: Whether to plot network structure (default: True)
        plot_function: Whether to plot function comparison (default: True)
        plot_splines: Whether to plot splines (default: True)
        title_suffix: Additional text to add to figure title (default: "")
        x_train: Training input data (optional, for consistent evaluation)
        y_train: Training target data (optional, for consistent evaluation)
    
    Returns:
        fig: Matplotlib figure
    """
    print(f"\nDEBUG in plot_results: model type = {type(model)}")
    print(f"DEBUG in plot_results: model.training = {model.training}")
    print(f"DEBUG in plot_results: number of layers = {len(layers)}")
    
    # Determine the target device for plotting
    try:
        device = next(model.parameters()).device
    except StopIteration:
        device = torch.device('cpu')
    
    # Explicitly move the entire model to the determined device
    model.to(device)
    
    if len(layers) > 0:
        input_dim = layers[0].d_in
    else:
        input_dim = 1
    num_blocks = len(layers)
    final_dim = getattr(model, 'final_dim', 1)
    
    # Check if we have routing to plot
    has_routing = any(hasattr(layer, 'routing_logits') and layer.routing_logits is not None for layer in layers)
    
    # Calculate layout based on what we're plotting
    num_rows = 0
    if plot_network or plot_splines:
        num_rows += 1
    if has_routing and CONFIG.get('use_residual_weights', True):
        num_rows += 1  # Add row for routing heatmaps
    if plot_function and dataset is not None:
        num_rows += 1
    
    if num_rows == 0:
        print("Nothing to plot!")
        return None
    
    # Calculate columns for top row
    if plot_network and plot_splines:
        total_cols = 1 + 2 * num_blocks
    elif plot_splines:
        total_cols = 2 * num_blocks
    else:
        total_cols = 1
    
    plt.rcParams.update({'xtick.labelsize': 8, 'ytick.labelsize': 8})
    
    # Calculate figure size
    if plot_network and plot_splines:
        fig_width = 4 * total_cols * 0.75
    else:
        fig_width = max(12, 3 * total_cols)
    
    fig_height = 5  # Base height
    if has_routing and CONFIG.get('use_residual_weights', True):
        fig_height += 4  # Add height for routing row
    if plot_function:
        fig_height += 7  # Add height for function plot
    
    fig = plt.figure(figsize=(fig_width, fig_height))
    
    # Add title with suffix if provided
    if title_suffix and "MNIST" not in title_suffix:
        fig.suptitle(title_suffix, fontsize=14, y=0.98)
    
    # Calculate row positions
    current_row = 0
    row_heights = []
    
    if plot_network or plot_splines:
        row_heights.append(5)  # Height for network/splines row
        current_row += 1
    
    if has_routing and CONFIG.get('use_residual_weights', True):
        row_heights.append(4)  # Height for routing row
        current_row += 1
    
    if plot_function and dataset is not None:
        row_heights.append(7)  # Height for function row
    
    # Create GridSpec with appropriate heights
    total_height = sum(row_heights)
    gs_main = gridspec.GridSpec(len(row_heights), 1, height_ratios=row_heights, hspace=0.3)
    
    current_row = 0
    
    # Plot network structure and/or splines
    if plot_network or plot_splines:
        gs_top = gridspec.GridSpecFromSubplotSpec(1, total_cols, subplot_spec=gs_main[current_row])
        
        col_offset = 0
        if plot_network and (plot_splines or not plot_function):
            # Only plot network structure if input dim is reasonable
            if input_dim <= 10:
                ax_net = fig.add_subplot(gs_top[0, 0])
                plot_network_structure_ax(ax_net, layers, input_dim, final_dim)
                col_offset = 1
        
        if plot_splines:
            for i, layer in enumerate(layers):
                j = i + 1
                
                # When no network plot, use sequential columns starting from 0
                if plot_network and input_dim <= 10:
                    col_phi = 2 * i + 1
                    col_Phi = 2 * i + 2
                else:
                    col_phi = 2 * i
                    col_Phi = 2 * i + 1
                
                # Plot phi spline
                ax_phi = fig.add_subplot(gs_top[0, col_phi])
                block_label = f"Block {j}"
                ax_phi.set_title(f"{block_label} $\\phi^{{({j})}}$", fontsize=11)
                with torch.no_grad():
                    phi_device = next(layer.phi.parameters()).device
                    # Get current domain of phi
                    x_min, x_max = layer.phi.in_min, layer.phi.in_max
                    x_vals = torch.linspace(x_min, x_max, 200).to(phi_device)
                    y_vals = layer.phi(x_vals)
                ax_phi.plot(x_vals.cpu(), y_vals.cpu(), 'c-', linewidth=2)
                ax_phi.grid(True, alpha=0.3)
                ax_phi.set_xlabel('Input', fontsize=9)
                ax_phi.set_ylabel('Output', fontsize=9)
                
                # Add theoretical range info if available
                domain_info = f"[{x_min:.2f}, {x_max:.2f}]"
                if hasattr(layer, 'input_range') and layer.input_range is not None:
                    domain_info += f"\nInput range: [{layer.input_range.min:.2f}, {layer.input_range.max:.2f}]"
                ax_phi.set_title(f"{block_label} $\\phi^{{({j})}}$ Domain: {domain_info}", fontsize=9)
                
                # Plot Phi spline
                ax_Phi = fig.add_subplot(gs_top[0, col_Phi])
                ax_Phi.set_title(f"{block_label} $\\Phi^{{({j})}}$", fontsize=11)
                with torch.no_grad():
                    # Get current domain of Phi
                    in_min, in_max = layer.Phi.in_min, layer.Phi.in_max
                    Phi_device = next(layer.Phi.parameters()).device
                    x_vals = torch.linspace(in_min, in_max, 200).to(Phi_device)
                    y_vals = layer.Phi(x_vals)
                    
                    # Get codomain info if trainable
                    if CONFIG['train_phi_codomain'] and hasattr(layer, 'phi_codomain_params') and layer.phi_codomain_params is not None:
                        cc = layer.phi_codomain_params.cc.item()
                        cr = layer.phi_codomain_params.cr.item()
                        codomain_str = f", Codomain: [{cc-cr:.2f}, {cc+cr:.2f}]"
                    else:
                        codomain_str = ""
                        
                ax_Phi.plot(x_vals.cpu(), y_vals.cpu(), 'm-', linewidth=2)
                ax_Phi.grid(True, alpha=0.3)
                ax_Phi.set_xlabel('Input', fontsize=9)
                ax_Phi.set_ylabel('Output', fontsize=9)
                
                # Add output range info if available
                title_str = f"{block_label} $\\Phi^{{({j})}}$ D: [{in_min:.2f}, {in_max:.2f}]{codomain_str}"
                if hasattr(layer, 'output_range') and layer.output_range is not None:
                    title_str += f"\nOut range: [{layer.output_range.min:.2f}, {layer.output_range.max:.2f}]"
                ax_Phi.set_title(title_str, fontsize=9)
        
        current_row += 1
    
    # Plot routing heatmaps if we have routing
    if has_routing and CONFIG.get('use_residual_weights', True):
        # Count how many layers have routing
        routing_layers = [(i, layer) for i, layer in enumerate(layers) 
                         if hasattr(layer, 'routing_logits') and layer.routing_logits is not None]
        
        if routing_layers:
            gs_routing = gridspec.GridSpecFromSubplotSpec(1, len(routing_layers), 
                                                         subplot_spec=gs_main[current_row])
            
            for idx, (layer_idx, layer) in enumerate(routing_layers):
                ax_route = fig.add_subplot(gs_routing[0, idx])
                plot_routing_heatmap(ax_route, layer, layer_idx)
            
            current_row += 1
    
    # Plot function comparison if dataset is provided
    if plot_function and dataset is not None:
        print("DEBUG: Plotting function comparison")
        print(f"DEBUG: Using model: {type(model)}")
        
        if input_dim == 1:
            # Use training data if provided, otherwise create test points
            if x_train is not None and y_train is not None:
                print("DEBUG: Using provided training data for plotting")
                x_test = x_train.to(device)
                y_true = y_train.to(device)
            else:
                print("DEBUG: Generating new test data for plotting")
                x_test, y_true = dataset.sample(200, device=device)
            
            print(f"DEBUG: About to evaluate model on test data")
            print(f"DEBUG: Test data shape: {x_test.shape}")
            with torch.no_grad():
                y_pred = model(x_test)
            print(f"DEBUG: Model predictions shape: {y_pred.shape}")
            print(f"DEBUG: Sample predictions: {y_pred[:5].flatten().cpu().numpy()}")
            
            if final_dim == 1:
                # For scalar output, plot the target vs. prediction
                gs_bottom = gridspec.GridSpecFromSubplotSpec(1, 1, subplot_spec=gs_main[current_row])
                ax_func = fig.add_subplot(gs_bottom[0, 0])
                ax_func.set_title("Function Comparison", fontsize=12)
                ax_func.plot(x_test.cpu(), y_true.cpu(), 'k--', label='Target f(x)')
                ax_func.plot(x_test.cpu(), y_pred.cpu(), 'r-', label='Network output')
                ax_func.grid(True)
                ax_func.legend()
            else:
                # For vector output, plot each dimension separately
                gs_bottom = gridspec.GridSpecFromSubplotSpec(1, final_dim, subplot_spec=gs_main[current_row])
                
                for d in range(final_dim):
                    ax_func = fig.add_subplot(gs_bottom[0, d])
                    ax_func.set_title(f"Output dim {d}", fontsize=12)
                    ax_func.plot(x_test.cpu(), y_true[:, d].cpu(), 'k--', label='Target f(x)')
                    ax_func.plot(x_test.cpu(), y_pred[:, d].cpu(), 'r-', label=f'Out dim {d}')
                    ax_func.grid(True)
                    ax_func.legend()
        elif input_dim == 2:
            # For 2D input, create grid for surface plots
            n = 50
            points, y_true = dataset.sample(n * n, device=device)
            
            if final_dim == 1:
                # For scalar output from 2D input, plot target and prediction as surfaces
                gs_bottom = gridspec.GridSpecFromSubplotSpec(1, 2, subplot_spec=gs_main[current_row])
                
                x = torch.linspace(0, 1, n)
                y = torch.linspace(0, 1, n)
                X, Y = torch.meshgrid(x, y, indexing='ij')
                
                Z_true = y_true.reshape(n, n)
                ax_target = fig.add_subplot(gs_bottom[0, 0], projection='3d')
                ax_target.set_title("Target Function", fontsize=12)
                ax_target.plot_surface(X.cpu().numpy(), Y.cpu().numpy(), Z_true.cpu().numpy(), cmap='viridis')
                
                ax_output = fig.add_subplot(gs_bottom[0, 1], projection='3d')
                ax_output.set_title("Network Output", fontsize=12)
                with torch.no_grad():
                    Z_pred = model(points).reshape(n, n)
                ax_output.plot_surface(X.cpu().numpy(), Y.cpu().numpy(), Z_pred.cpu().numpy(), cmap='viridis')
            else:
                # For vector output from 2D input, plot each output dimension
                gs_bottom = gridspec.GridSpecFromSubplotSpec(1, final_dim, subplot_spec=gs_main[current_row])
                
                with torch.no_grad():
                    Z_pred_all = model(points)
                
                x = torch.linspace(0, 1, n)
                y = torch.linspace(0, 1, n)
                X, Y = torch.meshgrid(x, y, indexing='ij')
                
                for d in range(final_dim):
                    ax_out = fig.add_subplot(gs_bottom[0, d], projection='3d')
                    ax_out.set_title(f"Output dim {d}", fontsize=12)
                    Z_pred_d = Z_pred_all[:, d].reshape(n, n)
                    Z_true_d = y_true[:, d].reshape(n, n)
                    ax_out.plot_surface(X.cpu().numpy(), Y.cpu().numpy(), Z_true_d.cpu().numpy(),
                                        cmap='viridis', alpha=0.5)
                    ax_out.plot_surface(X.cpu().numpy(), Y.cpu().numpy(), Z_pred_d.cpu().numpy(),
                                        cmap='autumn', alpha=0.5)
        else:
            # For high-dimensional inputs, use specialized plotting
            gs_bottom = gridspec.GridSpecFromSubplotSpec(1, 1, subplot_spec=gs_main[current_row])
            ax_func = fig.add_subplot(gs_bottom[0, 0])
            plot_high_dim_function(ax_func, model, dataset, device)
    
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=240, bbox_inches='tight')
        print(f"Figure saved to {save_path}")
    
    return fig


def plot_loss_curve(losses, save_path=None):
    """Plot training loss curve on logarithmic scale."""
    plt.figure(figsize=(10, 6))
    plt.semilogy(losses, label="Training Loss", linewidth=1.5)
    plt.xlabel("Epoch", fontsize=12)
    plt.ylabel("Loss (log scale)", fontsize=12)
    plt.title("Loss Curve - Logarithmic Scale", fontsize=14)
    plt.grid(True, which="both", ls="-", alpha=0.2)
    plt.legend(fontsize=12)
    
    # Add minor ticks for better readability
    plt.gca().xaxis.set_minor_locator(plt.MultipleLocator(5000))
    
    # Optionally add a text box with final loss value
    final_loss = losses[-1]
    textstr = f'Final Loss: {final_loss:.2e}'
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    plt.text(0.65, 0.95, textstr, transform=plt.gca().transAxes, fontsize=10,
            verticalalignment='top', bbox=props)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150)
        print(f"Loss curve saved to {save_path}")