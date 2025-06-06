"""Plotting utilities for Sprecher Networks."""

import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from mpl_toolkits.mplot3d import Axes3D


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


def plot_results(model, layers, dataset=None, save_path=None, 
                 plot_network=True, plot_function=True, plot_splines=True,
                 title_suffix=""):
    """
    Plot network structure, splines, and function approximation.
    
    Args:
        model: Trained Sprecher network
        layers: Model layers
        dataset: Dataset instance (optional, for function comparison)
        save_path: Path to save figure (optional)
        plot_network: Whether to plot network structure (default: True)
        plot_function: Whether to plot function comparison (default: True)
        plot_splines: Whether to plot splines (default: True)
        title_suffix: Additional text to add to figure title (default: "")
    
    Returns:
        fig: Matplotlib figure
    """
    if len(layers) > 0:
        input_dim = layers[0].d_in
    else:
        input_dim = 1
    num_blocks = len(layers)
    final_dim = getattr(model, 'final_dim', 1)
    
    # Calculate layout based on what we're plotting
    num_rows = 0
    if plot_network or plot_splines:
        num_rows += 1
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
        fig_width = max(12, 3 * total_cols)  # Larger cells when no network plot
    
    fig_height = 7 * num_rows if plot_function else 5
    
    fig = plt.figure(figsize=(fig_width, fig_height))
    
    # Add title with suffix if provided - but don't add it for MNIST to avoid overlap
    if title_suffix and "MNIST" not in title_suffix:
        fig.suptitle(title_suffix, fontsize=14, y=0.98)
    
    # Create layout based on what we're plotting
    if num_rows == 2:
        # Both splines/network and function plots
        gs_top = gridspec.GridSpec(1, total_cols)
        gs_top.update(left=0.05, right=0.95, top=0.95, bottom=0.55, wspace=0.4)
        gs_bottom_top = 0.45
    else:
        # Only splines/network or only function
        gs_top = gridspec.GridSpec(1, total_cols)
        gs_top.update(left=0.05, right=0.95, top=0.95, bottom=0.05, wspace=0.4)
        gs_bottom_top = None
    
    # Plot network structure and/or splines in top row
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
                device = next(layer.phi.parameters()).device
                x_vals = torch.linspace(layer.phi.fixed_in_range[0], layer.phi.fixed_in_range[1], 200).to(device)
                y_vals = layer.phi(x_vals)
            ax_phi.plot(x_vals.cpu(), y_vals.cpu(), 'c-', linewidth=2)
            ax_phi.grid(True, alpha=0.3)
            ax_phi.set_xlabel('Input', fontsize=9)
            ax_phi.set_ylabel('Output', fontsize=9)
            
            # Plot Phi spline
            ax_Phi = fig.add_subplot(gs_top[0, col_Phi])
            ax_Phi.set_title(f"{block_label} $\\Phi^{{({j})}}$", fontsize=11)
            with torch.no_grad():
                if layer.Phi.train_range and (layer.Phi.range_params is not None):
                    dc = layer.Phi.range_params.dc.item()
                    dr = layer.Phi.range_params.dr.item()
                    in_min = dc - dr
                    in_max = dc + dr
                else:
                    in_min, in_max = layer.Phi.fixed_in_range
                device = next(layer.Phi.parameters()).device
                x_vals = torch.linspace(in_min, in_max, 200).to(device)
                y_vals = layer.Phi(x_vals)
            ax_Phi.plot(x_vals.cpu(), y_vals.cpu(), 'm-', linewidth=2)
            ax_Phi.grid(True, alpha=0.3)
            ax_Phi.set_xlabel('Input', fontsize=9)
            ax_Phi.set_ylabel('Output', fontsize=9)
    
    # Plot function comparison if dataset is provided and requested
    if plot_function and dataset is not None and gs_bottom_top is not None:
        if input_dim == 1:
            # Create test points for 1D input
            x_test, y_true = dataset.sample(200, device=next(model.parameters()).device)
            with torch.no_grad():
                y_pred = model(x_test)
            
            if final_dim == 1:
                # For scalar output, plot the target vs. prediction
                gs_bottom = gridspec.GridSpec(1, 1)
                gs_bottom.update(left=0.05, right=0.95, top=gs_bottom_top, bottom=0.05)
                ax_func = fig.add_subplot(gs_bottom[0, 0])
                ax_func.set_title("Function Comparison", fontsize=12)
                ax_func.plot(x_test.cpu(), y_true.cpu(), 'k--', label='Target f(x)')
                ax_func.plot(x_test.cpu(), y_pred.cpu(), 'r-', label='Network output')
                ax_func.grid(True)
                ax_func.legend()
            else:
                # For vector output, plot each dimension separately
                gs_bottom = gridspec.GridSpec(1, final_dim)
                gs_bottom.update(left=0.05, right=0.95, top=gs_bottom_top, bottom=0.05, wspace=0.3)
                
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
            points, y_true = dataset.sample(n * n, device=next(model.parameters()).device)
            
            if final_dim == 1:
                # For scalar output from 2D input, plot target and prediction as surfaces
                gs_bottom = gridspec.GridSpec(1, 2)
                gs_bottom.update(left=0.05, right=0.95, top=gs_bottom_top, bottom=0.05, wspace=0.3)
                
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
                gs_bottom = gridspec.GridSpec(1, final_dim)
                gs_bottom.update(left=0.05, right=0.95, top=gs_bottom_top, bottom=0.05, wspace=0.3)
                
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
            gs_bottom = gridspec.GridSpec(1, 1)
            gs_bottom.update(left=0.05, right=0.95, top=gs_bottom_top, bottom=0.05)
            ax_func = fig.add_subplot(gs_bottom[0, 0])
            plot_high_dim_function(ax_func, model, dataset, next(model.parameters()).device)
    
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