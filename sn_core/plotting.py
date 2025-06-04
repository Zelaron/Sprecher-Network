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


def plot_results(model, layers, dataset=None, save_path=None):
    """
    Plot network structure, splines, and function approximation.
    
    Args:
        model: Trained Sprecher network
        layers: Model layers
        dataset: Dataset instance (optional, for function comparison)
        save_path: Path to save figure (optional)
    
    Returns:
        fig: Matplotlib figure
    """
    if len(layers) > 0:
        input_dim = layers[0].d_in
    else:
        input_dim = 1
    num_blocks = len(layers)
    final_dim = getattr(model, 'final_dim', 1)
    total_cols = 1 + 2 * num_blocks
    plt.rcParams.update({'xtick.labelsize': 8, 'ytick.labelsize': 8})
    
    # Increase figure height to accommodate 3D plots properly
    fig = plt.figure(figsize=(4 * total_cols * 0.75, 14))  # Taller figure
    
    # Create a grid for the top row (network structure and splines)
    width_ratios = [1.0]  # Network structure diagram keeps original width
    for i in range(1, total_cols):
        width_ratios.append(0.5)  # Spline subplots get 50% width
        
    # Create separate gridspecs for top and bottom rows
    gs_top = gridspec.GridSpec(1, total_cols, height_ratios=[1], width_ratios=width_ratios)
    gs_top.update(left=0.05, right=0.95, top=0.95, bottom=0.55, wspace=0.4)
    
    # Plot network structure and splines in top row
    ax_net = fig.add_subplot(gs_top[0, 0])
    plot_network_structure_ax(ax_net, layers, input_dim, final_dim)
    
    for i, layer in enumerate(layers):
        j = i + 1
        col_phi = 2 * i + 1
        col_Phi = 2 * i + 2
        ax_phi = fig.add_subplot(gs_top[0, col_phi])
        block_label = f"Block {j}"
        ax_phi.set_title(f"{block_label} $\\phi^{{({j})}}$", fontsize=11)
        with torch.no_grad():
            device = next(layer.phi.parameters()).device
            x_vals = torch.linspace(layer.phi.fixed_in_range[0], layer.phi.fixed_in_range[1], 200).to(device)
            y_vals = layer.phi(x_vals)
        ax_phi.plot(x_vals.cpu(), y_vals.cpu(), 'c-')
        ax_phi.grid(True)
        
        ax_Phi = fig.add_subplot(gs_top[0, col_Phi])
        ax_Phi.set_title(f"{block_label} $\\Phi^{{({j})}}$", fontsize=11)
        with torch.no_grad():
            if layer.Phi.train_range and (layer.Phi.range_params is not None):
                dc = layer.Phi.range_params.dc.item()  # dc: domain center for Φ
                dr = layer.Phi.range_params.dr.item()  # dr: domain radius for Φ
                in_min = dc - dr
                in_max = dc + dr
            else:
                in_min, in_max = layer.Phi.fixed_in_range
            device = next(layer.Phi.parameters()).device
            x_vals = torch.linspace(in_min, in_max, 200).to(device)
            y_vals = layer.Phi(x_vals)
        ax_Phi.plot(x_vals.cpu(), y_vals.cpu(), 'm-')
        ax_Phi.grid(True)
    
    # Plot function comparison if dataset is provided
    if dataset is not None:
        if input_dim == 1:
            # Create test points for 1D input
            x_test, y_true = dataset.sample(200, device=next(model.parameters()).device)
            with torch.no_grad():
                y_pred = model(x_test)
            
            if final_dim == 1:
                # For scalar output, plot the target vs. prediction
                gs_bottom = gridspec.GridSpec(1, 1)
                gs_bottom.update(left=0.05, right=0.95, top=0.45, bottom=0.05)
                ax_func = fig.add_subplot(gs_bottom[0, 0])
                ax_func.set_title("Function Comparison", fontsize=12)
                ax_func.plot(x_test.cpu(), y_true.cpu(), 'k--', label='Target f(x)')
                ax_func.plot(x_test.cpu(), y_pred.cpu(), 'r-', label='Network output')
                ax_func.grid(True)
                ax_func.legend()
            else:
                # For vector output, plot each dimension separately
                gs_bottom = gridspec.GridSpec(1, final_dim)
                gs_bottom.update(left=0.05, right=0.95, top=0.45, bottom=0.05, wspace=0.3)
                
                for d in range(final_dim):
                    ax_func = fig.add_subplot(gs_bottom[0, d])
                    ax_func.set_title(f"Output dim {d}", fontsize=12)
                    ax_func.plot(x_test.cpu(), y_true[:, d].cpu(), 'k--', label='Target f(x)')
                    ax_func.plot(x_test.cpu(), y_pred[:, d].cpu(), 'r-', label=f'Out dim {d}')
                    ax_func.grid(True)
                    ax_func.legend()
        else:
            # For 2D input, create grid for surface plots
            n = 50
            points, y_true = dataset.sample(n * n, device=next(model.parameters()).device)
            
            if final_dim == 1:
                # For scalar output from 2D input, plot target and prediction as surfaces
                gs_bottom = gridspec.GridSpec(1, 2)
                gs_bottom.update(left=0.05, right=0.95, top=0.45, bottom=0.05, wspace=0.3)
                
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
                gs_bottom.update(left=0.05, right=0.95, top=0.45, bottom=0.05, wspace=0.3)
                
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
    
    if save_path:
        fig.set_size_inches(16, 9) 
        fig.savefig(save_path, dpi=240)
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