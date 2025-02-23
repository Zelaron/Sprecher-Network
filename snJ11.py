import os
import torch
import torch.nn as nn
import numpy as np

# --- Force a non-interactive backend if Tcl/Tk is not properly installed ---
#import matplotlib
#matplotlib.use('Agg')  # Renders plots to files instead of interactive windows

import matplotlib.pyplot as plt
from tqdm import tqdm
from mpl_toolkits.mplot3d import Axes3D  # needed for 3D plotting
import matplotlib.gridspec as gridspec

##############################################################################
#                          CONFIGURABLE SETTINGS                             #
##############################################################################

# 'architecture' defines the hidden-layer sizes (i.e. the output dimension of each block).
# By default, let's do a single block of dimension 5:
architecture = [5,8,5]

# Desired dimension of the final output. 1 => scalar (Sprecher style).
# If > 1 => multi-output. By the logic below, that adds an extra block from architecture[-1]->final_dim.
FINAL_OUTPUT_DIM = 1

# Total training epochs
TOTAL_EPOCHS = 100000

# Whether to save the final plot
SAVE_FINAL_PLOT = True

# Image size in pixels (when saving)
FINAL_IMAGE_WIDTH, FINAL_IMAGE_HEIGHT = 3840, 2160

##############################################################################
#                          TARGET FUNCTION DEFINITION                        #
##############################################################################

def target_function(x):
    """
    Example target function. Adjust as needed.
    For 1D, uncomment the polynomial-like function.
    For 2D, we use an exponential-sinusoidal function on [0,1]^2.
    """
    # 1D Example:
    # return (x[:, [0]] - 0.5)**5 - (x[:, [0]] - (1/3))**3 + (1/5)*(x[:, [0]] - 0.1)**2
    
    # 2D Example:
    return (torch.exp(torch.sin(torch.pi * x[:, [0]]) + x[:, [1]]**2) - 1) / 7

##############################################################################
#                       UTILITY: DETERMINE INPUT DIMENSION                   #
##############################################################################

def get_input_dimension(target_function):
    """
    Tries dimensions 1 to 3 until target_function runs without error.
    Returns the dimension that works.
    """
    for dim in range(1, 4):
        test_x = torch.zeros(1, dim)
        try:
            result = target_function(test_x)
            if torch.isfinite(result).all():
                return dim
        except Exception:
            continue
    raise ValueError("Could not determine input dimension (or dimension > 3 not supported)")

##############################################################################
#                     SET SEEDS FOR REPRODUCIBILITY                          #
##############################################################################

SEED = 45
torch.manual_seed(SEED)
np.random.seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

##############################################################################
#                             SPLINE MODULE                                  #
##############################################################################

class SimpleSpline(nn.Module):
    """
    A piecewise-linear spline, with a user-specified number of knots.
    If monotonic=True, the spline is enforced to be non-decreasing.
    """
    def __init__(self, num_knots=30, in_range=(0, 1), out_range=(0, 1), monotonic=False):
        super().__init__()
        self.num_knots = num_knots
        self.in_min, self.in_max = in_range
        self.out_min, self.out_max = out_range
        self.monotonic = monotonic

        # Register a buffer for the knot positions so they're on the correct device
        self.register_buffer('knots', torch.linspace(self.in_min, self.in_max, num_knots))
        
        # Initialize spline coefficients
        torch.manual_seed(SEED)
        if monotonic:
            # Start with random increments, then do a cumsum
            increments = torch.rand(num_knots) * 0.1
            self.coeffs = nn.Parameter(torch.cumsum(increments, 0))
            with torch.no_grad():
                # Normalize to [out_min, out_max]
                self.coeffs.data = (self.coeffs - self.coeffs.min()) / (self.coeffs.max() - self.coeffs.min())
                self.coeffs.data = self.coeffs * (out_range[1] - out_range[0]) + out_range[0]
        else:
            # General spline: initialize around a linear ramp from out_min to out_max
            self.coeffs = nn.Parameter(
                torch.randn(num_knots) * 0.1 + torch.linspace(out_range[0], out_range[1], num_knots)
            )

    def forward(self, x):
        x = x.to(self.knots.device)
        x = torch.clamp(x, self.in_min, self.in_max)
        intervals = torch.searchsorted(self.knots, x) - 1
        intervals = torch.clamp(intervals, 0, self.num_knots - 2)
        t = (x - self.knots[intervals]) / (self.knots[intervals + 1] - self.knots[intervals])
        
        if self.monotonic:
            sorted_coeffs = torch.sort(self.coeffs)[0]
            return (1 - t) * sorted_coeffs[intervals] + t * sorted_coeffs[intervals + 1]
        else:
            return (1 - t) * self.coeffs[intervals] + t * self.coeffs[intervals + 1]
    
    def get_flatness_penalty(self):
        """
        A penalty on the second difference of the spline coefficients
        to encourage smoothness.
        """
        second_diff = self.coeffs[2:] - 2 * self.coeffs[1:-1] + self.coeffs[:-2]
        return torch.mean(second_diff**2)

##############################################################################
#                          SPRECHER LAYER BLOCK                              #
##############################################################################

class SprecherLayerBlock(nn.Module):
    """
    A single Sprecher block that transforms d_in -> d_out using:
      - monotonic spline phi
      - general spline Phi
      - shift parameter eta
      - weight matrix lambdas
    If is_final=True, we sum over the d_out outputs to produce a scalar.

    The forward pass is:
      1) shift x_i by eta*q
      2) apply phi
      3) multiply by lambdas and sum across i
      4) add q
      5) clamp, apply Phi
      6) if is_final, sum over q
    """
    def __init__(self, d_in, d_out, is_final=False):
        super().__init__()
        self.d_in = d_in
        self.d_out = d_out
        self.is_final = is_final
        
        # Two spline modules: monotonic phi, general Phi
        self.phi = SimpleSpline(num_knots=300, in_range=(0, 1), out_range=(0, 1), monotonic=True)
        self.Phi = SimpleSpline(num_knots=200, in_range=(-10.0, 12.0), out_range=(-10.0, 12.0))
        
        # Initialize Phi to a ramp from out_min to out_max
        with torch.no_grad():
            self.Phi.coeffs.data = torch.linspace(self.Phi.out_min, self.Phi.out_max, self.Phi.num_knots)
        
        # Weight matrix and shift
        self.lambdas = nn.Parameter(torch.ones(d_in, d_out))
        self.eta = nn.Parameter(torch.tensor(1.0 / 100.0))
        
        # Q-values for indexing
        self.register_buffer('q_values', torch.arange(d_out, dtype=torch.float32))

    def forward(self, x):
        # x: (batch_size, d_in)
        x_expanded = x.unsqueeze(-1)  # (batch_size, d_in, 1)
        q = self.q_values.view(1, 1, -1)  # (1, 1, d_out)
        
        # shift inputs by eta*q
        shifted = x_expanded + self.eta * q
        shifted = torch.clamp(shifted, 0, 1)
        
        # apply phi to the shifted inputs
        phi_out = self.phi(shifted)  # (batch_size, d_in, d_out)
        
        # weight by lambdas and sum across d_in
        weighted = phi_out * self.lambdas.unsqueeze(0)  # shape (1, d_in, d_out)
        s = weighted.sum(dim=1) + self.q_values  # shape (batch_size, d_out)
        
        # clamp, then apply Phi
        s = torch.clamp(s, self.Phi.in_min, self.Phi.in_max)
        activated = self.Phi(s)  # (batch_size, d_out)
        
        if self.is_final:
            # final block sums over d_out to produce a scalar
            return activated.sum(dim=1, keepdim=True)
        else:
            return activated

##############################################################################
#                     MULTI-LAYER SPRECHER NETWORK                           #
##############################################################################

class SprecherMultiLayerNetwork(nn.Module):
    """
    We create a sequence of blocks based on:
      - 'architecture': a list of length L of hidden-layer dimensions.
      - 'final_dim': the final output dimension (default=1 => scalar).
    
    RULES:
      1) If architecture == [] (empty), we create exactly 1 block:
         d_in -> final_dim.  If final_dim=1 => is_final=True => sum, else is_final=False.
      2) If architecture has L>=1 elements, we create the first L-1 blocks as non-final.
         For the last entry arch[L-1], call it out_dim:
           - If final_dim == 1 => we create exactly L blocks in total:
               the last block is from arch[L-2]->out_dim (or input_dim->out_dim if L=1),
               is_final=True => sums over out_dim => scalar output.
           - If final_dim > 1 => we create L blocks + 1 more block => L+1 blocks total:
               the last block is non-final => dimension out_dim,
               then an extra block out_dim-> final_dim, also non-final => multi-output dimension final_dim.
    """
    def __init__(self, input_dim, architecture, final_dim=1):
        super().__init__()
        self.input_dim = input_dim
        self.architecture = architecture
        self.final_dim = final_dim
        
        layers = []
        if len(architecture) == 0:
            # Single block from input_dim -> final_dim
            is_final = (final_dim == 1)
            layers.append(SprecherLayerBlock(d_in=input_dim, d_out=final_dim, is_final=is_final))
        
        else:
            L = len(architecture)
            # Create first L-1 blocks as non-final
            for i in range(L-1):
                d_in = input_dim if i==0 else architecture[i-1]
                d_out = architecture[i]
                layers.append(SprecherLayerBlock(d_in=d_in, d_out=d_out, is_final=False))
            
            # Last entry in architecture
            d_in_last = input_dim if L==1 else architecture[L-2]
            d_out_last = architecture[L-1]
            
            if self.final_dim == 1:
                # The last block is final => sum over d_out_last
                layers.append(SprecherLayerBlock(d_in=d_in_last, d_out=d_out_last, is_final=True))
            else:
                # final_dim>1 => we add two blocks
                layers.append(SprecherLayerBlock(d_in=d_in_last, d_out=d_out_last, is_final=False))
                layers.append(SprecherLayerBlock(d_in=d_out_last, d_out=self.final_dim, is_final=False))
        
        self.layers = nn.ModuleList(layers)
    
    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

##############################################################################
#                         DATA GENERATION                                    #
##############################################################################

def generate_data(n_samples=32):
    """
    Generates a grid of points in [0,1]^d, then applies target_function.
    """
    input_dim = get_input_dimension(target_function)
    if input_dim == 1:
        x = torch.linspace(0, 1, n_samples).unsqueeze(1)
        return x, target_function(x), input_dim
    else:
        x = torch.linspace(0, 1, n_samples)
        y = torch.linspace(0, 1, n_samples)
        X, Y = torch.meshgrid(x, y, indexing='ij')
        points = torch.stack([X.flatten(), Y.flatten()], dim=1)
        return points, target_function(points), input_dim

##############################################################################
#                           TRAINING FUNCTION                                #
##############################################################################

def train_network(target_function, architecture, total_epochs=100000, print_every=10000, device="cpu", final_dim=1):
    x_train, y_train, input_dim = generate_data()
    x_train = x_train.to(device)
    y_train = y_train.to(device)
    
    model = SprecherMultiLayerNetwork(input_dim=input_dim,
                                      architecture=architecture,
                                      final_dim=final_dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0003, weight_decay=1e-7)
    
    losses = []
    best_loss = float("inf")
    best_state = None
    
    pbar = tqdm(range(total_epochs), desc="Training Network")
    for epoch in pbar:
        optimizer.zero_grad()
        output = model(x_train)
        mse_loss = torch.mean((output - y_train) ** 2)
        
        # Add a smoothness penalty for each block
        flatness_penalty = 0.0
        for layer in model.layers:
            flatness_penalty += 0.01 * layer.phi.get_flatness_penalty() + 0.1 * layer.Phi.get_flatness_penalty()
        
        loss = mse_loss + flatness_penalty
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
        pbar.set_postfix({'loss': f'{loss.item():.2e}'})
        
        if (epoch + 1) % print_every == 0:
            print(f"Epoch {epoch+1}: Loss = {loss.item():.4e}")
        
        # Track best model
        if loss.item() < best_loss:
            best_loss = loss.item()
            best_state = model.state_dict()
    
    # Load best state
    if best_state is not None:
        model.load_state_dict(best_state)
    return model, losses, model.layers

##############################################################################
#                    PLOTTING: NETWORK SCHEMATIC + SPLINES                   #
##############################################################################

def plot_network_structure_ax(ax, layers, input_dim, final_dim=1):
    """
    Plots a schematic of the network structure on the given axis.
    - Each block is labeled "Block j" or "Block j (final)" with the corresponding
      φ^(j) and Φ^(j) functions noted.
    - If the block is final and final_dim=1, we draw an extra single "output" node
      to visually indicate the summation from d_out -> 1.
    """
    num_blocks = len(layers)
    if num_blocks == 0:
        ax.text(0.5, 0.5, "No network", ha='center', va='center', fontsize=12)
        ax.axis('off')
        return
    
    # Prepare x positions for each "layer" (input + each block)
    layer_x = np.linspace(0.2, 0.8, num_blocks + 1)
    
    if input_dim is None and num_blocks > 0:
        input_dim = layers[0].d_in
    
    # Plot input nodes
    if input_dim == 1:
        input_y = [0.5]
    else:
        input_y = np.linspace(0.2, 0.8, input_dim)
    
    # Draw input nodes
    for i, y in enumerate(input_y):
        ax.scatter(layer_x[0], y, c='black', s=100)
        ax.text(layer_x[0] - 0.05, y, f'$x_{{{i+1}}}$', ha='right', fontsize=12)
    
    prev_y = input_y
    
    for block_index, block in enumerate(layers):
        j = block_index + 1  # 1-based block index
        d_out = block.d_out
        
        # The block's outputs
        # We'll place them at layer_x[block_index + 1]
        new_x = layer_x[block_index + 1]
        
        color = 'red' if block.is_final else 'blue'
        
        # We always draw the block's d_out nodes (the "shifted sum" channels).
        # Then, if it's final and final_dim=1, we draw an extra node for the single output.
        
        # Position the block's d_out nodes
        if d_out == 1:
            new_y = [0.5]
        else:
            new_y = np.linspace(0.2, 0.8, d_out)
        
        # Draw them
        for ny in new_y:
            ax.scatter(new_x, ny, c=color, s=100)
        
        # Connect from previous layer to these
        for py in prev_y:
            for ny in new_y:
                ax.plot([layer_x[block_index], new_x], [py, ny], 'k-', alpha=0.5)
        
        # Label the block
        block_label = f"Block {j}"
        if block.is_final:
            block_label += " (final)"
        
        # Put block label near the top
        mid_x = 0.5 * (layer_x[block_index] + layer_x[block_index + 1])
        ax.text(mid_x, 0.92, block_label, ha='center', fontsize=10, color='green')
        
        # Put φ^(j) and Φ^(j) below
        ax.text(mid_x, 0.05,
                f"$\\phi^{{({j})}},\\,\\Phi^{{({j})}}$",
                ha='center', fontsize=10, color='green')
        
        # If this is the final block AND final_dim=1, draw a single output node
        # to illustrate the summation from d_out -> 1
        if block.is_final and final_dim == 1:
            # We'll place the sum node slightly to the right of new_x
            sum_x = new_x + 0.07
            sum_y = 0.5  # center vertically or we could do average of new_y
            ax.scatter(sum_x, sum_y, c='red', s=100)
            # Connect each d_out node to the sum node
            for ny in new_y:
                ax.plot([new_x, sum_x], [ny, sum_y], 'k-', alpha=0.5)
            ax.text(sum_x + 0.02, sum_y, 'output', ha='left', fontsize=12)
            
            # That is truly the final node, so no next block's edges
            # We'll skip updating prev_y, because there's no next block
            prev_y = [sum_y]
        
        else:
            # Not final => just update prev_y
            prev_y = new_y
    
    ax.set_title("Network Structure", fontsize=12)
    ax.axis('off')


def plot_results(model, layers):
    """
    Plots the network structure and the individual spline functions from each Sprecher block.
    For each block, both the monotonic spline φ^(j) and the general spline Φ^(j) are plotted
    in separate subplots. The top row includes the network schematic (first column)
    and two subplots per block. The bottom row shows either a 1D function comparison
    or a 3D surface plot for 2D input. If the final output dimension > 1, multiple
    lines/surfaces are plotted in the bottom row.
    """
    if len(layers) > 0:
        input_dim = layers[0].d_in
    else:
        input_dim = 1
    
    num_blocks = len(layers)
    final_dim = getattr(model, 'final_dim', 1)
    
    # 1 column for the network structure, plus 2 columns per block for φ and Φ
    total_cols = 1 + 2 * num_blocks  
    fig = plt.figure(figsize=(4 * total_cols, 10))
    gs = gridspec.GridSpec(2, total_cols, height_ratios=[1, 1.2])
    
    # First column: Network structure schematic
    ax_net = fig.add_subplot(gs[0, 0])
    plot_network_structure_ax(ax_net, layers, input_dim, final_dim)
    
    # For each block, plot φ^(j) and Φ^(j) in separate subplots
    for i, layer in enumerate(layers):
        j = i + 1
        col_phi = 2 * i + 1
        col_Phi = 2 * i + 2
        
        # Plot φ^(j)
        ax_phi = fig.add_subplot(gs[0, col_phi])
        block_label = f"Block {j}"
        if layer.is_final:
            block_label += " (final)"
        ax_phi.set_title(f"{block_label} $\\phi^{{({j})}}$", fontsize=12)
        with torch.no_grad():
            x_vals = torch.linspace(layer.phi.in_min, layer.phi.in_max, 200).to(layer.phi.knots.device)
            y_vals = layer.phi(x_vals)
        ax_phi.plot(x_vals.cpu(), y_vals.cpu(), 'c-')
        ax_phi.grid(True)
        
        # Plot Φ^(j)
        ax_Phi = fig.add_subplot(gs[0, col_Phi])
        ax_Phi.set_title(f"{block_label} $\\Phi^{{({j})}}$", fontsize=12)
        with torch.no_grad():
            x_vals = torch.linspace(layer.Phi.in_min, layer.Phi.in_max, 200).to(layer.Phi.knots.device)
            y_vals = layer.Phi(x_vals)
        ax_Phi.plot(x_vals.cpu(), y_vals.cpu(), 'm-')
        ax_Phi.grid(True)
    
    # Bottom row: Function comparison
    if input_dim == 1:
        # 1D input
        x_test = torch.linspace(0, 1, 200).reshape(-1, 1)
        x_test = x_test.to(next(model.parameters()).device)
        y_true = target_function(x_test)
        with torch.no_grad():
            y_pred = model(x_test)  # shape (200, final_dim)
        
        if final_dim == 1:
            # Single output => direct comparison
            ax_func = fig.add_subplot(gs[1, :])
            ax_func.set_title("Function Comparison", fontsize=12)
            ax_func.plot(x_test.cpu(), y_true.cpu(), 'k--', label='Target f(x)')
            ax_func.plot(x_test.cpu(), y_pred.cpu(), 'r-', label='Network output')
            ax_func.grid(True)
            ax_func.legend()
        else:
            # Multiple outputs => plot each dimension in a separate line
            for d in range(final_dim):
                ax_func = fig.add_subplot(gs[1, d % total_cols])
                ax_func.set_title(f"Output dim {d}", fontsize=12)
                ax_func.plot(x_test.cpu(), y_true.cpu(), 'k--', label='Target f(x)')
                ax_func.plot(x_test.cpu(), y_pred[:, d].cpu(), 'r-', label=f'Out dim {d}')
                ax_func.grid(True)
                ax_func.legend()
    
    else:
        # 2D input => 3D surface
        n = 50
        x = torch.linspace(0, 1, n)
        y = torch.linspace(0, 1, n)
        X, Y = torch.meshgrid(x, y, indexing='ij')
        points = torch.stack([X.flatten(), Y.flatten()], dim=1)
        
        Z_true = target_function(points).reshape(n, n)
        
        if final_dim == 1:
            # Single output => standard 3D surface for target vs network
            ax_target = fig.add_subplot(gs[1, :total_cols//2], projection='3d')
            ax_target.set_title("Target Function", fontsize=12)
            ax_target.plot_surface(X.cpu().numpy(), Y.cpu().numpy(), Z_true.cpu().numpy(), cmap='viridis')
            
            ax_output = fig.add_subplot(gs[1, total_cols//2:], projection='3d')
            ax_output.set_title("Network Output", fontsize=12)
            with torch.no_grad():
                Z_pred = model(points.to(next(model.parameters()).device)).reshape(n, n)
            ax_output.plot_surface(X.cpu().numpy(), Y.cpu().numpy(), Z_pred.cpu().numpy(), cmap='viridis')
        
        else:
            # Multiple outputs => plot multiple surfaces
            sub_cols = max(1, final_dim)
            width_per_plot = total_cols // sub_cols if sub_cols <= total_cols else 1
            with torch.no_grad():
                Z_pred_all = model(points.to(next(model.parameters()).device))  # (n*n, final_dim)
            
            for d in range(final_dim):
                col_start = d * width_per_plot
                col_end = col_start + width_per_plot
                if col_end > total_cols:
                    col_end = total_cols
                ax_out = fig.add_subplot(gs[1, col_start:col_end], projection='3d')
                ax_out.set_title(f"Output dim {d}", fontsize=12)
                Z_pred_d = Z_pred_all[:, d].reshape(n, n)
                ax_out.plot_surface(X.cpu().numpy(), Y.cpu().numpy(), Z_pred_d.cpu().numpy(), cmap='viridis')
    
    plt.tight_layout()
    return fig

##############################################################################
#                              MAIN EXECUTION                                #
##############################################################################

if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Build and train the network
    model, losses, layers = train_network(
        target_function,
        architecture=architecture,
        total_epochs=TOTAL_EPOCHS,
        print_every=TOTAL_EPOCHS // 10,
        device=device,
        final_dim=FINAL_OUTPUT_DIM
    )

    # Plot the combined results and capture the figure
    fig_results = plot_results(model, layers)

    # Optionally save the final plot with the desired resolution
    if SAVE_FINAL_PLOT:
        os.makedirs("plots", exist_ok=True)
        input_dim_guess = get_input_dimension(target_function)
        prefix = "OneVar" if input_dim_guess == 1 else "TwoVars"
        arch_str = "-".join(map(str, architecture)) if len(architecture) > 0 else "None"
        filename = f"{prefix}-{arch_str}-{TOTAL_EPOCHS}-epochs-outdim{FINAL_OUTPUT_DIM}.png"
        filepath = os.path.join("plots", filename)
        # set figure size so that, with dpi=240, the saved image is 3840x2160
        fig_results.set_size_inches(16, 9)
        fig_results.savefig(filepath, dpi=240)
        print(f"Final plot saved as {filepath}")

    # Show or close the figure (depending on the backend)
    plt.show()

    # Plot the loss curve
    plt.figure(figsize=(8, 5))
    plt.plot(losses, label="Training Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Loss Curve")
    plt.legend()
    if SAVE_FINAL_PLOT:
        plt.savefig("plots/loss_curve.png", dpi=150)
    plt.show()