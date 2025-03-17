import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# --- Force a non-interactive backend if Tcl/Tk is not properly installed ---
import matplotlib
matplotlib.use('Agg')  # Renders plots to files instead of interactive windows

import matplotlib.pyplot as plt
from tqdm import tqdm
from mpl_toolkits.mplot3d import Axes3D  # needed for 3D plotting
import matplotlib.gridspec as gridspec

##############################################################################
#                          CONFIGURABLE SETTINGS                             #
##############################################################################

# 'architecture' defines the hidden-layer sizes (i.e. the output dimension of each block).
architecture = [17, 17, 17, 17]

# Desired dimension of the final output.
FINAL_OUTPUT_DIM = 1

# Total training epochs
TOTAL_EPOCHS = 500000

# Whether to save the final plot
SAVE_FINAL_PLOT = True

# Image size in pixels (when saving)
FINAL_IMAGE_WIDTH, FINAL_IMAGE_HEIGHT = 3840, 2160

##############################################################################
#                          TARGET FUNCTION DEFINITION                        #
##############################################################################

def target_function(x):
    # 2D scalar function:
    return torch.exp(torch.sin(11*x[:, [0]])) + 3*x[:, [1]] + 4*torch.sin(8*x[:, [1]])
    
##############################################################################
#                     UTILITY: DETERMINE INPUT DIMENSION                   #
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
#                              SPLINE MODULE                                 #
##############################################################################

class SimpleSpline(nn.Module):
    """
    A piecewise-linear spline, with a user-specified number of knots.
    If monotonic=True, the spline is enforced to be non-decreasing.
    
    This implementation supports an adaptive range for non-monotonic splines.
    When adaptive_range is True, the domain (in_range) and codomain (out_range)
    are parameterized by a center and a radius, which are trained along with the spline coefficients.
    
    For the monotonic splines (e.g., self.phi), adaptive_range should be left False.
    """
    def __init__(self, num_knots=30, in_range=(0, 1), out_range=(0, 1), monotonic=False, adaptive_range=False):
        super().__init__()
        self.num_knots = num_knots
        self.monotonic = monotonic
        self.adaptive_range = adaptive_range

        if self.adaptive_range:
            # Register trainable parameters for the input range (domain)
            in_center_init = (in_range[0] + in_range[1]) / 2
            in_radius_init = (in_range[1] - in_range[0]) / 2
            self.in_center = nn.Parameter(torch.tensor(in_center_init, dtype=torch.float32))
            self.in_radius_param = nn.Parameter(torch.tensor(in_radius_init, dtype=torch.float32))
            # Register trainable parameters for the output range (codomain)
            out_center_init = (out_range[0] + out_range[1]) / 2
            out_radius_init = (out_range[1] - out_range[0]) / 2
            self.out_center = nn.Parameter(torch.tensor(out_center_init, dtype=torch.float32))
            self.out_radius_param = nn.Parameter(torch.tensor(out_radius_init, dtype=torch.float32))
        else:
            self.in_range = in_range
            self.out_range = out_range
            # Precompute knot positions for fixed range
            self.register_buffer('knots', torch.linspace(in_range[0], in_range[1], num_knots))

        # Initialize spline coefficients
        torch.manual_seed(SEED)
        if monotonic:
            increments = torch.rand(num_knots) * 0.1
            self.coeffs = nn.Parameter(torch.cumsum(increments, 0))
            with torch.no_grad():
                self.coeffs.data = (self.coeffs - self.coeffs.min()) / (self.coeffs.max() - self.coeffs.min())
                self.coeffs.data = self.coeffs * (out_range[1] - out_range[0]) + out_range[0]
        else:
            # Initialize around a linear ramp from out_range[0] to out_range[1]
            init_ramp = torch.linspace(out_range[0], out_range[1], num_knots)
            self.coeffs = nn.Parameter(torch.randn(num_knots) * 0.1 + init_ramp)

    @property
    def current_in_range(self):
        if self.adaptive_range:
            in_radius = F.softplus(self.in_radius_param)
            return (self.in_center - in_radius, self.in_center + in_radius)
        else:
            return self.in_range

    @property
    def current_out_range(self):
        if self.adaptive_range:
            out_radius = F.softplus(self.out_radius_param)
            return (self.out_center - out_radius, self.out_center + out_radius)
        else:
            return self.out_range

    def forward(self, x):
        if self.adaptive_range:
            in_min, in_max = self.current_in_range
            # Create base knots between 0 and 1 and then scale to current domain
            base_knots = torch.linspace(0, 1, self.num_knots, device=x.device)
            knots = in_min + (in_max - in_min) * base_knots
        else:
            in_min, in_max = self.in_range
            knots = self.knots

        # Clamp input x to the domain
        x = torch.clamp(x, in_min, in_max)
        # Determine the interval indices for each x value
        intervals = torch.searchsorted(knots, x) - 1
        intervals = torch.clamp(intervals, 0, self.num_knots - 2)
        # Compute relative position in each interval
        t = (x - knots[intervals]) / (knots[intervals + 1] - knots[intervals])
        
        if self.monotonic:
            sorted_coeffs = torch.sort(self.coeffs)[0]
            result = (1 - t) * sorted_coeffs[intervals] + t * sorted_coeffs[intervals + 1]
        else:
            result = (1 - t) * self.coeffs[intervals] + t * self.coeffs[intervals + 1]
        return result
    
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
      - general spline Phi (with adaptive range training if desired)
      - shift parameter eta
      - weight matrix lambdas
    If is_final=True, we sum over the d_out outputs to produce a scalar.
    """
    def __init__(self, d_in, d_out, is_final=False):
        super().__init__()
        self.d_in = d_in
        self.d_out = d_out
        self.is_final = is_final
        
        # Two spline modules: monotonic phi and adaptive general Phi
        self.phi = SimpleSpline(num_knots=300, in_range=(0, 1), out_range=(0, 1), monotonic=True, adaptive_range=False)
        # For the general spline, we use adaptive_range=True so that the in_range and out_range are trainable.
        self.Phi = SimpleSpline(num_knots=200, in_range=(-10.0, 12.0), out_range=(-10.0, 12.0), monotonic=False, adaptive_range=True)
        
        # Initialize Phi to a ramp from current out_min to current out_max (using initial values)
        with torch.no_grad():
            out_range = self.Phi.current_out_range
            self.Phi.coeffs.data = torch.linspace(out_range[0].item(), out_range[1].item(), self.Phi.num_knots)
        
        # Weight matrix and shift parameter
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
        
        # clamp s using Phi's current domain
        if self.Phi.adaptive_range:
            Phi_in_min, Phi_in_max = self.Phi.current_in_range
        else:
            Phi_in_min, Phi_in_max = self.Phi.in_range
        s = torch.clamp(s, Phi_in_min, Phi_in_max)
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
    Constructs a sequence of Sprecher blocks based on:
      - 'architecture': a list of hidden-layer dimensions.
      - 'final_dim': the final output dimension.
    
    Rules:
      1) If architecture == [] (empty), create a single block: d_in -> final_dim.
      2) If architecture has L>=1 elements:
         - For final_dim == 1, create L blocks with the last block marked as final.
         - For final_dim > 1, create L+1 blocks with an extra output block.
    """
    def __init__(self, input_dim, architecture, final_dim=1):
        super().__init__()
        self.input_dim = input_dim
        self.architecture = architecture
        self.final_dim = final_dim
        
        layers = []
        if len(architecture) == 0:
            is_final = (final_dim == 1)
            layers.append(SprecherLayerBlock(d_in=input_dim, d_out=final_dim, is_final=is_final))
        else:
            L = len(architecture)
            for i in range(L-1):
                d_in = input_dim if i == 0 else architecture[i-1]
                d_out = architecture[i]
                layers.append(SprecherLayerBlock(d_in=d_in, d_out=d_out, is_final=False))
            
            d_in_last = input_dim if L == 1 else architecture[L-2]
            d_out_last = architecture[L-1]
            
            if self.final_dim == 1:
                layers.append(SprecherLayerBlock(d_in=d_in_last, d_out=d_out_last, is_final=True))
            else:
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
    else:
        x = torch.linspace(0, 1, n_samples)
        y = torch.linspace(0, 1, n_samples)
        X, Y = torch.meshgrid(x, y, indexing='ij')
        x = torch.stack([X.flatten(), Y.flatten()], dim=1)
    
    y = target_function(x)
    if y.dim() == 1:
        y = y.unsqueeze(1)
    return x, y, input_dim

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
        
        if loss.item() < best_loss:
            best_loss = loss.item()
            best_state = model.state_dict()
    
    if best_state is not None:
        model.load_state_dict(best_state)
    return model, losses, model.layers

##############################################################################
#                    PLOTTING: NETWORK SCHEMATIC + SPLINES                   #
##############################################################################

def plot_network_structure_ax(ax, layers, input_dim, final_dim=1):
    num_blocks = len(layers)
    if num_blocks == 0:
        ax.text(0.5, 0.5, "No network", ha='center', va='center', fontsize=12)
        ax.axis('off')
        return
    
    layer_x = np.linspace(0.2, 0.8, num_blocks + 1)
    
    if input_dim is None and num_blocks > 0:
        input_dim = layers[0].d_in
    
    if input_dim == 1:
        input_y = [0.5]
    else:
        input_y = np.linspace(0.2, 0.8, input_dim)
    
    for i, y in enumerate(input_y):
        ax.scatter(layer_x[0], y, c='black', s=100)
        ax.text(layer_x[0] - 0.05, y, f'$x_{{{i+1}}}$', ha='right', fontsize=12)
    
    prev_y = input_y
    
    for block_index, block in enumerate(layers):
        j = block_index + 1
        d_out = block.d_out
        
        new_x = layer_x[block_index + 1]
        color = 'red' if block.is_final else 'blue'
        
        if d_out == 1:
            new_y = [0.5]
        else:
            new_y = np.linspace(0.2, 0.8, d_out)
        
        for ny in new_y:
            ax.scatter(new_x, ny, c=color, s=100)
        
        for py in prev_y:
            for ny in new_y:
                ax.plot([layer_x[block_index], new_x], [py, ny], 'k-', alpha=0.5)
        
        block_label = f"Block {j}"
        if block.is_final:
            block_label += " (final)"
        
        mid_x = 0.5 * (layer_x[block_index] + new_x)
        ax.text(mid_x, 0.92, block_label, ha='center', fontsize=10, color='green')
        ax.text(mid_x, 0.05,
                f"$\\phi^{{({j})}},\\,\\Phi^{{({j})}}$",
                ha='center', fontsize=10, color='green')
        
        if block.is_final and final_dim == 1:
            sum_x = new_x + 0.07
            sum_y = 0.5
            ax.scatter(sum_x, sum_y, c='red', s=100)
            for ny in new_y:
                ax.plot([new_x, sum_x], [ny, sum_y], 'k-', alpha=0.5)
            ax.text(sum_x + 0.02, sum_y, 'output', ha='left', fontsize=12)
            prev_y = [sum_y]
        else:
            prev_y = new_y
    
    ax.set_title("Network Structure", fontsize=12)
    ax.axis('off')

def plot_results(model, layers):
    if len(layers) > 0:
        input_dim = layers[0].d_in
    else:
        input_dim = 1
    
    num_blocks = len(layers)
    final_dim = getattr(model, 'final_dim', 1)
    
    total_cols = 1 + 2 * num_blocks  
    fig = plt.figure(figsize=(4 * total_cols, 10))
    gs = gridspec.GridSpec(2, total_cols, height_ratios=[1, 1.2])
    
    ax_net = fig.add_subplot(gs[0, 0])
    plot_network_structure_ax(ax_net, layers, input_dim, final_dim)
    
    for i, layer in enumerate(layers):
        j = i + 1
        col_phi = 2 * i + 1
        col_Phi = 2 * i + 2
        
        ax_phi = fig.add_subplot(gs[0, col_phi])
        block_label = f"Block {j}"
        if layer.is_final:
            block_label += " (final)"
        ax_phi.set_title(f"{block_label} $\\phi^{{({j})}}$", fontsize=12)
        with torch.no_grad():
            # For self.phi, use its fixed range and knots buffer
            x_vals = torch.linspace(layer.phi.in_range[0], layer.phi.in_range[1], 200).to(layer.phi.knots.device)
            y_vals = layer.phi(x_vals)
        ax_phi.plot(x_vals.cpu(), y_vals.cpu(), 'c-')
        ax_phi.grid(True)
        
        ax_Phi = fig.add_subplot(gs[0, col_Phi])
        ax_Phi.set_title(f"{block_label} $\\Phi^{{({j})}}$", fontsize=12)
        with torch.no_grad():
            if layer.Phi.adaptive_range:
                in_min, in_max = layer.Phi.current_in_range
                x_vals = torch.linspace(in_min.item(), in_max.item(), 200).to(layer.Phi.coeffs.device)
            else:
                x_vals = torch.linspace(layer.Phi.in_range[0], layer.Phi.in_range[1], 200).to(layer.Phi.knots.device)
            y_vals = layer.Phi(x_vals)
        ax_Phi.plot(x_vals.cpu(), y_vals.cpu(), 'm-')
        ax_Phi.grid(True)
    
    if input_dim == 1:
        x_test = torch.linspace(0, 1, 200).reshape(-1, 1).to(next(model.parameters()).device)
        y_true = target_function(x_test)
        if y_true.dim() == 1:
            y_true = y_true.unsqueeze(1)
        with torch.no_grad():
            y_pred = model(x_test)
        
        if final_dim == 1:
            ax_func = fig.add_subplot(gs[1, :])
            ax_func.set_title("Function Comparison", fontsize=12)
            ax_func.plot(x_test.cpu(), y_true.cpu(), 'k--', label='Target f(x)')
            ax_func.plot(x_test.cpu(), y_pred.cpu(), 'r-', label='Network output')
            ax_func.grid(True)
            ax_func.legend()
        else:
            for d in range(final_dim):
                ax_func = fig.add_subplot(gs[1, d % total_cols])
                ax_func.set_title(f"Output dim {d}", fontsize=12)
                ax_func.plot(x_test.cpu(), y_true[:, d].cpu(), 'k--', label='Target f(x)')
                ax_func.plot(x_test.cpu(), y_pred[:, d].cpu(), 'r-', label=f'Out dim {d}')
                ax_func.grid(True)
                ax_func.legend()
    
    else:
        n = 50
        x = torch.linspace(0, 1, n)
        y = torch.linspace(0, 1, n)
        X, Y = torch.meshgrid(x, y, indexing='ij')
        points = torch.stack([X.flatten(), Y.flatten()], dim=1)
        
        if final_dim == 1:
            Z_true = target_function(points).reshape(n, n)
            ax_target = fig.add_subplot(gs[1, :total_cols//2], projection='3d')
            ax_target.set_title("Target Function", fontsize=12)
            ax_target.plot_surface(X.cpu().numpy(), Y.cpu().numpy(), Z_true.cpu().numpy(), cmap='viridis')
            
            ax_output = fig.add_subplot(gs[1, total_cols//2:], projection='3d')
            ax_output.set_title("Network Output", fontsize=12)
            with torch.no_grad():
                Z_pred = model(points.to(next(model.parameters()).device)).reshape(n, n)
            ax_output.plot_surface(X.cpu().numpy(), Y.cpu().numpy(), Z_pred.cpu().numpy(), cmap='viridis')
        else:
            y_true_all = target_function(points)
            if y_true_all.dim() == 1:
                y_true_all = y_true_all.unsqueeze(1)
            sub_cols = max(1, final_dim)
            width_per_plot = total_cols // sub_cols if sub_cols <= total_cols else 1
            with torch.no_grad():
                Z_pred_all = model(points.to(next(model.parameters()).device))
            for d in range(final_dim):
                col_start = d * width_per_plot
                col_end = col_start + width_per_plot
                if col_end > total_cols:
                    col_end = total_cols
                ax_out = fig.add_subplot(gs[1, col_start:col_end], projection='3d')
                ax_out.set_title(f"Output dim {d}", fontsize=12)
                Z_pred_d = Z_pred_all[:, d].reshape(n, n)
                Z_true_d = y_true_all[:, d].reshape(n, n)
                ax_out.plot_surface(X.cpu().numpy(), Y.cpu().numpy(), Z_true_d.cpu().numpy(),
                                    cmap='viridis', alpha=0.5)
                ax_out.plot_surface(X.cpu().numpy(), Y.cpu().numpy(), Z_pred_d.cpu().numpy(),
                                    cmap='autumn', alpha=0.5)
    
    plt.tight_layout()
    return fig

##############################################################################
#                              MAIN EXECUTION                                #
##############################################################################

if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model, losses, layers = train_network(
        target_function,
        architecture=architecture,
        total_epochs=TOTAL_EPOCHS,
        print_every=TOTAL_EPOCHS // 10,
        device=device,
        final_dim=FINAL_OUTPUT_DIM
    )

    fig_results = plot_results(model, layers)

    if SAVE_FINAL_PLOT:
        os.makedirs("plots", exist_ok=True)
        input_dim_guess = get_input_dimension(target_function)
        prefix = "OneVar" if input_dim_guess == 1 else "TwoVars"
        arch_str = "-".join(map(str, architecture)) if len(architecture) > 0 else "None"
        filename = f"{prefix}-{arch_str}-{TOTAL_EPOCHS}-epochs-outdim{FINAL_OUTPUT_DIM}.png"
        filepath = os.path.join("plots", filename)
        fig_results.set_size_inches(16, 9)
        fig_results.savefig(filepath, dpi=240)
        print(f"Final plot saved as {filepath}")

    plt.show()

    plt.figure(figsize=(8, 5))
    plt.plot(losses, label="Training Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Loss Curve")
    plt.legend()
    if SAVE_FINAL_PLOT:
        plt.savefig("plots/loss_curve.png", dpi=150)
    plt.show()
