import os
import torch
import torch.nn as nn
import numpy as np
# --- Force a non-interactive backend if Tcl/Tk is not properly installed ---
# import matplotlib
# matplotlib.use('Agg') # Renders plots to files instead of interactive windows
import matplotlib.pyplot as plt
from tqdm import tqdm
from mpl_toolkits.mplot3d import Axes3D  # needed for 3D plotting
import matplotlib.gridspec as gridspec

##############################################################################
#                          CONFIGURABLE SETTINGS                             #
##############################################################################

# 'architecture' defines the hidden-layer sizes.
# architecture = [17, 17, 17, 17] # Works decently for a function of two variables
architecture = [10] # Good for a function of one variable

# Total training epochs
TOTAL_EPOCHS = 500000

# Whether to save the final plot
SAVE_FINAL_PLOT = True

# Image size in pixels (when saving)
FINAL_IMAGE_WIDTH, FINAL_IMAGE_HEIGHT = 3840, 2160

# Fixed seed for reproducibility
SEED = 45

# Variance penalty weight.
LAMBDA_VAR = 1.0

# Global outer spline (Φ) configuration.
# You can specify different ranges for the default input (domain) and output (codomain) of Φ, which will be trained.
PHI_IN_RANGE = (-10.0, 10.0)
PHI_OUT_RANGE = (-10.0, 10.0)

# Number of knots (control points defining spline segments) for the inner monotonic splines (φ)
PHI_KNOTS = 300

# Number of knots for the outer general splines (Φ)
PHI_CAPITAL_KNOTS = 200

##############################################################################
#                          TARGET FUNCTION DEFINITION                        #
##############################################################################

def target_function(x):
    # 1D scalar function example:
    return (x[:, [0]] - 0.5)**5 - (x[:, [0]] - (1/3))**3 + (1/5)*(x[:, [0]] - 0.1)**2
    
    # 2D scalar function example:
    # return torch.exp(torch.sin(11 * x[:, [0]])) + 3 * x[:, [1]] + 4 * torch.sin(8 * x[:, [1]])

    # 2D vector function example:
    # return torch.cat([
    #     (torch.exp(torch.sin(torch.pi * x[:, [0]]) + x[:, [1]]**2) - 1) / 7,
    #     (1/4)*x[:, [1]] + (1/5)*x[:, [1]]**2 - x[:, [0]]**3 + (1/5)*torch.sin(7*x[:, [0]])
    # ], dim=1)

##############################################################################
#                     UTILITY: DETERMINE INPUT DIMENSION                     #
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
#                     UTILITY: DETERMINE OUTPUT DIMENSION                    #
##############################################################################

def get_output_dimension(target_function, input_dim):
    """
    Determines the output dimension of the target function.
    """
    test_x = torch.zeros(1, input_dim)
    result = target_function(test_x)
    
    # Handle scalar outputs (1D tensors)
    if result.dim() == 1:
        return 1
    
    # For 2D tensors, return the size of the second dimension
    return result.shape[1]

##############################################################################
#                             SET SEEDS FOR REPRODUCIBILITY                  #
##############################################################################

torch.manual_seed(SEED)
np.random.seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

##############################################################################
#                        GLOBAL PHI RANGE PARAMETERS                         #
##############################################################################

class PhiRangeParams(nn.Module):
    """
    Global, trainable range parameters for all Φ splines.
    The domain and codomain are defined via a center and a radius:
      in_range = (in_center - in_radius, in_center + in_radius)
      out_range = (out_center - out_radius, out_center + out_radius)
    """
    def __init__(self, in_center=0.0, in_radius=1.0, out_center=0.0, out_radius=1.0):
        super().__init__()
        self.in_center = nn.Parameter(torch.tensor(in_center, dtype=torch.float32))
        self.in_radius = nn.Parameter(torch.tensor(in_radius, dtype=torch.float32))
        self.out_center = nn.Parameter(torch.tensor(out_center, dtype=torch.float32))
        self.out_radius = nn.Parameter(torch.tensor(out_radius, dtype=torch.float32))

##############################################################################
#                             SIMPLE SPLINE MODULE                           #
##############################################################################

class SimpleSpline(nn.Module):
    """
    A piecewise-linear spline.
    
    If train_range is False, the domains and codomains of the splines Phi are not trained.
    If train_range is True and a valid range_params (of type PhiRangeParams)
    is provided, then:
      - The input range (domain) is computed on the fly from normalized knots
        and the current in_center and in_radius.
      - After linear interpolation, the raw output is normalized using the
        initial (stored) min and max and then re-scaled to the current output range.
    """
    def __init__(self, num_knots=30, in_range=(0, 1), out_range=(0, 1), monotonic=False,
                 train_range=False, range_params=None):
        super().__init__()
        self.num_knots = num_knots
        self.monotonic = monotonic
        self.train_range = train_range
        self.fixed_in_range = in_range
        self.fixed_out_range = out_range
        self.range_params = range_params

        if self.train_range and (self.range_params is not None):
            # Create normalized knots in [0,1]; these will be scaled at runtime.
            self.register_buffer('normalized_knots', torch.linspace(0, 1, num_knots))
        else:
            self.in_min, self.in_max = in_range
            self.out_min, self.out_max = out_range
            self.register_buffer('knots', torch.linspace(self.in_min, self.in_max, num_knots))
        
        # Initialize spline coefficients
        torch.manual_seed(SEED)
        if monotonic:
            # For monotonic splines (φ), initialize with increasing values
            increments = torch.rand(num_knots) * 0.1
            self.coeffs = nn.Parameter(torch.cumsum(increments, 0))
            with torch.no_grad():
                if self.train_range and (self.range_params is not None):
                    out_center = self.range_params.out_center.item()
                    out_radius = self.range_params.out_radius.item()
                    out_min = out_center - out_radius
                    out_max = out_center + out_radius
                else:
                    out_min, out_max = out_range
                self.coeffs.data = (self.coeffs - self.coeffs.min()) / (self.coeffs.max() - self.coeffs.min() + 1e-8)
                self.coeffs.data = self.coeffs * (out_max - out_min) + out_min
        else:
            # For general splines (Φ), initialize with a balanced distribution
            if self.train_range and (self.range_params is not None):
                out_center = self.range_params.out_center.item()
                out_radius = self.range_params.out_radius.item()
                out_min = out_center - out_radius
                out_max = out_center + out_radius
            else:
                out_min, out_max = out_range
            self.coeffs = nn.Parameter(
                torch.randn(num_knots) * 0.1 + torch.linspace(out_min, out_max, num_knots)
            )
    
    def forward(self, x):
        if self.train_range and (self.range_params is not None):
            # Compute dynamic in_range from current trainable parameters.
            in_center = self.range_params.in_center
            in_radius = self.range_params.in_radius
            in_min = in_center - in_radius
            in_max = in_center + in_radius
            knots = self.normalized_knots * (in_max - in_min) + in_min
            # Also compute dynamic out_range.
            out_center = self.range_params.out_center
            out_radius = self.range_params.out_radius
            out_min = out_center - out_radius
            out_max = out_center + out_radius
        else:
            knots = self.knots
            in_min = self.in_min
            in_max = self.in_max
            out_min = self.out_min
            out_max = self.out_max
        
        x = x.to(knots.device)
        x = torch.clamp(x, in_min, in_max)
        
        # Find which segment each input value belongs to
        intervals = torch.searchsorted(knots, x) - 1
        intervals = torch.clamp(intervals, 0, self.num_knots - 2)
        
        # Compute interpolation parameter t in [0,1]
        t = (x - knots[intervals]) / (knots[intervals + 1] - knots[intervals])
        
        if self.monotonic:
            # For φ splines, ensure coefficients are sorted to maintain monotonicity
            sorted_coeffs = torch.sort(self.coeffs)[0]
            raw = (1 - t) * sorted_coeffs[intervals] + t * sorted_coeffs[intervals + 1]
        else:
            # For Φ splines, use coefficients directly
            raw = (1 - t) * self.coeffs[intervals] + t * self.coeffs[intervals + 1]
        
        # If train_range is enabled, re-scale the raw output.
        # We normalize raw using initial coeff min/max (stored during first call)
        # and then map to the current out_range.
        eps = 1e-8
        if self.train_range and (self.range_params is not None):
            if not hasattr(self, 'init_coeff_min') or not hasattr(self, 'init_coeff_max'):
                # Store initial min and max of coeffs.
                self.register_buffer('init_coeff_min', self.coeffs.data[0].clone())
                self.register_buffer('init_coeff_max', self.coeffs.data[-1].clone())
            coeff_min = self.init_coeff_min
            coeff_max = self.init_coeff_max
            raw_normalized = (raw - coeff_min) / (coeff_max - coeff_min + eps)
            final = out_min + (out_max - out_min) * raw_normalized
        else:
            final = raw
        return final
    
    def get_flatness_penalty(self):
        second_diff = self.coeffs[2:] - 2 * self.coeffs[1:-1] + self.coeffs[:-2]
        return torch.mean(second_diff ** 2)

##############################################################################
#                          SPRECHER LAYER BLOCK                              #
##############################################################################

class SprecherLayerBlock(nn.Module):
    """
    A single Sprecher block that transforms d_in -> d_out using:
      - A monotonic spline φ (with fixed (0,1) domain and range)
      - A general spline Φ (with trainable in/out ranges)
      - A trainable shift (η)
      - A trainable weight matrix (λ)
    If is_final is True, the block sums its outputs to produce a scalar.
    """
    def __init__(self, d_in, d_out, is_final=False, phi_range_params=None):
        super().__init__()
        self.d_in = d_in
        self.d_out = d_out
        self.is_final = is_final
        
        # The inner monotonic spline φ (with fixed range (0,1))
        # This represents the 'inner function' in Sprecher's theorem
        self.phi = SimpleSpline(num_knots=PHI_KNOTS, in_range=(0, 1), out_range=(0, 1), monotonic=True)
        
        # The outer general spline Φ with trainable range parameters
        # This represents the 'outer function' in Sprecher's theorem
        self.Phi = SimpleSpline(num_knots=PHI_CAPITAL_KNOTS, in_range=PHI_IN_RANGE, out_range=PHI_OUT_RANGE,
                                train_range=True, range_params=phi_range_params)
        with torch.no_grad():
            if phi_range_params is not None:
                out_center = phi_range_params.out_center.item()
                out_radius = phi_range_params.out_radius.item()
                out_min = out_center - out_radius
                out_max = out_center + out_radius
            else:
                out_min, out_max = PHI_OUT_RANGE
            self.Phi.coeffs.data = torch.linspace(out_min, out_max, self.Phi.num_knots)
            self.Phi.register_buffer('init_coeff_min', self.Phi.coeffs.data[0].clone())
            self.Phi.register_buffer('init_coeff_max', self.Phi.coeffs.data[-1].clone())
        
        # Weight matrix and shift parameter.
        self.lambdas = nn.Parameter(torch.ones(d_in, d_out))
        self.eta = nn.Parameter(torch.tensor(1.0 / 100.0))
        
        # Q-values for indexing.
        self.register_buffer('q_values', torch.arange(d_out, dtype=torch.float32))
    
    def forward(self, x):
        # x: (batch_size, d_in)
        x_expanded = x.unsqueeze(-1)  # shape: (batch_size, d_in, 1)
        q = self.q_values.view(1, 1, -1)  # shape: (1, 1, d_out)
        
        # Apply translation by η * q (part of Sprecher's construction)
        shifted = x_expanded + self.eta * q
        shifted = torch.clamp(shifted, 0, 1)
        
        # Apply inner monotonic spline φ to the shifted inputs
        phi_out = self.phi(shifted)  # shape: (batch_size, d_in, d_out)
        
        # Weight by λ and sum over input dimension (weighted superposition)
        weighted = phi_out * self.lambdas.unsqueeze(0)
        s = weighted.sum(dim=1) + self.q_values  # shape: (batch_size, d_out)
        
        # Pass through the general outer spline Φ
        activated = self.Phi(s)
        
        if self.is_final:
            # Sum outputs if this is the final block (produces scalar output)
            return activated.sum(dim=1, keepdim=True)
        else:
            return activated

##############################################################################
#                     MULTI-LAYER SPRECHER NETWORK                           #
##############################################################################

class SprecherMultiLayerNetwork(nn.Module):
    """
    Builds the Sprecher network with a given hidden-layer architecture and final output dimension.
    A single global PhiRangeParams instance is created and shared among all Φ splines.
    """
    def __init__(self, input_dim, architecture, final_dim=1):
        super().__init__()
        self.input_dim = input_dim
        self.architecture = architecture
        self.final_dim = final_dim
        
        # Compute centers and radii from the configured outer ranges.
        in_center = (PHI_IN_RANGE[0] + PHI_IN_RANGE[1]) / 2.0
        in_radius = (PHI_IN_RANGE[1] - PHI_IN_RANGE[0]) / 2.0
        out_center = (PHI_OUT_RANGE[0] + PHI_OUT_RANGE[1]) / 2.0
        out_radius = (PHI_OUT_RANGE[1] - PHI_OUT_RANGE[0]) / 2.0

        # Create global trainable parameters for all Φ splines
        self.phi_range_params = PhiRangeParams(in_center=in_center, in_radius=in_radius,
                                               out_center=out_center, out_radius=out_radius)
        
        layers = []
        if len(architecture) == 0:
            is_final = (final_dim == 1)
            layers.append(SprecherLayerBlock(d_in=input_dim, d_out=final_dim,
                                             is_final=is_final, phi_range_params=self.phi_range_params))
        else:
            L = len(architecture)
            for i in range(L - 1):
                d_in = input_dim if i == 0 else architecture[i - 1]
                d_out = architecture[i]
                layers.append(SprecherLayerBlock(d_in=d_in, d_out=d_out,
                                                 is_final=False, phi_range_params=self.phi_range_params))
            d_in_last = input_dim if L == 1 else architecture[L - 2]
            d_out_last = architecture[L - 1]
            if self.final_dim == 1:
                layers.append(SprecherLayerBlock(d_in=d_in_last, d_out=d_out_last,
                                                 is_final=True, phi_range_params=self.phi_range_params))
            else:
                layers.append(SprecherLayerBlock(d_in=d_in_last, d_out=d_out_last,
                                                 is_final=False, phi_range_params=self.phi_range_params))
                layers.append(SprecherLayerBlock(d_in=d_out_last, d_out=self.final_dim,
                                                 is_final=False, phi_range_params=self.phi_range_params))
        
        self.layers = nn.ModuleList(layers)
    
    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

##############################################################################
#                           DATA GENERATION                                  #
##############################################################################

def generate_data(n_samples=32):
    """
    Generates a grid of points in [0,1]^d and computes the target function.
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
    # Use two parameter groups: one for global Phi range parameters (with a higher LR)
    # and one for the rest of the parameters.
    params = [
        {"params": [p for n, p in model.named_parameters() if "phi_range_params" in n], "lr": 0.001},
        {"params": [p for n, p in model.named_parameters() if "phi_range_params" not in n], "lr": 0.0003}
    ]
    optimizer = torch.optim.Adam(params, weight_decay=1e-7)
    
    losses = []
    best_loss = float("inf")
    best_state = None
    
    pbar = tqdm(range(total_epochs), desc="Training Network")
    for epoch in pbar:
        optimizer.zero_grad()
        output = model(x_train)
        
        mse_loss = torch.mean((output - y_train) ** 2)
        
        # Calculate smoothness penalty (regularization term)
        flatness_penalty = 0.0
        for layer in model.layers:
            flatness_penalty += 0.01 * layer.phi.get_flatness_penalty() + 0.1 * layer.Phi.get_flatness_penalty()
        
        # Variance penalty: encourage network output variability to match that of the target
        std_target = torch.std(y_train)
        std_output = torch.std(output)
        var_loss = LAMBDA_VAR * (std_output - std_target) ** 2
        
        # Combine all loss components
        loss = mse_loss + flatness_penalty + var_loss
        loss.backward()
        
        # Capture gradients for the global range parameters for debugging
        ic_grad = model.phi_range_params.in_center.grad.item() if model.phi_range_params.in_center.grad is not None else 0.0
        ir_grad = model.phi_range_params.in_radius.grad.item() if model.phi_range_params.in_radius.grad is not None else 0.0
        oc_grad = model.phi_range_params.out_center.grad.item() if model.phi_range_params.out_center.grad is not None else 0.0
        or_grad = model.phi_range_params.out_radius.grad.item() if model.phi_range_params.out_radius.grad is not None else 0.0
        
        optimizer.step()
        losses.append(loss.item())
        
        pbar.set_postfix({
            'loss': f'{loss.item():.2e}',
            'ic': f'{model.phi_range_params.in_center.item():.3f}',
            'ir': f'{model.phi_range_params.in_radius.item():.3f}',
            'oc': f'{model.phi_range_params.out_center.item():.3f}',
            'or': f'{model.phi_range_params.out_radius.item():.3f}',
            'g_ic': f'{ic_grad:.3e}',
            'g_ir': f'{ir_grad:.3e}',
            'g_oc': f'{oc_grad:.3e}',
            'g_or': f'{or_grad:.3e}',
            'std_out': f'{std_output.item():.3f}',
            'std_tar': f'{std_target.item():.3f}',
            'var_loss': f'{var_loss.item():.2e}'
        })
        
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

def plot_results(model, layers):
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
            x_vals = torch.linspace(layer.phi.fixed_in_range[0], layer.phi.fixed_in_range[1], 200).to(layer.phi.coeffs.device)
            y_vals = layer.phi(x_vals)
        ax_phi.plot(x_vals.cpu(), y_vals.cpu(), 'c-')
        ax_phi.grid(True)
        
        ax_Phi = fig.add_subplot(gs_top[0, col_Phi])
        ax_Phi.set_title(f"{block_label} $\\Phi^{{({j})}}$", fontsize=11)
        with torch.no_grad():
            if layer.Phi.train_range and (layer.Phi.range_params is not None):
                in_center = layer.Phi.range_params.in_center.item()
                in_radius = layer.Phi.range_params.in_radius.item()
                in_min = in_center - in_radius
                in_max = in_center + in_radius
            else:
                in_min, in_max = layer.Phi.fixed_in_range
            x_vals = torch.linspace(in_min, in_max, 200).to(layer.Phi.coeffs.device)
            y_vals = layer.Phi(x_vals)
        ax_Phi.plot(x_vals.cpu(), y_vals.cpu(), 'm-')
        ax_Phi.grid(True)
    
    # Create a separate gridspec for the bottom row (function plots)
    # For multi-dimensional outputs, create equal-sized columns
    if input_dim == 1:
        # Create test points for 1D input
        x_test = torch.linspace(0, 1, 200).reshape(-1, 1).to(next(model.parameters()).device)
        y_true = target_function(x_test)
        if y_true.dim() == 1:
            y_true = y_true.unsqueeze(1)
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
        x = torch.linspace(0, 1, n)
        y = torch.linspace(0, 1, n)
        X, Y = torch.meshgrid(x, y, indexing='ij')
        points = torch.stack([X.flatten(), Y.flatten()], dim=1)
        
        if final_dim == 1:
            # For scalar output from 2D input, plot target and prediction as surfaces
            gs_bottom = gridspec.GridSpec(1, 2)
            gs_bottom.update(left=0.05, right=0.95, top=0.45, bottom=0.05, wspace=0.3)
            
            Z_true = target_function(points).reshape(n, n)
            ax_target = fig.add_subplot(gs_bottom[0, 0], projection='3d')
            ax_target.set_title("Target Function", fontsize=12)
            ax_target.plot_surface(X.cpu().numpy(), Y.cpu().numpy(), Z_true.cpu().numpy(), cmap='viridis')
            
            ax_output = fig.add_subplot(gs_bottom[0, 1], projection='3d')
            ax_output.set_title("Network Output", fontsize=12)
            with torch.no_grad():
                Z_pred = model(points.to(next(model.parameters()).device)).reshape(n, n)
            ax_output.plot_surface(X.cpu().numpy(), Y.cpu().numpy(), Z_pred.cpu().numpy(), cmap='viridis')
        else:
            # For vector output from 2D input, plot each output dimension 
            gs_bottom = gridspec.GridSpec(1, final_dim)
            gs_bottom.update(left=0.05, right=0.95, top=0.45, bottom=0.05, wspace=0.3)
            
            y_true_all = target_function(points)
            if y_true_all.dim() == 1:
                y_true_all = y_true_all.unsqueeze(1)
                
            with torch.no_grad():
                Z_pred_all = model(points.to(next(model.parameters()).device))
            
            for d in range(final_dim):
                ax_out = fig.add_subplot(gs_bottom[0, d], projection='3d')
                ax_out.set_title(f"Output dim {d}", fontsize=12)
                Z_pred_d = Z_pred_all[:, d].reshape(n, n)
                Z_true_d = y_true_all[:, d].reshape(n, n)
                ax_out.plot_surface(X.cpu().numpy(), Y.cpu().numpy(), Z_true_d.cpu().numpy(),
                                    cmap='viridis', alpha=0.5)
                ax_out.plot_surface(X.cpu().numpy(), Y.cpu().numpy(), Z_pred_d.cpu().numpy(),
                                    cmap='autumn', alpha=0.5)
    
    return fig

##############################################################################
#                              MAIN EXECUTION                                #
##############################################################################

if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Automatically determine input and output dimensions
    input_dim = get_input_dimension(target_function)
    final_dim = get_output_dimension(target_function, input_dim)
    
    model, losses, layers = train_network(
        target_function,
        architecture=architecture,
        total_epochs=TOTAL_EPOCHS,
        print_every=TOTAL_EPOCHS // 10,
        device=device,
        final_dim=final_dim
    )
    
    fig_results = plot_results(model, layers)
    
    if SAVE_FINAL_PLOT:
        os.makedirs("plots", exist_ok=True)
        prefix = "OneVar" if input_dim == 1 else "TwoVars"
        arch_str = "-".join(map(str, architecture)) if len(architecture) > 0 else "None"
        filename = f"{prefix}-{arch_str}-{TOTAL_EPOCHS}-epochs-outdim{final_dim}.png"
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