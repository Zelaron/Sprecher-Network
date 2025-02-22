import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from mpl_toolkits.mplot3d import Axes3D  # needed for 3D plotting
import matplotlib.gridspec as gridspec

# Settings for plotting resolution
NUM_PLOT_POINTS = 200
NUM_SURFACE_POINTS = 50

# --- Architecture Configuration ---
# Set the network architecture by specifying the number of nodes in each hidden layer.
# For example:
#   architecture = [7]       -> one hidden layer with 7 nodes
#   architecture = [5, 8]    -> two hidden layers with 5 nodes in the first and 8 nodes in the second
architecture = [5, 8]  # Change this to [7] for one hidden layer

# -----------------------------
# Target function definition:
# Uncomment the desired version.
def target_function(x):
    # For 1D:
    # return (x[:, [0]] - 0.5)**5 - (x[:, [0]] - (1/3))**3 + (1/5)*(x[:, [0]] - 0.1)**2
    # For 2D:
    return (torch.exp(torch.sin(torch.pi * x[:, [0]]) + x[:, [1]]**2) - 1) / 7

# --- Utility: Determine input dimension ---
def get_input_dimension(target_function):
    """
    Try dimensions 1 to 3 until target_function runs without error.
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

# --- Set seeds for reproducibility ---
SEED = 45
torch.manual_seed(SEED)
np.random.seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# --- Spline Module ---
class SimpleSpline(nn.Module):
    def __init__(self, num_knots=30, in_range=(0, 1), out_range=(0, 1), monotonic=False):
        super().__init__()
        self.num_knots = num_knots
        self.in_min, self.in_max = in_range
        self.out_min, self.out_max = out_range
        self.monotonic = monotonic
        # Register knots as a buffer so they move with the module when transferring devices
        self.register_buffer('knots', torch.linspace(self.in_min, self.in_max, num_knots))
        
        torch.manual_seed(SEED)
        if monotonic:
            increments = torch.rand(num_knots) * 0.1
            self.coeffs = nn.Parameter(torch.cumsum(increments, 0))
            with torch.no_grad():
                self.coeffs.data = (self.coeffs - self.coeffs.min()) / (self.coeffs.max() - self.coeffs.min())
                self.coeffs.data = self.coeffs * (out_range[1] - out_range[0]) + out_range[0]
        else:
            self.coeffs = nn.Parameter(
                torch.randn(num_knots) * 0.1 + torch.linspace(out_range[0], out_range[1], num_knots)
            )

    def forward(self, x):
        # Ensure x is on the same device as the knots
        x = x.to(self.knots.device)
        x = torch.clamp(x, self.in_min, self.in_max)
        intervals = torch.clamp(torch.searchsorted(self.knots, x) - 1, 0, self.num_knots - 2)
        t = (x - self.knots[intervals]) / (self.knots[intervals + 1] - self.knots[intervals])
        if self.monotonic:
            sorted_coeffs = torch.sort(self.coeffs)[0]
            return (1 - t) * sorted_coeffs[intervals] + t * sorted_coeffs[intervals + 1]
        else:
            return (1 - t) * self.coeffs[intervals] + t * self.coeffs[intervals + 1]
    
    def get_flatness_penalty(self):
        second_diff = self.coeffs[2:] - 2 * self.coeffs[1:-1] + self.coeffs[:-2]
        return torch.mean(second_diff**2)

# --- Sprecher Layer Block ---
class SprecherLayerBlock(nn.Module):
    def __init__(self, d_in, d_out, is_final=False):
        """
        Implements a single transformation block.
        For non-final layers, the block maps an input vector of dimension d_in
        to an output vector of dimension d_out using a pair of spline functions.
        For the final block (is_final=True), d_out should be 1, and the output is summed to produce a scalar.
        """
        super().__init__()
        self.d_in = d_in
        self.d_out = d_out
        self.is_final = is_final
        
        # Inner spline activation (monotonic) analogous to φ
        self.phi = SimpleSpline(num_knots=300, in_range=(0, 1), out_range=(0, 1), monotonic=True)
        # Outer spline transformation analogous to Φ
        self.Phi = SimpleSpline(num_knots=200, in_range=(-10.0, 12.0), out_range=(-10.0, 12.0))
        with torch.no_grad():
            self.Phi.coeffs.data = torch.linspace(self.Phi.out_min, self.Phi.out_max, self.Phi.num_knots)
        
        # Learnable weight parameters: a matrix mapping d_in inputs to d_out outputs.
        self.lambdas = nn.Parameter(torch.ones(d_in, d_out))
        # Learnable shift parameter η.
        self.eta = nn.Parameter(torch.tensor(1.0 / 100.0))
        # q values (offsets) registered as a buffer; these serve as node-specific biases.
        self.register_buffer('q_values', torch.arange(d_out, dtype=torch.float32))

    def forward(self, x):
        # x shape: [batch, d_in]
        batch_size = x.shape[0]
        # Expand x to shape [batch, d_in, d_out] so that each node in the output gets an offset.
        x_expanded = x.unsqueeze(-1)  # shape: [batch, d_in, 1]
        q = self.q_values.view(1, 1, -1)  # shape: [1, 1, d_out]
        shifted = x_expanded + self.eta * q
        shifted = torch.clamp(shifted, 0, 1)
        phi_out = self.phi(shifted)  # shape: [batch, d_in, d_out]
        # Weight each input channel and sum over the d_in dimension.
        weighted = phi_out * self.lambdas.unsqueeze(0)  # shape: [batch, d_in, d_out]
        s = weighted.sum(dim=1) + self.q_values  # shape: [batch, d_out]
        s = torch.clamp(s, self.Phi.in_min, self.Phi.in_max)
        activated = self.Phi(s)  # shape: [batch, d_out]
        if self.is_final:
            # For the final block, output is a scalar (summing over the single node if needed)
            return activated.sum(dim=1, keepdim=True)
        else:
            return activated

# --- Multi-Layer Sprecher Network ---
class SprecherMultiLayerNetwork(nn.Module):
    def __init__(self, input_dim, architecture):
        """
        Builds a network with a configurable number of hidden layers.
        architecture: list of integers, where each integer specifies the number of nodes in that hidden layer.
        The final output layer always produces a single scalar output.
        """
        super().__init__()
        self.input_dim = input_dim
        self.architecture = architecture
        layers = []
        if len(architecture) == 0:
            # No hidden layers: direct mapping from input to output.
            layers.append(SprecherLayerBlock(d_in=input_dim, d_out=1, is_final=True))
        else:
            # First hidden layer: input_dim -> architecture[0]
            layers.append(SprecherLayerBlock(d_in=input_dim, d_out=architecture[0], is_final=False))
            # Additional hidden layers (if any)
            for i in range(1, len(architecture)):
                layers.append(SprecherLayerBlock(d_in=architecture[i-1], d_out=architecture[i], is_final=False))
            # Final output layer: maps from last hidden layer to one output.
            layers.append(SprecherLayerBlock(d_in=architecture[-1], d_out=1, is_final=True))
        self.layers = nn.ModuleList(layers)
    
    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

# --- Data Generation ---
def generate_data(n_samples=32):
    """
    Generates training data based on the active target_function.
    Automatically determines if the target_function expects 1D or 2D inputs.
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

# --- Training Function ---
def train_network(target_function, architecture, total_epochs=100000, print_every=10000, device="cpu"):
    x_train, y_train, input_dim = generate_data()
    x_train = x_train.to(device)
    y_train = y_train.to(device)
    
    model = SprecherMultiLayerNetwork(input_dim=input_dim, architecture=architecture).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0003, weight_decay=1e-7)
    
    losses = []
    best_loss = float("inf")
    best_state = None
    
    pbar = tqdm(range(total_epochs), desc="Training Network")
    for epoch in pbar:
        optimizer.zero_grad()
        output = model(x_train)
        mse_loss = torch.mean((output - y_train) ** 2)
        # Accumulate flatness penalties for all spline modules in the network.
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

# --- Plotting Functions for Combined Figure ---
def plot_network_structure_ax(ax, layers, input_dim):
    """
    Plots a schematic of the network structure on the given axis.
    """
    # Determine total number of layers: input + hidden layers + output.
    if len(layers) == 1 and layers[0].is_final:
        total_layers = 2
    else:
        total_layers = len(layers) + 1  # input layer + blocks
    layer_x = np.linspace(0.1, 0.9, total_layers)
    
    # Plot input nodes.
    if input_dim == 1:
        input_y = [0.5]
    else:
        input_y = np.linspace(0.2, 0.8, input_dim)
    for i, y in enumerate(input_y):
        ax.scatter(layer_x[0], y, c='black', s=100)
        ax.text(layer_x[0] - 0.05, y, f'x_{i+1}', ha='right', va='center')
    
    prev_y = input_y
    # Plot each subsequent layer.
    for idx, layer in enumerate(layers):
        if layer.is_final:
            node_count = 1
            color = 'red'
            label = 'output'
            current_y = [0.5]
        else:
            node_count = layer.q_values.shape[0]
            color = 'blue'
            label = f'Hidden {idx+1}'
            current_y = np.linspace(0.2, 0.8, node_count)
        # Ensure x is repeated for each node.
        ax.scatter([layer_x[idx+1]] * len(current_y), current_y, c=color, s=100)
        for j, y in enumerate(current_y):
            if not layer.is_final:
                ax.text(layer_x[idx+1] + 0.05, y, f'q_{idx+1}{j}', ha='left', va='center')
            else:
                ax.text(layer_x[idx+1] + 0.05, y, label, ha='left', va='center')
        # Draw connections from previous layer to current layer.
        for y_prev in prev_y:
            for y_curr in current_y:
                ax.plot([layer_x[idx], layer_x[idx+1]], [y_prev, y_curr], 'k-', alpha=0.3)
        # Label the edge between layers.
        if idx == 0:
            edge_label = "φ"
        elif layer.is_final:
            edge_label = "Φ"
        else:
            edge_label = f"ψ_{idx}"
        mid_x = (layer_x[idx] + layer_x[idx+1]) / 2
        ax.text(mid_x, 0.15, edge_label, ha='center', fontsize=14)
        prev_y = current_y
    ax.axis('off')

def plot_results(model, layers):
    # Determine input dimension from first layer if available.
    input_dim = layers[0].d_in if hasattr(layers[0], 'd_in') else get_input_dimension(target_function)
    num_splines = len(layers)
    top_cols = 1 + num_splines  # first column for network structure, rest for spline plots

    fig = plt.figure(figsize=(4 * top_cols, 10))
    gs = gridspec.GridSpec(2, top_cols, height_ratios=[1, 1.2])
    
    # Top row, first column: Network Structure
    ax_net = fig.add_subplot(gs[0, 0])
    plot_network_structure_ax(ax_net, layers, input_dim)
    ax_net.set_title("Network Structure with Parameters", fontsize=12)
    
    # Top row, remaining columns: Spline Plots (one per block)
    for i, layer in enumerate(layers):
        ax_spline = fig.add_subplot(gs[0, i+1])
        if layer.is_final:
            ax_spline.set_title("Final Layer Outer Spline (Φ)", fontsize=12)
            with torch.no_grad():
                x_vals = torch.linspace(layer.Phi.in_min, layer.Phi.in_max, NUM_PLOT_POINTS).to(layer.Phi.knots.device)
                y_vals = layer.Phi(x_vals)
            ax_spline.plot(x_vals.cpu(), y_vals.cpu(), 'm-')
        else:
            ax_spline.set_title(f"Hidden Layer {i+1} Inner Spline (φ)", fontsize=12)
            with torch.no_grad():
                x_vals = torch.linspace(layer.phi.in_min, layer.phi.in_max, NUM_PLOT_POINTS).to(layer.phi.knots.device)
                y_vals = layer.phi(x_vals)
            ax_spline.plot(x_vals.cpu(), y_vals.cpu(), 'c-')
        ax_spline.grid(True)
    
    # Bottom row: Function Comparison
    if input_dim == 1:
        ax_func = fig.add_subplot(gs[1, :])
        ax_func.set_title("Function Comparison", fontsize=12)
        x_test = torch.linspace(0, 1, NUM_PLOT_POINTS).reshape(-1, 1)
        y_true = target_function(x_test)
        with torch.no_grad():
            y_pred = model(x_test)
        ax_func.plot(x_test.cpu(), y_true.cpu(), 'k--', label='Target f(x)')
        ax_func.plot(x_test.cpu(), y_pred.cpu(), 'r-', label='Network output')
        ax_func.grid(True)
        ax_func.legend()
    else:
        # For 2D case, create 3D axes for target function and network output.
        half = top_cols // 2 if top_cols > 2 else 1
        ax_target = fig.add_subplot(gs[1, :half], projection='3d')
        ax_target.set_title("Target Function", fontsize=12)
        n = NUM_SURFACE_POINTS
        x = torch.linspace(0, 1, n)
        y = torch.linspace(0, 1, n)
        X, Y = torch.meshgrid(x, y, indexing='ij')
        points = torch.stack([X.flatten(), Y.flatten()], dim=1)
        Z_true = target_function(points).reshape(n, n)
        ax_target.plot_surface(X.cpu().numpy(), Y.cpu().numpy(), Z_true.cpu().numpy(), cmap='viridis')
        
        ax_output = fig.add_subplot(gs[1, half:], projection='3d')
        ax_output.set_title("Network Output", fontsize=12)
        with torch.no_grad():
            Z_pred = model(points.to(next(model.parameters()).device)).reshape(n, n)
        ax_output.plot_surface(X.cpu().numpy(), Y.cpu().numpy(), Z_pred.cpu().numpy(), cmap='viridis')
    
    plt.tight_layout()
    plt.show()

# --- Main Execution ---
device = "cuda" if torch.cuda.is_available() else "cpu"

model, losses, layers = train_network(
    target_function, architecture=architecture, total_epochs=100000, print_every=10000, device=device
)

# Now pass the actual model (an nn.Module) to the plotting function.
plot_results(model, layers)

# Plot loss curve.
plt.figure(figsize=(8, 5))
plt.plot(losses, label="Training Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Loss Curve")
plt.legend()
plt.show()