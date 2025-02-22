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
architecture = [5, 8, 7]  # Example with three hidden layers

# -----------------------------
# Target function definition:
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
        super().__init__()
        self.d_in = d_in
        self.d_out = d_out
        self.is_final = is_final
        
        self.phi = SimpleSpline(num_knots=300, in_range=(0, 1), out_range=(0, 1), monotonic=True)
        self.Phi = SimpleSpline(num_knots=200, in_range=(-10.0, 12.0), out_range=(-10.0, 12.0))
        with torch.no_grad():
            self.Phi.coeffs.data = torch.linspace(self.Phi.out_min, self.Phi.out_max, self.Phi.num_knots)
        
        self.lambdas = nn.Parameter(torch.ones(d_in, d_out))
        self.eta = nn.Parameter(torch.tensor(1.0 / 100.0))
        self.register_buffer('q_values', torch.arange(d_out, dtype=torch.float32))

    def forward(self, x):
        batch_size = x.shape[0]
        x_expanded = x.unsqueeze(-1)
        q = self.q_values.view(1, 1, -1)
        shifted = x_expanded + self.eta * q
        shifted = torch.clamp(shifted, 0, 1)
        phi_out = self.phi(shifted)
        weighted = phi_out * self.lambdas.unsqueeze(0)
        s = weighted.sum(dim=1) + self.q_values
        s = torch.clamp(s, self.Phi.in_min, self.Phi.in_max)
        activated = self.Phi(s)
        if self.is_final:
            return activated.sum(dim=1, keepdim=True)
        else:
            return activated

# --- Multi-Layer Sprecher Network ---
class SprecherMultiLayerNetwork(nn.Module):
    def __init__(self, input_dim, architecture):
        super().__init__()
        self.input_dim = input_dim
        self.architecture = architecture
        layers = []
        if len(architecture) == 0:
            layers.append(SprecherLayerBlock(d_in=input_dim, d_out=1, is_final=True))
        else:
            layers.append(SprecherLayerBlock(d_in=input_dim, d_out=architecture[0], is_final=False))
            for i in range(1, len(architecture)):
                layers.append(SprecherLayerBlock(d_in=architecture[i-1], d_out=architecture[i], is_final=False))
            layers.append(SprecherLayerBlock(d_in=architecture[-1], d_out=1, is_final=True))
        self.layers = nn.ModuleList(layers)
    
    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

# --- Data Generation ---
def generate_data(n_samples=32):
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

# --- Updated Plotting Function for Network Structure ---
def plot_network_structure_ax(ax, layers, input_dim):
    """
    Plots a schematic of the network structure on the given axis using LaTeX-formatted subscripts/superscripts.
    Displays lambda values (from the first hidden block) above the input nodes and η near the output.
    This version now handles any number (>= 1) of hidden layers properly.
    """
    num_blocks = len(layers)
    if num_blocks == 0:
        # No layers at all (direct mapping not using any blocks)
        ax.text(0.5, 0.5, "No network", ha='center', va='center', fontsize=12)
        ax.axis('off')
        return
    
    # If the last layer is marked is_final, we consider that the final block
    if layers[-1].is_final:
        num_hidden = num_blocks - 1
    else:
        num_hidden = num_blocks
    
    # Prepare x positions for each "layer" (input + each block)
    layer_x = np.linspace(0.2, 0.8, num_blocks + 1)
    
    # Plot input nodes
    input_y = np.linspace(0.2, 0.8, input_dim)
    for i, y in enumerate(input_y):
        ax.scatter(layer_x[0], y, c='black', s=100)
        ax.text(layer_x[0] - 0.05, y, f'$x_{{{i+1}}}$', ha='right', fontsize=12)
        # Display lambda values from the first hidden block above input nodes
        if num_hidden >= 1:
            # layers[0] is the first hidden block if there's at least one hidden block
            first_block = layers[0]
            if i < first_block.lambdas.shape[0]:
                lambda_val = first_block.lambdas[i, 0].item()
            else:
                lambda_val = 0.0
            ax.text(layer_x[0] + 0.05, y + 0.0125, f'$\\lambda_{{{i+1}}}={lambda_val:.3f}$',
                    ha='center', va='bottom', fontsize=10)
    
    # For each block, plot the layer of nodes and connect from the previous layer
    # layer_sizes[i] = number of nodes at layer i. We'll just get d_in from the block itself.
    # We'll track the previous layer's y positions for connections.
    prev_y = input_y
    for i, block in enumerate(layers):
        d_in = block.d_in
        d_out = block.d_out
        
        # We'll space out the new layer's nodes
        new_y = np.linspace(0.2, 0.8, d_out)
        
        # Color for final layer's node(s) = red, else blue
        color = 'red' if block.is_final else 'blue'
        
        # Plot nodes of this layer
        for j, yy in enumerate(new_y):
            ax.scatter(layer_x[i+1], yy, c=color, s=100)
            if block.is_final:
                # Label output if final
                if d_out == 1:
                    ax.text(layer_x[i+1] + 0.02, yy, 'output', ha='left', fontsize=12)
                else:
                    ax.text(layer_x[i+1] + 0.02, yy, f'output_{j}', ha='left', fontsize=12)
            else:
                # Label hidden nodes with q^{(i+1)}_j
                ax.text(layer_x[i+1] + 0.02, yy, f'$q^{{{i+1}}}_{{{j}}}$', ha='left', fontsize=10)
        
        # Connect lines from the previous layer to this layer
        for y_in in prev_y:
            for y_out in new_y:
                ax.plot([layer_x[i], layer_x[i+1]], [y_in, y_out], 'k-', alpha=0.5)
        
        # Label the spline between layers
        mid_x = 0.5 * (layer_x[i] + layer_x[i+1])
        # If we're not at the final block, label \varphi_{i+1}, else label \Phi_{num_hidden} (or f if no hidden layers)
        if i < num_blocks - 1:
            ax.text(mid_x, 0.1, f'$\\varphi_{{{i+1}}}$', ha='center', fontsize=14)
        else:
            if num_hidden > 0:
                ax.text(mid_x, 0.1, f'$\\Phi_{{{num_hidden}}}$', ha='center', fontsize=14)
            else:
                # If there are zero hidden layers, just label f
                ax.text(mid_x, 0.1, '$f$', ha='center', fontsize=14)
        
        # The new_y positions become prev_y for the next iteration
        prev_y = new_y
    
    # Label η near the output if we have at least one hidden layer
    if num_hidden > 0:
        # We mimic the original code's approach: use the last hidden block's eta
        eta_val = layers[num_hidden - 1].eta.item()
        ax.text(layer_x[-1] + 0.02, 0.1, f'$\\eta={eta_val:.3e}$', ha='left', fontsize=10)
    
    ax.axis('off')

# --- Plotting Functions for Combined Figure ---
def plot_results(model, layers):
    input_dim = layers[0].d_in if hasattr(layers[0], 'd_in') else get_input_dimension(target_function)
    num_splines = len(layers)
    top_cols = 1 + num_splines

    fig = plt.figure(figsize=(4 * top_cols, 10))
    gs = gridspec.GridSpec(2, top_cols, height_ratios=[1, 1.2])
    
    ax_net = fig.add_subplot(gs[0, 0])
    plot_network_structure_ax(ax_net, layers, input_dim)
    ax_net.set_title("Network Structure with Parameters", fontsize=12)
    
    for i, layer in enumerate(layers):
        ax_spline = fig.add_subplot(gs[0, i+1])
        if layer.is_final:
            ax_spline.set_title("Final Layer Outer Spline ($\\Phi$)", fontsize=12)
            with torch.no_grad():
                x_vals = torch.linspace(layer.Phi.in_min, layer.Phi.in_max, NUM_PLOT_POINTS).to(layer.Phi.knots.device)
                y_vals = layer.Phi(x_vals)
            ax_spline.plot(x_vals.cpu(), y_vals.cpu(), 'm-')
        else:
            ax_spline.set_title(f"Hidden Layer {i+1} Inner Spline ($\\varphi$)", fontsize=12)
            with torch.no_grad():
                x_vals = torch.linspace(layer.phi.in_min, layer.phi.in_max, NUM_PLOT_POINTS).to(layer.phi.knots.device)
                y_vals = layer.phi(x_vals)
            ax_spline.plot(x_vals.cpu(), y_vals.cpu(), 'c-')
        ax_spline.grid(True)
    
    if input_dim == 1:
        ax_func = fig.add_subplot(gs[1, :])
        ax_func.set_title("Function Comparison", fontsize=12)
        x_test = torch.linspace(0, 1, NUM_PLOT_POINTS).reshape(-1, 1)
        x_test = x_test.to(next(model.parameters()).device)
        y_true = target_function(x_test)
        with torch.no_grad():
            y_pred = model(x_test)
        ax_func.plot(x_test.cpu(), y_true.cpu(), 'k--', label='Target f(x)')
        ax_func.plot(x_test.cpu(), y_pred.cpu(), 'r-', label='Network output')
        ax_func.grid(True)
        ax_func.legend()
    else:
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
    target_function, architecture=architecture, total_epochs=20000, print_every=10000, device=device
)

plot_results(model, layers)

plt.figure(figsize=(8, 5))
plt.plot(losses, label="Training Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Loss Curve")
plt.legend()
plt.show()