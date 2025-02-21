import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from mpl_toolkits.mplot3d import Axes3D  # needed for 3D plotting

# Settings for plotting resolution
NUM_PLOT_POINTS = 200
NUM_SURFACE_POINTS = 50

# --- Architecture Configuration ---
# Set the network architecture by specifying the number of nodes per hidden layer.
# For one hidden layer (with 7 nodes):
# architecture = [7]
# For two hidden layers (with 5 nodes in the first and 8 in the second):
# architecture = [5, 8]
architecture = [5,8]  # Change this to [5, 8] to use two hidden layers

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

# --- Learnable Spline Module ---
class LearnableSpline(nn.Module):
    def __init__(self, num_knots=30, in_range=(0, 1), out_range=(0, 1), monotonic=False):
        super().__init__()
        self.num_knots = num_knots
        self.in_min, self.in_max = in_range
        self.out_min, self.out_max = out_range
        self.monotonic = monotonic
        # Create knots over the input range
        self.register_buffer('knots', torch.linspace(self.in_min, self.in_max, num_knots))
        
        torch.manual_seed(SEED)
        if monotonic:
            increments = torch.rand(num_knots) * 0.1
            self.coeffs = nn.Parameter(torch.cumsum(increments, 0))
            with torch.no_grad():
                self.coeffs.data = (self.coeffs - self.coeffs.min()) / (self.coeffs.max() - self.coeffs.min())
                self.coeffs.data = self.coeffs.data * (out_range[1] - out_range[0]) + out_range[0]
        else:
            self.coeffs = nn.Parameter(
                torch.randn(num_knots) * 0.1 + torch.linspace(out_range[0], out_range[1], num_knots)
            )

    def forward(self, x):
        # Clamp x to the valid range
        x = torch.clamp(x, self.in_min, self.in_max)
        flat_x = x.view(-1)
        # Find the interval index for each x value
        indices = torch.clamp(torch.searchsorted(self.knots, flat_x) - 1, 0, self.num_knots - 2)
        # Compute the interpolation factor
        t = (flat_x - self.knots[indices]) / (self.knots[indices+1] - self.knots[indices])
        # Linear interpolation between knot coefficients
        flat_y = (1 - t) * self.coeffs[indices] + t * self.coeffs[indices+1]
        return flat_y.view_as(x)
    
    def get_flatness_penalty(self):
        second_diff = self.coeffs[2:] - 2 * self.coeffs[1:-1] + self.coeffs[:-2]
        return torch.mean(second_diff**2)

# --- Spline Layer Module ---
class SplineLayer(nn.Module):
    def __init__(self, in_features, out_features, spline_params=None):
        """
        A layer that applies a linear transformation followed by a learnable spline activation.
        """
        super().__init__()
        self.linear = nn.Linear(in_features, out_features)
        # Default spline parameters for hidden layers
        if spline_params is None:
            spline_params = {
                'num_knots': 300, 
                'in_range': (-10.0, 12.0), 
                'out_range': (-10.0, 12.0), 
                'monotonic': False
            }
        self.spline = LearnableSpline(**spline_params)
    
    def forward(self, x):
        z = self.linear(x)
        return self.spline(z)

# --- Spline Network Module ---
class SplineNetwork(nn.Module):
    def __init__(self, input_dim, architecture, 
                 spline_params_hidden=None, spline_params_output=None):
        """
        Builds a network with an arbitrary number of hidden layers as specified by 'architecture'.
        Each hidden layer is a SplineLayer and the output layer is a linear mapping followed by a learnable spline.
        For a one-hidden-layer network, the overall activation nonlinearity is two splines (as before).
        For a two-hidden-layer network, there will be three spline functions in total.
        """
        super().__init__()
        self.hidden_layers = nn.ModuleList()
        prev_dim = input_dim
        # Use default spline parameters for hidden layers if not provided
        if spline_params_hidden is None:
            spline_params_hidden = {
                'num_knots': 300, 
                'in_range': (-10.0, 12.0), 
                'out_range': (-10.0, 12.0), 
                'monotonic': False
            }
        for _hidden in architecture:
            self.hidden_layers.append(SplineLayer(prev_dim, _hidden, spline_params=spline_params_hidden))
            prev_dim = _hidden
        # Output layer: linear mapping then spline activation.
        self.out_linear = nn.Linear(prev_dim, 1)
        if spline_params_output is None:
            spline_params_output = {
                'num_knots': 200, 
                'in_range': (-10.0, 12.0), 
                'out_range': (-10.0, 12.0), 
                'monotonic': False
            }
        self.out_spline = LearnableSpline(**spline_params_output)
    
    def forward(self, x):
        for layer in self.hidden_layers:
            x = layer(x)
        x = self.out_linear(x)
        x = self.out_spline(x)
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

# --- End-to-End Training ---
def train_network(target_function, architecture, total_epochs=100000, print_every=10000, device="cpu"):
    x_train, y_train, input_dim = generate_data()
    x_train = x_train.to(device)
    y_train = y_train.to(device)
    
    model = SplineNetwork(input_dim, architecture).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0003, weight_decay=1e-7)
    loss_history = []
    
    pbar = tqdm(range(total_epochs), desc="Training Network")
    for epoch in pbar:
        optimizer.zero_grad()
        output = model(x_train)
        mse_loss = torch.mean((output - y_train) ** 2)
        # Add flatness penalties from all spline modules in the network
        penalty = 0.0
        for layer in model.hidden_layers:
            penalty += 0.01 * layer.spline.get_flatness_penalty()
        penalty += 0.1 * model.out_spline.get_flatness_penalty()
        total_loss = mse_loss + penalty
        total_loss.backward()
        optimizer.step()
        
        loss_history.append(total_loss.item())
        if (epoch + 1) % print_every == 0:
            pbar.set_postfix({'loss': f'{total_loss.item():.2e}'})
    
    return model, loss_history

# --- Plotting Results ---
def plot_results(model, loss_history):
    input_dim = get_input_dimension(target_function)
    num_hidden = len(model.hidden_layers)
    
    # Determine labels for hidden activations based on the number of hidden layers.
    if num_hidden == 1:
        hidden_label = 'φ'
    else:
        hidden_labels = [f'φ₍{i+1}₎' for i in range(num_hidden)]
    
    plt.figure(figsize=(18, 12))
    
    # --- Plot Network Structure with Parameters ---
    plt.subplot(231)
    plt.title("Network Structure with Parameters")
    # Layout: input layer, then each hidden layer, then output layer.
    num_layers = num_hidden + 2  # input, hidden(s), output
    x_positions = np.linspace(0.1, 0.9, num_layers)
    
    # Plot input nodes
    input_nodes = get_input_dimension(target_function)
    input_y_positions = np.linspace(0.2, 0.8, input_nodes)
    for i, y in enumerate(input_y_positions):
        plt.scatter([x_positions[0]], [y], c='black', s=100)
        plt.text(x_positions[0] - 0.05, y, f'x_{i+1}', ha='right')
    layer_positions = [(x_positions[0], input_y_positions)]
    
    # Plot hidden layers
    for idx, layer in enumerate(model.hidden_layers):
        hidden_nodes = layer.linear.out_features
        hidden_y_positions = np.linspace(0.2, 0.8, hidden_nodes)
        for j, y in enumerate(hidden_y_positions):
            plt.scatter([x_positions[idx+1]], [y], c='blue', s=100)
            # Label with q (or with a subscript indicating layer and node if more than one hidden layer)
            if num_hidden == 1:
                plt.text(x_positions[idx+1] + 0.05, y, f'q={j}', ha='left')
            else:
                plt.text(x_positions[idx+1] + 0.05, y, f'q₍{idx+1},{j}₎', ha='left')
        # Draw connections from previous layer to current hidden layer
        prev_x, prev_y_positions = layer_positions[-1]
        for y_prev in prev_y_positions:
            for y_curr in hidden_y_positions:
                plt.plot([prev_x, x_positions[idx+1]], [y_prev, y_curr], 'k-', alpha=0.3)
        layer_positions.append((x_positions[idx+1], hidden_y_positions))
    
    # Plot output node
    output_x = x_positions[-1]
    output_y = np.array([0.5])
    plt.scatter([output_x], [0.5], c='red', s=100)
    plt.text(output_x + 0.05, 0.5, 'output', ha='left')
    # Draw connections from last hidden layer to output
    prev_x, prev_y_positions = layer_positions[-1]
    for y_prev in prev_y_positions:
        plt.plot([prev_x, output_x], [y_prev, 0.5], 'k-', alpha=0.3)
    
    # Label the activation functions on the edges
    for i in range(num_layers - 1):
        mid_x = (x_positions[i] + x_positions[i+1]) / 2
        if i == 0:
            # From input to first hidden layer: label with hidden activation function
            if num_hidden == 1:
                label = hidden_label
            else:
                label = hidden_labels[0]
        elif i < num_hidden:
            # Between hidden layers, use the corresponding hidden label
            if num_hidden == 1:
                label = 'Φ'
            else:
                label = hidden_labels[i]
        else:
            label = 'Φ'
        plt.text(mid_x, 0.15, label, ha='center')
    
    plt.axis('off')
    
    # --- Plot Spline Functions ---
    # For one hidden layer, there are two splines (hidden and output).
    # For two hidden layers, there will be three splines (one per hidden layer, plus output).
    num_splines = num_hidden + 1
    if num_hidden == 1:
        spline_titles = ["Hidden Activation φ", "Output Activation Φ"]
    else:
        spline_titles = [f"Hidden Activation {hidden_labels[i]}" for i in range(num_hidden)] + ["Output Activation Φ"]
    
    for i in range(num_splines):
        plt.subplot(2, 3, 2 + i)  # subplots 232, 233, etc.
        plt.title(spline_titles[i])
        if i < num_hidden:
            spline_module = model.hidden_layers[i].spline
        else:
            spline_module = model.out_spline
        with torch.no_grad():
            x_vals = torch.linspace(spline_module.in_min, spline_module.in_max, NUM_PLOT_POINTS)
            y_vals = spline_module(x_vals)
        plt.plot(x_vals.cpu(), y_vals.cpu())
        plt.grid(True)
    
    # --- Plot Function Comparison ---
    if input_dim == 1:
        plt.subplot(235)
        plt.title("Function Comparison")
        x_test = torch.linspace(0, 1, NUM_PLOT_POINTS).reshape(-1, 1)
        y_true = target_function(x_test)
        with torch.no_grad():
            y_pred = model(x_test)
        plt.plot(x_test.cpu(), y_true.cpu(), '--', label='Target f(x)')
        plt.plot(x_test.cpu(), y_pred.cpu(), '-', label='Network output')
        plt.grid(True)
        plt.legend()
    else:
        # 2D case: use 3D surface plots for target and network output
        ax1 = plt.subplot(235, projection='3d')
        n = NUM_SURFACE_POINTS
        x = torch.linspace(0, 1, n)
        y = torch.linspace(0, 1, n)
        X, Y = torch.meshgrid(x, y, indexing='ij')
        points = torch.stack([X.flatten(), Y.flatten()], dim=1)
        Z_true = target_function(points).reshape(n, n)
        ax1.plot_surface(X.cpu().numpy(), Y.cpu().numpy(), Z_true.cpu().numpy(), cmap='viridis')
        ax1.set_title('Target Function')
        
        ax2 = plt.subplot(236, projection='3d')
        with torch.no_grad():
            Z_pred = model(points.to(next(model.parameters()).device)).reshape(n, n)
        ax2.plot_surface(X.cpu().numpy(), Y.cpu().numpy(), Z_pred.cpu().numpy(), cmap='viridis')
        ax2.set_title('Network Output')
    
    plt.tight_layout()
    plt.show()
    
    # --- Plot Loss Curve ---
    plt.figure()
    plt.plot(loss_history)
    plt.title("Training Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.grid(True)
    plt.show()

# --- Main Execution ---
device = "cuda" if torch.cuda.is_available() else "cpu"

model, loss_history = train_network(target_function, architecture, total_epochs=100000, print_every=10000, device=device)
plot_results(model, loss_history)
