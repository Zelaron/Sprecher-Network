architecture = [5, 8]  # two hidden layers – first with 5 nodes (which will output a vector) and second with 8 nodes that produces the final scalar output
```)
that controls whether the network is one‐ or two‐layer. In the one‐layer case the SprecherLayer works just as before (using two splines, φ and Φ), while in the two–hidden–layer network the first layer is “hidden” (it only uses its φ spline as an activation, without summing over nodes) and the second (final) layer uses both φ and Φ—thus overall you will see three splines in the plots (the first layer’s φ, then the second layer’s φ and its output‐spline Φ). The network is trained end‐to‐end with standard backpropagation. Also, the network‐structure subplot now adapts to the number of hidden layers, drawing four “columns” (input, hidden layer 1, hidden layer 2, output) for two–hidden–layer networks (with appropriate labels such as φ₁, φ₂, and Φ₂) while retaining the original style for one hidden layer.

Here is the full updated code:

```python
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from mpl_toolkits.mplot3d import Axes3D  # needed for 3D plotting

# =============================================================================
# Settings for plotting resolution and reproducibility
# =============================================================================
NUM_PLOT_POINTS = 200
NUM_SURFACE_POINTS = 50

# Set a random seed for reproducibility
SEED = 45
torch.manual_seed(SEED)
np.random.seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# =============================================================================
# Architecture configuration
# -----------------------------------------------------------------------------
# Define the network architecture as a list.
# For one hidden layer, e.g., architecture = [7]
# For two hidden layers, e.g., architecture = [5, 8]
architecture = [5, 8]  # Change this variable to switch architectures

# =============================================================================
# Target Function Definition
# -----------------------------------------------------------------------------
def target_function(x):
    # For 1D target function, uncomment the next line:
    # return (x[:, [0]] - 0.5)**5 - (x[:, [0]] - (1/3))**3 + (1/5)*(x[:, [0]] - 0.1)**2
    # For 2D target function, use:
    return (torch.exp(torch.sin(torch.pi * x[:, [0]]) + x[:, [1]]**2) - 1) / 7

# =============================================================================
# Utility: Determine Input Dimension
# -----------------------------------------------------------------------------
def get_input_dimension(target_function):
    """
    Determines the number of input dimensions expected by the target function.
    Tries dimensions 1 to 3 until the function runs without error.
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

# =============================================================================
# Data Generation
# -----------------------------------------------------------------------------
def generate_data(n_samples=32):
    """
    Generates training data for the target function.
    Automatically determines whether the target function expects 1D or 2D inputs.
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

# =============================================================================
# Spline Module
# -----------------------------------------------------------------------------
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
                self.coeffs.data = self.coeffs.data * (out_range[1] - out_range[0]) + out_range[0]
        else:
            self.coeffs = nn.Parameter(
                torch.randn(num_knots) * 0.1 + torch.linspace(out_range[0], out_range[1], num_knots)
            )

    def forward(self, x):
        # Ensure x is on the same device as the knots and within range
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

# =============================================================================
# Sprecher Layer Module
# -----------------------------------------------------------------------------
class SprecherLayer(nn.Module):
    def __init__(self, input_dim=2, hidden_dim=None, in_range=(0, 1), out_range=(0, 1), final_layer=True):
        """
        If final_layer is True, the output will be a scalar (summing over hidden nodes),
        otherwise the layer outputs a vector (to be used as input to the next layer).
        """
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = (2 * input_dim + 1) if hidden_dim is None else hidden_dim
        self.final_layer = final_layer
        
        # Initialize the two spline functions.
        # phi: a monotonic spline mapping inputs from in_range to in_range.
        self.phi = SimpleSpline(num_knots=300, in_range=in_range, out_range=in_range, monotonic=True)
        # Phi: a spline mapping with a broader output range.
        self.Phi = SimpleSpline(num_knots=200, in_range=(-10.0, 12.0), out_range=(-10.0, 12.0))
        with torch.no_grad():
            self.Phi.coeffs.data = torch.linspace(self.Phi.out_min, self.Phi.out_max, self.Phi.num_knots)
          
        # Initialize lambda parameters (one per input dimension)
        lambdas = torch.ones(input_dim)
        gamma = 10
        for p in range(1, input_dim):
            beta_sum = 0
            for r in range(1, 10):
                beta_r = (input_dim**r - 1) / (input_dim - 1)
                beta_sum += gamma**(-(p - 1) * beta_r)
            lambdas[p] = beta_sum
        self.lambdas = nn.Parameter(lambdas)
        
        # A learnable shift parameter η.
        self.eta = nn.Parameter(torch.tensor([1.0 / (gamma * (gamma - 1))]))
        self.eta.data = self.eta.data * 0.05  # Adjust scaling for more manageable inputs

    def forward(self, x):
        # x is of shape [batch, input_dim]
        q_values = torch.arange(self.hidden_dim, device=x.device, dtype=x.dtype)
        # Expand x so that we add a unique offset (η * q) per hidden node.
        shifted_x = x.unsqueeze(-1) + self.eta * q_values.view(1, 1, -1)
        shifted_x = torch.clamp(shifted_x, 0, 1)
        phi_out = self.phi(shifted_x)  # Apply the inner spline function
        # Multiply each input feature by its corresponding lambda and sum over the input dimension.
        inner = (self.lambdas.view(1, -1, 1) * phi_out).sum(dim=1) + q_values
        inner_clamped = torch.clamp(inner, self.Phi.in_min, self.Phi.in_max)
        spline_out = self.Phi(inner_clamped)
        if self.final_layer:
            # In the final layer, sum over hidden nodes to produce a scalar output.
            return spline_out.sum(dim=1, keepdim=True)
        else:
            # For a hidden (nonfinal) layer, return the vector (without summing) so that subsequent
            # layers can operate on each node's activation separately.
            return spline_out

# =============================================================================
# Spline Network: Composing Sprecher Layers
# -----------------------------------------------------------------------------
class SplineNetwork(nn.Module):
    def __init__(self, architecture, input_dim):
        """
        architecture: list specifying the number of nodes in each hidden layer.
                      For one hidden layer: [num_nodes]
                      For two hidden layers: [num_nodes_layer1, num_nodes_layer2]
        input_dim: number of input features.
        """
        super().__init__()
        self.architecture = architecture
        self.num_layers = len(architecture)
        self.layers = nn.ModuleList()
        if self.num_layers == 1:
            # One hidden layer: use a single SprecherLayer that produces a scalar output.
            self.layers.append(SprecherLayer(input_dim=input_dim, hidden_dim=architecture[0], final_layer=True))
        elif self.num_layers == 2:
            # Two hidden layers: first layer produces a vector output (final_layer=False),
            # second layer is the final layer that outputs a scalar.
            self.layers.append(SprecherLayer(input_dim=input_dim, hidden_dim=architecture[0], final_layer=False))
            self.layers.append(SprecherLayer(input_dim=architecture[0], hidden_dim=architecture[1], final_layer=True))
        else:
            raise NotImplementedError("Only 1 or 2 hidden layers are supported.")
    
    def forward(self, x):
        out = x
        for layer in self.layers:
            out = layer(out)
        return out

# =============================================================================
# Training Function (End-to-End)
# -----------------------------------------------------------------------------
def train_network(target_function, architecture, total_epochs=100000, print_every=10000, device="cpu"):
    """
    Trains the SplineNetwork end-to-end on the target function.
    Returns the trained model, loss history, and the list of SprecherLayer modules.
    """
    x_train, y_train, input_dim = generate_data()
    x_train = x_train.to(device)
    y_train = y_train.to(device)

    model = SplineNetwork(architecture, input_dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0003, weight_decay=1e-7)
    loss_history = []
    best_loss = float("inf")
    best_state = None

    pbar = tqdm(range(total_epochs), desc="Training Network")
    for epoch in pbar:
        optimizer.zero_grad()
        output = model(x_train)
        mse_loss = torch.mean((output - y_train) ** 2)
        penalty = 0
        # Apply regularization penalties for each SprecherLayer in the network.
        for layer in model.layers:
            if layer.lambdas.numel() > 1:
                lambda_reg = torch.mean((layer.lambdas[1:] - layer.lambdas[:-1]) ** 2)
            else:
                lambda_reg = 0.0
            penalty += 0.01 * lambda_reg \
                       + 0.01 * layer.phi.get_flatness_penalty() \
                       + 0.1 * layer.Phi.get_flatness_penalty()
        loss = mse_loss + penalty
        loss.backward()
        optimizer.step()
        loss_history.append(loss.item())
        pbar.set_postfix({'loss': f'{loss.item():.2e}'})

        if (epoch + 1) % print_every == 0:
            with torch.no_grad():
                current_output = model(x_train)
                print(f"Epoch {epoch+1}: MSE loss = {mse_loss.item():.2e}, Total loss = {loss.item():.2e}")
            if loss.item() < best_loss:
                best_loss = loss.item()
                best_state = model.state_dict()

    if best_state is not None:
        model.load_state_dict(best_state)
    return model, loss_history, model.layers

# =============================================================================
# Plotting Function: Results and Network Structure
# -----------------------------------------------------------------------------
def plot_results(model, layers):
    input_dim = layers[0].input_dim
    num_hidden_layers = len(layers)
    
    plt.figure(figsize=(15, 10))
    
    # -------------------------------
    # Plot Network Structure with Parameters
    # -------------------------------
    plt.subplot(231)
    plt.title("Network Structure with Parameters")
    # Determine x positions based on number of layers.
    if num_hidden_layers == 1:
        layer_x = [0.2, 0.5, 0.8]  # input, hidden, output
    elif num_hidden_layers == 2:
        layer_x = [0.2, 0.4, 0.6, 0.8]  # input, hidden1, hidden2, output
    
    # Plot input nodes
    if input_dim == 1:
        input_y = [0.5]  # center the single input node
    else:
        input_y = np.linspace(0.2, 0.8, input_dim)
    for i, y in enumerate(input_y):
        plt.scatter([layer_x[0]], [y], c='black', s=100)
        plt.text(layer_x[0] - 0.05, y, f'$x_{{{i+1}}}$', ha='right', fontsize=12)
        # For the first layer, display lambda values above the input nodes.
        lambda_val = layers[0].lambdas[i].item() if i < layers[0].lambdas.numel() else 0.0
        plt.text(layer_x[0] + 0.05, y + 0.0125, f'$\\lambda_{{{i+1}}}={lambda_val:.3f}$', ha='center', va='bottom', fontsize=10)

    # Plot hidden layers and output node
    if num_hidden_layers == 1:
        # One hidden layer
        hidden_y = np.linspace(0.2, 0.8, layers[0].hidden_dim)
        for i, y in enumerate(hidden_y):
            plt.scatter([layer_x[1]], [y], c='blue', s=100)
            plt.text(layer_x[1] + 0.05, y, f'$q_{{{i}}}$', ha='left', fontsize=10)
            # Draw connections from each input node to this hidden node
            for j, in_y in enumerate(input_y):
                plt.plot([layer_x[0], layer_x[1]], [in_y, y], 'k-', alpha=0.5)
        # Output node
        plt.scatter([layer_x[2]], [0.5], c='red', s=100)
        plt.text(layer_x[2] + 0.05, 0.5, 'output', ha='left', fontsize=12)
        # Draw connections from hidden nodes to output node
        for h_y in hidden_y:
            plt.plot([layer_x[1], layer_x[2]], [h_y, 0.5], 'k-', alpha=0.5)
        # Label edges with activation function symbols
        mid_x1 = (layer_x[0] + layer_x[1]) / 2
        mid_x2 = (layer_x[1] + layer_x[2]) / 2
        plt.text(mid_x1, 0.15, '$\\varphi$', ha='center', fontsize=14)
        plt.text(mid_x2, 0.15, '$\\Phi$', ha='center', fontsize=14)
        # Show η value near output node
        eta_val = layers[0].eta.item()
        plt.text(layer_x[2] + 0.05, 0.15, f'$\\eta={eta_val:.3e}$', ha='left', fontsize=10)
    elif num_hidden_layers == 2:
        # Two hidden layers: plot first hidden layer, then second hidden layer, then output.
        # First hidden layer:
        hidden1_y = np.linspace(0.2, 0.8, layers[0].hidden_dim)
        for i, y in enumerate(hidden1_y):
            plt.scatter([layer_x[1]], [y], c='blue', s=100)
            plt.text(layer_x[1] + 0.02, y, f'$q^1_{{{i}}}$', ha='left', fontsize=10)
            for j, in_y in enumerate(input_y):
                plt.plot([layer_x[0], layer_x[1]], [in_y, y], 'k-', alpha=0.5)
        # Second hidden layer:
        hidden2_y = np.linspace(0.2, 0.8, layers[1].hidden_dim)
        for i, y in enumerate(hidden2_y):
            plt.scatter([layer_x[2]], [y], c='blue', s=100)
            plt.text(layer_x[2] + 0.02, y, f'$q^2_{{{i}}}$', ha='left', fontsize=10)
            for h1_y in hidden1_y:
                plt.plot([layer_x[1], layer_x[2]], [h1_y, y], 'k-', alpha=0.5)
        # Output node:
        plt.scatter([layer_x[3]], [0.5], c='red', s=100)
        plt.text(layer_x[3] + 0.02, 0.5, 'output', ha='left', fontsize=12)
        for h2_y in hidden2_y:
            plt.plot([layer_x[2], layer_x[3]], [h2_y, 0.5], 'k-', alpha=0.5)
        # Label the edges with activation function symbols:
        mid_x1 = (layer_x[0] + layer_x[1]) / 2
        mid_x2 = (layer_x[1] + layer_x[2]) / 2
        mid_x3 = (layer_x[2] + layer_x[3]) / 2
        plt.text(mid_x1, 0.1, '$\\varphi_1$', ha='center', fontsize=14)
        plt.text(mid_x2, 0.1, '$\\varphi_2$', ha='center', fontsize=14)
        plt.text(mid_x3, 0.1, '$\\Phi_2$', ha='center', fontsize=14)
        # Show η value from final layer near output
        eta_val = layers[1].eta.item()
        plt.text(layer_x[3] + 0.02, 0.1, f'$\\eta={eta_val:.3e}$', ha='left', fontsize=10)
    
    plt.axis('off')
    
    # -------------------------------
    # Plotting the Spline Functions
    # -------------------------------
    if num_hidden_layers == 1:
        # For one hidden layer, plot the two splines from the single SprecherLayer.
        # Subplot for Outer Function Φ:
        plt.subplot(232)
        plt.title("Outer Function $\\Phi$")
        with torch.no_grad():
            x_Phi = torch.linspace(layers[0].Phi.in_min, layers[0].Phi.in_max, NUM_PLOT_POINTS)
            x_Phi = x_Phi.to(layers[0].Phi.knots.device)
            y_Phi = layers[0].Phi(x_Phi)
        plt.plot(x_Phi.cpu(), y_Phi.cpu())
        plt.grid(True)
        # Subplot for Inner Function φ:
        plt.subplot(233)
        plt.title("Inner Function $\\varphi$")
        with torch.no_grad():
            x_phi = torch.linspace(layers[0].phi.in_min, layers[0].phi.in_max, NUM_PLOT_POINTS)
            x_phi = x_phi.to(layers[0].phi.knots.device)
            y_phi = layers[0].phi(x_phi)
        plt.plot(x_phi.cpu(), y_phi.cpu())
        plt.grid(True)
    elif num_hidden_layers == 2:
        # For two hidden layers, we show three splines:
        # First hidden layer's φ
        plt.subplot(232)
        plt.title("Hidden Layer 1: $\\varphi_1$")
        with torch.no_grad():
            x_phi1 = torch.linspace(layers[0].phi.in_min, layers[0].phi.in_max, NUM_PLOT_POINTS)
            x_phi1 = x_phi1.to(layers[0].phi.knots.device)
            y_phi1 = layers[0].phi(x_phi1)
        plt.plot(x_phi1.cpu(), y_phi1.cpu())
        plt.grid(True)
        # Second hidden layer's φ
        plt.subplot(233)
        plt.title("Hidden Layer 2: $\\varphi_2$")
        with torch.no_grad():
            x_phi2 = torch.linspace(layers[1].phi.in_min, layers[1].phi.in_max, NUM_PLOT_POINTS)
            x_phi2 = x_phi2.to(layers[1].phi.knots.device)
            y_phi2 = layers[1].phi(x_phi2)
        plt.plot(x_phi2.cpu(), y_phi2.cpu())
        plt.grid(True)
        # Second hidden layer's Outer Spline Φ (final layer)
        plt.subplot(234)
        plt.title("Hidden Layer 2: $\\Phi_2$")
        with torch.no_grad():
            x_Phi2 = torch.linspace(layers[1].Phi.in_min, layers[1].Phi.in_max, NUM_PLOT_POINTS)
            x_Phi2 = x_Phi2.to(layers[1].Phi.knots.device)
            y_Phi2 = layers[1].Phi(x_Phi2)
        plt.plot(x_Phi2.cpu(), y_Phi2.cpu())
        plt.grid(True)
    
    # -------------------------------
    # Plotting Function Comparison (Target vs. Network Output)
    # -------------------------------
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
        # For 2D input, use 3D surface plots.
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
            Z_pred = model(points.to(layers[0].Phi.knots.device)).reshape(n, n)
        ax2.plot_surface(X.cpu().numpy(), Y.cpu().numpy(), Z_pred.cpu().numpy(), cmap='viridis')
        ax2.set_title('Network Output')
    
    plt.tight_layout()
    plt.show()

# =============================================================================
# Main Execution
# -----------------------------------------------------------------------------
device = "cuda" if torch.cuda.is_available() else "cpu"

# Train the network using the specified architecture.
model, loss_history, layers = train_network(
    target_function, architecture, total_epochs=100000, print_every=10000, device=device
)

# Create a simple model function for easy evaluation.
def model_fn(x):
    return model(x)

# Plot the network structure, spline functions, and function comparison.
plot_results(model_fn, layers)

# Plot the loss curves over training epochs.
plt.figure(figsize=(8, 6))
plt.plot(loss_history, label="Total Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training Loss Curve")
plt.legend()
plt.show()
