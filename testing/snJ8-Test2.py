import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from mpl_toolkits.mplot3d import Axes3D  # needed for 3D plotting

# Settings for plotting resolution
NUM_PLOT_POINTS = 200
NUM_SURFACE_POINTS = 50

# -----------------------------
# Set the network architecture here.
# For one hidden layer, e.g.: architecture = [7]
# For two hidden layers, e.g.: architecture = [5, 8]
architecture = [5, 8]

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

# --- SplineLayer Module ---
class SplineLayer(nn.Module):
    def __init__(self, in_features, out_features, is_output=False):
        """
        A generalized spline-based layer.
        For each node i (with fixed bias q_i, initialized as i), we compute:
          shifted = clamp(x + η * q_i)
          φ_out = φ(shifted)
          weighted_sum = Σ_j (λ_j * φ_out_j)
          inner = weighted_sum + q_i
          activated = Ψ(inner)
        For hidden layers (is_output==False), the output is the vector of activated node values.
        For the output layer (is_output==True), we sum over the nodes to produce a scalar.
        """
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.is_output = is_output
        
        # Inner activation spline φ (monotonic)
        self.phi = SimpleSpline(num_knots=300, in_range=(0, 1), out_range=(0, 1), monotonic=True)
        # Outer activation spline Ψ (non-monotonic)
        self.Psi = SimpleSpline(num_knots=200, in_range=(-10.0, 12.0), out_range=(-10.0, 12.0), monotonic=False)
        with torch.no_grad():
            self.Psi.coeffs.data = torch.linspace(self.Psi.out_min, self.Psi.out_max, self.Psi.num_knots)
        
        # Initialize λ parameters (one per input feature)
        lambdas = torch.ones(in_features)
        gamma = 10
        for p in range(1, in_features):
            beta_sum = 0
            for r in range(1, 10):
                beta_r = (in_features**r - 1) / (in_features - 1)
                beta_sum += gamma**(-(p - 1) * beta_r)
            lambdas[p] = beta_sum
        self.lambdas = nn.Parameter(lambdas)
        
        # η parameter (scalar)
        self.eta = nn.Parameter(torch.tensor([1.0 / (gamma * (gamma - 1))]))
        self.eta.data = self.eta.data * 0.05  # scale for manageable inputs
        
        # Fixed q values for each node (used as bias); not learnable.
        q_values = torch.arange(out_features, dtype=torch.float32)
        self.register_buffer('q_values', q_values)
        
    def forward(self, x):
        # x: shape (B, in_features)
        B = x.shape[0]
        # Compute shifted input: shape (B, in_features, out_features)
        shifted = x.unsqueeze(-1) + self.eta * self.q_values.view(1, 1, -1)
        shifted = torch.clamp(shifted, 0, 1)
        phi_out = self.phi(shifted)  # shape: (B, in_features, out_features)
        # Weighted sum over input features:
        weighted_sum = (self.lambdas.view(1, self.in_features, 1) * phi_out).sum(dim=1)  # shape: (B, out_features)
        # Add fixed q values:
        inner = weighted_sum + self.q_values.view(1, -1)
        inner_clamped = torch.clamp(inner, self.Psi.in_min, self.Psi.in_max)
        activated = self.Psi(inner_clamped)  # shape: (B, out_features)
        if self.is_output:
            # Sum over nodes to produce a scalar output per sample.
            return activated.sum(dim=1, keepdim=True)
        else:
            return activated  # pass on the full vector
        
    def get_flatness_penalty(self):
        # Combine flatness penalties from both splines.
        return self.phi.get_flatness_penalty() + self.Psi.get_flatness_penalty()

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

# --- End-to-End Network Training ---
def train_network(target_function, architecture, total_epochs=100000, print_every=10000, device="cpu"):
    """
    Builds a network based on the given architecture.
    For a one-hidden-layer network, a single SplineLayer (with is_output=True) is used.
    For a multi–hidden–layer network, the layers are built sequentially:
       - The first hidden layer uses in_features = input_dim and out_features = architecture[0]
       - Intermediate layers (if any) are non-output layers.
       - The final layer is an output layer which takes the last hidden layer’s output and maps it to a scalar.
    The network is then trained end-to-end using MSE loss and a regularizer (flatness penalty and λ regularization).
    """
    x_train, y_train, input_dim = generate_data()
    x_train = x_train.to(device)
    y_train = y_train.to(device)
    
    layers = []
    if len(architecture) == 0:
        raise ValueError("Architecture must have at least one hidden layer")
    
    if len(architecture) == 1:
        # Single hidden layer: use one SplineLayer that acts as the output layer.
        layer = SplineLayer(in_features=input_dim, out_features=architecture[0], is_output=True).to(device)
        layers.append(layer)
    else:
        # For two (or more) hidden layers:
        # First hidden layer (non-output)
        layer1 = SplineLayer(in_features=input_dim, out_features=architecture[0], is_output=False).to(device)
        layers.append(layer1)
        # For intermediate hidden layers:
        for i in range(1, len(architecture)):
            # For all but the last element, make a non-output layer.
            if i < len(architecture) - 1:
                layer = SplineLayer(in_features=architecture[i-1], out_features=architecture[i], is_output=False).to(device)
                layers.append(layer)
            else:
                # For the last hidden layer, build two layers:
                # First, a non-output layer mapping from previous hidden dimension to the current.
                layer = SplineLayer(in_features=architecture[i-1], out_features=architecture[i], is_output=False).to(device)
                layers.append(layer)
                # Then, an output layer mapping from the last hidden layer to a scalar.
                output_layer = SplineLayer(in_features=architecture[i], out_features=1, is_output=True).to(device)
                layers.append(output_layer)
    
    # Define the combined forward pass.
    def combined_forward(x):
        x = x.to(device)
        out = x
        for layer in layers:
            out = layer(out)
        return out
         
    # Collect all parameters.
    all_params = []
    for layer in layers:
        all_params += list(layer.parameters())
    
    optimizer = torch.optim.Adam(all_params, lr=0.0003, weight_decay=1e-7)
    loss_history = []
    pbar = tqdm(range(total_epochs), desc="Training Network")
    for epoch in pbar:
        optimizer.zero_grad()
        output = combined_forward(x_train)
        mse_loss = torch.mean((output - y_train)**2)
        reg_loss = 0
        for layer in layers:
            if hasattr(layer, 'lambdas'):
                if layer.lambdas.numel() > 1:
                    reg_loss = reg_loss + torch.mean((layer.lambdas[1:] - layer.lambdas[:-1])**2)
            if hasattr(layer, 'get_flatness_penalty'):
                reg_loss = reg_loss + layer.get_flatness_penalty()
        loss = mse_loss + 0.01 * reg_loss
        loss.backward()
        optimizer.step()
        loss_history.append(loss.item())
        pbar.set_postfix({'loss': f'{loss.item():.2e}'})
        if (epoch + 1) % print_every == 0:
            with torch.no_grad():
                print(f"Epoch {epoch+1}, Loss: {loss.item():.2e}")
    return combined_forward, loss_history, layers, input_dim

# --- Plotting Results ---
def plot_results(model, layers, input_dim):
    plt.figure(figsize=(15, 10))
    
    # --- Plot Network Structure with Parameters ---
    plt.subplot(231)
    plt.title("Network Structure with Parameters")
    # Determine the number of layers to display: input, hidden layer(s), and output.
    if len(layers) == 1:
        num_layers = 3
        layer_positions = [0.2, 0.5, 0.8]  # input, hidden, output
        hidden_layers = [layers[0]]
        spline_labels = ['φ', 'Φ']
    else:
        num_layers = len(layers) + 1
        layer_positions = np.linspace(0.1, 0.9, num_layers)
        hidden_layers = layers[:-1]
        # For edges between layers: label the first few with φ₁, φ₂, ... then the final edge with Φ.
        spline_labels = []
        for i in range(num_layers - 1):
            if i < num_layers - 2:
                spline_labels.append(f'φ_{i+1}')
            else:
                spline_labels.append('Φ')
    
    # Plot input nodes.
    if input_dim == 1:
        input_ys = [0.5]
    else:
        input_ys = np.linspace(0.2, 0.8, input_dim)
    for i, y in enumerate(input_ys):
        plt.scatter([layer_positions[0]], [y], c='black', s=100)
        plt.text(layer_positions[0] - 0.05, y, f'$x_{{{i+1}}}$', ha='right')
    
    # Plot hidden layers and output node.
    all_layer_nodes = []
    if len(layers) == 1:
        # Single hidden layer.
        hidden_ys = np.linspace(0.2, 0.8, layers[0].out_features)
        all_layer_nodes.append(hidden_ys)
        for i, y in enumerate(hidden_ys):
            plt.scatter([layer_positions[1]], [y], c='blue', s=100)
            plt.text(layer_positions[1] + 0.05, y, f'$q_{{{i}}}$', ha='left')
            for in_y in input_ys:
                plt.plot([layer_positions[0], layer_positions[1]], [in_y, y], 'k-', alpha=0.5)
        plt.scatter([layer_positions[2]], [0.5], c='red', s=100)
        plt.text(layer_positions[2] + 0.05, 0.5, 'output', ha='left')
        for y in hidden_ys:
            plt.plot([layer_positions[1], layer_positions[2]], [y, 0.5], 'k-', alpha=0.5)
    else:
        current_layer_nodes = input_ys
        for l in range(1, num_layers-1):
            layer_index = l - 1
            num_nodes = hidden_layers[layer_index].out_features
            node_ys = np.linspace(0.2, 0.8, num_nodes)
            all_layer_nodes.append(node_ys)
            for i, y in enumerate(node_ys):
                plt.scatter([layer_positions[l]], [y], c='blue', s=100)
                plt.text(layer_positions[l] + 0.05, y, f'$q^{{({l})}}_{{{i}}}$', ha='left')
                for prev_y in current_layer_nodes:
                    plt.plot([layer_positions[l-1], layer_positions[l]], [prev_y, y], 'k-', alpha=0.5)
            current_layer_nodes = node_ys
        plt.scatter([layer_positions[-1]], [0.5], c='red', s=100)
        plt.text(layer_positions[-1] + 0.05, 0.5, 'output', ha='left')
        for y in current_layer_nodes:
            plt.plot([layer_positions[-2], layer_positions[-1]], [y, 0.5], 'k-', alpha=0.5)
    
    # Add spline labels on edges.
    for i in range(len(spline_labels)):
        mid_x = (layer_positions[i] + layer_positions[i+1]) / 2
        plt.text(mid_x, 0.15, spline_labels[i], ha='center')
    
    # Show η value (taken from the final layer if multiple layers, else from the single layer)
    if len(layers) == 1:
        eta_val = layers[0].eta.item()
    else:
        eta_val = layers[-1].eta.item()
    plt.text(layer_positions[-1] + 0.05, 0.15, f'$\\eta$={eta_val:.3e}', ha='left')
    plt.axis('off')
    
    # --- Plot Splines ---
    if len(layers) == 1:
        # One hidden layer: plot the outer (Ψ) and inner (φ) splines from the single layer.
        plt.subplot(232)
        plt.title("Outer Function Ψ")
        with torch.no_grad():
            x_Psi = torch.linspace(layers[0].Psi.in_min, layers[0].Psi.in_max, NUM_PLOT_POINTS)
            x_Psi = x_Psi.to(layers[0].Psi.knots.device)
            y_Psi = layers[0].Psi(x_Psi)
        plt.plot(x_Psi.cpu(), y_Psi.cpu())
        plt.grid(True)
        
        plt.subplot(233)
        plt.title("Inner Function φ")
        with torch.no_grad():
            x_phi = torch.linspace(layers[0].phi.in_min, layers[0].phi.in_max, NUM_PLOT_POINTS)
            x_phi = x_phi.to(layers[0].phi.knots.device)
            y_phi = layers[0].phi(x_phi)
        plt.plot(x_phi.cpu(), y_phi.cpu())
        plt.grid(True)
    else:
        # Two hidden layers (network has three spline sets):
        # Plot first hidden layer’s inner spline φ.
        plt.subplot(232)
        plt.title("Inner Function φ (Layer 1)")
        with torch.no_grad():
            x_phi1 = torch.linspace(layers[0].phi.in_min, layers[0].phi.in_max, NUM_PLOT_POINTS)
            x_phi1 = x_phi1.to(layers[0].phi.knots.device)
            y_phi1 = layers[0].phi(x_phi1)
        plt.plot(x_phi1.cpu(), y_phi1.cpu())
        plt.grid(True)
        
        # Plot second hidden layer’s inner spline φ.
        plt.subplot(233)
        plt.title("Inner Function φ (Layer 2)")
        with torch.no_grad():
            x_phi2 = torch.linspace(layers[1].phi.in_min, layers[1].phi.in_max, NUM_PLOT_POINTS)
            x_phi2 = x_phi2.to(layers[1].phi.knots.device)
            y_phi2 = layers[1].phi(x_phi2)
        plt.plot(x_phi2.cpu(), y_phi2.cpu())
        plt.grid(True)
        
        # Plot outer function Ψ from the output layer.
        plt.subplot(234)
        plt.title("Outer Function Ψ (Output Layer)")
        with torch.no_grad():
            x_Psi = torch.linspace(layers[-1].Psi.in_min, layers[-1].Psi.in_max, NUM_PLOT_POINTS)
            x_Psi = x_Psi.to(layers[-1].Psi.knots.device)
            y_Psi = layers[-1].Psi(x_Psi)
        plt.plot(x_Psi.cpu(), y_Psi.cpu())
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
    else:  # 2D case: use 3D surface plots
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
            Z_pred = model(points.to(layers[0].Psi.knots.device)).reshape(n, n)
        ax2.plot_surface(X.cpu().numpy(), Y.cpu().numpy(), Z_pred.cpu().numpy(), cmap='viridis')
        ax2.set_title('Network Output')
    
    plt.tight_layout()
    plt.show()

# --- Main Execution ---
device = "cuda" if torch.cuda.is_available() else "cpu"

combined_forward, loss_history, layers, input_dim = train_network(
    target_function, architecture, total_epochs=100000, print_every=10000, device=device
)

# Create a model function for easy evaluation.
model = lambda x: combined_forward(x)

plot_results(model, layers, input_dim)

# Plot loss curves for the entire network.
plt.figure()
plt.plot(loss_history, label="Training Loss")
plt.legend()
plt.title("Loss Curve")
plt.show()