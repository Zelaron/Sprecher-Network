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
# Define the hidden-layer architecture.
# For one hidden layer with 7 nodes, use:
#   architecture = [7]
# For two hidden layers with 5 and 8 nodes, use:
#   architecture = [5, 8]
architecture = [5,8]  # <-- change this list to switch architectures

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
                self.coeffs.data = self.coeffs.data * (out_range[1] - out_range[0]) + out_range[0]
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

# --- Hidden Spline Layer (for hidden layers) ---
class HiddenSplineLayer(nn.Module):
    def __init__(self, in_features, out_features, 
                 phi_knots=300, psi_knots=200, 
                 phi_in_range=(0, 1), phi_out_range=(0, 1),
                 psi_in_range=(-10.0, 12.0), psi_out_range=(-10.0, 12.0)):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        
        # Initialize lambda parameters (one per input feature)
        self.lambdas = nn.Parameter(torch.ones(in_features))
        gamma = 10
        for p in range(1, in_features):
            beta_sum = 0
            for r in range(1, 10):
                beta_r = (in_features**r - 1) / (in_features - 1)
                beta_sum += gamma**(-(p - 1) * beta_r)
            self.lambdas.data[p] = beta_sum
        # Learnable shift parameter η.
        self.eta = nn.Parameter(torch.tensor([1.0 / (gamma * (gamma - 1))]))
        self.eta.data = self.eta.data * 0.05
        
        # Inner spline: monotonic activation for input shift
        self.phi = SimpleSpline(num_knots=phi_knots, in_range=phi_in_range, out_range=phi_out_range, monotonic=True)
        # Hidden activation spline: non-monotonic, learnable activation
        self.psi = SimpleSpline(num_knots=psi_knots, in_range=psi_in_range, out_range=psi_out_range, monotonic=False)
        with torch.no_grad():
            self.psi.coeffs.data = torch.linspace(psi_out_range[0], psi_out_range[1], psi_knots)
    
    def forward(self, x):
        # x shape: [batch, in_features]
        batch = x.size(0)
        device = x.device
        q_values = torch.arange(self.out_features, device=device, dtype=x.dtype)
        # Expand x to [batch, in_features, 1] and add a learnable shift for each node
        shifted = x.unsqueeze(-1) + self.eta * q_values.view(1, 1, -1)
        shifted = torch.clamp(shifted, self.phi.in_min, self.phi.in_max)
        phi_out = self.phi(shifted)  # shape: [batch, in_features, out_features]
        inner = (self.lambdas.view(1, -1, 1) * phi_out).sum(dim=1) + q_values  # shape: [batch, out_features]
        # Apply hidden activation spline (ψ) elementwise
        out = self.psi(inner)
        return out  # shape: [batch, out_features]

# --- Output Spline Layer (for final output) ---
class OutputSplineLayer(nn.Module):
    def __init__(self, in_features, 
                 phi_knots=300, Phi_knots=200,
                 phi_in_range=(0, 1), phi_out_range=(0, 1),
                 Phi_in_range=(-10.0, 12.0), Phi_out_range=(-10.0, 12.0)):
        super().__init__()
        self.in_features = in_features
        
        # Initialize lambda parameters (one per input feature)
        self.lambdas = nn.Parameter(torch.ones(in_features))
        gamma = 10
        for p in range(1, in_features):
            beta_sum = 0
            for r in range(1, 10):
                beta_r = (in_features**r - 1) / (in_features - 1)
                beta_sum += gamma**(-(p - 1) * beta_r)
            self.lambdas.data[p] = beta_sum
        self.eta = nn.Parameter(torch.tensor([1.0 / (gamma * (gamma - 1))]))
        self.eta.data = self.eta.data * 0.05
        
        # Inner spline: monotonic
        self.phi = SimpleSpline(num_knots=phi_knots, in_range=phi_in_range, out_range=phi_out_range, monotonic=True)
        # Outer spline: non-monotonic; used to combine the hidden outputs to a scalar
        self.Phi = SimpleSpline(num_knots=Phi_knots, in_range=Phi_in_range, out_range=Phi_out_range, monotonic=False)
        with torch.no_grad():
            self.Phi.coeffs.data = torch.linspace(Phi_out_range[0], Phi_out_range[1], Phi_knots)
    
    def forward(self, x):
        # x shape: [batch, in_features]
        batch = x.size(0)
        device = x.device
        q_values = torch.arange(self.in_features, device=device, dtype=x.dtype)
        shifted = x.unsqueeze(-1) + self.eta * q_values.view(1, 1, -1)
        shifted = torch.clamp(shifted, self.phi.in_min, self.phi.in_max)
        phi_out = self.phi(shifted)  # shape: [batch, in_features, in_features]
        inner = (self.lambdas.view(1, -1, 1) * phi_out).sum(dim=1) + q_values  # shape: [batch, in_features]
        inner = torch.clamp(inner, self.Phi.in_min, self.Phi.in_max)
        out = self.Phi(inner)  # shape: [batch, in_features]
        out = out.sum(dim=1, keepdim=True)  # combine nodes to produce a scalar output
        return out  # shape: [batch, 1]

# --- Spline Block: Composes hidden layers and output layer ---
class SplineBlock(nn.Module):
    def __init__(self, input_dim, architecture):
        super().__init__()
        if len(architecture) == 0:
            raise ValueError("Architecture list must have at least one hidden layer size.")
        layers = []
        prev_dim = input_dim
        # Each element in architecture defines a hidden layer.
        for h in architecture:
            layers.append(HiddenSplineLayer(in_features=prev_dim, out_features=h))
            prev_dim = h
        # Always append the final output layer.
        layers.append(OutputSplineLayer(in_features=prev_dim))
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

# --- Iterative (Boosting) Training ---
def train_network_iterative(target_function, n_blocks=1, total_epochs=100000, print_every=10000, device="cpu"):
    blocks = []
    all_losses = []
    
    x_train, y_train, input_dim = generate_data()
    x_train = x_train.to(device)
    y_train = y_train.to(device)

    residual = y_train.clone()
    epochs_per_block = total_epochs // n_blocks

    for r in range(n_blocks):
        print(f"\nTraining Block {r + 1}/{n_blocks}")
        # Create a SplineBlock based on the global 'architecture' list.
        block = SplineBlock(input_dim=input_dim, architecture=architecture).to(device)
        
        # Lower learning rate for smoother training
        optimizer = torch.optim.Adam(block.parameters(), lr=0.0003, weight_decay=1e-7)
        
        best_loss = float("inf")
        best_state = None
        block_losses = []
        
        pbar = tqdm(range(epochs_per_block), desc="Training Block")
        for epoch in pbar:
            optimizer.zero_grad()
            
            output = block(x_train)
            mse_loss = torch.mean((output - residual) ** 2)
            
            # Regularization for lambdas and flatness penalties for all spline modules
            lambda_reg = 0.0
            flatness_penalty = 0.0
            for layer in block.layers:
                if hasattr(layer, 'lambdas') and layer.lambdas.numel() > 1:
                    lambda_reg += torch.mean((layer.lambdas[1:] - layer.lambdas[:-1]) ** 2)
                if hasattr(layer, 'phi'):
                    flatness_penalty += layer.phi.get_flatness_penalty()
                if hasattr(layer, 'Phi'):
                    flatness_penalty += layer.Phi.get_flatness_penalty()
                if hasattr(layer, 'psi'):
                    flatness_penalty += layer.psi.get_flatness_penalty()
            
            loss = mse_loss + 0.01 * lambda_reg + 0.01 * flatness_penalty
            loss.backward()
            optimizer.step()

            block_losses.append(loss.item())
            pbar.set_postfix({'loss': f'{loss.item():.2e}'})

            if (epoch + 1) % print_every == 0:
                with torch.no_grad():
                    # Inspect the first layer's inner computation for debugging.
                    first_layer = block.layers[0]
                    q_values = torch.arange(first_layer.out_features, device=x_train.device, dtype=x_train.dtype)
                    shifted = x_train.unsqueeze(-1) + first_layer.eta * q_values.view(1, 1, -1)
                    shifted = torch.clamp(shifted, first_layer.phi.in_min, first_layer.phi.in_max)
                    phi_out = first_layer.phi(shifted)
                    inner = (first_layer.lambdas.view(1, -1, 1) * phi_out).sum(dim=1) + q_values
                    inner_clamped = torch.clamp(inner, first_layer.psi.in_min, first_layer.psi.out_max if hasattr(first_layer, 'psi') else first_layer.phi.out_max)
                    print(f"Epoch {epoch+1}: First layer inner values - min: {inner_clamped.min():.3f}, max: {inner_clamped.max():.3f}")
                    
            if loss.item() < best_loss:
                best_loss = loss.item()
                best_state = {k: v.clone() for k, v in block.state_dict().items()}
        
        if best_state is not None:
            block.load_state_dict(best_state)
        blocks.append(block)
        all_losses.append(block_losses)

        with torch.no_grad():
            output = block(x_train)
        residual = residual - output

    def combined_forward(x):
        x = x.to(device)
        output = torch.zeros_like(blocks[0](x))
        for block in blocks:
            output = output + block(x)
        return output
    
    return combined_forward, all_losses, blocks

# --- Plotting Results ---
def plot_results(model, blocks):
    # Use the first block for extracting spline functions and network structure
    block = blocks[0]
    input_dim = get_input_dimension(target_function)
    
    # Determine number of layers in the network: input, hidden layers, output
    num_hidden = len(architecture)
    total_layers = num_hidden + 2  # including input and output
    layer_x = np.linspace(0.1, 0.9, total_layers)
    
    plt.figure(figsize=(18, 12))
    
    # --- Plot Network Structure with Parameters ---
    plt.subplot(231)
    plt.title("Network Structure with Parameters")
    
    # Plot input nodes
    if input_dim == 1:
        input_y = [0.5]
    else:
        input_y = np.linspace(0.2, 0.8, input_dim)
    for i, y in enumerate(input_y):
        plt.scatter([layer_x[0]], [y], c='black', s=100)
        plt.text(layer_x[0] - 0.05, y, f'$x_{i+1}$', ha='right', fontsize=12)
        # Show lambda values from first hidden layer for corresponding input node if available
        first_hidden = block.layers[0]
        if i < first_hidden.lambdas.numel():
            lambda_val = first_hidden.lambdas[i].item()
            plt.text(layer_x[0] + 0.05, y + 0.0125, f'$\\lambda_{{{i+1}}}={lambda_val:.3f}$', ha='center', fontsize=10)
    
    # Plot hidden layers
    for l in range(1, total_layers - 1):
        # Determine number of nodes in this hidden layer
        if l == 1:
            num_nodes = block.layers[0].out_features
        else:
            if l-1 < len(architecture):
                num_nodes = block.layers[l-1].out_features
            else:
                num_nodes = 1
        hidden_y = np.linspace(0.2, 0.8, num_nodes)
        for i, y in enumerate(hidden_y):
            plt.scatter([layer_x[l]], [y], c='blue', s=100)
            plt.text(layer_x[l] + 0.05, y, f'$q_{{{l},{i}}}$', ha='left', fontsize=10)
            # Draw connections from previous layer nodes to this node
            if l == 1:
                prev_nodes = input_y
                for prev_y in prev_nodes:
                    plt.plot([layer_x[0], layer_x[l]], [prev_y, y], 'k-', alpha=0.5)
            else:
                if l-1 < len(architecture):
                    prev_num = block.layers[l-1].out_features
                    prev_hidden_y = np.linspace(0.2, 0.8, prev_num)
                    for prev_y in prev_hidden_y:
                        plt.plot([layer_x[l-1], layer_x[l]], [prev_y, y], 'k-', alpha=0.5)
    
    # Plot output node
    plt.scatter([layer_x[-1]], [0.5], c='red', s=100)
    plt.text(layer_x[-1] + 0.05, 0.5, 'output', ha='left', fontsize=12)
    # Draw connections from last hidden layer to output
    last_hidden_num = block.layers[-2].out_features if len(block.layers) >=2 else block.layers[0].out_features
    last_hidden_y = np.linspace(0.2, 0.8, last_hidden_num)
    for y in last_hidden_y:
        plt.plot([layer_x[-2], layer_x[-1]], [y, 0.5], 'k-', alpha=0.5)
    
    # Show function labels for each set of edges
    if len(architecture) == 1:
        # One hidden layer: label from input->hidden as φ and hidden->output as Φ
        mid_x1 = (layer_x[0] + layer_x[1]) / 2
        mid_x2 = (layer_x[1] + layer_x[-1]) / 2
        plt.text(mid_x1, 0.15, '$\\phi$', ha='center', fontsize=14)
        plt.text(mid_x2, 0.15, '$\\Phi$', ha='center', fontsize=14)
    elif len(architecture) >= 2:
        # Two hidden layers: label input->hidden1 as $\\phi$, hidden1->hidden2 as $\\psi$, and hidden2->output as $\\Phi$
        mid_x1 = (layer_x[0] + layer_x[1]) / 2
        mid_x2 = (layer_x[1] + layer_x[2]) / 2
        mid_x3 = (layer_x[2] + layer_x[-1]) / 2
        plt.text(mid_x1, 0.15, '$\\phi$', ha='center', fontsize=14)
        plt.text(mid_x2, 0.15, '$\\psi$', ha='center', fontsize=14)
        plt.text(mid_x3, 0.15, '$\\Phi$', ha='center', fontsize=14)
    else:
        plt.text((layer_x[0]+layer_x[1])/2, 0.15, '$\\phi$', ha='center', fontsize=14)
    
    # Show η value near output
    eta_val = block.layers[0].eta.item()
    plt.text(layer_x[-1] + 0.05, 0.15, f'$\\eta={eta_val:.3e}$', ha='left', fontsize=10)
    plt.axis('off')
    
    # --- Plot Spline Functions ---
    if len(architecture) == 1:
        # For one hidden layer, plot hidden activation spline (ψ) and output spline (Φ)
        plt.subplot(232)
        plt.title("Hidden Activation Spline (ψ)")
        with torch.no_grad():
            x_psi = torch.linspace(block.layers[0].psi.in_min, block.layers[0].psi.in_max, NUM_PLOT_POINTS)
            y_psi = block.layers[0].psi(x_psi)
        plt.plot(x_psi.cpu(), y_psi.cpu())
        plt.grid(True)
        
        plt.subplot(233)
        plt.title("Output Spline (Φ)")
        with torch.no_grad():
            x_Phi = torch.linspace(block.layers[-1].Phi.in_min, block.layers[-1].Phi.in_max, NUM_PLOT_POINTS)
            y_Phi = block.layers[-1].Phi(x_Phi)
        plt.plot(x_Phi.cpu(), y_Phi.cpu())
        plt.grid(True)
    elif len(architecture) >= 2:
        # For two hidden layers, plot first hidden layer's activation (ψ), second hidden layer's activation (ψ), and output spline (Φ)
        plt.subplot(232)
        plt.title("Hidden Layer 1 Activation (ψ)")
        with torch.no_grad():
            x_psi1 = torch.linspace(block.layers[0].psi.in_min, block.layers[0].psi.in_max, NUM_PLOT_POINTS)
            y_psi1 = block.layers[0].psi(x_psi1)
        plt.plot(x_psi1.cpu(), y_psi1.cpu())
        plt.grid(True)
        
        plt.subplot(233)
        plt.title("Hidden Layer 2 Activation (ψ)")
        with torch.no_grad():
            x_psi2 = torch.linspace(block.layers[1].psi.in_min, block.layers[1].psi.in_min + (block.layers[1].psi.in_max - block.layers[1].psi.in_min), NUM_PLOT_POINTS)
            y_psi2 = block.layers[1].psi(x_psi2)
        plt.plot(x_psi2.cpu(), y_psi2.cpu())
        plt.grid(True)
        
        plt.subplot(234)
        plt.title("Output Spline (Φ)")
        with torch.no_grad():
            x_Phi = torch.linspace(block.layers[-1].Phi.in_min, block.layers[-1].Phi.in_max, NUM_PLOT_POINTS)
            y_Phi = block.layers[-1].Phi(x_Phi)
        plt.plot(x_Phi.cpu(), y_Phi.cpu())
        plt.grid(True)
    else:
        plt.subplot(232)
        plt.title("Inner Spline")
        with torch.no_grad():
            x_phi = torch.linspace(block.layers[0].phi.in_min, block.layers[0].phi.in_max, NUM_PLOT_POINTS)
            y_phi = block.layers[0].phi(x_phi)
        plt.plot(x_phi.cpu(), y_phi.cpu())
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
        x_vals = torch.linspace(0, 1, n)
        y_vals = torch.linspace(0, 1, n)
        X, Y = torch.meshgrid(x_vals, y_vals, indexing='ij')
        points = torch.stack([X.flatten(), Y.flatten()], dim=1)
        Z_true = target_function(points).reshape(n, n)
        ax1.plot_surface(X.cpu().numpy(), Y.cpu().numpy(), Z_true.cpu().numpy(), cmap='viridis')
        ax1.set_title('Target Function')
        
        ax2 = plt.subplot(236, projection='3d')
        with torch.no_grad():
            Z_pred = model(points.to(block.layers[0].phi.knots.device)).reshape(n, n)
        ax2.plot_surface(X.cpu().numpy(), Y.cpu().numpy(), Z_pred.cpu().numpy(), cmap='viridis')
        ax2.set_title('Network Output')
    
    plt.tight_layout()
    plt.show()

# --- Main Execution ---
device = "cuda" if torch.cuda.is_available() else "cpu"

combined_forward, all_losses, blocks = train_network_iterative(
    target_function, n_blocks=1, total_epochs=100000, print_every=10000, device=device
)

# Create a model function for easy evaluation
model = lambda x: combined_forward(x)

plot_results(model, blocks)

# Plot loss curves for each block
plt.figure()
for i, loss_curve in enumerate(all_losses):
    plt.plot(loss_curve, label=f"Block {i+1}")
plt.legend()
plt.title("Loss by Block")
plt.show()