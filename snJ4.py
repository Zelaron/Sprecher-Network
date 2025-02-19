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
# Target function definition:
# Uncomment the desired version.
# (The 1D version is active below; to use 2D, comment out the 1D line and uncomment the 2D line.)

def target_function(x):
    # For 1D:
    # return (x[:, [0]] - 0.5)**5 - (x[:, [0]] - (1/3))**3 + (1/5)*(x[:, [0]] - 0.1)**2
    # For 2D:
    return (torch.exp(torch.sin(torch.pi * x[:, [0]]) + x[:, [1]]**2) - 1) / 7
# -----------------------------

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

# --- Sprecher Layer ---
class SprecherLayer(nn.Module):
    def __init__(self, input_dim=2, hidden_dim=None, in_range=(0, 1), out_range=(0, 1)):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = 2 * input_dim + 1 if hidden_dim is None else hidden_dim
        
        # The inner spline φ: enforce monotonicity for stability.
        self.phi = SimpleSpline(num_knots=1000, in_range=in_range, out_range=in_range, monotonic=True)
        # The outer spline Φ:
        self.Phi = SimpleSpline(num_knots=400, in_range=(-10.0, 12.0), out_range=(-10.0, 12.0))
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
        q_values = torch.arange(self.hidden_dim, device=x.device)
        shifted_x = x.unsqueeze(-1) + self.eta * q_values.view(1, 1, -1)
        shifted_x = torch.clamp(shifted_x, 0, 1)
        phi_out = self.phi(shifted_x)
        inner = (self.lambdas.view(1, -1, 1) * phi_out).sum(dim=1) + q_values
        inner_clamped = torch.clamp(inner, self.Phi.in_min, self.Phi.in_max)
        return self.Phi(inner_clamped).sum(dim=1, keepdim=True)

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
def train_network_iterative(target_function, n_layers=3, total_epochs=100000, print_every=10000, device="cpu"):
    layers = []
    all_losses = []
    
    x_train, y_train, input_dim = generate_data()
    x_train = x_train.to(device)
    y_train = y_train.to(device)

    residual = y_train.clone()
    epochs_per_layer = total_epochs // n_layers

    for r in range(n_layers):
        print(f"\nTraining Layer {r + 1}/{n_layers}")
        layer = SprecherLayer(input_dim=input_dim).to(device)
        optimizer = torch.optim.Adam(layer.parameters(), lr=0.001, weight_decay=1e-7)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=1000
        )
        
        best_loss = float("inf")
        best_state = None
        layer_losses = []
        
        pbar = tqdm(range(epochs_per_layer), desc="Training Layer")
        for epoch in pbar:
            optimizer.zero_grad()
            
            output = layer(x_train)
            mse_loss = torch.mean((output - residual) ** 2)
            # Fix: Avoid NaN by checking if lambdas has more than one element
            if layer.lambdas.numel() > 1:
                lambda_reg = torch.mean((layer.lambdas[1:] - layer.lambdas[:-1]) ** 2)
            else:
                lambda_reg = 0.0
            flatness_penalty_phi = layer.phi.get_flatness_penalty()
            flatness_penalty_Phi = layer.Phi.get_flatness_penalty()
            
            loss = mse_loss + 0.01 * lambda_reg + 0.001 * flatness_penalty_phi + 0.01 * flatness_penalty_Phi
            loss.backward()
            optimizer.step()
            scheduler.step(loss)

            layer_losses.append(loss.item())
            pbar.set_postfix({'loss': f'{loss.item():.2e}'})

            if (epoch + 1) % print_every == 0:
                with torch.no_grad():
                    q_values = torch.arange(layer.hidden_dim, device=x_train.device)
                    shifted_x = x_train.unsqueeze(-1) + layer.eta * q_values.view(1, 1, -1)
                    phi_out = layer.phi(shifted_x)
                    inner = (layer.lambdas.view(1, -1, 1) * phi_out).sum(dim=1) + q_values
                    inner_clamped = torch.clamp(inner, layer.Phi.in_min, layer.Phi.in_max)
                    print(f"Epoch {epoch+1}:")
                    print(f"Values going into Phi - min: {inner_clamped.min():.3f}, max: {inner_clamped.max():.3f}")
                    if inner_clamped.min() < layer.Phi.in_min or inner_clamped.max() > layer.Phi.in_max:
                        print("WARNING: Values exceed Phi's input range!")

            if loss.item() < best_loss:
                best_loss = loss.item()
                best_state = layer.state_dict().copy()
        
        # If no best state was recorded (e.g. due to NaNs), simply use the current state.
        if best_state is None:
            best_state = layer.state_dict()
        layer.load_state_dict(best_state)
        layers.append(layer)
        all_losses.append(layer_losses)

        with torch.no_grad():
            output = layer(x_train)
        residual = residual - output

    def combined_forward(x):
        x = x.to(device)
        output = torch.zeros_like(layers[0](x))
        for layer in layers:
            output = output + layer(x)
        return output
    
    return combined_forward, all_losses, layers

# --- Plotting Results ---
def plot_results(model, layers):
    # Get input dimension from the first layer (assumed same for all)
    input_dim = layers[0].input_dim

    plt.figure(figsize=(15, 10))
    
    # --- Plot Network Structure ---
    plt.subplot(231)
    plt.title("Network Structure")
    layer_x = [0.2, 0.5, 0.8]
    # Plot input nodes (adjust number based on input_dim)
    input_y = np.linspace(0.2, 0.8, input_dim)
    for y in input_y:
        plt.scatter([layer_x[0]], [y], c='black', s=100)
    hidden_y = np.linspace(0.2, 0.8, layers[0].hidden_dim)
    for i, y in enumerate(hidden_y):
        plt.scatter([layer_x[1]], [y], c='blue', s=100)
        plt.text(layer_x[1], y - 0.05, f'q={i}', ha='center', va='top')
        for in_y in input_y:
            plt.plot([layer_x[0], layer_x[1]], [in_y, y], 'k-', alpha=0.5)
    # Output node (always one)
    plt.scatter([layer_x[2]], [0.5], c='red', s=100)
    for h_y in hidden_y:
        plt.plot([layer_x[1], layer_x[2]], [h_y, 0.5], 'k-', alpha=0.5)
    plt.axis('off')
    
    # --- Plot Outer Spline Φ ---
    plt.subplot(232)
    plt.title("Outer Function Φ")
    with torch.no_grad():
        x_Phi = torch.linspace(layers[0].Phi.in_min, layers[0].Phi.in_max, NUM_PLOT_POINTS)
        x_Phi = x_Phi.to(layers[0].Phi.knots.device)
        y_Phi = layers[0].Phi(x_Phi)
    plt.plot(x_Phi.cpu(), y_Phi.cpu())
    plt.grid(True)
    
    # --- Plot Inner Spline φ ---
    plt.subplot(233)
    plt.title("Inner Function φ")
    with torch.no_grad():
        x_phi = torch.linspace(layers[0].phi.in_min, layers[0].phi.in_max, NUM_PLOT_POINTS)
        x_phi = x_phi.to(layers[0].phi.knots.device)
        y_phi = layers[0].phi(x_phi)
    plt.plot(x_phi.cpu(), y_phi.cpu())
    plt.grid(True)
    
    # --- Plot Function Comparison ---
    if input_dim == 1:
        plt.subplot(235)
        plt.title("Function Comparison")
        x_test = torch.linspace(layers[0].phi.in_min, layers[0].phi.in_max, NUM_PLOT_POINTS).reshape(-1, 1)
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
            Z_pred = model(points.to(layers[0].Phi.knots.device)).reshape(n, n)
        ax2.plot_surface(X.cpu().numpy(), Y.cpu().numpy(), Z_pred.cpu().numpy(), cmap='viridis')
        ax2.set_title('Network Output')
    
    plt.tight_layout()
    plt.show()

# --- Main Execution ---
device = "cuda" if torch.cuda.is_available() else "cpu"

combined_forward, all_losses, layers = train_network_iterative(
    target_function, n_layers=1, total_epochs=100000, print_every=20000, device=device
)

# Create a model function for easy evaluation
model = lambda x: combined_forward(x)

plot_results(model, layers)

# Plot loss curves for each layer
for i, layer_loss in enumerate(all_losses):
    plt.plot(layer_loss, label=f"Layer {i+1}")
plt.legend()
plt.title("Loss by Layer")
plt.show()