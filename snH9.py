import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from mpl_toolkits.mplot3d import Axes3D

NUM_PLOT_POINTS = 200
NUM_SURFACE_POINTS = 50

def target_function(x):
    original_target = torch.exp(torch.sin(torch.pi * x[:, [0]]) + x[:, [1]]**2)
    return (original_target - 1) / 7  # Rescale to [-1,1] range, approximately

def original_target_function(x):
    return torch.exp(torch.sin(torch.pi * x[:, [0]]) + x[:, [1]]**2)

# Set seeds for reproducibility
SEED = 45
torch.manual_seed(SEED)
np.random.seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

class SimpleSpline(nn.Module):
    def __init__(self, num_knots=30, in_range=(0, 1), out_range=(0, 1), monotonic=False):
        super().__init__()
        self.num_knots = num_knots
        self.in_min, self.in_max = in_range
        self.out_min, self.out_max = out_range
        self.monotonic = monotonic
        # Register knots as a buffer so they move with the module when calling .to(device)
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
        # Ensure x is on the same device as the knots buffer.
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

class SprecherLayer(nn.Module):
    def __init__(self, input_dim=2, hidden_dim=None, in_range=(0, 1), out_range=(0, 1)):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = 2 * input_dim + 1 if hidden_dim is None else hidden_dim
        
        self.phi = SimpleSpline(num_knots=1000, in_range=in_range, out_range=in_range, monotonic=True)
        self.Phi = SimpleSpline(num_knots=400, in_range=(-10.0, 12.0), out_range=(-10.0, 12.0))

        with torch.no_grad():
            self.Phi.coeffs.data = torch.linspace(self.Phi.out_min, self.Phi.out_max, self.Phi.num_knots)
          
        lambdas = torch.ones(input_dim)
        gamma = 10
        for p in range(1, input_dim):
            beta_sum = 0
            for r in range(1, 10):
                beta_r = (input_dim**r - 1) / (input_dim - 1)
                beta_sum += gamma**(-(p - 1) * beta_r)
            lambdas[p] = beta_sum
        self.lambdas = nn.Parameter(lambdas)

        self.eta = nn.Parameter(torch.tensor([1.0 / (gamma * (gamma - 1))]))
        self.eta.data = self.eta.data * 0.05  # Adjust eta scaling factor for more manageable inputs

    def forward(self, x):
        q_values = torch.arange(self.hidden_dim, device=x.device)
        shifted_x = x.unsqueeze(-1) + self.eta * q_values.view(1, 1, -1)
        shifted_x = torch.clamp(shifted_x, 0, 1)
        
        phi_out = self.phi(shifted_x)
        inner = (self.lambdas.view(1, -1, 1) * phi_out).sum(dim=1) + q_values
        
        # Clamp values to avoid out-of-range issues
        inner_clamped = torch.clamp(inner, self.Phi.in_min, self.Phi.in_max)
        
        return self.Phi(inner_clamped).sum(dim=1, keepdim=True)

def generate_data(n_samples=32):
    x = torch.linspace(0, 1, n_samples)
    y = torch.linspace(0, 1, n_samples)
    X, Y = torch.meshgrid(x, y, indexing='ij')
    points = torch.stack([X.flatten(), Y.flatten()], dim=1)
    return points, target_function(points)

def train_network_iterative(target_function, n_layers=3, total_epochs=100000, print_every=10000, device="cpu"):
    layers = []
    all_losses = []
    
    x_train, y_train = generate_data()
    x_train = x_train.to(device)
    y_train = y_train.to(device)

    residual = y_train.clone()
    
    epochs_per_layer = total_epochs // n_layers

    for r in range(n_layers):
        print(f"\nTraining Layer {r + 1}/{n_layers}")
        layer = SprecherLayer().to(device)
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
            lambda_reg = torch.mean((layer.lambdas[1:] - layer.lambdas[:-1]) ** 2)
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

def plot_results(model, layers):
    plt.figure(figsize=(15, 10))
    
    plt.subplot(231)
    plt.title("Network Structure")
    layer_x = [0.2, 0.5, 0.8]
    
    input_y = np.linspace(0.2, 0.8, 2)
    for y in input_y:
        plt.scatter([layer_x[0]], [y], c='black', s=100)

    hidden_y = np.linspace(0.2, 0.8, layers[0].hidden_dim)
    for i, y in enumerate(hidden_y):
        plt.scatter([layer_x[1]], [y], c='blue', s=100)
        plt.text(layer_x[1], y - 0.05, f'q={i}', ha='center', va='top')
        for in_y in input_y:
            plt.plot([layer_x[0], layer_x[1]], [in_y, y], 'k-', alpha=0.5)
    
    plt.scatter([layer_x[2]], [0.5], c='red', s=100)
    for h_y in hidden_y:
        plt.plot([layer_x[1], layer_x[2]], [h_y, 0.5], 'k-', alpha=0.5)
    
    plt.axis('off')
    
    plt.subplot(232)
    plt.title("Outer Function Φ")
    with torch.no_grad():
        x_Phi = torch.linspace(layers[0].Phi.in_min, layers[0].Phi.in_max, NUM_PLOT_POINTS)
        # Ensure x_Phi is on the same device as Phi
        x_Phi = x_Phi.to(layers[0].Phi.knots.device)
        y_Phi = layers[0].Phi(x_Phi)
    plt.plot(x_Phi.cpu(), y_Phi.cpu())
    plt.grid(True)
    
    plt.subplot(233)
    plt.title("Inner Function φ")
    with torch.no_grad():
        x_phi = torch.linspace(layers[0].phi.in_min, layers[0].phi.in_max, NUM_PLOT_POINTS)
        x_phi = x_phi.to(layers[0].phi.knots.device)
        y_phi = layers[0].phi(x_phi)
    plt.plot(x_phi.cpu(), y_phi.cpu())
    plt.grid(True)
    
    x = torch.linspace(0, 1, NUM_SURFACE_POINTS)
    y = torch.linspace(0, 1, NUM_SURFACE_POINTS)
    X, Y = torch.meshgrid(x, y, indexing='ij')
    points = torch.stack([X.flatten(), Y.flatten()], dim=1)
    
    ax1 = plt.subplot(235, projection='3d')
    Z_true = original_target_function(points).reshape(NUM_SURFACE_POINTS, NUM_SURFACE_POINTS)
    ax1.plot_surface(X, Y, Z_true.detach().cpu(), cmap='viridis')
    ax1.set_title('Target Function')
    
    ax2 = plt.subplot(236, projection='3d')
    with torch.no_grad():
        # Move points to the model's device before evaluation.
        Z_pred = model(points.to(layers[0].Phi.knots.device)).reshape(NUM_SURFACE_POINTS, NUM_SURFACE_POINTS)
    ax2.plot_surface(X, Y, Z_pred.detach().cpu(), cmap='viridis')
    ax2.set_title('Network Output')
    
    plt.tight_layout()
    plt.show()

# Example Usage:
device = "cuda" if torch.cuda.is_available() else "cpu"

combined_forward, all_losses, layers = train_network_iterative(
    target_function, n_layers=3, total_epochs=100000, print_every=10000, device=device
)

# Create a model function for easy plotting
model = lambda x: combined_forward(x)

plot_results(model, layers)

for i, layer_loss in enumerate(all_losses):
    plt.plot(layer_loss, label=f"Layer {i+1}")
plt.legend()
plt.title("Loss by Layer")
plt.show()
