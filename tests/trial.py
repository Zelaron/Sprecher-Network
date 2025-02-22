import os
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from mpl_toolkits.mplot3d import Axes3D  # needed for 3D plotting
import matplotlib.gridspec as gridspec

# --- Set architecture and TOTAL_EPOCHS manually ---
architecture = [5, 8]
TOTAL_EPOCHS = 100000

# --- Configuration Flags and Settings ---
SAVE_FINAL_PLOT = True  # Set to True to save the final results plot as an image
FINAL_IMAGE_WIDTH, FINAL_IMAGE_HEIGHT = 3840, 2160  # in pixels

# --- Target function definition ---
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

# === Spline Module ===
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
        
        # Optional: Reduced number of knots for speed
        self.phi = SimpleSpline(num_knots=100, in_range=(0, 1), out_range=(0, 1), monotonic=True)
        self.Phi = SimpleSpline(num_knots=50, in_range=(-10.0, 12.0), out_range=(-10.0, 12.0))
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

# --- Training Function with Early Stopping and Scheduler ---
def train_network(target_function, architecture, total_epochs=100000, print_every=10000, device="cpu"):
    # Generate training data
    x_train, y_train, input_dim = generate_data()
    x_train = x_train.to(device)
    y_train = y_train.to(device)
    
    # Generate validation data (1000 random points)
    val_x = torch.rand(1000, input_dim).to(device)
    val_y = target_function(val_x).to(device)
    
    # Initialize model and optimizer
    model = SprecherMultiLayerNetwork(input_dim=input_dim, architecture=architecture).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0003, weight_decay=1e-7)
    
    # Learning rate scheduler
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=10, min_lr=1e-6
    )
    
    # Early stopping parameters
    check_every = 100  # Check validation loss every 100 epochs
    patience = 5000   # Stop if no improvement for 5000 epochs
    min_delta = 1e-6  # Minimum improvement threshold
    best_val_loss = float("inf")
    wait = 0
    best_state = None
    
    losses = []
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
        
        # Validation check and early stopping
        if (epoch + 1) % check_every == 0:
            with torch.no_grad():
                val_output = model(val_x)
                val_loss = torch.mean((val_output - val_y) ** 2).item()
            
            # Update scheduler
            scheduler.step(val_loss)
            
            # Early stopping logic
            if val_loss < best_val_loss - min_delta:
                best_val_loss = val_loss
                wait = 0
                best_state = model.state_dict()  # Save best model based on validation loss
            else:
                wait += check_every
                if wait >= patience:
                    print(f"Early stopping at epoch {epoch+1} with best validation loss: {best_val_loss:.4e}")
                    break
        
        if (epoch + 1) % print_every == 0:
            print(f"Epoch {epoch+1}: Loss = {loss.item():.4e}")
    
    # Load the best model state
    if best_state is not None:
        model.load_state_dict(best_state)
    return model, losses, model.layers

# --- Updated Plotting Function for Network Structure ---
def plot_network_structure_ax(ax, layers, input_dim):
    """
    Plots a schematic of the network structure on the given axis using LaTeX-formatted subscripts/superscripts.
    Displays lambda values (from the first hidden block) above the input nodes and η near the output.
    Now ensures a single output node is vertically centered if d_out == 1.
    """
    num_blocks = len(layers)
    if num_blocks == 0:
        ax.text(0.5, 0.5, "No network", ha='center', va='center', fontsize=12)
        ax.axis('off')
        return
    
    # Determine number of hidden blocks
    if layers[-1].is_final:
        num_hidden = num_blocks - 1
    else:
        num_hidden = num_blocks
    
    # Prepare x positions for each "layer" (input + each block)
    layer_x = np.linspace(0.2, 0.8, num_blocks + 1)
    
    # Plot input nodes
    if input_dim == 1:
        input_y = [0.5]
    else:
        input_y = np.linspace(0.2, 0.8, input_dim)
    
    for i, y in enumerate(input_y):
        ax.scatter(layer_x[0], y, c='black', s=100)
        ax.text(layer_x[0] - 0.05, y, f'$x_{{{i+1}}}$', ha='right', fontsize=12)
        # Display lambda values from the first hidden block above input nodes
        if num_hidden >= 1:
            first_block = layers[0]
            if i < first_block.lambdas.shape[0]:
                lambda_val = first_block.lambdas[i, 0].item()
            else:
                lambda_val = 0.0
            ax.text(layer_x[0] + 0.05, y + 0.0125,
                    f'$\\lambda_{{{i+1}}}={lambda_val:.3f}$',
                    ha='center', va='bottom', fontsize=10)
    
    # For each block, plot the layer of nodes and connect from the previous layer
    prev_y = input_y
    for i, block in enumerate(layers):
        d_in = block.d_in
        d_out = block.d_out
        
        # Center a single output node; else space them out
        if block.is_final and d_out == 1:
            new_y = [0.5]
        else:
            new_y = np.linspace(0.2, 0.8, d_out)
        
        color = 'red' if block.is_final else 'blue'
        
        for j, yy in enumerate(new_y):
            ax.scatter(layer_x[i+1], yy, c=color, s=100)
            if block.is_final:
                if d_out == 1:
                    ax.text(layer_x[i+1] + 0.02, yy, 'output', ha='left', fontsize=12)
                else:
                    ax.text(layer_x[i+1] + 0.02, yy, f'output_{j}', ha='left', fontsize=12)
            else:
                ax.text(layer_x[i+1] + 0.02, yy, f'$q^{{{i+1}}}_{{{j}}}$', ha='left', fontsize=10)
        
        for y_in in prev_y:
            for y_out in new_y:
                ax.plot([layer_x[i], layer_x[i+1]], [y_in, y_out], 'k-', alpha=0.5)
        
        # Label the spline between layers
        mid_x = 0.5 * (layer_x[i] + layer_x[i+1])
        if i < num_blocks - 1:
            ax.text(mid_x, 0.1, f'$\\varphi_{{{i+1}}}$', ha='center', fontsize=14)
        else:
            if num_hidden > 0:
                ax.text(mid_x, 0.1, f'$\\Phi_{{{num_hidden}}}$', ha='center', fontsize=14)
            else:
                ax.text(mid_x, 0.1, '$f$', ha='center', fontsize=14)
        
        prev_y = new_y
    
    # Label η near the output if we have at least one hidden layer
    if num_hidden > 0:
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
    ax_net.set_title("Network Structure", fontsize=12)
    
    for i, layer in enumerate(layers):
        ax_spline = fig.add_subplot(gs[0, i+1])
        if layer.is_final:
            ax_spline.set_title("Final Layer ($\\Phi$)", fontsize=12)
            with torch.no_grad():
                x_vals = torch.linspace(layer.Phi.in_min, layer.Phi.in_max, 200).to(layer.Phi.knots.device)
                y_vals = layer.Phi(x_vals)
            ax_spline.plot(x_vals.cpu(), y_vals.cpu(), 'm-')
        else:
            ax_spline.set_title(f"Hidden Layer {i+1} ($\\varphi$)", fontsize=12)
            with torch.no_grad():
                x_vals = torch.linspace(layer.phi.in_min, layer.phi.in_max, 200).to(layer.phi.knots.device)
                y_vals = layer.phi(x_vals)
            ax_spline.plot(x_vals.cpu(), y_vals.cpu(), 'c-')
        ax_spline.grid(True)
    
    if input_dim == 1:
        ax_func = fig.add_subplot(gs[1, :])
        ax_func.set_title("Function Comparison", fontsize=12)
        x_test = torch.linspace(0, 1, 200).reshape(-1, 1)
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
        n = 50
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
    # Return the figure for further processing (showing or saving)
    return fig

# --- Main Execution ---
device = "cuda" if torch.cuda.is_available() else "cpu"

model, losses, layers = train_network(
    target_function, architecture=architecture, total_epochs=TOTAL_EPOCHS, print_every=TOTAL_EPOCHS//10, device=device
)

# Plot the combined results and capture the figure
fig_results = plot_results(model, layers)

# Optionally save the final plot with the desired resolution inside the "plots" directory
if SAVE_FINAL_PLOT:
    # Create the plots directory if it doesn't exist
    os.makedirs("plots", exist_ok=True)
    # Determine the filename based on input dimension, architecture, and total epochs
    input_dim_guess = get_input_dimension(target_function)
    prefix = "OneVar" if input_dim_guess == 1 else "TwoVars"
    arch_str = "-".join(map(str, architecture))
    filename = f"{prefix}-{arch_str}-{TOTAL_EPOCHS}-epochs.png"
    filepath = os.path.join("plots", filename)
    # Set figure size so that, with dpi=240, the saved image is 3840x2160 pixels
    fig_results.set_size_inches(16, 9)
    fig_results.savefig(filepath, dpi=240)
    print(f"Final plot saved as {filepath}")

plt.show()

# Plot the loss curve
plt.figure(figsize=(8, 5))
plt.plot(losses, label="Training Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Loss Curve")
plt.legend()
plt.show()
