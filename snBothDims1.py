import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from mpl_toolkits.mplot3d import Axes3D

NUM_PLOT_POINTS = 200
NUM_SURFACE_POINTS = 50
INPUT_RANGE = (0, 1)

def target_function(x):
   # For 1D:
   return (x[:,[0]]-(1/2))**5 - (x[:,[0]]-(1/3))**3 + (1/5)*(x[:,[0]]-(1/10))**2
   # For 2D:
   # return torch.exp(torch.sin(torch.pi*x[:,[0]]) + x[:,[1]]**2)

def get_input_dimension(target_function):
   # Try dimensions 1-3
   for dim in range(1, 4):
       test_x = torch.zeros(1, dim)
       try:
           result = target_function(test_x)
           # Additional check: make sure the computation actually worked
           if torch.isfinite(result).all():
               return dim
       except:
           continue
   raise ValueError("Could not determine input dimension or dimension > 3 not supported")

input_dim = get_input_dimension(target_function)
hidden_dims = 2 * input_dim + 1

class SimpleSpline(nn.Module):
   def __init__(self, num_knots=30, in_range=(0,1), out_range=(0,1)):
       super().__init__()
       self.num_knots = num_knots
       self.in_min, self.in_max = in_range
       self.out_min, self.out_max = out_range
       self.knots = torch.linspace(self.in_min, self.in_max, num_knots)
       self.coeffs = nn.Parameter(torch.linspace(out_range[0], out_range[1], num_knots))

   def forward(self, x):
       x = torch.clamp(x, self.in_min, self.in_max)
       intervals = torch.clamp(torch.searchsorted(self.knots, x) - 1, 0, self.num_knots - 2)
       t = (x - self.knots[intervals]) / (self.knots[intervals + 1] - self.knots[intervals])
       return (1 - t) * self.coeffs[intervals] + t * self.coeffs[intervals + 1]

class SprecherNet(nn.Module):
   def __init__(self, input_dim=1):
       super().__init__()
       self.input_dim = input_dim
       self.hidden_dim = hidden_dims
       self.phi = SimpleSpline(num_knots=200, in_range=INPUT_RANGE, out_range=(0,1))
       self.Phi = SimpleSpline(num_knots=100, in_range=(-3,3), out_range=(-3,3))
       self.lambdas = nn.Parameter(torch.ones(input_dim))
       self.eta = nn.Parameter(torch.tensor(0.1))
       
   def forward(self, x):
       q_values = torch.arange(self.hidden_dim, device=x.device)
       shifted_x = x.unsqueeze(-1) + self.eta * q_values.view(1, 1, -1)
       phi_out = self.phi(shifted_x)
       inner = (self.lambdas.view(1, -1, 1) * phi_out).sum(dim=1) + q_values
       return self.Phi(inner).sum(dim=1, keepdim=True)

def generate_data(target_function, n_samples=1000):
   input_dim = get_input_dimension(target_function)
   if input_dim == 1:
       x = torch.rand(n_samples, 1)
   else:  # input_dim == 2
       # Create a grid of points for better visualization
       n = int(np.sqrt(n_samples))
       x = torch.linspace(0, 1, n)
       y = torch.linspace(0, 1, n)
       X, Y = torch.meshgrid(x, y, indexing='ij')
       x = torch.stack([X.flatten(), Y.flatten()], dim=1)
   return x, target_function(x), input_dim

def train_network(model, target_function, n_epochs=40000):
   optimizer = torch.optim.Adam(model.parameters())
   x_train, y_train, _ = generate_data(target_function)
   
   losses = []
   pbar = tqdm(range(n_epochs), desc="Training")
   for epoch in pbar:
       optimizer.zero_grad()
       loss = torch.mean((model(x_train) - y_train)**2)
       loss.backward()
       optimizer.step()
       losses.append(loss.item())
       pbar.set_postfix({'loss': f'{loss.item():.2e}'})
   
   return losses

def plot_results(model):
   plt.figure(figsize=(15, 10))
   
   # Use GridSpec for better subplot organization
   gs = plt.GridSpec(2, 3)
   
   # Network structure
   plt.subplot(gs[0, 0])
   plt.title("Network Structure")
   layer_x = [0.2, 0.5, 0.8]

   def get_node_positions(n_nodes):
       return np.linspace(0.2, 0.8, n_nodes) if n_nodes > 1 else [0.5]

   # Use actual input dimension
   input_y = get_node_positions(model.input_dim)
   hidden_y = get_node_positions(model.hidden_dim)
   output_y = get_node_positions(1)

   # Plot layers and connections
   for y in input_y:
       plt.scatter([layer_x[0]], [y], c='black', s=100)

   for i, y in enumerate(hidden_y):
       plt.scatter([layer_x[1]], [y], c='blue', s=100)
       plt.text(layer_x[1], y - 0.05, f'q={i}', ha='center', va='top')
       for in_y in input_y:
           plt.plot([layer_x[0], layer_x[1]], [in_y, y], 'k-', alpha=0.5)

   for y in output_y:
       plt.scatter([layer_x[2]], [y], c='red', s=100)
       for h_y in hidden_y:
           plt.plot([layer_x[1], layer_x[2]], [h_y, y], 'k-', alpha=0.5)

   # Adjust text positions to avoid overlap
   lambda_str = '[' + ', '.join(f"{l:.3f}" for l in model.lambdas.data) + ']'
   plt.text(layer_x[0] + 0.05, 0.05, f'λ={lambda_str}')
   plt.text(layer_x[1] - 0.1, 0.15, f'η={model.eta.item():.3f}')
   plt.axis('off')

   # Plot functions
   plt.subplot(gs[0, 1])
   plt.title(f"Outer Function Φ: [{model.Phi.in_min}, {model.Phi.in_max}]→[{model.Phi.out_min}, {model.Phi.out_max}]")
   with torch.no_grad():
       x_Phi = torch.linspace(model.Phi.in_min, model.Phi.in_max, NUM_PLOT_POINTS)
       y_Phi = model.Phi(x_Phi)
   plt.plot(x_Phi, y_Phi)
   plt.grid(True)

   plt.subplot(gs[0, 2])
   plt.title(f"Inner Function φ: [{model.phi.in_min}, {model.phi.in_max}]→[{model.phi.out_min}, {model.phi.out_max}]")
   with torch.no_grad():
       x_phi = torch.linspace(model.phi.in_min, model.phi.in_max, NUM_PLOT_POINTS)
       y_phi = model.phi(x_phi)
   plt.plot(x_phi, y_phi)
   plt.grid(True)

   # Function comparison
   if model.input_dim == 1:
       plt.subplot(gs[1, :])
       plt.title("Function Comparison")
       x_test = torch.linspace(model.phi.in_min, model.phi.in_max, NUM_PLOT_POINTS).reshape(-1, 1)
       y_true = target_function(x_test)
       with torch.no_grad():
           y_pred = model(x_test)
       plt.plot(x_test, y_true, '--', label='Target f(x)')
       plt.plot(x_test, y_pred, '-', label='Network output')
       plt.grid(True)
       plt.legend()
   else:  # 2D case
       # Create a higher resolution grid of test points
       n = NUM_SURFACE_POINTS
       x = torch.linspace(model.phi.in_min, model.phi.in_max, n)
       y = torch.linspace(model.phi.in_min, model.phi.in_max, n)
       X, Y = torch.meshgrid(x, y, indexing='ij')
       grid_points = torch.stack([X.flatten(), Y.flatten()], dim=1)
       
       y_true = target_function(grid_points).reshape(n, n)
       with torch.no_grad():
           y_pred = model(grid_points).reshape(n, n)
       
       # Plot target function
       ax1 = plt.subplot(gs[1, :3//2], projection='3d')
       ax1.plot_surface(X.numpy(), Y.numpy(), y_true.numpy(), cmap='viridis')
       ax1.set_title('Target Function')
       
       # Plot network output
       ax2 = plt.subplot(gs[1, 3//2:], projection='3d')
       ax2.plot_surface(X.numpy(), Y.numpy(), y_pred.numpy(), cmap='viridis')
       ax2.set_title('Network Output')

   plt.tight_layout()
   plt.show()

model = SprecherNet(input_dim=input_dim)
losses = train_network(model, target_function)
plot_results(model)
