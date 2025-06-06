"""Dataset registry and implementations for Sprecher Network experiments."""

import torch
import numpy as np


class Dataset:
    """Base class for SN datasets."""
    
    def sample(self, n: int, device='cpu'):
        """Return (x, y) shaped (n, input_dim) and (n, output_dim)."""
        x = self.generate_inputs(n, device)
        y = self.evaluate(x)
        return x, y
    
    def generate_inputs(self, n: int, device='cpu'):
        """Generate random input samples."""
        # Default: uniform random in [0, 1]^input_dim
        return torch.rand(n, self.input_dim, device=device)
    
    def evaluate(self, x):
        """Evaluate the function at given inputs x."""
        raise NotImplementedError
    
    @property
    def input_dim(self) -> int:
        """Input dimension."""
        raise NotImplementedError
    
    @property
    def output_dim(self) -> int:
        """Output dimension."""
        raise NotImplementedError
    
    def __str__(self):
        return self.__class__.__name__


class Toy1DPoly(Dataset):
    """1D polynomial: (x - 3/10)^5 - (x - 1/3)^3 + (1/5)(x - 1/10)^2"""
    
    @property
    def input_dim(self):
        return 1
    
    @property
    def output_dim(self):
        return 1
    
    def generate_inputs(self, n, device='cpu'):
        """For 1D functions, use linspace for better visualization."""
        return torch.linspace(0, 1, n, device=device).unsqueeze(1)
    
    def evaluate(self, x):
        """Evaluate the polynomial at x."""
        return (x - 3/10)**5 - (x - 1/3)**3 + (1/5)*(x - 1/10)**2


class Toy1DComplex(Dataset):
    """Complex 1D function with multiple sines and exponentials."""
    
    @property
    def input_dim(self):
        return 1
    
    @property
    def output_dim(self):
        return 1
    
    def generate_inputs(self, n, device='cpu'):
        """For 1D functions, use linspace for better visualization."""
        return torch.linspace(0, 1, n, device=device).unsqueeze(1)
    
    def evaluate(self, x):
        """Evaluate the complex function at x."""
        e = torch.exp(torch.tensor(1.0, device=x.device))
        y = (torch.sin((torch.exp(x) - 1.0) / (2.0 * (e - 1.0))) + 
             torch.sin(1.0 - (4.0 * (torch.exp(x + 0.1) - 1.0)) / (5.0 * (e - 1.0))) + 
             torch.sin(2.0 + (torch.exp(x + 0.2) - 1.0) / (e - 1.0)) + 
             torch.sin(3.0 + (torch.exp(x + 0.3) - 1.0) / (5.0 * (e - 1.0))) + 
             torch.sin(4.0 - (6.0 * (torch.exp(x + 0.4) - 1.0)) / (5.0 * (e - 1.0))))
        return y


class Toy2D(Dataset):
    """2D function: exp(sin(11x)) + 3y + 4sin(8y)"""
    
    @property
    def input_dim(self):
        return 2
    
    @property
    def output_dim(self):
        return 1
    
    def generate_inputs(self, n, device='cpu'):
        """Generate grid if n is perfect square, otherwise random."""
        sqrt_n = int(np.sqrt(n))
        if sqrt_n * sqrt_n == n:
            x = torch.linspace(0, 1, sqrt_n, device=device)
            y = torch.linspace(0, 1, sqrt_n, device=device)
            X, Y = torch.meshgrid(x, y, indexing='ij')
            return torch.stack([X.flatten(), Y.flatten()], dim=1)
        else:
            return torch.rand(n, 2, device=device)
    
    def evaluate(self, x):
        """Evaluate the 2D function at x."""
        return torch.exp(torch.sin(11 * x[:, [0]])) + 3 * x[:, [1]] + 4 * torch.sin(8 * x[:, [1]])


class Toy2DVector(Dataset):
    """2D vector function with 2 outputs."""
    
    @property
    def input_dim(self):
        return 2
    
    @property
    def output_dim(self):
        return 2
    
    def generate_inputs(self, n, device='cpu'):
        """Generate grid if n is perfect square, otherwise random."""
        sqrt_n = int(np.sqrt(n))
        if sqrt_n * sqrt_n == n:
            x_vals = torch.linspace(0, 1, sqrt_n, device=device)
            y_vals = torch.linspace(0, 1, sqrt_n, device=device)
            X, Y = torch.meshgrid(x_vals, y_vals, indexing='ij')
            return torch.stack([X.flatten(), Y.flatten()], dim=1)
        else:
            return torch.rand(n, 2, device=device)
    
    def evaluate(self, x):
        """Evaluate the 2D vector function at x."""
        y1 = (torch.exp(torch.sin(torch.pi * x[:, [0]]) + x[:, [1]]**2) - 1) / 7
        y2 = (1/4)*x[:, [1]] + (1/5)*x[:, [1]]**2 - x[:, [0]]**3 + (1/5)*torch.sin(7*x[:, [0]])
        return torch.cat([y1, y2], dim=1)


class Toy100D(Dataset):
    """100D function: exp of mean squared sine (KAN Section 3)."""
    
    @property
    def input_dim(self):
        return 100
    
    @property
    def output_dim(self):
        return 1
    
    def evaluate(self, x):
        """Evaluate the 100D function at x."""
        # f(x_1,...,x_100) = exp(1/100 * sum(sin^2(pi*x_i/2)))
        return torch.exp(torch.mean(torch.sin(np.pi * x / 2)**2, dim=1, keepdim=True))


class SpecialBessel(Dataset):
    """Special function: Bessel function approximation."""
    
    @property
    def input_dim(self):
        return 2
    
    @property
    def output_dim(self):
        return 1
    
    def generate_inputs(self, n, device='cpu'):
        """Generate grid if n is perfect square, otherwise random."""
        sqrt_n = int(np.sqrt(n))
        if sqrt_n * sqrt_n == n:
            x = torch.linspace(0, 1, sqrt_n, device=device)
            y = torch.linspace(0, 1, sqrt_n, device=device)
            X, Y = torch.meshgrid(x, y, indexing='ij')
            return torch.stack([X.flatten(), Y.flatten()], dim=1)
        else:
            return torch.rand(n, 2, device=device)
    
    def evaluate(self, x):
        """Evaluate the Bessel-like function at x."""
        r = torch.sqrt(x[:, [0]]**2 + x[:, [1]]**2)
        return torch.sin(10 * r) / (1 + 10 * r)


class FeynmanUV(Dataset):
    """Feynman dataset: UV radiation formula (Planck's law)."""
    
    @property
    def input_dim(self):
        return 2
    
    @property
    def output_dim(self):
        return 1
    
    def generate_inputs(self, n, device='cpu'):
        """Generate grid if n is perfect square, otherwise random."""
        sqrt_n = int(np.sqrt(n))
        if sqrt_n * sqrt_n == n:
            x = torch.linspace(0.1, 1, sqrt_n, device=device)  # Avoid zero
            y = torch.linspace(0.1, 1, sqrt_n, device=device)
            X, Y = torch.meshgrid(x, y, indexing='ij')
            return torch.stack([X.flatten(), Y.flatten()], dim=1)
        else:
            return torch.rand(n, 2, device=device) * 0.9 + 0.1  # Avoid zero
    
    def evaluate(self, x):
        """Evaluate the Feynman UV formula at x."""
        T = x[:, [0]]  # Temperature parameter
        nu = x[:, [1]]  # Frequency parameter
        scaling_factor = 5.0
        y = (8 * torch.pi * nu**3) / (torch.exp(scaling_factor * nu / T) - 1)
        return y / (8 * torch.pi)  # Normalize


class Poisson2D(Dataset):
    """2D Poisson equation manufactured solution."""
    
    @property
    def input_dim(self):
        return 2
    
    @property
    def output_dim(self):
        return 1
    
    def generate_inputs(self, n, device='cpu'):
        """Generate grid if n is perfect square, otherwise random."""
        sqrt_n = int(np.sqrt(n))
        if sqrt_n * sqrt_n == n:
            x = torch.linspace(0, 1, sqrt_n, device=device)
            y = torch.linspace(0, 1, sqrt_n, device=device)
            X, Y = torch.meshgrid(x, y, indexing='ij')
            return torch.stack([X.flatten(), Y.flatten()], dim=1)
        else:
            return torch.rand(n, 2, device=device)
    
    def evaluate(self, x):
        """Evaluate the Poisson solution at x."""
        return torch.sin(torch.pi * x[:, [0]]) * torch.sin(torch.pi * x[:, [1]])


class Toy4Dto5D(Dataset):
    """4D to 5D vector function with mixed operations."""
    
    @property
    def input_dim(self):
        return 4
    
    @property
    def output_dim(self):
        return 5
    
    def evaluate(self, x):
        """Evaluate the 4D to 5D function at x."""
        y1 = torch.sin(2 * np.pi * x[:, 0]) * torch.cos(np.pi * x[:, 1])
        y2 = torch.exp(-2 * (x[:, 0]**2 + x[:, 1]**2))
        y3 = x[:, 2]**3 - x[:, 3]**2 + 0.5 * torch.sin(5 * x[:, 2])
        y4 = torch.sigmoid(3 * (x[:, 0] + x[:, 1] - x[:, 2] + x[:, 3]))
        y5 = 0.5 * torch.sin(4 * np.pi * x[:, 0] * x[:, 3]) + 0.5 * torch.cos(3 * np.pi * x[:, 1] * x[:, 2])
        
        return torch.stack([y1, y2, y3, y4, y5], dim=1)


# Dataset registry
DATASETS = {
    "toy_1d_poly": Toy1DPoly(),
    "toy_1d_complex": Toy1DComplex(),
    "toy_2d": Toy2D(),
    "toy_2d_vector": Toy2DVector(),
    "toy_100d": Toy100D(),
    "special_bessel": SpecialBessel(),
    "feynman_uv": FeynmanUV(),
    "poisson": Poisson2D(),
    "toy_4d_to_5d": Toy4Dto5D(),
}


def get_dataset(name: str) -> Dataset:
    """Get dataset by name."""
    if name not in DATASETS:
        raise ValueError(f"Unknown dataset '{name}'. Available: {list(DATASETS.keys())}")
    return DATASETS[name]