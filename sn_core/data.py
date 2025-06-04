"""Dataset registry and implementations for Sprecher Network experiments."""

import torch
import numpy as np


class Dataset:
    """Base class for SN datasets."""
    
    def sample(self, n: int, device='cpu'):
        """Return (x, y) shaped (n, input_dim) and (n, output_dim)."""
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
    
    def sample(self, n, device='cpu'):
        x = torch.linspace(0, 1, n, device=device).unsqueeze(1)
        y = (x - 3/10)**5 - (x - 1/3)**3 + (1/5)*(x - 1/10)**2
        return x, y


class Toy1DComplex(Dataset):
    """Complex 1D function with multiple sines and exponentials."""
    
    @property
    def input_dim(self):
        return 1
    
    @property
    def output_dim(self):
        return 1
    
    def sample(self, n, device='cpu'):
        x = torch.linspace(0, 1, n, device=device).unsqueeze(1)
        e = torch.exp(torch.tensor(1.0, device=device))
        y = (torch.sin((torch.exp(x) - 1.0) / (2.0 * (e - 1.0))) + 
             torch.sin(1.0 - (4.0 * (torch.exp(x + 0.1) - 1.0)) / (5.0 * (e - 1.0))) + 
             torch.sin(2.0 + (torch.exp(x + 0.2) - 1.0) / (e - 1.0)) + 
             torch.sin(3.0 + (torch.exp(x + 0.3) - 1.0) / (5.0 * (e - 1.0))) + 
             torch.sin(4.0 - (6.0 * (torch.exp(x + 0.4) - 1.0)) / (5.0 * (e - 1.0))))
        return x, y


class Toy2D(Dataset):
    """2D function: exp(sin(11x)) + 3y + 4sin(8y)"""
    
    @property
    def input_dim(self):
        return 2
    
    @property
    def output_dim(self):
        return 1
    
    def sample(self, n, device='cpu'):
        # Create grid if n is a perfect square, otherwise random
        sqrt_n = int(np.sqrt(n))
        if sqrt_n * sqrt_n == n:
            x = torch.linspace(0, 1, sqrt_n, device=device)
            y = torch.linspace(0, 1, sqrt_n, device=device)
            X, Y = torch.meshgrid(x, y, indexing='ij')
            x = torch.stack([X.flatten(), Y.flatten()], dim=1)
        else:
            x = torch.rand(n, 2, device=device)
        
        y = torch.exp(torch.sin(11 * x[:, [0]])) + 3 * x[:, [1]] + 4 * torch.sin(8 * x[:, [1]])
        return x, y


class Toy2DVector(Dataset):
    """2D vector function with 2 outputs."""
    
    @property
    def input_dim(self):
        return 2
    
    @property
    def output_dim(self):
        return 2
    
    def sample(self, n, device='cpu'):
        # Create grid if n is a perfect square, otherwise random
        sqrt_n = int(np.sqrt(n))
        if sqrt_n * sqrt_n == n:
            x_vals = torch.linspace(0, 1, sqrt_n, device=device)
            y_vals = torch.linspace(0, 1, sqrt_n, device=device)
            X, Y = torch.meshgrid(x_vals, y_vals, indexing='ij')
            x = torch.stack([X.flatten(), Y.flatten()], dim=1)
        else:
            x = torch.rand(n, 2, device=device)
        
        y = torch.cat([
            (torch.exp(torch.sin(torch.pi * x[:, [0]]) + x[:, [1]]**2) - 1) / 7,
            (1/4)*x[:, [1]] + (1/5)*x[:, [1]]**2 - x[:, [0]]**3 + (1/5)*torch.sin(7*x[:, [0]])
        ], dim=1)
        return x, y


class Toy100D(Dataset):
    """100D function: sum of sinusoids (KAN Section 3)."""
    
    @property
    def input_dim(self):
        return 100
    
    @property
    def output_dim(self):
        return 1
    
    def sample(self, n, device='cpu'):
        x = torch.rand(n, 100, device=device)
        # Sum of sinusoids with different frequencies
        y = torch.zeros(n, 1, device=device)
        for i in range(100):
            y += torch.sin((i + 1) * x[:, [i]])
        y = y / 100  # Normalize
        return x, y


class SpecialBessel(Dataset):
    """Special function: Bessel function approximation."""
    
    @property
    def input_dim(self):
        return 2
    
    @property
    def output_dim(self):
        return 1
    
    def sample(self, n, device='cpu'):
        # Create grid if n is a perfect square
        sqrt_n = int(np.sqrt(n))
        if sqrt_n * sqrt_n == n:
            x = torch.linspace(0, 1, sqrt_n, device=device)
            y = torch.linspace(0, 1, sqrt_n, device=device)
            X, Y = torch.meshgrid(x, y, indexing='ij')
            x = torch.stack([X.flatten(), Y.flatten()], dim=1)
        else:
            x = torch.rand(n, 2, device=device)
        
        # Approximate Bessel-like function
        r = torch.sqrt(x[:, [0]]**2 + x[:, [1]]**2)
        y = torch.sin(10 * r) / (1 + 10 * r)
        return x, y


class FeynmanUV(Dataset):
    """Feynman dataset: UV radiation formula (Planck's law)."""
    
    @property
    def input_dim(self):
        return 2
    
    @property
    def output_dim(self):
        return 1
    
    def sample(self, n, device='cpu'):
        # Create grid if n is a perfect square
        sqrt_n = int(np.sqrt(n))
        if sqrt_n * sqrt_n == n:
            x = torch.linspace(0.1, 1, sqrt_n, device=device)  # Avoid zero
            y = torch.linspace(0.1, 1, sqrt_n, device=device)
            X, Y = torch.meshgrid(x, y, indexing='ij')
            x = torch.stack([X.flatten(), Y.flatten()], dim=1)
        else:
            x = torch.rand(n, 2, device=device) * 0.9 + 0.1  # Avoid zero
        
        # Feynman UV formula with better scaling
        # x[:, 0] represents temperature (scaled)
        # x[:, 1] represents frequency (scaled)
        T = x[:, [0]]  # Temperature parameter
        nu = x[:, [1]]  # Frequency parameter
        
        # Use a scaled version that gives interesting behavior
        # The key is to keep nu/T in a reasonable range
        scaling_factor = 5.0
        y = (8 * torch.pi * nu**3) / (torch.exp(scaling_factor * nu / T) - 1)
        
        # Normalize to reasonable range
        y = y / (8 * torch.pi)  # This makes max ≈ 1 when nu/T is optimal
        
        return x, y


class Poisson2D(Dataset):
    """2D Poisson equation manufactured solution."""
    
    @property
    def input_dim(self):
        return 2
    
    @property
    def output_dim(self):
        return 1
    
    def sample(self, n, device='cpu'):
        # Create grid if n is a perfect square
        sqrt_n = int(np.sqrt(n))
        if sqrt_n * sqrt_n == n:
            x = torch.linspace(0, 1, sqrt_n, device=device)
            y = torch.linspace(0, 1, sqrt_n, device=device)
            X, Y = torch.meshgrid(x, y, indexing='ij')
            x = torch.stack([X.flatten(), Y.flatten()], dim=1)
        else:
            x = torch.rand(n, 2, device=device)
        
        # Manufactured solution: u = sin(pi*x)*sin(pi*y)
        y = torch.sin(torch.pi * x[:, [0]]) * torch.sin(torch.pi * x[:, [1]])
        return x, y


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
}


def get_dataset(name: str) -> Dataset:
    """Get dataset by name."""
    if name not in DATASETS:
        raise ValueError(f"Unknown dataset '{name}'. Available: {list(DATASETS.keys())}")
    return DATASETS[name]
