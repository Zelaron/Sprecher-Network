# Sprecher Networks

A PyTorch implementation of Sprecher Networks (SNs), a novel neural architecture inspired by David Sprecher's 1965 constructive proof of the Kolmogorov-Arnold representation theorem. This implementation accompanies the paper "Sprecher Networks: A Parameter-Efficient Architecture Inspired by the Kolmogorov-Arnold-Sprecher Theorem" by Hägg et al.

## Overview

Sprecher Networks offer a fundamentally different approach to function approximation compared to Multi-Layer Perceptrons (MLPs) and Kolmogorov-Arnold Networks (KANs). By using shared learnable splines (monotonic φ and general Φ) within structured blocks that incorporate explicit shifts and mixing weights, SNs achieve remarkable parameter efficiency with scaling of O(LN + LG) compared to O(LN²) for MLPs or O(LN²G) for KANs, where L is depth, N is width, and G is spline resolution.

The key innovation is the use of weight vectors rather than matrices, maintaining fidelity to Sprecher's original construction while dramatically reducing parameter count. This architectural choice can be understood as a principled form of weight sharing, analogous to how convolutional networks share weights across spatial locations.

## Installation

Clone the repository and set up your environment:

```
git clone https://github.com/zelaron/sprecher-network.git
cd sprecher-network
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

## Quick Start

### Basic Function Approximation

Run a single experiment with default settings:

```
python sn_experiments.py
```

Customize the architecture and training parameters:

```
python sn_experiments.py --dataset toy_2d --arch 10,10,10 --epochs 5000 --save_plots
```

### MNIST Classification

Sprecher Networks can also handle classification tasks. The MNIST example demonstrates competitive performance with dramatically fewer parameters than equivalent MLPs:

```
# Train on MNIST (interactive mode selection)
python sn_mnist.py

# The script will prompt for mode:
# 0: Train a new model
# 1: Test existing model
# 2: Classify a single image (requires 'digit.png')
# 3: Visualize learned splines
```

A single-layer network with architecture 784→[100]→10 achieves ~92% accuracy with only ~45,000 parameters, compared to over 100,000 for a comparable MLP.

### Batch Experiments

Run a comprehensive sweep of predefined configurations:

```
python sn_sweeps.py
```

## Available Datasets

The implementation includes various test functions to demonstrate SN capabilities:

- **1D Functions**: `toy_1d_poly` (polynomial), `toy_1d_complex` (multi-frequency composition)
- **2D Functions**: `toy_2d` (scalar), `toy_2d_vector` (vector-valued)
- **High-Dimensional**: `toy_100d` (100D exponential of mean squared sine)
- **Physics-Inspired**: `special_bessel`, `feynman_uv` (Planck's law), `poisson` (PDE solution)
- **Multi-Input/Output**: `toy_4d_to_5d` (demonstrates vector output capabilities)

## Architecture and Configuration

### Network Notation

Sprecher Networks use the notation `input_dim → [hidden_widths] → output_dim`. For example:
- `2→[10,15,10]→1` denotes a network with 2D input, three hidden layers of widths 10, 15, and 10, and scalar output
- `784→[100]→10` denotes a network suitable for MNIST with one hidden layer

### Key Configuration Parameters

The main configuration options are accessible through `sn_core/config.py`:

```
CONFIG = {
    'train_phi_range': True,      # Enable trainable domain/codomain for Φ splines
    'use_residual_weights': True, # Enable residual connections
    'seed': 45,                   # Random seed for reproducibility
    'use_advanced_scheduler': True, # Plateau-aware learning rate scheduling
    'weight_decay': 1e-6,         # L2 regularization
    'max_grad_norm': 1.0,         # Gradient clipping threshold
}
```

### Command Line Arguments

- `--dataset`: Choose from available datasets
- `--arch`: Hidden layer widths as comma-separated values
- `--phi_knots`: Number of knots for monotonic φ splines (default: 100)
- `--Phi_knots`: Number of knots for general Φ splines (default: 100)
- `--epochs`: Training epochs (default: 4000)
- `--seed`: Random seed
- `--device`: Device selection (auto/cpu/cuda)
- `--save_plots`: Save visualizations to files
- `--no_show`: Suppress plot display (useful for batch runs)

## Implementation Structure

### File Organization

```
sprecher-network/
├── sn_experiments.py   # CLI for single experiments
├── sn_sweeps.py        # Batch sweep runner
├── sn_mnist.py         # MNIST classification example
├── requirements.txt    # Dependencies
├── README.md           # This file
└── sn_core/            # Core package
    ├── __init__.py     # Package exports
    ├── model.py        # Network architecture (splines, blocks, composition)
    ├── train.py        # Training loop with optimization
    ├── data.py         # Dataset implementations
    ├── plotting.py     # Visualization utilities
    └── config.py       # Global configuration
```

### Spline Implementation

The monotonic inner splines φ use a log-space parameterization to ensure strict monotonicity throughout training, while the general outer splines Φ can develop complex shapes to compensate for the constrained parameter space. This often results in characteristic oscillations in Φ, particularly in deeper networks.

### Training Features

The implementation includes several advanced training techniques:
- Gradient clipping to handle the challenging optimization landscape created by shared splines
- Plateau-aware cosine annealing that increases learning rate when stuck in local minima
- Optional residual connections that adapt based on dimensional compatibility
- Regularization options for smoother splines and better generalization

## Visualization

The package generates comprehensive visualizations saved to the `plots/` directory:
- Network architecture diagrams showing the block structure
- Learned spline functions φ and Φ for each block
- Function approximation comparisons (for regression tasks)
- Training loss curves with logarithmic scaling

Visualizations are automatically saved with descriptive filenames indicating the dataset, architecture, and training configuration.

## Extending the Implementation

### Adding Custom Datasets

Create new datasets by extending the base Dataset class:

```
from sn_core.data import Dataset, DATASETS

class MyDataset(Dataset):
    @property
    def input_dim(self):
        return 3  # Your input dimension
    
    @property
    def output_dim(self):
        return 2  # Your output dimension
    
    def evaluate(self, x):
        # Define your target function
        return your_function(x)

# Register the dataset
DATASETS["my_dataset"] = MyDataset()
```

### Modifying Architecture Defaults

While the architecture is flexible through command-line arguments, you can modify default behaviors by editing the configuration in `sn_core/config.py`. This includes training parameters, regularization strengths, and architectural choices like residual connections.

## Theoretical Background

Sprecher Networks are based on David Sprecher's 1965 constructive proof showing that any continuous multivariate function can be represented as a composition of univariate functions. The key formula for a single Sprecher block is:

```
y_q = Φ(Σᵢ λᵢ φ(xᵢ + ηq) + q)
```

where φ is a monotonic function, Φ is a general continuous function, λ are mixing weights (vectors, not matrices), and η is a learnable shift parameter. The index q provides the necessary diversity despite the extreme parameter sharing.

## License

MIT License - see the LICENSE file for details.

## Acknowledgments

This implementation was developed as part of research at Stockholm University and KTH Royal Institute of Technology. Special thanks to the mathematical foundations laid by David Sprecher and the recent renewed interest in Kolmogorov-Arnold representations sparked by the KAN paper.
