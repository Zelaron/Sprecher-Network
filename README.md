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

Run a single experiment on a toy dataset:

```
# Run with default settings (toy_1d_poly dataset)
python sn_experiments.py

# Customize the architecture, save plots, and disable residual connections
python sn_experiments.py --dataset toy_2d --arch 10,10,10 --epochs 5000 --save_plots --no_residual

# High-dimensional example (100D input) with normalization on first block
python sn_experiments.py --dataset toy_100d --arch 100 --epochs 100 --norm_first
```

### MNIST Classification

Sprecher Networks can also handle classification tasks. Use the `sn_mnist.py` script with the `--mode` flag (with the other flags generally being optional):

```
# Train a new model on MNIST for 5 epochs
python sn_mnist.py --mode train --arch 100 --epochs 5 --save_plots --norm_first

# Test the accuracy (automatically uses saved configuration)
python sn_mnist.py --mode test

# Run inference on a single image
python sn_mnist.py --mode infer --image digit.png

# Visualize the learned splines from a trained model
python sn_mnist.py --mode plot
```

A single-layer network with architecture 784→[100]→10 achieves ~92% accuracy with only ~45,000 parameters. Test/infer/plot modes automatically load the model configuration from the checkpoint.

### Batch Experiments

The `sn_sweeps.py` script can run a comprehensive suite of experiments in parallel.

```
# List all available sweeps
python sn_sweeps.py --list

# Run all sweeps in parallel using all available CPU cores
python sn_sweeps.py

# Run a few specific sweeps sequentially (1 worker)
python sn_sweeps.py --sweeps toy_1d_poly feynman_uv --parallel 1
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
    # Model settings
    'train_phi_codomain': True,
    'use_residual_weights': True,
    'seed': 45,

    # Normalization settings (defaults when not overridden by CLI flags)
    'use_normalization': True,      # Master switch for normalization
    'norm_type': 'batch',           # Default type: 'batch' or 'layer'
    'norm_position': 'after',       # Default position: 'before' or 'after' blocks
    'norm_skip_first': True,        # Skip normalization for the first block

    # Training settings
    'use_advanced_scheduler': False,
    'weight_decay': 1e-6,
    'max_grad_norm': 1.0,
}
```

### Command Line Arguments

**Common Arguments (`sn_experiments.py` & `sn_mnist.py`)**

These flags control the core model and training for both main scripts.

- `--arch`: Hidden layer widths as comma-separated values (e.g., `15,15`).
- `--phi_knots` / `--Phi_knots`: Number of knots for φ and Φ splines.
- `--epochs`: Number of training epochs.
- `--seed`: Random seed.
- `--device`: Device selection (`auto`/`cpu`/`cuda`).
- `--save_plots`: Save visualizations to files in the `plots/` directory.
- `--no_show`: Suppress interactive plot display (useful for batch runs).
- `--use_advanced_scheduler`: Use the `PlateauAwareCosineAnnealingLR` scheduler.

**Feature & Normalization Control (Common)**

- `--no_residual`: Disables residual connections.
- `--no_norm`: A simple flag to disable all normalization.
- `--norm_type [batch|layer|none]`: Specify the type of normalization to use.
- `--norm_position [before|after]`: Position normalization relative to Sprecher blocks.
- `--norm_first`: Enable normalization for the first block (recommended for high-dimensional inputs like MNIST).

**Script-Specific Arguments**

- **For `sn_mnist.py`:**
    - `--mode [train|test|infer|plot]`: Selects the operation mode.
    - `--batch_size`: The training and testing batch size.
    - `--lr`: The learning rate for the Adam optimizer.
    - `--retrain`: Deletes an existing model file and retrains from scratch.
    - `--image`: The image file to use for inference mode.

- **For `sn_sweeps.py`:**
    - `--list`: Lists all available sweep names.
    - `--sweeps [NAME ...]`: Specifies which sweeps to run (default: all).
    - `--parallel N`: Number of sweeps to run in parallel.
    - `--fail-fast`: Stops all jobs if a single sweep fails.

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
- **Flexible normalization:** Supports batch norm and layer norm, which can be strategically placed to stabilize training and improve convergence.
- **Gradient clipping:** Handles the challenging optimization landscape created by shared splines.
- **Plateau-aware cosine annealing:** Increases learning rate when stuck in local minima.
- **Optional residual connections:** Adapt based on dimensional compatibility.
- **Regularization options:** For smoother splines and better generalization.
- **Robust checkpointing:** Ensures plots perfectly reflect the best model state.

## Visualization

The package generates comprehensive visualizations saved to the `plots/` directory:
- Network architecture diagrams showing the block structure.
- Learned spline functions φ and Φ for each block.
- Function approximation comparisons (for regression tasks).
- Training loss and accuracy curves.

Visualizations are automatically saved with descriptive filenames indicating the dataset, architecture, and training configuration.

## Extending the Implementation

### Adding Custom Datasets

Create new datasets by extending the base `Dataset` class in `sn_core/data.py`:

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

While the architecture is flexible through command-line arguments, you can modify default behaviors by editing the configuration in `sn_core/config.py`. This includes training parameters, regularization strengths, and architectural choices like residual connections or normalization.

## Theoretical Background

Sprecher Networks are based on David Sprecher's 1965 constructive proof showing that any continuous multivariate function can be represented as a composition of univariate functions. The key formula for a single Sprecher block is:

`y_q = Φ(Σᵢ λᵢ φ(xᵢ + ηq) + q)`

where φ is a monotonic function, Φ is a general continuous function, λ are mixing weights (vectors, not matrices), and η is a learnable shift parameter. The index q provides the necessary diversity despite the extreme parameter sharing.

## License

MIT License - see the LICENSE file for details.

## Acknowledgments

This implementation was developed as part of research at Stockholm University and KTH Royal Institute of Technology. Special thanks to the mathematical foundations laid by David Sprecher and the recent renewed interest in Kolmogorov-Arnold representations sparked by the KAN paper.
