# Sprecher Network

Sprecher Network is a PyTorch implementation of a universal function approximator based on Sprecher's constructive proof of the Kolmogorov–Arnold representation theorem. It uses custom spline layers for function approximation with a clean, modular architecture.

## Overview

This project implements a neural network that:
- Combines custom monotonic (inner φ) and non-monotonic (outer Φ) spline layers
- Leverages Sprecher's construction to approximate complex functions
- Supports multiple datasets including 1D/2D functions, high-dimensional problems, and PDE solutions
- Provides a modular architecture with separate components for models, training, data, and visualization
- Includes both CLI for single experiments and batch sweep functionality
- Employs advanced training techniques including plateau-aware learning rate scheduling

## Features

- **Modular Design**: Clean separation of concerns with `sn_core` package
- **Dataset Registry**: Pre-implemented datasets including toy functions, Feynman equations, and PDE solutions
- **Flexible CLI**: Easy experimentation with different architectures and hyperparameters
- **Batch Sweeps**: Automated running of multiple configurations
- **Comprehensive Visualizations**: Network structure, spline functions, function comparisons, and loss curves
- **Residual Connections**: Optional ResNet-style skip connections for better gradient flow
- **Adaptive Spline Ranges**: Trainable domain and codomain parameters for Φ splines

## Installation

1. **Clone the repository:**
   ```
   git clone https://github.com/zelaron/sprecher-network.git
   cd sprecher-network
   ```

2. **Create and activate a virtual environment:**
   ```
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies:**
   ```
   pip install -r requirements.txt
   ```

## Usage

### Single Experiment

Run a single experiment using the CLI:

```
# Default configuration (reproduces original behavior)
python sn_experiments.py

# Custom configuration
python sn_experiments.py --dataset toy_2d --arch 10,10,10,10 --epochs 4000 --seed 42

# Full options
python sn_experiments.py \
    --dataset special_bessel \
    --arch 5,5,5 \
    --phi_knots 50 \
    --Phi_knots 50 \
    --epochs 6000 \
    --seed 0 \
    --save_plots
```

### Batch Sweeps

Run predefined sweep configurations:

```
python sn_sweeps.py
```

This will run multiple experiments with different datasets and architectures, saving all plots to the `plots` directory.

### Available Datasets

- `toy_1d_poly`: 1D polynomial function
- `toy_1d_complex`: Complex 1D function with sines and exponentials
- `toy_2d`: 2D function exp(sin(11x)) + 3y + 4sin(8y)
- `toy_2d_vector`: 2D vector-valued function
- `toy_100d`: 100-dimensional sum of sinusoids
- `special_bessel`: Bessel-like special function
- `feynman_uv`: Feynman UV radiation formula
- `poisson`: 2D Poisson equation manufactured solution

### CLI Arguments

- `--dataset`: Dataset name (default: toy_2d)
- `--arch`: Architecture as comma-separated values (default: 10,10,10,10)
- `--phi_knots`: Number of knots for φ splines (default: 100)
- `--Phi_knots`: Number of knots for Φ splines (default: 100)
- `--epochs`: Number of training epochs (default: 4000)
- `--seed`: Random seed (default: 45)
- `--device`: Device selection: auto, cpu, or cuda (default: auto)
- `--save_plots`: Save plots to files
- `--no_show`: Don't display plots (useful for batch runs)

## Package Structure

```
sprecher-network/
├── sn_core/             # Core package
│   ├── __init__.py      # Package initialization
│   ├── model.py         # Sprecher network layers and models
│   ├── train.py         # Training utilities
│   ├── data.py          # Dataset registry
│   └── plotting.py      # Visualization functions
├── sn_experiments.py    # CLI for single experiments
├── sn_sweeps.py         # Batch sweep runner
├── requirements.txt     # Package dependencies
└── README.md            # This file
```

## Customization

### Adding New Datasets

Create a new dataset class in `sn_core/data.py`:

```
class MyDataset(Dataset):
    @property
    def input_dim(self):
        return 2  # Your input dimension
    
    @property
    def output_dim(self):
        return 1  # Your output dimension
    
    def sample(self, n, device='cpu'):
        # Generate n samples
        x = torch.rand(n, self.input_dim, device=device)
        y = your_function(x)  # Your target function
        return x, y

# Register in DATASETS dictionary
DATASETS["my_dataset"] = MyDataset()
```

### Modifying Training Parameters

For advanced modifications (e.g., Q-values factor, residual weights, scheduler settings), edit the global configuration in `sn_core/model.py`:

```
Q_VALUES_FACTOR = 1.0          # Sprecher Q-values scaling - a value of 1.0 is true to Sprecher's original formula
                               # but may cause an initial plateau; try a value of 0.1 for optimization
USE_RESIDUAL_WEIGHTS = True    # Enable skip connections
TRAIN_PHI_RANGE = True         # Train Φ spline ranges
```

## Output Files

During training, the network saves visualizations in the `plots` directory:
- Network structure and spline function plots
- Function approximation comparisons
- Training loss curves

File naming convention: `{dims}Vars-{dataset}-{architecture}-{epochs}-epochs-outdim{output_dim}.png`

## Hardware

The code automatically detects GPU availability and uses CUDA if available. You can override this with the `--device` flag.

## Theory

This implementation is based on:
- Sprecher's constructive proof of the Kolmogorov-Arnold representation theorem
- Modern neural network techniques including residual connections and adaptive learning rates
- Piecewise linear spline approximations with monotonicity constraints

The network approximates multivariate functions through compositions of univariate functions (φ and Φ), providing a theoretically grounded approach to function approximation.

## License

MIT License - see the [LICENSE](LICENSE) file for details.