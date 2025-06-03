# Sprecher Network
Sprecher Network is a PyTorch implementation of a universal function approximator based on Sprecher's constructive proof of the Kolmogorov–Arnold representation theorem. It uses custom spline layers for iterative (boosting-style) function approximation.
## Overview
This project implements a neural network that:
- Combines custom monotonic (inner φ) and non-monotonic (outer Φ) spline layers
- Leverages Sprecher's construction (φ and Φ functions) to approximate complex functions
- Supports both 1D and 2D function approximation with automatic input dimension detection
- Employs an iterative, layer-by-layer training process
- Provides detailed visualizations for network structure, spline functions, function comparison, and training loss curves

## Output Files
During training, the network produces and saves visualizations (e.g., network structure plots, spline function plots, and loss curves) in the `plots` directory.

## Installation
1. **Create and activate a virtual environment:**
   ```
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```
2. **Install dependencies:**
   ```
   pip install -r requirements.txt
   ```
Required packages:
- torch>=1.7.1
- numpy>=1.19.2
- matplotlib>=3.3.4
- tqdm>=4.56.0
## Usage
Run the main script:
```
python sprecher-mse.py
```

## Customization
Users are encouraged to experiment by modifying the code, particularly in the sections labeled "**CONFIGURABLE SETTINGS**" and "**TARGET FUNCTION DEFINITION**". You can adjust training parameters (e.g., number of layers, epochs, learning rates) or alter the target function to explore different function approximation tasks.

**Note:** The code uses a fixed random seed (SEED=45) for reproducibility by calling `torch.manual_seed(SEED)`, `np.random.seed(SEED)`, and related functions. If you prefer different random initializations on each run, you can remove or comment out these seed-setting lines in the code.

## Hardware
The code automatically detects whether a GPU is available and uses it if possible.

## License
MIT License - see the [LICENSE](LICENSE) file for details.
