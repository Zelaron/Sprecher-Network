# Sprecher Network
Sprecher Network is a PyTorch implementation of a universal function approximator based on Sprecher's constructive proof of the Kolmogorov–Arnold representation theorem. It uses custom spline layers for iterative (boosting-style) function approximation.
## Overview
This project implements a neural network that:
- Combines custom monotonic (inner φ) and non-monotonic (outer Φ) spline layers
- Leverages Sprecher's construction (φ and Φ functions) to approximate complex functions
- Supports both 1D and 2D function approximation with automatic input dimension detection
- Employs an iterative, layer-by-layer training process
- Provides detailed visualizations for network structure, spline functions, function comparison, and training loss curves
## Installation
1. **Create and activate a virtual environment:**
   wwwwww
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   wwwwww
2. **Install dependencies:**
   wwwwww
   pip install -r requirements.txt
   wwwwww
Required packages:
- torch>=1.7.1
- numpy>=1.19.2
- matplotlib>=3.3.4
- tqdm>=4.56.0
## Usage
Run the main script:
wwwwww
python snJ4.py
wwwwww
## Customization
- Modify the target function in the code (uncomment the desired version for 1D or 2D).
- Adjust training parameters such as the number of layers, epochs, and learning rate.
- Tweak visualization settings (e.g., NUM_PLOT_POINTS and NUM_SURFACE_POINTS).
- Change the random seed (SEED=45) for different but reproducible results.
## License
MIT License - see the [LICENSE](LICENSE) file for details.
