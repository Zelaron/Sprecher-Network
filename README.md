# Sprecher Networks

A PyTorch implementation of **Sprecher Networks (SNs)**, a neural architecture inspired by David Sprecher's 1965 constructive proof of the Kolmogorov-Arnold representation theorem. This implementation accompanies the paper *"Sprecher Networks: A Parameter-Efficient Kolmogorov-Arnold Architecture"* by Hägg, Kohn, Marchetti, and Shapiro.

## Overview

Sprecher Networks are built by composing layers (blocks) that each implement the transformation:

**y_q = Φ(Σᵢ λᵢ φ(xᵢ + ηq) + αq)**

where φ is a **shared monotonic spline**, Φ is a **shared general spline**, λ is a **weight vector** (not a matrix), and η is a learnable shift. The index q ∈ {0, …, d_out−1} provides output diversity despite extreme parameter sharing.

**Key properties:**
- **Parameter efficiency:** O(LN + LG) scaling vs. O(LN²) for MLPs or O(LN²G) for KANs, where L = depth, N = width, G = spline knots
- **Memory efficiency:** Sequential evaluation reduces peak memory from O(N²) to O(N)
- **Interpretability:** Learned splines can be visualized directly
- **Lateral mixing:** Optional intra-block communication between output dimensions with only O(N) additional parameters

**Scalability advantage over "Sprecher-inspired" alternatives:**
Recent architectures like GS-KAN and SaKAN claim Sprecher-inspired efficiency but retain O(N²) weight matrices. In contrast, SNs use only vector weights, achieving genuinely linear scaling. In a scalability benchmark (64-dimensional input, depth 3) on an M2 MacBook with 8GB unified memory, we double layer width until architectures run out of memory:
- At width 4096: GS-KAN fails (OOM)
- At width 8192: Standard KAN fails (OOM)
- At width 16384: MLP and SaKAN fail (OOM). **SN is the sole survivor** with 49K parameters vs 538M+ for competitors—a ~10,000× difference.

The benchmark then trains SN at the surviving width, confirming the capacity is utilized (loss drops from 3.4 to 0.068 over 400 epochs).

## Installation

```
git clone https://github.com/zelaron/sprecher-network.git
cd sprecher-network
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

## Quick Start

### Function Approximation

```
# Default: toy_1d_poly dataset, architecture [15,15], 4000 epochs
python sn_experiments.py

# 2D function with custom architecture and cubic splines
python sn_experiments.py --dataset toy_2d --arch 10,10,10 --spline_type cubic --epochs 5000 --save_plots

# High-dimensional input (100D) with first-block normalization
python sn_experiments.py --dataset toy_100d --arch 100 --epochs 100 --norm_first
```

### MNIST Classification

```
# Train on MNIST
python sn_mnist.py --mode train --arch 100 --epochs 5 --norm_first --save_plots

# Test accuracy (loads saved config automatically)
python sn_mnist.py --mode test

# Inference on a single image
python sn_mnist.py --mode infer --image digit.png

# Visualize learned splines
python sn_mnist.py --mode plot
```

A single-layer network `784→[100]→10` achieves ~92% accuracy with ~45,000 parameters.

### Batch Experiments

```
# List available sweeps
python sn_sweeps.py --list

# Run all sweeps in parallel
python sn_sweeps.py

# Run specific sweeps sequentially
python sn_sweeps.py --sweeps toy_1d_poly feynman_uv --parallel 1
```

## Architecture

### Network Notation

SNs use the notation `input_dim → [hidden_widths] → output_dim`:
- `2→[10,15,10]→1` - 2D input, three hidden layers (10, 15, 10), scalar output (summed)
- `784→[100]→10` - MNIST-sized input, one hidden layer, 10-class output

### Block Structure

Each Sprecher block contains:
- **Monotonic spline φ**: Shared across all input dimensions, maps to [0,1]
- **General spline Φ**: Shared across all outputs, unrestricted codomain
- **Weight vector λ**: Combines transformed inputs (vector, not matrix)
- **Shift parameter η**: Controls index-dependent input shifts
- **Lateral mixing** (optional): Cyclic or bidirectional mixing between outputs before Φ

### Spline Types

Both φ and Φ support:
- `linear` / `pwl` - Piecewise linear (fast, discontinuous derivatives)
- `cubic` - C¹ piecewise-cubic Hermite (PCHIP), smooth with well-defined 2nd derivatives

```
# Set both splines to cubic
python sn_experiments.py --spline_type cubic

# Different types for φ and Φ
python sn_experiments.py --phi_spline_type linear --Phi_spline_type cubic
```

### Residual Connections

Two styles are supported:
- `node` (default): Cyclic assignment with pooling/broadcasting for dimension changes
- `linear`: Standard linear projection matrix when dimensions differ

```
python sn_experiments.py --residual_style linear
python sn_experiments.py --no_residual  # disable entirely
```

### Lateral Mixing

Enables intra-block communication between output dimensions:
- `cyclic`: Each output receives from its cyclic neighbor
- `bidirectional`: Each output mixes with both neighbors

```
python sn_experiments.py --lateral_type bidirectional
python sn_experiments.py --no_lateral  # disable
```

### Normalization

```
# Batch normalization (default)
python sn_experiments.py --norm_type batch --norm_position after

# Layer normalization
python sn_experiments.py --norm_type layer

# Include normalization on first block (recommended for high-dim inputs)
python sn_experiments.py --norm_first

# Disable normalization
python sn_experiments.py --no_norm
```

## Datasets

| Name | Input Dim | Output Dim | Description |
|------|-----------|------------|-------------|
| `toy_1d_poly` | 1 | 1 | Polynomial function |
| `toy_1d_complex` | 1 | 1 | Multi-frequency composition |
| `toy_2d` | 2 | 1 | 2D scalar function |
| `toy_2d_vector` | 2 | 5 | 2D vector-valued function |
| `toy_100d` | 100 | 1 | High-dimensional regression |
| `toy_4d_to_5d` | 4 | 5 | Multi-input/output |
| `special_bessel` | 2 | 1 | Bessel function approximation |
| `feynman_uv` | 2 | 1 | Planck's law (physics) |
| `poisson` | 2 | 1 | Poisson PDE solution |

## Benchmarks

### Scalability: SN vs Competitors

Compare memory scaling against MLP, KAN, GS-KAN, and SaKAN:

```
python -m benchmarks.benchmark_scalability
```

The script starts at width 512 with depth 3 on 64-dimensional input, then doubles the width until only one architecture survives without running out of memory. On our reference system (M2 MacBook, 8GB unified memory), SN is the sole survivor at width 16384, using 49K parameters and 7.4 MB while competitors would require 538M+ parameters. The script then trains the survivor to verify the capacity is actually utilized.

### Ablation Study

Evaluate the contribution of each architectural component:

```
python -m benchmarks.ablations
```

Tests progressive addition of: cyclic residuals → lateral mixing → domain tracking → resampling.

### SN vs KAN: Barebones Comparisons

The `benchmarks/` directory contains eight head-to-head comparisons between SNs and KANs under strict parameter parity, with no residuals, no normalization, and no lateral mixing. Each benchmark file includes recommended commands in its docstring. For example:

```
# Softstair-Wavepacket (10D) - run with 20 seeds
for s in 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19; do
  python -m benchmarks.benchmark_softstair_wavepacket --seed $s \
    --sn_phi_knots 650 --sn_Phi_knots 714
done

# Shared-Warped-Ridge (16D)
for s in 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19; do
  python -m benchmarks.benchmark_shared_warped_ridge --seed $s \
    --phi-knots 340 --Phi-knots 350
done
```

See the individual benchmark files for full command-line options and parameter choices that achieve near-exact parameter parity.

### SN vs KAN: Poisson PINN

Fair comparison on 2D Poisson problem with manufactured solution:

```
python -m benchmarks.pinn_sn_vs_kan_poisson --model both --epochs 5000 --seed 0 --target_params 900
```

Both networks use cubic splines, identical parameter counts, same optimizer. SN's residuals and lateral mixing are disabled for fairness.

### SN vs KAN: Dense-Head Shift

Multi-head regression benchmark favoring SN's shift-based structure:

```
for s in 0 1 2 3 4 5 6 7 8 9; do
  python -m benchmarks.kan_sn_densehead_shift_bench \
    --dims 12 --heads 64 --test_size 50000 \
    --alpha 0.08 --beta 0.7 --q_bias 0.15 \
    --epochs 4000 --device cpu --seed $s \
    --sn_arch 32,32 --sn_phi_knots 60 --sn_Phi_knots 60 \
    --sn_norm_type batch --sn_norm_position after --sn_norm_skip_first \
    --sn_residual_style linear --sn_no_lateral \
    --sn_freeze_domains_after 400 --sn_domain_margin 0.01 \
    --kan_arch 3,24,4 --kan_degree 3 \
    --kan_bn_type batch --kan_bn_position after --kan_bn_skip_first \
    --kan_residual_type linear --kan_outside linear \
    --equalize_params --prefer_leq
done
```

### SN vs KAN: Monotonic Quantile-Shifted Index

Multi-output benchmark with monotonicity constraints:

```
for s in 0 1 2 3 4 5 6 7 8 9; do
  python -m benchmarks.kan_sn_monoindex_bench \
    --dims 20 --n_quantiles 9 --test_size 50000 \
    --epochs 4000 --device cpu --seed $s \
    --sn_arch 24,24 --sn_phi_knots 85 --sn_Phi_knots 84 \
    --sn_norm_type batch --sn_norm_position after --sn_norm_skip_first \
    --sn_residual_style linear --sn_no_lateral \
    --sn_freeze_domains_after 300 --sn_domain_margin 0.01 \
    --kan_arch 4,4 --kan_degree 3 \
    --kan_bn_type batch --kan_bn_position after --kan_bn_skip_first \
    --kan_residual_type linear --kan_outside linear \
    --equalize_params --prefer_leq
done
```

## Command Reference

### Core Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--dataset` | `toy_1d_poly` | Dataset name |
| `--arch` | `15,15` | Hidden layer widths (comma-separated) |
| `--phi_knots` | `100` | Number of knots for φ splines |
| `--Phi_knots` | `100` | Number of knots for Φ splines |
| `--epochs` | `4000` | Training epochs |
| `--seed` | `45` | Random seed |
| `--device` | `auto` | Device: `auto`, `cpu`, `cuda` |

### Spline Arguments

| Argument | Description |
|----------|-------------|
| `--spline_type [linear\|cubic]` | Set both φ and Φ spline types |
| `--phi_spline_type [linear\|cubic]` | Override φ spline type |
| `--Phi_spline_type [linear\|cubic]` | Override Φ spline type |

### Feature Toggles

| Argument | Description |
|----------|-------------|
| `--no_residual` | Disable residual connections |
| `--residual_style [node\|linear]` | Residual connection style |
| `--no_lateral` | Disable lateral mixing |
| `--lateral_type [cyclic\|bidirectional]` | Lateral mixing type |
| `--no_norm` | Disable normalization |
| `--norm_type [batch\|layer\|none]` | Normalization type |
| `--norm_position [before\|after]` | Normalization position |
| `--norm_first` | Enable normalization on first block |

### Output Arguments

| Argument | Description |
|----------|-------------|
| `--save_plots` | Save plots to `plots/` directory |
| `--no_show` | Suppress interactive plot display |
| `--export_params [all\|TYPES]` | Export parameters to file |

### Advanced Arguments

| Argument | Description |
|----------|-------------|
| `--low_memory_mode` | Sequential computation (O(N) memory) |
| `--use_advanced_scheduler` | Plateau-aware cosine annealing LR |
| `--debug_domains` | Print domain debugging info |
| `--track_violations` | Track spline domain violations |

## Project Structure

```
sprecher-network/
├── sn_experiments.py                    # Main CLI for function approximation
├── sn_mnist.py                          # MNIST classification example
├── sn_sweeps.py                         # Batch experiment runner
├── requirements.txt                     # Dependencies
├── README.md
├── sn_core/                             # Core package
│   ├── __init__.py                      # Package exports
│   ├── model.py                         # Splines, blocks, network architecture
│   ├── train.py                         # Training loop, schedulers, BN utilities
│   ├── data.py                          # Dataset implementations
│   ├── plotting.py                      # Visualization utilities
│   ├── config.py                        # Global configuration
│   └── export.py                        # Parameter export utilities
└── benchmarks/                          # Comparison benchmarks
    ├── ablations.py                     # Feature ablation study
    ├── benchmark_scalability.py         # Memory scaling vs competitors
    ├── pinn_sn_vs_kan_poisson.py        # PINN benchmark
    ├── benchmark_softstair_wavepacket.py
    ├── benchmark_shared_warped_ridge.py
    ├── benchmark_shared_warp_chirp.py
    ├── benchmark_motif_chirp.py
    ├── kan_sn_inputshift_bump_bench.py
    ├── kan_sn_oscillatory_headshift_bench.py
    ├── benchmark_barebones_sn_vs_kan_pwl_vs_pchip.py
    ├── quantile_harmonics.py
    ├── kan_sn_densehead_shift_bench.py  # Dense-head benchmark
    └── kan_sn_monoindex_bench.py        # Monotonic index benchmark
```

## Configuration

Global defaults are in `sn_core/config.py`:

```
CONFIG = {
    'train_phi_codomain': True,       # Learn Φ codomain parameters
    'use_residual_weights': True,     # Enable residual connections
    'use_lateral_mixing': True,       # Enable lateral mixing
    'lateral_mixing_type': 'cyclic',  # 'cyclic' or 'bidirectional'
    'use_normalization': True,        # Enable normalization
    'norm_type': 'batch',             # 'batch', 'layer', or 'none'
    'norm_position': 'after',         # 'before' or 'after' blocks
    'norm_skip_first': True,          # Skip norm on first block
    'use_theoretical_domains': True,  # Dynamic domain tracking
    'weight_decay': 1e-6,
    'max_grad_norm': 1.0,
    'seed': 45,
}
```

## Extending the Implementation

### Custom Datasets

```
from sn_core.data import Dataset, DATASETS

class MyDataset(Dataset):
    @property
    def input_dim(self):
        return 3
    
    @property
    def output_dim(self):
        return 2
    
    def evaluate(self, x):
        # x: [N, 3] → return: [N, 2]
        return your_function(x)

DATASETS["my_dataset"] = MyDataset()
```

### Programmatic Usage

```
from sn_core import SprecherMultiLayerNetwork, train_network, get_dataset

# Build network
model = SprecherMultiLayerNetwork(
    input_dim=2,
    architecture=[10, 10],
    final_dim=1,
    phi_knots=100,
    Phi_knots=100,
    norm_type='batch',
    phi_spline_type='cubic',
    Phi_spline_type='cubic',
)

# Train
dataset = get_dataset('toy_2d')
snapshot, losses = train_network(
    dataset=dataset,
    architecture=[10, 10],
    total_epochs=5000,
    device='cuda',
)
```

## Theoretical Background

Sprecher (1965) showed that any continuous f:[0,1]ⁿ → ℝ can be represented as:

**f(x) = Σ_q Φ(Σₚ λₚ φ(xₚ + ηq) + q)**

with a single monotonic φ, a continuous Φ, and scalar weights λₚ. The sum over q runs from 0 to 2n. This architecture directly implements this formula with learnable components and extends it to deep compositions.

**Key insight:** The use of weight *vectors* rather than *matrices* maintains fidelity to Sprecher's construction while providing O(LN + LG) parameter scaling-dramatically more efficient than MLPs' O(LN²) or KANs' O(LN²G).

## Citation

If you use this code, please cite:

```
@article{hagg2025sprecher,
  title={Sprecher Networks: A Parameter-Efficient Kolmogorov-Arnold Architecture},
  author={H{\"a}gg, Christian and Kohn, Kathl{\'e}n and Marchetti, Giovanni Luca and Shapiro, Boris},
  year={2025}
}
```

## License

MIT License - see LICENSE file for details.

## Acknowledgments

Developed at Stockholm University and KTH Royal Institute of Technology. We thank David Sprecher for the foundational mathematical construction and the KAN authors for renewing interest in Kolmogorov-Arnold representations.
