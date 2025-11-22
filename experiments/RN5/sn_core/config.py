"""Configuration for Sprecher Networks.

This module holds all run-time configuration used across the project.
"""

# Core configuration
CONFIG = {
    # ------------------------------
    # Model settings
    # ------------------------------
    'train_phi_codomain': True,   # Whether to train Φ codomain parameters (cc, cr)
    'use_residual_weights': True, # Toggle residual connections globally
    'residual_style': 'node',     # 'node' (default, original) or 'linear' (standard residuals)
    'seed': 45,

    # ------------------------------
    # Spline settings
    # ------------------------------
    # NOTE:
    #   - 1  = piecewise-linear splines (original implementation, CURRENT DEFAULT)
    #   - 3  = cubic splines (to be used e.g. for PINN experiments where we
    #          want non-trivial second derivatives of the spline functions)
    #   - Other odd degrees may be supported in code, but only 1 and 3 are
    #     intended / tested by default.
    #
    # IMPORTANT:
    #   When these are left at 1, all behavior should remain IDENTICAL to the
    #   original piecewise-linear Sprecher Network implementation (same domains,
    #   same φ / Φ, same training dynamics).
    'phi_spline_degree': 1,   # Degree of φ splines (1 = original PWL, 3 = cubic, etc.)
    'Phi_spline_degree': 1,   # Degree of Φ splines (1 = original PWL, 3 = cubic, etc.)

    # ------------------------------
    # Training settings
    # ------------------------------
    'use_advanced_scheduler': False,
    'weight_decay': 1e-6,
    'max_grad_norm': 1.0,

    # ------------------------------
    # Normalization settings
    # ------------------------------
    'use_normalization': True,    # Global switch for normalization
    'norm_type': 'batch',         # Default normalization type when enabled: 'none', 'batch', 'layer'
    'norm_position': 'after',     # 'before' or 'after' each block
    'norm_skip_first': True,      # Skip normalization for the first block

    # ------------------------------
    # Scheduler settings (used when use_advanced_scheduler=True)
    # ------------------------------
    'scheduler_type': 'plateau_cosine',  # Currently informational
    'scheduler_base_lr': 1e-4,
    'scheduler_max_lr': 1e-2,
    'scheduler_patience': 500,
    'scheduler_threshold': 1e-5,

    # ------------------------------
    # Domain settings
    # ------------------------------
    'use_theoretical_domains': True, # Use theoretical domain computation
    'domain_safety_margin': 0.0,     # Safety margin around computed domains
    'debug_domains': False,          # Print domain information during training

    # ------------------------------
    # Domain violation tracking
    # ------------------------------
    'track_domain_violations': False,     # Track out-of-domain evaluations
    'verbose_domain_violations': False,   # Print violations as they occur

    # ------------------------------
    # Checkpoint & debugging
    # ------------------------------
    'debug_checkpoint_loading': False,    # Detailed logs during checkpoint save/load

    # ------------------------------
    # Parameter export
    # ------------------------------
    'export_params': False,               # False, True/'all', or comma-separated list
    'export_params_dir': 'params',

    # ------------------------------
    # Lateral mixing settings
    # ------------------------------
    'use_lateral_mixing': True,     # Enable intra-block lateral connections
    'lateral_mixing_type': 'cyclic',# 'cyclic' or 'bidirectional'
    'lateral_scale_init': 0.1,      # Initial global scale for lateral mixing
    'lateral_weight_init': 0.1,     # Initial per-channel mixing weights

    # ------------------------------
    # Memory optimization
    # ------------------------------
    'low_memory_mode': False,   # O(B × max(d_in, d_out)) memory forward
    'memory_debug': False,      # Print CUDA memory usage profiles when enabled
}

# MNIST-specific defaults (used by the MNIST example script, kept unchanged)
MNIST_CONFIG = {
    'architecture': [100],
    'phi_knots': 50,
    'Phi_knots': 50,
    'learning_rate': 0.001,
    'weight_decay': 1e-6,
    'batch_size': 64,
    'epochs': 3,
    'model_file': 'sn_mnist_model.pth',
    'data_directory': './data',
}

# Mathematical constants (not configuration)
Q_VALUES_FACTOR = 1.0

# Parameter categories for export
PARAM_CATEGORIES = [
    'lambda', 'eta', 'spline', 'residual',
    'codomain', 'norm', 'output', 'lateral'
]