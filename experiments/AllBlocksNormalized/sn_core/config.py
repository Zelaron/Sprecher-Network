"""Configuration for Sprecher Networks."""

# Core configuration
CONFIG = {
    # Model settings
    'train_phi_codomain': True,  # Whether to train Φ codomain parameters
    'use_residual_weights': True,
    'seed': 45,
    
    # Training settings
    'use_advanced_scheduler': False,
    'weight_decay': 1e-6,
    'max_grad_norm': 1.0,
    
    # Normalization settings
    'use_normalization': True,  # Whether to use normalization
    'norm_type': 'batch',  # Default normalization type when enabled
    'norm_position': 'after',  # Default position when enabled
    'norm_skip_blocks': 'none',  # Which blocks to skip: 'none', 'first', 'last', 'edges', 'all'
    
    # Scheduler settings
    'scheduler_type': 'plateau_cosine',  # Type of scheduler when use_advanced_scheduler=True
    'scheduler_base_lr': 0.0001,
    'scheduler_max_lr': 0.01,
    'scheduler_patience': 500,
    'scheduler_threshold': 1e-5,
    
    # Domain settings
    'use_theoretical_domains': True,  # Use theoretical domain computation
    'domain_safety_margin': 0.0,  # No safety margin by default (exact domains)
    'debug_domains': False,  # Print domain information during training
    
    # Domain violation tracking
    'track_domain_violations': False,  # Enable to track out-of-domain evaluations
    'verbose_domain_violations': False,  # Print violations as they occur
    
    # Checkpoint debugging
    'debug_checkpoint_loading': False,  # Enable detailed logging during checkpoint operations
}

# MNIST-specific settings
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
