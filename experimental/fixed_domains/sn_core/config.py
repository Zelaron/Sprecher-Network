"""Configuration for Sprecher Networks."""

# Core configuration
CONFIG = {
    # Model settings
    'train_phi_codomain': True,  # Whether to train Φ codomain parameters
    'use_residual_weights': True,
    'seed': 45,
    
    # Training settings
    'use_advanced_scheduler': True,
    'weight_decay': 1e-6,
    'max_grad_norm': 1.0,
    
    # Scheduler settings (only used if use_advanced_scheduler=True)
    'scheduler_base_lr': 0.0001,
    'scheduler_max_lr': 0.01,
    'scheduler_patience': 500,
    'scheduler_threshold': 1e-5,
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
PHI_RANGE = (-10.0, 10.0)  # Initial domain/codomain for general splines Φ
Q_VALUES_FACTOR = 1.0