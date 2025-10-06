"""Sprecher Network (SN) core package."""

from .model import (
    Interval,
    TheoreticalRange,
    PhiCodomainParams,
    SimpleSpline,
    SprecherLayerBlock,
    SprecherMultiLayerNetwork,
    compute_batchnorm_bounds,
    compute_layernorm_bounds,
    test_domain_tightness
)
from .train import (
    train_network,
    PlateauAwareCosineAnnealingLR,
    recalculate_bn_stats,
    has_batchnorm,
    # Export BN-only mode helpers so callers can freeze BN at eval
    # without switching the entire model to eval().
    # (Implemented in sn_core.train)
    set_bn_eval,
    set_bn_train,
)
from .data import get_dataset, DATASETS
from .plotting import plot_results, plot_loss_curve
from .config import CONFIG, MNIST_CONFIG, Q_VALUES_FACTOR, PARAM_CATEGORIES
from .export import export_parameters, parse_param_types

__all__ = [
    'Interval',
    'TheoreticalRange',
    'PhiCodomainParams',
    'SimpleSpline',
    'SprecherLayerBlock',
    'SprecherMultiLayerNetwork',
    'compute_batchnorm_bounds',
    'compute_layernorm_bounds',
    'test_domain_tightness',
    'train_network',
    'PlateauAwareCosineAnnealingLR',
    'recalculate_bn_stats',
    'has_batchnorm',
    'set_bn_eval',
    'set_bn_train',
    'get_dataset',
    'DATASETS',
    'plot_results',
    'plot_loss_curve',
    'CONFIG',
    'MNIST_CONFIG',
    'Q_VALUES_FACTOR',
    'PARAM_CATEGORIES',
    'export_parameters',
    'parse_param_types'
]