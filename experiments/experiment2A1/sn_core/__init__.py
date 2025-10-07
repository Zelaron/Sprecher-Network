"""Sprecher Network (SN) core package."""

from contextlib import contextmanager
import torch

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
    # without switching the entire model to eval() globally.
    set_bn_eval,
    set_bn_train,
    # BN safety utilities for verification / checkpoint checks
    freeze_bn_running_stats,
    use_batch_stats_without_updating_bn,
)
from .data import get_dataset, DATASETS
from .plotting import plot_results, plot_loss_curve
from .config import CONFIG, MNIST_CONFIG, Q_VALUES_FACTOR, PARAM_CATEGORIES
from .export import export_parameters, parse_param_types


@contextmanager
def evaluation_mode(model: torch.nn.Module):
    """
    Temporarily run the given model in eval() under torch.no_grad(), and then
    restore its previous train/eval state when done.

    Usage:
        with evaluation_mode(model):
            y = model(x)       # no grad, BN uses running stats
            ...compute metrics...

    This helper is intentionally tiny and local: it does NOT change any global
    flags and does not alter training behavior outside the 'with' block.
    """
    was_training = model.training
    try:
        model.eval()
        with torch.no_grad():
            yield
    finally:
        model.train(was_training)


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
    'freeze_bn_running_stats',
    'use_batch_stats_without_updating_bn',
    'evaluation_mode',
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