"""Sprecher Network (SN) core package."""

from .model import (
    TheoreticalRange,
    PhiCodomainParams,
    SimpleSpline,
    SprecherLayerBlock,
    SprecherMultiLayerNetwork
)
from .train import train_network, PlateauAwareCosineAnnealingLR, recalculate_bn_stats, has_batchnorm
from .data import get_dataset, DATASETS
from .plotting import plot_results, plot_loss_curve
from .config import CONFIG, MNIST_CONFIG, Q_VALUES_FACTOR, PARAM_CATEGORIES
from .export import export_parameters, parse_param_types

__all__ = [
    'TheoreticalRange',
    'PhiCodomainParams',
    'SimpleSpline',
    'SprecherLayerBlock',
    'SprecherMultiLayerNetwork',
    'train_network',
    'PlateauAwareCosineAnnealingLR',
    'recalculate_bn_stats',
    'has_batchnorm',
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