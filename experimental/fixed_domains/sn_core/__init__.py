"""Sprecher Network (SN) core package."""

from .model import (
    PhiCodomainParams,
    SimpleSpline,
    SprecherLayerBlock,
    SprecherMultiLayerNetwork
)
from .train import train_network, PlateauAwareCosineAnnealingLR
from .data import get_dataset, DATASETS
from .plotting import plot_results, plot_loss_curve
from .config import CONFIG, MNIST_CONFIG, PHI_RANGE, Q_VALUES_FACTOR

__all__ = [
    'PhiCodomainParams',
    'SimpleSpline',
    'SprecherLayerBlock',
    'SprecherMultiLayerNetwork',
    'train_network',
    'PlateauAwareCosineAnnealingLR',
    'get_dataset',
    'DATASETS',
    'plot_results',
    'plot_loss_curve',
    'CONFIG',
    'MNIST_CONFIG',
    'PHI_RANGE',
    'Q_VALUES_FACTOR'
]