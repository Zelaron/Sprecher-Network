"""Sprecher Network (SN) core package."""

from .model import (
    TheoreticalRange,
    PhiCodomainParams,
    SimpleSpline,
    SprecherLayerBlock,
    SprecherMultiLayerNetwork,
    ExactLayerNorm,
    RangePreservingNorm
)
from .train import train_network, PlateauAwareCosineAnnealingLR
from .data import get_dataset, DATASETS
from .plotting import plot_results, plot_loss_curve
from .config import CONFIG, MNIST_CONFIG, Q_VALUES_FACTOR

__all__ = [
    'TheoreticalRange',
    'PhiCodomainParams',
    'SimpleSpline',
    'SprecherLayerBlock',
    'SprecherMultiLayerNetwork',
    'ExactLayerNorm',
    'RangePreservingNorm',
    'train_network',
    'PlateauAwareCosineAnnealingLR',
    'get_dataset',
    'DATASETS',
    'plot_results',
    'plot_loss_curve',
    'CONFIG',
    'MNIST_CONFIG',
    'Q_VALUES_FACTOR'
]