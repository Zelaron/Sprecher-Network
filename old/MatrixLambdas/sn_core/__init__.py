"""Sprecher Network (SN) core package."""

from .model import (
    PhiRangeParams,
    SimpleSpline,
    SprecherLayerBlock,
    SprecherMultiLayerNetwork
)
from .train import train_network, PlateauAwareCosineAnnealingLR
from .data import get_dataset, DATASETS
from .plotting import plot_results, plot_loss_curve

__all__ = [
    'PhiRangeParams',
    'SimpleSpline',
    'SprecherLayerBlock',
    'SprecherMultiLayerNetwork',
    'train_network',
    'PlateauAwareCosineAnnealingLR',
    'get_dataset',
    'DATASETS',
    'plot_results',
    'plot_loss_curve'
]