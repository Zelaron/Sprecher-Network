"""Sprecher Network (SN) core package.

This init module also exposes lightweight helpers to run *any* forward pass
in safe evaluation mode (model.eval() + no_grad) while **restoring** the
original training/eval state afterwards.
"""

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
    # BN-only mode helpers (sometimes useful for debugging)
    set_bn_eval,
    set_bn_train,
    # Expose BN safety context managers used throughout training/verification
    freeze_bn_running_stats,
    use_batch_stats_without_updating_bn,
)
from .data import get_dataset, DATASETS
from .plotting import plot_results, plot_loss_curve
from .config import CONFIG, MNIST_CONFIG, Q_VALUES_FACTOR, PARAM_CATEGORIES
from .export import export_parameters, parse_param_types

# ---------------------------------------------------------------------
# Evaluation-mode helpers
# ---------------------------------------------------------------------
from contextlib import contextmanager
import torch


@contextmanager
def eval_mode(model):
    """
    Temporarily put `model` into eval() **and** disable autograd, then restore
    its previous training/eval state afterwards.

    Usage:
        with eval_mode(model):
            y = model(x)

    Guarantees:
      - Inside the block:   model.training == False, gradients off.
      - After the block:    model.training is restored to its original value.
    """
    was_training = model.training
    try:
        model.eval()
        with torch.no_grad():
            yield model
    finally:
        if was_training:
            model.train()
        else:
            # Ensure we leave it explicitly in eval if it originally was eval
            model.eval()


def forward_eval(model, *args, **kwargs):
    """
    Convenience wrapper to run a single forward pass in eval()+no_grad(),
    restoring the previous state afterwards.

    Returns:
        The model output tensor(s) from the forward pass.
    """
    with eval_mode(model):
        return model(*args, **kwargs)


def batched_forward_eval(model, x, batch_size=8192, device=None):
    """
    Run `model(x)` in eval()+no_grad() in mini-batches to avoid large-memory spikes.
    The model's train/eval mode is restored upon return.

    Args:
        model:      torch.nn.Module
        x:          Input tensor of shape [N, ...]
        batch_size: Integer > 0 (default: 8192)
        device:     Optional device for the forward passes; if None, inferred
                    from the model's parameters; if the model has no params,
                    use x.device.

    Returns:
        Concatenated model outputs on CPU, dtype preserved from the model outputs.
    """
    if device is None:
        try:
            device = next(model.parameters()).device
        except StopIteration:
            device = x.device

    N = x.shape[0]
    outs = []
    with eval_mode(model):
        for start in range(0, N, max(1, int(batch_size))):
            end = min(N, start + max(1, int(batch_size)))
            xb = x[start:end].to(device)
            yb = model(xb)
            outs.append(yb.detach().cpu())
    return torch.cat(outs, dim=0)


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
    'get_dataset',
    'DATASETS',
    'plot_results',
    'plot_loss_curve',
    'CONFIG',
    'MNIST_CONFIG',
    'Q_VALUES_FACTOR',
    'PARAM_CATEGORIES',
    'export_parameters',
    'parse_param_types',
    # Eval-mode helpers
    'eval_mode',
    'forward_eval',
    'batched_forward_eval',
]