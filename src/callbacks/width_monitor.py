# Inspired by https://github.com/Lightning-AI/lightning/blob/master/src/pytorch_lightning/utilities/grads.py
# However, they compute grad at every iteration (I think), and the .item() calls incur a lot of overhead
# (6-7% slow down on GPT-2 small). Instead we only compute for iterations where we need to log, and don't
# call .item() explicitly.

from typing import Any
from collections import OrderedDict

from lightning.pytorch import Callback, Trainer
from lightning.pytorch.utilities import rank_zero_only
from lightning.pytorch.utilities.model_summary import get_human_readable_count
from src.models.layers.fouriermask import FourierMaskLR

import torch
import torch.nn as nn


class WidthMonitor(Callback):
    """Monitor the scales of weights and gradients.
    """

    def __init__(self, val=True, train=False):
        super().__init__()
        self.val = val
        self.train = train



    @rank_zero_only
    def on_validation_epoch_start(self, trainer, pl_module, *args, **kwargs):
        model = pl_module.model
        named_parameters = {}
        modules = {}
        ln_modules = (FourierMaskLR,)
        for mn, m in model.named_modules():
            if isinstance(m, ln_modules):
                min_widths = m.min_widths
                max_widths = m.max_widths
                for pn, p in m.named_parameters():
                    if 'widths' in pn and 'widths_base' not in pn:
                        fpn = '%s.%s' % (mn, pn) if mn else pn # full param name
                        named_parameters[fpn] = p #torch.stack([m.get_width(0), m.get_width(1)])
                        modules[fpn] = m

        stats = {}
        width, grad_l1_norm = [], []
        width_max = []
        ranks = []
        areas = []
        largest_width_mean = 0
        smallest_width_mean = 10.0
        with torch.no_grad():
            for param_name, param in named_parameters.items():
                m = modules[param_name]
                w1, w2 = m.get_width(0), m.get_width(1)
                param_after_clamp = torch.stack([w1, w2])
                nonzero_width_mask = m.get_nonzero_width_mask()
                rank = nonzero_width_mask.mean() * m.total_rank_ratio
                ranks.append(rank)
                
                param_abs = param_after_clamp.abs()
                param_abs_mean = param_abs.mean() * m.total_rank_ratio
                
                if param_abs_mean > largest_width_mean:
                    largest_width_mean = param_abs_mean
                if param_abs_mean < smallest_width_mean:
                    smallest_width_mean = param_abs_mean

                stats[f'stats/{param_name}_mean'] = param_abs_mean
                stats[f'stats/{param_name}_rank'] = rank
                width.append(param_abs_mean)
                width_max.append(param_abs.max())
                if False: #param.grad is not None:
                    # Gradient is already unscaled by the AMP loss scaler at this point
                    # https://github.com/Lightning-AI/lightning/pull/9606
                    param_grad_abs = param.grad.abs()
                    param_grad_abs_mean = param_grad_abs.mean()
                    stats[f'stats/{param_name}_grad_max'] = param_grad_abs.max()
                    stats[f'stats/{param_name}_grad_mean'] = param_grad_abs_mean
                    grad_l1_norm.append(param_grad_abs_mean * param.grad.numel())
            if len(width)>0:
                stats['width_mean'] = torch.stack(width).mean()
                stats['rank_mean'] = torch.stack(ranks).mean()
                width_max = torch.stack(width_max).max()
            if 'additional_flop_count' in kwargs:
                stats['flops'] += kwargs['additional_flop_count']
            if grad_l1_norm:
                stats['total_grad_l1_norm'] = torch.stack(grad_l1_norm).mean()
            # Sort by params name

            if len(width)>0:
                stats = OrderedDict(sorted(stats.items()))
                pl_module.log("width_mean", stats['width_mean'], prog_bar=True, rank_zero_only=True)
                pl_module.log("largest_wm", largest_width_mean, prog_bar=True, rank_zero_only=True)
                pl_module.log("smallest_wm", smallest_width_mean, prog_bar=True, rank_zero_only=True)
                pl_module.log("width_max", width_max, prog_bar=False, rank_zero_only=True)
                pl_module.log("rank_mean", stats['rank_mean'], prog_bar=False, rank_zero_only=True)
                if trainer.loggers is not None:
                    for logger in trainer.loggers:
                        logger.log_metrics(stats, step=trainer.fit_loop.epoch_loop._batches_that_stepped)
                if grad_l1_norm:
                    pl_module.log("grad", stats['total_grad_l1_norm'], prog_bar=True, rank_zero_only=True)

