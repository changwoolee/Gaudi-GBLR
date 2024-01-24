import torch
from torch.optim import Optimizer
""" Cosine Scheduler

Cosine LR schedule with warmup, cycle/restarts, noise, k-decay.

Hacked together by / Copyright 2021 Ross Wightman
"""
import logging
import math
import numpy as np
import torch

from src.optim.scheduler import Scheduler
from timm.scheduler.step_lr import StepLRScheduler


_logger = logging.getLogger(__name__)


class MyCosineLRScheduler(Scheduler):
    """
    Cosine decay with restarts.
    This is described in the paper https://arxiv.org/abs/1608.03983.

    Inspiration from
    https://github.com/allenai/allennlp/blob/master/allennlp/training/learning_rate_schedulers/cosine.py

    k-decay option based on `k-decay: A New Method For Learning Rate Schedule` - https://arxiv.org/abs/2004.05909
    """

    def __init__(
            self,
            optimizer: torch.optim.Optimizer,
            t_initial: int,
            lr_min: float = 0.,
            cycle_mul: float = 1.,
            cycle_decay: float = 1.,
            cycle_limit: int = 1,
            warmup_t=0,
            warmup_lr_init=0,
            warmup_prefix=False,
            t_in_epochs=True,
            noise_range_t=None,
            noise_pct=0.67,
            noise_std=1.0,
            noise_seed=42,
            k_decay=1.0,
            initialize=True,
            exclude_special_groups=False,
    ) -> None:
        super().__init__(
            optimizer,
            param_group_field="lr",
            t_in_epochs=t_in_epochs,
            noise_range_t=noise_range_t,
            noise_pct=noise_pct,
            noise_std=noise_std,
            noise_seed=noise_seed,
            initialize=initialize,
        )

        assert t_initial > 0
        assert lr_min >= 0
        if t_initial == 1 and cycle_mul == 1 and cycle_decay == 1:
            _logger.warning(
                "Cosine annealing scheduler will have no effect on the learning "
                "rate since t_initial = t_mul = eta_mul = 1.")
        self.t_initial = t_initial
        self.lr_min = lr_min
        self.cycle_mul = cycle_mul
        self.cycle_decay = cycle_decay
        self.cycle_limit = cycle_limit
        self.warmup_t = warmup_t
        self.warmup_lr_init = warmup_lr_init
        self.warmup_prefix = warmup_prefix
        self.k_decay = k_decay
        self.exclude_special_groups = exclude_special_groups
        if self.exclude_special_groups:
            _logger.info("Exclude Special Groups.")
        if self.warmup_t:
            _logger.info("Warmup LR Init: {}".format(self.warmup_lr_init))
            self.warmup_lr_init = np.array(self.warmup_lr_init)
            self.warmup_steps = (np.array(self.base_values) - self.warmup_lr_init) / self.warmup_t #[(v - warmup_lr_init) / self.warmup_t for v in self.base_values]
            super().update_groups(self.warmup_lr_init)
        else:
            self.warmup_steps = [1 for _ in self.base_values]

    def _get_lr(self, t):
        if t < self.warmup_t:
            #lrs = [self.warmup_lr_init + t * s for s in self.warmup_steps]
            lrs = list(self.warmup_lr_init + t * self.warmup_steps)
            if self.exclude_special_groups:
                lrs[2:] = self.base_values[2:]
        else:
            if self.warmup_prefix:
                t = t - self.warmup_t

            if self.cycle_mul != 1:
                i = math.floor(math.log(1 - t / self.t_initial * (1 - self.cycle_mul), self.cycle_mul))
                t_i = self.cycle_mul ** i * self.t_initial
                t_curr = t - (1 - self.cycle_mul ** i) / (1 - self.cycle_mul) * self.t_initial
            else:
                i = t // self.t_initial
                t_i = self.t_initial
                t_curr = t - (self.t_initial * i)

            gamma = self.cycle_decay ** i
            lr_max_values = [v * gamma for v in self.base_values]
            k = self.k_decay

            if i < self.cycle_limit:
                lrs = [
                    self.lr_min + 0.5 * (lr_max - self.lr_min) * (1 + math.cos(math.pi * t_curr ** k / t_i ** k))
                    for lr_max in lr_max_values
                ]
            else:
                lrs = [self.lr_min for _ in self.base_values]

        return lrs

    def get_cycle_length(self, cycles=0):
        cycles = max(1, cycles or self.cycle_limit)
        if self.cycle_mul == 1.0:
            return self.t_initial * cycles
        else:
            return int(math.floor(-self.t_initial * (self.cycle_mul ** cycles - 1) / (1 - self.cycle_mul)))

# We need to subclass torch.optim.lr_scheduler._LRScheduler, or Pytorch-lightning will complain
class TimmCosineLRScheduler(MyCosineLRScheduler, torch.optim.lr_scheduler._LRScheduler):
    """ Wrap timm.scheduler.CosineLRScheduler so we can call scheduler.step() without passing in epoch.
    It supports resuming as well.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._last_epoch = -1
        self.step(epoch=0)

    def step(self, epoch=None):
        if epoch is None:
            self._last_epoch += 1
        else:
            self._last_epoch = epoch
        # We call either step or step_update, depending on whether we're using the scheduler every
        # epoch or every step.
        # Otherwise, lightning will always call step (i.e., meant for each epoch), and if we set
        # scheduler interval to "step", then the learning rate update will be wrong.
        if self.t_in_epochs:
            super().step(epoch=self._last_epoch)
        else:
            super().step_update(num_updates=self._last_epoch)


class TimmStepLRScheduler(StepLRScheduler, torch.optim.lr_scheduler._LRScheduler):
    def __init__(self, optimizer, decay_t, decay_rate=1.0, 
                       warmup_t=0, warmup_lr_init=0, t_in_epochs=True, noise_range_t=None,
                       noise_pct=0.67, noise_std=1.0,  noise_seed=42, initialize=True, **kwargs):
        super().__init__(optimizer, decay_t, decay_rate=decay_rate,
                         warmup_t=warmup_t, warmup_lr_init=warmup_lr_init, t_in_epochs=t_in_epochs, noise_range_t=noise_range_t,
                         noise_pct=noise_pct, noise_std=noise_std, noise_seed=noise_seed, initialize=initialize)

        self._last_epoch = -1
        self.step(epoch=0)

    def step(self, epoch=None):
        if epoch is None:
            self._last_epoch += 1
        else:
            self._last_epoch = epoch
        # We call either step or step_update, depending on whether we're using the scheduler every
        # epoch or every step.
        # Otherwise, lightning will always call step (i.e., meant for each epoch), and if we set
        # scheduler interval to "step", then the learning rate update will be wrong.
        if self.t_in_epochs:
            super().step(epoch=self._last_epoch)
        else:
            super().step_update(num_updates=self._last_epoch)

