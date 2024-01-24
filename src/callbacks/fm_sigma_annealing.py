from typing import Any
from collections import OrderedDict

from lightning.pytorch import Callback, Trainer
from lightning.pytorch.utilities import rank_zero_only

import torch
import torch.nn as nn
from src.models.layers.gblr import GaudiGBLR



class SigmaAnnealing(Callback):
    """Monitor the scales of weights and gradients.
    """

    def __init__(self, sigma_init, sigma_final, start_epoch, end_epoch, test_sigma=None):
        super().__init__()
        self.sigma_init = sigma_init
        self.sigma_final = sigma_final
        self.start_epoch = start_epoch
        self.end_epoch = end_epoch
        assert start_epoch < end_epoch
        self.test_sigma = test_sigma

    @rank_zero_only
    def log(self, trainer, sigma):
        if trainer.loggers is not None:
            for logger in trainer.loggers:
                logger.log_metrics({'sigma': sigma}, step=trainer.fit_loop.epoch_loop._batches_that_stepped)


    def on_train_epoch_start(self, trainer, pl_module):
        current_epoch = trainer.current_epoch
        model = pl_module.model
        if current_epoch < self.start_epoch:
            sigma = self.sigma_init
        elif current_epoch > self.end_epoch:
            sigma = self.sigma_final
        else:
            sigma = self.sigma_init + (self.sigma_final - self.sigma_init) * (current_epoch - self.start_epoch) / (self.end_epoch - self.start_epoch)

        #pl_module.log("sigma", sigma, prog_bar=True, rank_zero_only=True, logger=True)

        for mn, m in model.named_modules():
            if isinstance(m, GaudiGBLR):
                if isinstance(m.sigma, torch.Tensor):
                    m.sigma.data = torch.ones_like(m.sigma) * sigma
                else:
                    m.sigma = sigma


    def on_test_epoch_start(self, trainer, pl_module):
        if self.test_sigma is None:
            return
        model = pl_module.model
        for mn, m in model.named_modules():
            if isinstance(m, GaudiGBLR):
                if isinstance(m.sigma, torch.Tensor):
                    m.sigma.data = sigma
                else:
                    m.sigma = sigma



class SigmaAnnealingBatch(Callback):
    """Monitor the scales of weights and gradients.
    """

    def __init__(self, sigma_init, sigma_final, at_epoch=0, **kwargs):
        super().__init__()
        self.sigma_init = sigma_init
        self.sigma_final = sigma_final
        self.at_epoch = at_epoch

    @rank_zero_only
    def log(self, trainer, sigma):
        if trainer.loggers is not None:
            for logger in trainer.loggers:
                logger.log_metrics({'sigma': sigma}, step=trainer.fit_loop.epoch_loop._batches_that_stepped)


    def on_train_batch_start(self, trainer, pl_module, batch, batch_idx):
        current_epoch = trainer.current_epoch
        if current_epoch != self.at_epoch:
            return

        num_training_batches = trainer.num_training_batches

        model = pl_module.model

        sigma = self.sigma_init + (self.sigma_final - self.sigma_init) * batch_idx / num_training_batches

        pl_module.log("sigma", sigma, prog_bar=True, rank_zero_only=True, logger=True)

        for mn, m in model.named_modules():
            if isinstance(m, GaudiGBLR):
                if isinstance(m.sigma, torch.Tensor):
                    m.sigma.data = torch.ones_like(m.sigma) * sigma
                else:
                    m.sigma = sigma



