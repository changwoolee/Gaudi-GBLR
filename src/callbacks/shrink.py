from typing import Any
from collections import OrderedDict

from lightning.pytorch import Callback, Trainer
from src.models.layers.gblr import GaudiGBLR

import torch
import torch.nn as nn
import torch.nn.functional as F

import logging

_logger = logging.getLogger(__name__)


class Softshrink(Callback):
    """Monitor the scales of weights and gradients.
    """


    def __init__(self, thres, rate=1):
        super().__init__()
        self.thres = thres
        _logger.info("thres: {}".format(thres))
        self.shrink_fn = F.softshrink
        self.rate=rate

    def get_thres(self, thres, trainer, pl_module, batch_idx):
        return self.thres

    def get_lambd(self, thres, trainer, pl_module, batch_idx):
        opt = pl_module.optimizers()
        lr = opt.param_groups[-1]['lr']
        lambd = self.get_thres(thres, trainer, pl_module, batch_idx) * lr
        return lambd

    def on_train_batch_end(self, trainer: Trainer, pl_module, outputs, batch, batch_idx) -> None:
        if batch_idx % self.rate != self.rate - 1:
            return 
        with torch.no_grad():
            model = pl_module.model
            named_parameters = {}
            modules = {}
            target_modules = (GaudiGBLR)

            lambd = self.get_lambd(self.thres, trainer, pl_module, batch_idx)

            for mn, m in model.named_modules():
                if isinstance(m, target_modules) and m.widths.requires_grad:
                    m.widths.data = self.shrink_fn(m.widths.data, lambd).clamp_(0.0,1.0)
                    m.locations.data = m.locations.data - m.locations.data.floor()



class TargetedSoftshrink(Softshrink):
    def __init__(self, thres, target_width, cont=False, p=1.0, freeze=False, rate=1, **kwargs):
        super().__init__(thres)
        _logger.info("target width: {}".format(target_width))
        self.target_width = target_width
        self.cont = cont
        self.p = p
        self.freeze = freeze
        self.rate = rate


    @torch.no_grad()
    def get_thres(self, thres, trainer, pl_module, batch_idx):
        target_modules = (GaudiGBLR)
        width_sum = 0.
        count = 0
        if self.target_width > 0:
            for mn, m in pl_module.model.named_modules():
                if isinstance(m, target_modules):
                    width_sum += (((m.widths[0:1,:] * m.widths[1:2,:]) > 0).float() * m.widths).mean()
                    count += 1
            if count == 0:
                return 0.0
            width_mean = width_sum / count
            if width_mean < self.target_width:
                thres = 0.0 
            else:
                thres_max = super().get_thres(thres, trainer, pl_module, batch_idx)
                if self.cont:
                    thres = thres_max * ((width_mean - self.target_width) / (1.0 - self.target_width)) ** self.p
                else: 
                    thres = thres_max
        else:
            thres = super().get_thres(thres, trainer, pl_module, batch_idx)

        return thres

    def on_train_batch_end(self, trainer: Trainer, pl_module, outputs, batch, batch_idx) -> None:
        if batch_idx % self.rate != self.rate - 1:
            return 
        with torch.no_grad():
            model = pl_module.model
            named_parameters = {}
            modules = {}
            target_modules = (GaudiGBLR)

            lambd = self.get_lambd(self.thres, trainer, pl_module, batch_idx)

            for mn, m in model.named_modules():
                if isinstance(m, target_modules) and m.widths.requires_grad:
                    m.widths.data[0,:] = self.shrink_fn(m.widths.data[0,:], lambd).clamp_(m.min_widths[0], m.max_widths[0])
                    m.widths.data[1,:] = self.shrink_fn(m.widths.data[1,:], lambd).clamp_(m.min_widths[1], m.max_widths[1])
                    m.locations.data = m.locations.data - m.locations.data.floor()
                    if self.freeze and lambd == 0:
                        m.widths.requires_grad_(False)
                        m.locations.requires_grad_(False)



class AdaptiveSoftshrink(TargetedSoftshrink):
    """Monitor the scales of weights and gradients.
    """

    def __init__(self, thres, init_thres=None, mult_factor=None, start_epoch=0, end_epoch=10, **kwargs):
        super().__init__(thres, **kwargs)
        if init_thres is not None and mult_factor is None:
            self.init = init_thres
        elif mult_factor is not None:
            self.init = thres * mult_factor
        else:
            raise ValueError("Either init_thres or mult_factor must be given.")
        self.start_epoch = start_epoch
        self.end_epoch = end_epoch
        _logger.info("init_thres: {}, start: {}, end: {}".format(init_thres, start_epoch, end_epoch))

    @torch.no_grad()
    def get_thres(self, thres, trainer, pl_module, batch_idx):
        #opt = pl_module.optimizers()
        #lr = opt.param_groups[-1]['lr']

        if trainer.current_epoch < self.start_epoch:
            thres = self.init
        elif trainer.current_epoch >= self.end_epoch:
            thres = super().get_thres(thres, trainer, pl_module, batch_idx)
        else:
            total_decay_epoch = self.end_epoch - self.start_epoch
            total_decay_batch = total_decay_epoch * trainer.num_training_batches
            current_batch_idx = (trainer.current_epoch - self.start_epoch) * trainer.num_training_batches + batch_idx
            t = abs(total_decay_batch - current_batch_idx) / total_decay_batch
            thres_ = super().get_thres(thres, trainer, pl_module, batch_idx)
            thres = (self.init - thres_) * t + thres_  
        return thres





class MeanShrink(Callback):
    def __init__(self, target_width, lambd=1.0, rate=1, **kwargs):
        super().__init__()
        self.target = target_width
        self.lambd = lambd
        self.rate = rate
        self.modules = {}
        _logger.info("target: {}".format(self.target))

    def on_train_batch_end(self, trainer: Trainer, pl_module, outputs, batch, batch_idx) -> None:
        if batch_idx % self.rate != self.rate - 1:
            return 
        with torch.no_grad():
            model = pl_module.model
            named_parameters = {}
            modules = {}
            target_modules = (GaudiGBLR)

            width_sum = 0.
            count = 0
            if len(self.modules) == 0:
                for mn, m in model.named_modules():
                    if isinstance(m, target_modules) and m.widths.requires_grad:
                        if mn not in self.modules:
                            self.modules[mn] = m

            for mn in self.modules:
                widths = self.modules[mn].widths
                width_sum += (((widths[0:1,:] * widths[1:2,:]) > 0).float() * widths).mean()
                count += 1
            if count > 0:
                avg_width = width_sum /count
                opt = pl_module.optimizers()
                lr = opt.param_groups[-1]['lr']
                for mn in self.modules:
                    if avg_width > self.target:
                        self.modules[mn].widths.data = (self.modules[mn].widths.data - self.lambd * (avg_width - self.target)).clamp(0.0, 1.0)
                    else:
                        self.modules[mn].widths.data = self.modules[mn].widths.data.clamp(0.0, 1.0)
                    self.modules[mn].locations.data = self.modules[mn].locations.data - self.modules[mn].locations.data.floor()


