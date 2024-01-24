# Adapted from https://github.com/rwightman/pytorch-image-models/blob/master/benchmark.py
from typing import Any, List, Sequence

import torch
from lightning.pytorch import Callback, Trainer, LightningModule
from lightning.pytorch.utilities import rank_zero_only
from lightning.pytorch.utilities.parsing import AttributeDict

from src.utils.flops import has_deepspeed_profiling, has_fvcore_profiling
from src.utils.flops import profile_deepspeed, profile_fvcore, profile_fvcore_sinc_gaussian, profile_fvcore_gaudi_conv

from src.models.layers.fouriermask import FourierMaskLR, FourierMaskConv2d
from src.utils import utils
log = utils.get_logger(__name__)


class NumParamsGPT2Gaudi(Callback):
    def __init__(self):
        super().__init__()

    @rank_zero_only
    #def on_train_batch_end(self, trainer, pl_module, *args, **kwargs):
    def on_validation_epoch_start(self, trainer, pl_module):
        num_params_dict = {}
        count_dict = {}
        counts = 0
        with torch.no_grad():
            for mn, m in pl_module.model.named_modules():
                if isinstance(m, FourierMaskLR):
                    num_params = int(m.get_num_params())
                    counts += num_params * 1024  

        if counts > 0:
            counts += (805306368 + 603979776) * 12
            ratio = counts / 9.66E+10
            pl_module.log('flops_ratio', ratio, rank_zero_only=True, prog_bar=True)

        
class NumParamsGPT2MediumGaudi(Callback):
    def __init__(self):
        super().__init__()

    @rank_zero_only
    #def on_train_batch_end(self, trainer, pl_module, *args, **kwargs):
    def on_validation_epoch_start(self, trainer, pl_module):
        num_params_dict = {}
        count_dict = {}
        counts = 0
        with torch.no_grad():
            for mn, m in pl_module.model.named_modules():
                if isinstance(m, FourierMaskLR):
                    num_params = int(m.get_num_params())
                    counts += num_params * 1024  

        if counts > 0:
            counts += (1073741824 + 1073741824) * 24
            ratio = counts / 3.35E+11
            pl_module.log('flops_ratio', ratio, rank_zero_only=True, prog_bar=True)

        



class FlopCount(Callback):
    """Counter the number of FLOPs used by the model
    """
    def __init__(self, profilers: List[str] = ['fvcore', 'deepspeed'],
                 input_size: tuple = (3, 224, 224), device=None,
                 sinc_gaussian=False,
                 baseline_complexity=None,
                 gaudi_conv=False):
        if not isinstance(profilers, Sequence):
            profilers = [profilers]
        if any(p not in ['fvcore', 'deepspeed'] for p in profilers):
            raise NotImplementedError('Only support fvcore and deepspeed profilers')
        if 'fvcore' in profilers and not has_fvcore_profiling:
            raise ImportError('fvcore is not installed. Install it by running `pip install fvcore`')
        elif 'deepspeed' in profilers and not has_deepspeed_profiling:
            raise ImportError('deepspeed is not installed')
        super().__init__()
        self.profilers = profilers
        self.input_size = tuple(input_size)
        self.device = device
        self.sinc_gaussian = sinc_gaussian
        self.baseline_complexity = baseline_complexity
        self.done = False
        self.gaudi_conv = gaudi_conv


    @rank_zero_only
    def on_validation_epoch_start(self, trainer, pl_module):
    #def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
    #    if batch_idx != 16:
    #        return
        with torch.no_grad():
            self.done=True
            if 'fvcore' in self.profilers:
                if self.sinc_gaussian:
                    _, macs, _, acts, comp_ratio = profile_fvcore_sinc_gaussian(pl_module, input_size=self.input_size,
                                                      detailed=False,
                                                      baseline_complexity=self.baseline_complexity,
                                                      )
                    pl_module.log("flop_ratio", comp_ratio, prog_bar=True, rank_zero_only=True)
                elif self.gaudi_conv:
                    _, macs, _, acts, comp_ratio = profile_fvcore_gaudi_conv(pl_module, input_size=self.input_size,
                                                      detailed=False,
                                                      baseline_complexity=self.baseline_complexity,
                                                      )
                    pl_module.log("flop_ratio", comp_ratio, prog_bar=True, rank_zero_only=True)
                else:
                    _, macs, _, acts = profile_fvcore(pl_module, input_size=self.input_size,
                                                      detailed=True)
                    trainer.logger.log_hyperparams({'GMACs': macs * 1e-9, 'MActs': acts * 1e-6})

                    print(macs * 1e-9, acts * 1e-6)
            if 'deepspeed' in self.profilers:
                macs, _= profile_deepspeed(pl_module, input_size=self.input_size,
                                           detailed=True)
                if 'fvcore' not in self.profilers:  # fvcore's MACs seem more accurate
                    trainer.logger.log_hyperparams({'GMACs': macs * 1e-9})


    @rank_zero_only
    def on_test_epoch_start(self, trainer, pl_module):
        self.on_validation_epoch_start(trainer, pl_module)



