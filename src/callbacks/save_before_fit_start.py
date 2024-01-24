import torch
import lightning.pytorch as pl
from pathlib import Path
from src.utils import utils
from lightning.pytorch.utilities import rank_zero_only

log = utils.get_logger(__name__)

class SaveBeforeFitStart(pl.callbacks.Callback):
    def __init__(self, save_dir):
        super().__init__()
        self.save_dir = save_dir
    
    @rank_zero_only
    def on_fit_start(self, trainer, pl_module):
        path = str(Path(self.save_dir) / 'finetune_start_point.ckpt')
        log.info("Saving finetuing starting point...")
        #trainer.save_checkpoint(path)
        torch.save(pl_module.model.state_dict(), path)


