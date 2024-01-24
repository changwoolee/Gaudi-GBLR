import rootutils

root = rootutils.setup_root(
    search_from=__file__,
    indicator=[".git", "pyproject.toml"],
    pythonpath=True,
    dotenv=True,
)

from typing import List, Optional
from pathlib import Path

import torch

import hydra
from omegaconf import OmegaConf, DictConfig
from lightning import (
    Callback,
    LightningDataModule,
    LightningModule,
    Trainer,
    seed_everything,
)

from lightning.pytorch.loggers import Logger

from src.utils import utils

log = utils.get_logger(__name__)


def load_checkpoint(path, device='cpu'):
    path = Path(path).expanduser()
    if path.is_dir():
        path /= 'checkpoint_last.pt'
    # dst = f'cuda:{torch.cuda.current_device()}'
    log.info(f'Loading checkpoint from {str(path)}')
    state_dict = torch.load(path, map_location=device)
    # T2T-ViT checkpoint is nested in the key 'state_dict_ema'
    if state_dict.keys() == {'state_dict_ema'}:
        state_dict = state_dict['state_dict_ema']
    return state_dict


def evaluate(config: DictConfig) -> None:
    """Example of inference with trained model.
    It loads trained image classification model from checkpoint.
    Then it loads example image and predicts its label.
    """

    # load model from checkpoint
    # model __init__ parameters will be loaded from ckpt automatically
    # you can also pass some parameter explicitly to override it

    # We want to add fields to config so need to call OmegaConf.set_struct
    OmegaConf.set_struct(config, False)

    # load Lightning model
    
    checkpoint_type = config.eval.get('checkpoint_type', 'lightning')
    if checkpoint_type not in ['lightning', 'pytorch', 'timm']:
        raise NotImplementedError(f'checkpoint_type ${checkpoint_type} not supported')

    if checkpoint_type == 'lightning':
        cls = hydra.utils.get_class(config.task._target_)
        trained_model = cls.load_from_checkpoint(checkpoint_path=config.eval.ckpt,
                                                 #root_dir=config.root_dir,
                                                 #data_dir=config.data_dir,
                                                 #datamodule=config.datamodule,
                                                 #log_dir=config.log_dir,
                                                 #output_dir=config.output_dir,
                                                 #work_dir=config.work_dir,
                                                 #logger=None, #config.logger,
                                                 #callbacks=config.callbacks,
                                                 #trainer=config.trainer,
                                                 #strict=False,
                                                 )
    elif checkpoint_type == 'pytorch':
        trained_model: LightningModule = hydra.utils.instantiate(config.task, cfg=config,
                                                                 _recursive_=False)
        load_return = trained_model.load_state_dict(load_checkpoint(config.eval.ckpt,
                                                                          device=trained_model.device),
                                                          strict=False)
        log.info(load_return)
    else:
        trained_model: LightningModule = hydra.utils.instantiate(config.task, cfg=config, _recursive_=False)



    # datamodule: LightningDataModule = hydra.utils.instantiate(config.datamodule)
    datamodule: LightningDataModule = trained_model._datamodule
    datamodule.prepare_data()
    datamodule.setup()

    # print model hyperparameters
    log.info(f'Model hyperparameters: {trained_model.hparams}')

    # Init Lightning callbacks
    callbacks: List[Callback] = []
    if "callbacks" in config:
        for _, cb_conf in config["callbacks"].items():
            if cb_conf is not None and "_target_" in cb_conf:
                log.info(f"Instantiating callback <{cb_conf._target_}>")
                callbacks.append(hydra.utils.instantiate(cb_conf))

    # Init Lightning loggers
    logger: List[Logger] = []
    if "logger" in config:
        for _, lg_conf in config["logger"].items():
            if "_target_" in lg_conf:
                log.info(f"Instantiating logger <{lg_conf._target_}>")
                logger.append(hydra.utils.instantiate(lg_conf))

    # Init Lightning trainer
    log.info(f"Instantiating trainer <{config.trainer._target_}>")
    trainer: Trainer = hydra.utils.instantiate(
        config.trainer, callbacks=callbacks, logger=logger,  _convert_="partial"
    )

    # Evaluate the model
    log.info("Starting evaluation!")
    if config.eval.get('run_val', True):
        trainer.validate(model=trained_model, datamodule=datamodule)


    if config.eval.get('run_test', True):
        trainer.test(model=trained_model, datamodule=datamodule)

    # Make sure everything closed properly
    log.info("Finalizing!")
    utils.finish(
        config=config,
        model=trained_model,
        datamodule=datamodule,
        trainer=trainer,
        callbacks=callbacks,
        logger=logger,
    )

@hydra.main(config_path=root / "configs/", config_name="config.yaml", version_base="1.2")
def main(cfg: DictConfig) -> None:
    evaluate(cfg)

if __name__ == "__main__":
    main()
