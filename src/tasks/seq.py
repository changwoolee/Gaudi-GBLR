from typing import Any, List

import torch
import torch.nn as nn
import hydra
from lightning import LightningModule, LightningDataModule
from torchmetrics import MetricCollection

from einops import rearrange

from omegaconf import OmegaConf

from src.utils.utils import get_logger
from src.optim.param_grouping import group_parameters_for_optimizer
from src.utils.checkpoint import load_checkpoint

logger = get_logger(__name__)


class SequenceModel(LightningModule):

    def __init__(self, cfg, model_cfg=None):
        """If model_cfg is passed, it will take precedence over cfg.model
        """
        super().__init__()
        # this line ensures params passed to LightningModule will be saved to ckpt
        # it also allows to access params with 'self.hparams' attribute
        self.save_hyperparameters(cfg)
        self.cfg = cfg
        self.model_cfg = model_cfg or self.cfg.model

        self.instantiate_datamodule()
        self.instantiate_model()
        self.warmstart()
        self.instantiate_loss()
        self.instantiate_metrics()

    def instantiate_datamodule(self):
        logger.info(f"Instantiating datamodule <{self.cfg.datamodule._target_}>")
        # Calling this self.datamodule will mess with PL since it also assigns self.datamodule
        self._datamodule: LightningDataModule = hydra.utils.instantiate(self.cfg.datamodule)
        self._datamodule.prepare_data()
        self._datamodule.setup()

    def instantiate_model(self):
        if hasattr(self._datamodule, 'num_classes'):
            self.model_cfg.num_classes = self._datamodule.num_classes
        if (hasattr(self._datamodule, 'vocab_size')
            and self.model_cfg.get('embedding_cfg', None) is not None):
            self.model_cfg.embedding_cfg.num_embeddings = self._datamodule.vocab_size
        logger.info(f"Instantiating model <{self.model_cfg._target_}>")
        recursive = getattr(self.model_cfg, '_recursive_', False)
        self.model = hydra.utils.instantiate(self.model_cfg, _recursive_=recursive)
        # Mixup / Cutmix
        if hasattr(self.cfg.train, 'mixup'):
            if hasattr(self._datamodule, 'num_classes'):
                self.cfg.train.mixup.num_classes = self._datamodule.num_classes
            self.mixup = hydra.utils.instantiate(self.cfg.train.mixup)
        else:
            self.mixup = None

    def instantiate_loss(self):
        loss_fn_cfg = self.cfg.train.get('loss_fn', {'_target_': 'torch.nn.CrossEntropyLoss'})
        self.loss_fn = hydra.utils.instantiate(loss_fn_cfg)
        reg_fn_cfg = self.cfg.train.get('reg_fn', None)
        self.reg_fn = hydra.utils.instantiate(reg_fn_cfg)
        loss_fn_val_cfg = self.cfg.train.get('loss_fn_val', loss_fn_cfg)
        self.loss_fn_val = hydra.utils.instantiate(loss_fn_val_cfg)

    def instantiate_metrics(self):
        # use separate metric instance for train, val and test step
        # to ensure a proper reduction over the epoch
        if 'eval' in self.cfg and 'metrics' in self.cfg.eval:
            metrics_cfg = self.cfg.eval.metrics
        else:
            metrics_cfg = {'acc': {'_target_': 'torchmetrics.Accuracy'}}
        for k in metrics_cfg:
            if 'acc' in k and 'task' not in metrics_cfg[k]:
                metrics_cfg[k]['task'] = 'multiclass'
                metrics_cfg[k]['num_classes'] = self._datamodule.num_classes
        metrics = MetricCollection({name: hydra.utils.instantiate(cfg)
                                    for name, cfg in metrics_cfg.items()})
        self.train_metrics = metrics.clone(prefix='train/')
        self.val_metrics = metrics.clone(prefix='val/')
        self.test_metrics = metrics.clone(prefix='test/')

    def warmstart(self):
        if self.cfg.train.get('warmstart', None) is not None:
            logger.info(f"Warm-starting with weights from {self.cfg.train.warmstart.path}")
            strict = self.cfg.train.warmstart.get('strict', True)
            state_dict = load_checkpoint(self.cfg.train.warmstart.path)
            if self.cfg.train.warmstart.get('post_process', None) is not None:
                state_dict = hydra.utils.instantiate(self.cfg.train.warmstart.post_process,
                                                     state_dict)
            load_return = self.model.load_state_dict(state_dict, strict=False)
            logger.info(load_return)

    def forward(self, *args, **kwargs):
        return self.model(*args, **kwargs)

    def step(self, batch: Any, is_train=True):
        try:
            x, y, lengths = batch
        except ValueError:
            x, y = batch
            lengths = None
        if is_train and self.mixup is not None:
            x, y = self.mixup(x, y)
        targets = y.argmax(dim=-1) if is_train and self.mixup is not None else y  # In case of Mixup
        output = self.forward(x) if lengths is None else self.forward(x, lengths=lengths)
        loss = self.loss_fn(output, y) if is_train else self.loss_fn_val(output, y)
        if self.reg_fn is not None:
            reg = self.reg_fn(self)
        else:
            reg = None
        return loss, output, targets, reg
        #output = self.forward(x) if lengths is None else self.forward(x, lengths=lengths)
        #loss = self.loss_fn(output, y) if is_train else self.loss_fn_val(output, y)
        #return loss, output, y

    def shared_step(self, batch: Any, batch_idx: int, phase='train'):
        loss, output, targets, reg = self.step(batch, is_train=(phase == 'train'))
        with torch.no_grad():
            metrics = getattr(self, f'{phase}_metrics')(output, targets)
        self.log(f"{phase}/loss", loss, on_step=False, on_epoch=True, prog_bar=False, sync_dist=True)
        #if phase == 'train' and reg is not None:
        #    self.log(f"{phase}/reg", reg, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
        if reg is None:
            reg = 0.
        self.log_dict(metrics, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
        return {"loss": loss + reg, "output": output, "targets": targets}

    def training_step(self, batch: Any, batch_idx: int):
        return self.shared_step(batch, batch_idx, phase='train')

    def validation_step(self, batch: Any, batch_idx: int):
        return self.shared_step(batch, batch_idx, phase='val')

    def test_step(self, batch: Any, batch_idx: int):
        return self.shared_step(batch, batch_idx, phase='test')

    def configure_optimizers(self):
        if 'optimizer_param_grouping' in self.cfg.train:  # Set zero weight decay for some params
            parameters = group_parameters_for_optimizer(self.model, self.cfg.train.optimizer,
                                                        **self.cfg.train.optimizer_param_grouping)
        else:
            # parameters = self.model.parameters()
            parameters = self.parameters() # [21-09-08] AG: this will train task specific parameters such as Retrieval head for AAN
        optimizer = hydra.utils.instantiate(self.cfg.train.optimizer, parameters)


        # Log optimizer info
        for i, g in enumerate(optimizer.param_groups):
            ntensors = len(g['params'])
            nparams = sum(p.numel() for p in g['params'])
            hparams = {k: v for k, v in g.items() if k != 'params'}
            logger.info(f'Optimizer group {i}: {ntensors} tensors, {nparams} parameters, {hparams}')
            #print(f'Optimizer group {i}: {ntensors} tensors, {nparams} parameters, {hparams}')

        if 'scheduler' not in self.cfg.train:
            return optimizer
        else:
            # lr_scheduler should be called either every step (default) or every epoch
            lr_scheduler = hydra.utils.instantiate(self.cfg.train.scheduler, optimizer)
            return [optimizer], {'scheduler': lr_scheduler,
                                 'interval': self.cfg.train.get('scheduler_interval', 'step'),
                                 'monitor': self.cfg.train.get('scheduler_monitor', 'val/loss')}


class SequenceDualModel(SequenceModel):

    def step(self, batch: Any, is_train=True):
        x1, x2, y, lengths1, lengths2 = batch
        output = self.forward(x1, x2, lengths1=lengths1, lengths2=lengths2)
        loss = self.loss_fn(output, y) if is_train else self.loss_fn_val(output, y)
        output = torch.argmax(output, dim=1)
        return loss, output, y, None


class SequenceLMModel(SequenceModel):

    def instantiate_model(self):
        if (hasattr(self._datamodule, 'vocab_size')
            and self.model_cfg.get('embedding_cfg', None) is not None):
            self.model_cfg.embedding_cfg.num_embeddings = self._datamodule.vocab_size
        logger.info(f"Instantiating model <{self.model_cfg._target_}>")
        # Huggingface models need the config object to be instantiated first
        config = hydra.utils.instantiate(self.model_cfg.pop('config'), _recursive_=False)
        self.model = hydra.utils.instantiate(self.model_cfg, config, _recursive_=False)

    def step(self, batch: Any, is_train=True):
        x, y = batch
        output = self.forward(x).logits
        output = rearrange(output, '... C -> (...) C')
        y = rearrange(y, '... -> (...)')
        loss = self.loss_fn(output, y) if is_train else self.loss_fn_val(output, y)
        return loss, output, y, None



class GPT2Wiki103(SequenceLMModel):
    def instantiate_model(self):
        from transformers import AutoTokenizer, AutoModelWithLMHead

        model_name = self.model_cfg.model_name #"Graphcore/gpt2-wikitext-103"
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelWithLMHead.from_pretrained(model_name)
        self.replace_linear_layers()
        

    def replace_linear_layers(self):
        if self.model_cfg.layer_type is None:
            return

        from copy import deepcopy
        from transformers.modeling_utils import Conv1D
        from src.models.layers.gblr import GaudiGBLR
        from src.models.modules.olb import find_olb
        from src.models.layers.fastlinear import LowRank
        from src.models.layers.monarch_linear import MonarchLinear

        new_model = deepcopy(self.model)
        for mn, m in self.model.named_modules():
            if isinstance(m, Conv1D) and 'transformer.h' in mn and not 'attn.c_proj' in mn:
                logger.info(mn)
                M = m.weight.T
                device = M.device
                in_features, out_features = m.weight.size()
                #budget_in_ratio = self.model_cfg.budget_in_ratio
                if self.model_cfg.layer_type == 'lr':
                    rank = int(self.model_cfg.rank_ratio * min(in_features, out_features))
                    logger.info("Rank: {} / {}".format(rank, min(in_features, out_features)))
                    new_layer = LowRank(in_features=in_features,
                                        out_features=out_features,
                                        rank=rank)
                    new_layer.set_weights_from_projection(M)
                    if m.bias is not None:
                        new_layer.bias.data = m.bias.data

                elif self.model_cfg.layer_type == 'monarch':
                    nblocks = self.model_cfg.nblocks
                    new_layer = MonarchLinear(in_features=in_features, 
                                              out_features=out_features, 
                                              nblocks=nblocks)
                    if self.model_cfg.decompose:
                        from src.ops.blockdiag_butterfly_einsum import blockdiag_butterfly_project_einsum_rank
                        min_f = min(in_features, out_features)
                        if nblocks == 3:
                            rank = min_f // 8
                        elif nblocks == 6:
                            rank = min_f // 32
                        else:
                            rank = min_f // nblocks // nblocks
                        w1, w2 = blockdiag_butterfly_project_einsum_rank(M, nblocks, nblocks, rank)
                        print(new_layer.blkdiag1.size(), w1.size())
                        print(new_layer.blkdiag2.size(), w2.size())
                        new_layer.blkdiag1.data = w1
                        new_layer.blkdiag2.data = w2
                        new_layer = new_layer.to(device)
                        del M
                    if m.bias is not None:
                        new_layer.bias.data = m.bias.data

                elif self.model_cfg.layer_type == 'gaudi':
                    #if self.model_cfg.project_only:
                    #    self.model_cfg.gaudi_params['width_init'] = 'lr0.25' 
                    new_layer = GaudiGBLR(in_features, out_features, **self.model_cfg.gaudi_params).to(device)
                    if self.model_cfg.decompose:
                        if self.model_cfg.project_only:
                            new_layer.set_weights_from_projection(M)
                            if m.bias is not None:
                                new_layer.bias.data = m.bias.data
                        else:
                            budget = int(self.model_cfg.rank_ratio * min(in_features, out_features) * (in_features + out_features))
                            #M = M.cuda()
                            w,l,U,Vt = find_olb(M=M, budget=budget,
                                                thres_row_list=[0.98],
                                                thres_col_list=[0.98],
                                                weight_lr=0.005,
                                                structure_lr_base=self.model_cfg.structure_lr_base,
                                                verbose=False,
                                                niter=1000,
                                                sched_params={'start_factor': 1.0, 'end_factor': 0.01},
                                                )
                            w = w.flip(0)
                            l = l.flip(0)
                            new_layer.lr_weight1.data = Vt.to(device).data
                            new_layer.lr_weight2.data = U.to(device).data
                            new_layer.widths.data = w.to(device).data
                            new_layer.locations.data = l.to(device).data
                            del M
                            if m.bias is not None:
                                new_layer.bias.data = m.bias.data

                elif self.model_cfg.layer_type == 'None':
                    continue

                else:
                    raise NotImplementedError()               

                with torch.no_grad():
                    parent_name = ".".join(mn.split(".")[:-1])
                    child_name = mn.split(".")[-1]
                    for new_mn, new_m in new_model.named_modules():
                        if new_mn == parent_name:
                            new_m.add_module(child_name, deepcopy(new_layer))

        self.model = new_model
        #if self.load_from_decomposed is not None:
        #    sd = torch.load(self.load_from_decomposed, map_location=device)
        #    del sd['classifier.weight']
        #    del sd['classifier.bias']
        #    missing, unexpected = self.model.load_state_dict(sd, strict=False)
        #    log.info("Parameters restored from {}".format(self.load_from_decomposed))
        #    if missing is not None:
        #        log.info("Missing Keys: {}".format(missing))
        #    if unexpected is not None:
        #        log.info("Unexpected Keys: {}".format(unexpected))



class ConvNetImageNet(SequenceModel):
    def instantiate_model(self):
        model_name = self.model_cfg.model_name #"Graphcore/gpt2-wikitext-103"
        self.model = torch.hub.load('pytorch/vision:v0.10.0', model_name, pretrained=self.model_cfg.pretrained)
        self.replace_linear_layers()
         # Mixup / Cutmix
        if hasattr(self.cfg.train, 'mixup'):
            if hasattr(self._datamodule, 'num_classes'):
                self.cfg.train.mixup.num_classes = self._datamodule.num_classes
            self.mixup = hydra.utils.instantiate(self.cfg.train.mixup)
        else:
            self.mixup = None       

    def replace_linear_layers(self):
        if self.model_cfg.layer_type is None:
            return

        from copy import deepcopy
        from src.models.layers.gblr import GaudiGBLR, GaudiGBLRConv2d, GaudiGBLRConv2dIntegrated

        new_model = deepcopy(self.model)
        for mn, m in self.model.named_modules():
            if isinstance(m, nn.Conv2d) and 'layer' in mn and 'downsample' not in mn:
                logger.info(mn)
                M = m.weight
                device = M.device
                out_features, in_features, k1, k2 = M.size() 
                conv_params = {'stride': m.stride,
                               'padding': m.padding,
                               'dilation': m.dilation,
                               'groups': m.groups,
                               'padding_mode': m.padding_mode,
                               #'dtype': m.dtype,
                               #'device': m.device,
                               }
                               
                if self.model_cfg.layer_type == 'gaudi':
                    if self.model_cfg.per_kernel:
                        new_layer = GaudiGBLRConv2d(m, gaudi_params=self.model_cfg.gaudi_params, init=self.model_cfg.init).to(device)
                    else:
                        new_layer = GaudiGBLRConv2dIntegrated(m, gaudi_params=self.model_cfg.gaudi_params, init=self.model_cfg.init).to(device)
                

                elif self.model_cfg.layer_type == 'None':
                    continue

                else:
                    raise NotImplementedError()               

                with torch.no_grad():
                    parent_name = ".".join(mn.split(".")[:-1])
                    child_name = mn.split(".")[-1]
                    for new_mn, new_m in new_model.named_modules():
                        if new_mn == parent_name:
                            new_m.add_module(child_name, deepcopy(new_layer))

        self.model = new_model
        #if self.load_from_decomposed is not None:
        #    sd = torch.load(self.load_from_decomposed, map_location=device)
        #    del sd['classifier.weight']
        #    del sd['classifier.bias']
        #    missing, unexpected = self.model.load_state_dict(sd, strict=False)
        #    log.info("Parameters restored from {}".format(self.load_from_decomposed))
        #    if missing is not None:
        #        log.info("Missing Keys: {}".format(missing))
        #    if unexpected is not None:
        #        log.info("Unexpected Keys: {}".format(unexpected))


