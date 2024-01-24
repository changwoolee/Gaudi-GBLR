from datetime import datetime
from typing import Optional

from copy import deepcopy

import datasets
import torch
import torch.nn as nn
import hydra
from pytorch_lightning import LightningDataModule, LightningModule, Trainer, seed_everything
from torch.utils.data import DataLoader
from transformers import (
    AdamW,
    AutoConfig,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    get_linear_schedule_with_warmup,
)
from src.models.layers.fouriermask import FourierMaskLR
from src.models.modules.olb import find_olb
from src.models.layers.fastlinear import LowRank
from src.models.layers.monarch_linear import MonarchLinear, get_nblocks
from src.utils import utils
from src.optim.param_grouping import group_parameters_for_optimizer
log = utils.get_logger(__name__)
from omegaconf import OmegaConf, DictConfig


class GLUETransformer(LightningModule):
    def __init__(
        self,
        cfg,
        learning_rate: float = 2e-5,
        adam_epsilon: float = 1e-8,
        warmup_steps: int = 0,
        weight_decay: float = 0.0,
        train_batch_size: int = 32,
        eval_batch_size: int = 32,
        eval_splits: Optional[list] = None,
        layer_type=None,
        budget_in_ratio=0.5,
        nblocks=2,
        decompose=False,
        gaudi_params=None,
        structure_lr_base=80,
        load_from_decomposed=None,
        equal_sized=False,
        scale_gaudi_lr=False,
        **kwargs,
    ):
        super().__init__()
        self._datamodule = hydra.utils.instantiate(cfg.datamodule)
        self._datamodule.prepare_data()
        self._datamodule.setup()
        self.decompose = decompose
        self.load_from_decomposed = load_from_decomposed
        task_name = cfg.datamodule.task_name
        model_name_or_path = cfg.datamodule.model_name_or_path
        num_labels = self._datamodule.num_labels
        eval_splits = self._datamodule.eval_splits
        self.save_hyperparameters({
                                    "model_name_or_path": model_name_or_path,
                                    "task_name": task_name,
                                    "cfg": cfg,
                                    "learning_rate": learning_rate,
                                    "adam_epsilon": adam_epsilon,
                                    "warmup_steps": warmup_steps,
                                    "weight_decay": weight_decay,
                                    "train_batch_size": train_batch_size,
                                    "eval_batch_size": eval_batch_size,
                                    "eval_splits": eval_splits,
                                    "num_labels": num_labels,
                                    "eval_splits": eval_splits,
                                    'layer_type': layer_type,
                                    'budget_in_ratio': budget_in_ratio,
                                    'gaudi_params': gaudi_params,
                                    'nblocks': nblocks,
                                    'structure_lr_base': structure_lr_base,
                                    'equal_sized': equal_sized,
                                    'scale_gaudi_lr': scale_gaudi_lr,
                                    **kwargs},
                                    ignore=["decompose", "load_from_decomposed"],
                                    )

        self.config = AutoConfig.from_pretrained(model_name_or_path, num_labels=num_labels)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name_or_path, config=self.config)
        self.metric = datasets.load_metric(
            "glue", self.hparams.task_name, experiment_id=datetime.now().strftime("%d-%m-%Y_%H-%M-%S")
        )

        self.replace_linear_layers()


    def replace_linear_layers(self):
        if self.hparams.layer_type is None:
            return

        new_model = deepcopy(self.model)
        for mn, m in self.model.named_modules():
            if isinstance(m, nn.Linear) and 'bert.encoder.layer' in mn and not 'attention.output' in mn:
                log.info(mn)
                M = m.weight
                device = M.device
                budget_in_ratio = self.hparams.budget_in_ratio
                if self.hparams.layer_type == 'lr':
                    #rank = int(budget_in_ratio * m.in_features * m.out_features / (m.in_features + m.out_features))
                    if 'attention' in mn:
                        rank = int(0.25 * min(m.in_features, m.out_features))
                    else:
                        rank = int(0.5 * min(m.in_features, m.out_features))
                    log.info("Rank: {} / {}".format(rank, min(m.in_features, m.out_features)))
                    new_layer = LowRank(in_features=m.in_features,
                                        out_features=m.out_features,
                                        rank=rank)
                    new_layer.set_weights_from_projection(M)
                    if m.bias is not None:
                        new_layer.bias.data = m.bias.data

                elif self.hparams.layer_type == 'monarch':
                    #nblocks = get_nblocks(m.in_features, m.out_features, budget_in_ratio)
                    if 'attention' in mn:
                        nblocks = 4
                    else:
                        nblocks = 2
                    new_layer = MonarchLinear(in_features=m.in_features, 
                                              out_features=m.out_features, 
                                              nblocks=nblocks)
                    if self.decompose:
                        new_layer.bias.data = torch.zeros_like(new_layer.bias)
                        M = M.cuda()
                        new_layer = new_layer.cuda()
                        I = torch.eye(new_layer.in_features_extended).cuda()
                        opt = torch.optim.Adam([new_layer.blkdiag1, new_layer.blkdiag2], lr=1e-3)
                        loss = torch.tensor(0.0)
                        for t in range(1000):
                            opt.zero_grad()
                            loss = torch.mean((M.T.detach() - new_layer(I))**2)
                            loss.backward()
                            opt.step()
                        log.info("Num Monarch Blocks: {}, Loss: {}".format(nblocks, loss.item()))
                        new_layer = new_layer.to(device)
                    if m.bias is not None:
                        new_layer.bias.data = m.bias.data

                elif self.hparams.layer_type == 'gaudi':
                    if self.hparams.scale_gaudi_lr:
                        lr = min(2e-3, 0.5 / len(self._datamodule.train_dataloader()))
                        self.hparams.gaudi_params['width_learning_rate'] = lr
                        self.hparams.gaudi_params['location_learning_rate'] = lr
                        log.info("Gaudi Structural Parameters Scaled to {:e}".format(lr))
                    new_layer = FourierMaskLR(m.in_features, m.out_features, **self.hparams.gaudi_params).to(device)
                    if self.decompose:
                        #budget = int(6/7*budget_in_ratio * min(m.in_features, m.out_features) * (m.in_features + m.out_features))
                        if self.hparams.equal_sized:
                            budget = int(0.35 * min(m.in_features, m.out_features) * (m.in_features + m.out_features))
                        else: 
                            if 'attention' in mn:
                                budget = int(0.23 * min(m.in_features, m.out_features) * (m.in_features + m.out_features))
                            else:
                                budget = int(0.48 * min(m.in_features, m.out_features) * (m.in_features + m.out_features))
                        M = M.cuda()
                        w,l,U,Vt = find_olb(M=M, budget=budget,
                                            thres_row_list=[0.98],
                                            thres_col_list=[0.98],
                                            weight_lr=0.005,
                                            structure_lr_base=self.hparams.structure_lr_base,
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
                        if m.bias is not None:
                            new_layer.bias.data = m.bias.data


                else:
                    raise NotImplementedError()               

                with torch.no_grad():
                    parent_name = ".".join(mn.split(".")[:-1])
                    child_name = mn.split(".")[-1]
                    for new_mn, new_m in new_model.named_modules():
                        if new_mn == parent_name:
                            new_m.add_module(child_name, deepcopy(new_layer))

        self.model = new_model
        if self.load_from_decomposed is not None:
            sd = torch.load(self.load_from_decomposed, map_location=device)
            del sd['classifier.weight']
            del sd['classifier.bias']
            missing, unexpected = self.model.load_state_dict(sd, strict=False)
            log.info("Parameters restored from {}".format(self.load_from_decomposed))
            if missing is not None:
                log.info("Missing Keys: {}".format(missing))
            if unexpected is not None:
                log.info("Unexpected Keys: {}".format(unexpected))



    def forward(self, **inputs):
        return self.model(**inputs)

    def training_step(self, batch, batch_idx):
        outputs = self(**batch)
        loss = outputs[0]
        return loss


    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        labels = batch["labels"]
        outputs = self(**batch)
        val_loss, logits = outputs[:2]

        if self.hparams.num_labels > 1:
            preds = torch.argmax(logits, axis=1)
        elif self.hparams.num_labels == 1:
            preds = logits.squeeze()

        labels = batch["labels"]

        return {"loss": val_loss, "preds": preds, "labels": labels}

    def test_step(self, batch, batch_idx, dataloader_idx=0):
        self.validation_step(batch, batch_idx, dataloader_idx)

    def validation_epoch_end(self, outputs):
        if self.hparams.task_name == "mnli":
            for i, output in enumerate(outputs):
                # matched or mismatched
                split = self.hparams.eval_splits[i].split("_")[-1]
                preds = torch.cat([x["preds"] for x in output]).detach().cpu().numpy()
                labels = torch.cat([x["labels"] for x in output]).detach().cpu().numpy()
                loss = torch.stack([x["loss"] for x in output]).mean()
                self.log(f"val_loss_{split}", loss, prog_bar=True)
                split_metrics = {
                    f"{k}_{split}": v for k, v in self.metric.compute(predictions=preds, references=labels).items()
                }
                self.log_dict(split_metrics, prog_bar=True)
            return loss

        preds = torch.cat([x["preds"] for x in outputs]).detach().cpu().numpy()
        labels = torch.cat([x["labels"] for x in outputs]).detach().cpu().numpy()
        loss = torch.stack([x["loss"] for x in outputs]).mean()
        self.log("val_loss", loss, prog_bar=True)
        self.log_dict(self.metric.compute(predictions=preds, references=labels), prog_bar=True)

    def configure_optimizers(self):
        """Prepare optimizer and schedule (linear warmup and decay)"""
        if self.hparams.layer_type != "gaudi":
            model = self.model
            no_decay = ["bias", "LayerNorm.weight"]
            optimizer_grouped_parameters = [
                {
                    "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
                    "weight_decay": self.hparams.weight_decay,
                },
                {
                    "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
                    "weight_decay": 0.0,
                },
            ]
            optimizer = torch.optim.AdamW(optimizer_grouped_parameters, lr=self.hparams.learning_rate, eps=self.hparams.adam_epsilon)


        else:

            parameters = group_parameters_for_optimizer(self.model, 
                                                        optimizer_cfg=OmegaConf.create({
                                                            'weight_decay': self.hparams.weight_decay,
                                                            'lr': self.hparams.learning_rate,
                                                            'eps': self.hparams.adam_epsilon,
                                                            }),
                                                        bias_weight_decay=False,
                                                        normalization_weight_decay=False)

            optimizer = torch.optim.AdamW(parameters, lr=self.hparams.learning_rate, eps=self.hparams.adam_epsilon, weight_decay=self.hparams.weight_decay)
            # Log optimizer info
            for i, g in enumerate(optimizer.param_groups):
                ntensors = len(g['params'])
                nparams = sum(p.numel() for p in g['params'])
                hparams = {k: v for k, v in g.items() if k != 'params'}
                log.info(f'Optimizer group {i}: {ntensors} tensors, {nparams} parameters, {hparams}')


        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=self.hparams.warmup_steps,
            num_training_steps=self.trainer.estimated_stepping_batches,
        )
        scheduler = {"scheduler": scheduler, "interval": "step", "frequency": 1}
        return [optimizer], [scheduler]
