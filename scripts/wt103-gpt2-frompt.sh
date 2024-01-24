#!/bin/bash

export HYDRA_FULL_ERROR=1


python3 run.py  experiment=wt103/gpt2-ft-gaudi +task_name="gpt2-dense" trainer.devices=2 model.layer_type='gaudi' train.optimizer.lr=1.5e-3 datamodule.batch_size=8 trainer.strategy="ddp" model.model_name="gpt2" model.gaudi_params.no_gaussian=True
