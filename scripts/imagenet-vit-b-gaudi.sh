#!/bin/bash

export PROJECT_ROOT='./'
export HYDRA_FULL_ERROR=1


target=0.12
thres=0.04
rpc=1
lr=1e-3
python3 run.py experiment=imagenet/vit/vit-b-gaudi +task_name="imagenet-vit-b-gaudi" callbacks.shrink.thres=$thres callbacks.shrink.target_width=$target trainer.devices=8 trainer.num_nodes=1 datamodule.num_workers=4 +trainer.strategy="ddp"  model.mlp_cfg.linear1_cfg.width_learning_rate=$lr 
