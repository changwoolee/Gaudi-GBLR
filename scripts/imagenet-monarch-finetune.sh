#!/bin/bash
export PROJECT_ROOT='./'
export HYDRA_FULL_ERROR=1
export IMAGENET_DIR=./data/ILSVRC/Data/CLS-LOC

epoch=35
for budget in 4 6 8
do
	python3 -u run.py experiment=imagenet/vit/vit-b-monarch +task_name="imagenet-vit-s-monarch-finetune" datamodule.data_dir=$IMAGENET_DIR  model.drop_path_rate=0.1 train.scheduler.t_initial=$epoch model.mlp_cfg.linear1_cfg.nblocks=$budget 
done


