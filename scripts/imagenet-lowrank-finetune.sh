#!/bin/bash

export PROJECT_ROOT='./'
export HYDRA_FULL_ERROR=1
export IMAGENET_DIR=./data/ILSVRC/Data/CLS-LOC


epoch=35
for budget in 0.125 0.15 0.175 0.2
do
	python3 -u run.py experiment=imagenet/vit/vit-b-lowrank +task_name="imagenet-vit-s-lowrank-finetune" datamodule.data_dir=$IMAGENET_DIR  model.drop_path_rate=0.1 train.scheduler.t_initial=$epoch callbacks.init_from_pretrained.budget=$budget 
done


