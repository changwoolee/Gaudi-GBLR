#!/bin/bash

export PROJECT_ROOT="./"
export HYDRA_FULL_ERROR=1

# CIFAR-10 / 100
#common_args='model.img_size=32 datamodule.image_size=32 datamodule.batch_size=1024 model.drop_rate=0.0 model.drop_path_rate=0.1'

#python3 run.py experiment=cifar10/mixer/mixers +task_name="cifar10-mixer-s-baseline" $common_args datamodule=cifar10 train.optimizer_param_grouping.bias_weight_decay=False train.optimizer_param_grouping.normalization_weight_decay=False
#python3 run.py experiment=cifar10/mixer/mixers +task_name="cifar100-mixer-s-baseline" $common_args datamodule=cifar100 train.optimizer_param_grouping.bias_weight_decay=False train.optimizer_param_grouping.normalization_weight_decay=False
#python3 run.py experiment=cifar10/vit/vit-s +task_name="cifar10-vit-s-baseline" $common_args datamodule=cifar10 train.optimizer_param_grouping.bias_weight_decay=False train.optimizer_param_grouping.normalization_weight_decay=False
#python3 run.py experiment=cifar10/vit/vit-s +task_name="cifar100-vit-s-baseline" $common_args datamodule=cifar100 train.optimizer_param_grouping.bias_weight_decay=False train.optimizer_param_grouping.normalization_weight_decay=False

# ImageNet
common_args='model.img_size=224 datamodule.image_size=224 datamodule.batch_size=1024 model.drop_rate=0.0 model.drop_path_rate=0.1'
python3 run.py experiment=imagenet/vit/vit-b +task_name="imagenet-vit-b-baseline" $common_args  trainer.devices=8 trainer.num_nodes=1 datamodule.num_workers=4 +trainer.strategy="ddp"  
