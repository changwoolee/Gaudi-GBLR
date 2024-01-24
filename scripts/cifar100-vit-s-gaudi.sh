#!/bin/bash

export PROJECT_ROOT='./'
export HYDRA_FULL_ERROR=1

common_args='model.img_size=32 datamodule.image_size=32  '

sleep 5


target=0.05
for thres in 0.02 0.03 0.04 0.05
do
	python3 -u run.py experiment=cifar10/vit/vit-s-gaudi +task_name="cifar100-vit-s-gaudi-initalg" $common_args datamodule=cifar100  callbacks.shrink.thres=$thres callbacks.shrink.target_width=$target 
done


