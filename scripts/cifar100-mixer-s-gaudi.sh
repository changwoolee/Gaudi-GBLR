#!/bin/bash

export PROJECT_ROOT='./'
export HYDRA_FULL_ERROR=1

common_args='model.img_size=32 datamodule.image_size=32 datamodule.batch_size=1024'

thres=0.2
for target in 0.1 0.2 0.3 0.4
do
	python3 -u run.py experiment=cifar10/mixer/mixers-gaudi +task_name="cifar100-mixer-s-gaudi" $common_args datamodule=cifar100 callbacks.shrink.thres=$thres callbacks.shrink.target_width=$target 
done


