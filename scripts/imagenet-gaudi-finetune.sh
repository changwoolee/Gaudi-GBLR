#!/bin/bash

export PROJECT_ROOT='./'
export HYDRA_FULL_ERROR=1

for budget in 0.125 0.15 0.175 0.2
do
	python3 run.py experiment=imagenet/vit/vit-b-gaudi-finetune +task_name="imagenet-vit-s-gaudi-finetune" callbacks.init_from_pretrained.budget=$budget   
done

