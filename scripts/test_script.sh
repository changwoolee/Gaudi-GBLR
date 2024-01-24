#!/bin/bash


python3 -u run.py  experiment=imagenet/vit/vit-b +task_name="None" logger="csv" datamodule.data_dir=$IMAGENET_DIR trainer.devices=1 datamodule.num_workers=4 +mode_type="eval" +eval.run_test=False +eval.checkpoint_type="timm"  +eval.eval_runtime=True
python3 -u run.py  experiment=imagenet/vit/vit-b-gaudi +task_name="None" logger="csv" datamodule.data_dir=$IMAGENET_DIR trainer.devices=1 datamodule.num_workers=4 +mode_type="eval" +eval.ckpt="./saved_models/imagenet-gaudi/last.ckpt" +eval.run_test=False  +eval.eval_runtime=True
