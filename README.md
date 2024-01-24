<div align="center">

## Differentiable Learning of Generalized Structured Matrices for Efficient Deep Neural Networks 

 [[Paper📖]](https://openreview.net/forum?id=pAVJKp3Dvn) [[Blog🔥]](http://changwoolee.github.io/blog/2024/gaudi-gblr/) 

by *Changwoo Lee* and *Hun-Seok Kim* from University of Michigan

 </div>

#### An Official Implementation of **Gaudi-GBLR**, (G)eneralized (B)lock (L)ow-(R)ank Matrix with (Gau)ssian-(Di)richlet Function for End-to-End ViT and GPT Compression.

## News

- 2024/01/16 Paper accepted to ICLR 2024!

## Introduction

Deep Neural Networks (DNNs) are getting bigger and bigger. A way to reduce the complexity of the DNNs is to use the *structured* weight matrices, like Low-Rank or Sparse matrices. 

This repo introduces a **Gaudi-GBLR** matrix, which is a (G)eneralized (B)lock (L)ow-(R)ank Matrix with (Gau)ssian-(Di)richlet Function. 

* *Expressiveness.*  **Gaudi-GBLR** includes other structured matrices such as Low-Rank, Block-Low-Rank, and Block-Sparse matrices.
* *Differentiability.* The structure of the **Gaudi-GBLR** is **differentiable**! The efficient structure of the weight matrix is learned from data. So the efficient DNN can be learned from scratch!
* *Layer-wise Optimization.* The structure (and the complexity) is optimized in a layer-wise manner. The less important, the less complexity.

To this end, the efficient weight structures of the ViTs and GPT2 are found in this repo.

## Dependencies

Please refer to `environment.yml` for the full dependency list.

## Requirements


An environment variable `PROJECT_ROOT='./'` must be set beforehand.

The ImageNet dataset is assumed to be prepared in `./data/imagenet`.
Otherwise, User can specify the path by `datamodule.data_path=PATH_TO_DATASET`.

## Model Zoo



## Example Runs

All scripts can be found in `./scripts/`.

1. Baseline Models (with 8 GPUs)
```bash
python3 run.py experiment=imagenet/vit/vit-b trainer.device=8 trainer.num_nodes=1 datamodule.num_workers=4 +trainer.strategy="ddp"
```

2. ImageNet Fine-tuning

Assuming that the ImageNet baseline ViT-Base model obtained in 1. is saved in `./saved_models/imagenet-vit-b/last.ckpt`:
```bash
python3 run.py experiment=imagenet/vit/vit-b-gaudi-finetune callbacks.init_from_pretrained.budget=0.15
```

3. WikiText103 Fine-tuning
```bash
python3 run.py  experiment=wt103/gpt2-ft-gaudi trainer.devices=2 model.layer_type='gaudi' datamodule.batch_size=8 trainer.strategy="ddp" model.model_name="gpt2" model.gaudi_params.no_gaussian=True
```

4. Training From Scratch
```bash
python3 run.py experiment=imagenet/vit/vit-b-gaudi callbacks.shrink.thres=0.04 callbacks.shrink.target_width=0.12 trainer.devices=8 trainer.num_nodes=1 datamodule.num_workers=4 +trainer.strategy="ddp"  model.mlp_cfg.linear1_cfg.width_learning_rate=1e-3
```

## Acknowledgment 

This repo is heavily based on the Monarch and Pixelated Butterfly Repository (https://github.com/HazyResearch/fly). Feel free to check out the amazing research projects from Hazy Research!

## Citation

```
@article{lee2023differentiable,
  title={Differentiable Learning of Generalized Structured Matrices for Efficient Deep Neural Networks},
  author={Lee, Changwoo and Kim, Hun-Seok},
  journal={arXiv preprint arXiv:2310.18882},
  year={2023}
}
```
