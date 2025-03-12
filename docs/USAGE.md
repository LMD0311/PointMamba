# Preparation

## Environment

This codebase was tested with the following environment configurations. It may work with other versions.
- Ubuntu 20.04
- CUDA 11.7
- Python 3.9
- PyTorch 1.13.1 + cu117

## Installation

We recommend using Anaconda for the installation process:
```shell
# Clone the repository
$ git clone https://github.com/LMD0311/PointMamba.git
$ cd PointMamba

# Create virtual env and install PyTorch
$ conda create -n pointmamba python=3.9
$ conda activate pointmamba
(pointmamba) $ pip install torch==1.13.1+cu117 torchvision==0.14.1+cu117 torchaudio==0.13.1 --extra-index-url https://download.pytorch.org/whl/cu117

# Install basic required packages
(pointmamba) $ pip install -r requirements.txt

# Chamfer Distance & emd
(pointmamba) $ cd ./extensions/chamfer_dist && python setup.py install --user
(pointmamba) $ cd ./extensions/emd && python setup.py install --user

# PointNet++
(pointmamba) $ pip install "git+https://github.com/erikwijmans/Pointnet2_PyTorch.git#egg=pointnet2_ops&subdirectory=pointnet2_ops_lib"

# GPU kNN
(pointmamba) $ pip install --upgrade https://github.com/unlimblue/KNN_CUDA/releases/download/0.2/KNN_CUDA-0.2-py3-none-any.whl

# Mamba
(pointmamba) $ pip install causal-conv1d>=1.2.0
(pointmamba) $ cd mamba & pip install .
```
You can double-check the [environment](./environment.yml) file for the same requirements.

# Training

## Pre-train

```shell
CUDA_VISIBLE_DEVICES=<GPU> python main.py --config cfgs/pretrain.yaml --exp_name <name>
```
## Classification on ModelNet40

```shell
CUDA_VISIBLE_DEVICES=<GPU> python main.py --finetune_model --config cfgs/finetune_modelnet.yaml --ckpts <path/to/pre-trained/model> --exp_name <name>
```
## Classification on ScanObjectNN

```shell
CUDA_VISIBLE_DEVICES=<GPU> python main.py --finetune_model --config cfgs/finetune_scan_objbg.yaml --ckpts <path/to/pre-trained/model> --exp_name <name>
```