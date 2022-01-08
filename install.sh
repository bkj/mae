#!/bin/bash

# run.sh

# --
# Setup environment

conda create -y -n mae_env python=3.8
conda activate mae_env

conda install -y -c pytorch \
  pytorch torchvision cudatoolkit=11.3 

conda install -y jupyter


pip install timm==0.3.2

pip install git+https://github.com/bkj/rcode

pip install tensorboard
pip install requests
pip install matplotlib
pip install setuptools==59.5.0

# --
# Get models

wget -nc https://dl.fbaipublicfiles.com/mae/visualize/mae_visualize_vit_large.pth