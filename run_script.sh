#!/bin/bash
set -e
source ./venv/bin/activate
#env CUDA_VISIBLE_DEVICES=3 python -u train.py --config configs/beta_vae.yaml
#env CUDA_VISIBLE_DEVICES=3 python -u train.py --config configs/vq_vae.yaml
env CUDA_VISIBLE_DEVICES=3 python -u train.py --config configs/vae.yaml
