#!/bin/bash
set -e
source ./venv/bin/activate
env CUDA_VISIBLE_DEVICES=1,2,3 torchrun --standalone --nnodes=1 --nproc_per_node=3 --log_dir=output train_distributed.py --config configs/beta_vae_AID.yaml
env CUDA_VISIBLE_DEVICES=1,2,3 torchrun --standalone --nnodes=1 --nproc_per_node=3 --log_dir=output train_distributed.py --config configs/beta_vae_audio.yaml
#env CUDA_VISIBLE_DEVICES=1,2,3 torchrun --standalone --nnodes=1 --nproc_per_node=3 --log_dir=output train_distributed.py --config configs/beta_vae.yaml
#env CUDA_VISIBLE_DEVICES=1,2,3 torchrun --standalone --nnodes=1 --nproc_per_node=3 --log_dir=output train_distributed.py --config configs/vq_vae.yaml
#env CUDA_VISIBLE_DEVICES=1,2,3 torchrun --standalone --nnodes=1 --nproc_per_node=3 --log_dir=output train_distributed.py --config configs/vae.yaml
