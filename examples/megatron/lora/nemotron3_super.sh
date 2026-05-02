#!/usr/bin/env bash
set -euo pipefail

# Experimental Nemotron 3 Super LoRA path.
#
# This expects the HF checkpoint to have already been converted to Megatron
# torch_dist format with NVIDIA Megatron-Bridge.
#
# The target end-state is a merged final artifact (`--merge_lora true`,
# `--save_safetensors true`).  Run a tiny smoke save before launching a full
# training job; if Nemotron-H safetensors export fails, temporarily fall back to
# MCore adapter checkpoints plus a post-training `megatron export --to_mcore
# --merge_lora true` merge step.

MODEL_ID=${MODEL_ID:-nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-BF16}
MCORE_MODEL=${MCORE_MODEL:-/path/to/NVIDIA-Nemotron-3-Super-120B-A12B-BF16-mcore}
DATASET=${DATASET:-AI-ModelScope/alpaca-gpt4-data-en#1000}

NPROC_PER_NODE=${NPROC_PER_NODE:-8} \
megatron sft \
    --model "${MODEL_ID}" \
    --model_type nemotron_h \
    --use_hf true \
    --mcore_model "${MCORE_MODEL}" \
    --dataset "${DATASET}" \
    --template default \
    --tuner_type lora \
    --target_modules linear_qkv linear_proj linear_fc1 linear_fc2 in_proj out_proj \
    --lora_rank 8 \
    --lora_alpha 16 \
    --save_safetensors true \
    --merge_lora true \
    --torch_dtype bfloat16 \
    --max_length 2048 \
    --micro_batch_size 1 \
    --global_batch_size 16 \
    --tensor_model_parallel_size 1 \
    --pipeline_model_parallel_size 1 \
    --expert_model_parallel_size 1 \
    --expert_tensor_parallel_size 1 \
    --sequence_parallel true \
    --attention_backend fused \
    --mtp_loss_scaling_factor 0.3 \
    --train_iters 50 \
    --eval_iters 10 \
    --save_interval 50 \
    --eval_interval 50 \
    --output_dir megatron_output/NVIDIA-Nemotron-3-Super-120B-A12B-BF16-lora
