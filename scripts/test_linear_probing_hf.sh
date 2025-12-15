#!/bin/bash

export MODEL_PATH=/path/to/vtp-l-hf
export DATA_PATH=/path/to/imagenet/val
export OUTPUT_DIR=output/linear_probing_results
export BATCH_SIZE=128
export EPOCHS=10
export PRECISION=bf16
export NUM_GPUS=$(nvidia-smi --list-gpus 2>/dev/null | wc -l)

# Multi-GPU with DDP
torchrun --nproc_per_node=$NUM_GPUS \
    tools/test_linear_probing_hf.py \
    --model_path $MODEL_PATH \
    --imagenet_root $IMAGENET_ROOT \
    --output_dir $OUTPUT_DIR \
    --batch_size $BATCH_SIZE \
    --epochs $EPOCHS \
    --precision $PRECISION \
    --use_ddp
