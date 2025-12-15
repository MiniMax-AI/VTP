#!/bin/bash

export MODEL_PATH=/path/to/vtp-l-hf
export DATA_PATH=/path/to/imagenet/val
export OUTPUT_PATH=output/reconstruction/vtp-l-hf
export BATCH_SIZE=50
export PRECISION=bf16
export NUM_GPUS=$(nvidia-smi --list-gpus 2>/dev/null | wc -l)

torchrun --nproc_per_node=$NUM_GPUS \
    tools/test_reconstruction_hf.py \
    --model_path $MODEL_PATH \
    --data_path $DATA_PATH \
    --output_path $OUTPUT_PATH \
    --batch_size $BATCH_SIZE \
    --precision $PRECISION \
    --use_ddp
