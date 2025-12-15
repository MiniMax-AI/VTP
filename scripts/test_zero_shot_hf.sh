#!/bin/bash

export MODEL_PATH=/path/to/vtp-l-hf
export DATA_PATH=/path/to/imagenet/val
export BATCH_SIZE=50
export PRECISION=bf16
export DEVICE=cuda:0

python tools/test_zero_shot_hf.py \
    --model_path $MODEL_PATH \
    --data_path $DATA_PATH \
    --batch_size $BATCH_SIZE \
    --precision $PRECISION \
    --device $DEVICE
