#! /bin/bash

export CUDA_VISIBLE_DEVICES=0

DATA_DIR=data/processed
BART_DIR=/.../bart-large-chinese
MODEL_NAME=bart-baseline
SEED=10
mkdir -p checkpoints/$MODEL_NAME

python train.py $DATA_DIR \
    --fp16 \
    --seed 10 \
    --user-dir bart-zh \
    --arch hf_bart \
    --task translation_hf_bart \
    --hf-model-name $BART_DIR \
    --optimizer adam \
    --clip-norm 0.1 \
    --lr 1e-5 \
    --lr-scheduler inverse_sqrt \
    --warmup-updates 4000 \
    --criterion label_smoothed_cross_entropy \
    --label-smoothing 0.1 \
    --update-freq 4 \
    --max-tokens 4096 \
    --max-epoch 30 \
    --num-workers 10 \
    --save-dir checkpoints/seed$SEED \
    --log-format simple \
    --log-interval 50 \
    --left-pad-source \
    --maximize-best-checkpoint-metric \
    2>&1 | tee -a checkpoints/seed$SEED/training.log
