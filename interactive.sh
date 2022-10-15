#! /usr/bin
export CUDA_VISIBLE_DEVICES=0

DATA_SET=data/processed

python interactive.py $DATA_SET \
    --user-dir bart-zh \
    --task translation_hf_bart \
    --path .../checkpoint16.pt \
    --beam 5 \
    --left-pad-source \
    --buffer-size 64 \
    --batch-size 64