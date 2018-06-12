#!/bin/sh

CUDA_VISIBLE_DEVICES=0 python nmt.py \
    --cuda \
    --mode train \
    --vocab "path/to/train/vocab" \
    --save_to "path/to/saved/models" \
    --train_src "path/to/train/article" \
    --train_tgt "path/to/train/title" \
    --dev_src "path/to/dev/article" \
    --dev_tgt "path/to/dev/title" \
    --test_src "path/to/test/article" \
    --test_tgt "path/to/test/title" \
    --train_ent_x "path/to/train/entailment/premise" \
    --train_ent_y "path/to/train/entailment/hypothesis" \
    --train_ent_label "path/to/train/entailment/label" \
    2>&1 | tee -a train.mtl.log
