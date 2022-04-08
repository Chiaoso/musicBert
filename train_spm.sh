#!/bin/bash
# 

TOTAL_NUM_UPDATES=250000
WARMUP_UPDATES=50000
PEAK_LR=0.00005
TOKENS_PER_SAMPLE=8192
MAX_POSITIONS=8192
BATCH_SIZE=64
MAX_SENTENCES=4
MUSICBERT_PATH=checkpoints/checkpoint_last_musicbert_small.pt
HEAD_NAME=spm_head


# for spm
CUDA_VISIBLE_DEVICES=0 fairseq-train spm_data_bin/0 --user-dir musicbert \
    --restore-file $MUSICBERT_PATH \
    --max-update $TOTAL_NUM_UPDATES \
    --batch-size $MAX_SENTENCES \
    --max-positions $MAX_POSITIONS \
    --max-tokens $((${TOKENS_PER_SAMPLE} * ${MAX_SENTENCES})) \
    --task sentence_prediction_multilabel \
    --reset-optimizer --reset-dataloader --reset-meters \
    --required-batch-size-multiple 1 \
    --num-workers 0 \
    --init-token 0 --separator-token 2 \
    --arch musicbert_small \
    --criterion sentence_prediction_multilabel \
    --classification-head-name $HEAD_NAME \
    --num-classes 256 \
    --dropout 0.1 --attention-dropout 0.1 --weight-decay 0.01 \
    --optimizer adam --adam-betas "(0.9, 0.98)" --adam-eps 1e-6 --clip-norm 0.0 \
    --lr-scheduler polynomial_decay --lr $PEAK_LR --total-num-update $TOTAL_NUM_UPDATES --warmup-updates $WARMUP_UPDATES \
    --log-format simple --log-interval 100 \
    --best-checkpoint-metric f1_score_micro --maximize-best-checkpoint-metric \
    --shorten-method "truncate" \
    --checkpoint-suffix _spm.pt \
    --no-epoch-checkpoints \
    --find-unused-parameters