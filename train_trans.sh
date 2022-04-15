#!/bin/bash
# 

TOTAL_NUM_UPDATES=250000
WARMUP_UPDATES=50000
PEAK_LR=0.00005
TOKENS_PER_SAMPLE=8192
MAX_POSITIONS=8192
MAX_SENTENCES=4
MUSICBERT_PATH=checkpoints/checkpoint_last_musicbert_small.pt
HEAD_NAME=trans_head

# for translation
CUDA_VISIBLE_DEVICES=0 fairseq-train trans_data_bin/tokenized.txt-label --user-dir musicbert \
    --max-update $TOTAL_NUM_UPDATES \
    --batch-size $MAX_SENTENCES \
    --max-positions $MAX_POSITIONS \
    --max-tokens $((${TOKENS_PER_SAMPLE} * ${MAX_SENTENCES})) \
    --reset-optimizer --reset-dataloader --reset-meters \
    --criterion label_smoothed_cross_entropy \
    --arch musicbert_small \
    --label-smoothing 0.1 \
    --dropout 0.1 --attention-dropout 0.1 --weight-decay 0.1 \
    --optimizer adam --adam-betas "(0.9, 0.98)" --adam-eps 1e-6 --clip-norm 0.0 \
    --lr-scheduler polynomial_decay --lr $PEAK_LR --total-num-update $TOTAL_NUM_UPDATES --warmup-updates $WARMUP_UPDATES \
    --log-format simple --log-interval 100 \
    --best-checkpoint-metric accuracy --maximize-best-checkpoint-metric \
    --no-epoch-checkpoints \
    --disable-validation \
    --find-unused-parameters \