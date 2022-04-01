TOTAL_NUM_UPDATES=250000
WARMUP_UPDATES=50000
PEAK_LR=0.00005
TOKENS_PER_SAMPLE=8192
MAX_POSITIONS=8192
BATCH_SIZE=64
MAX_SENTENCES=1
MUSICBERT_PATH=checkpoints/checkpoint_last_musicbert_small.pt
HEAD_NAME=trans_head

CUDA_VISIBLE_DEVICES=0 fairseq-train \
    spm_data_bin/0 \
    --user-dir musicbert \
    --arch musicbert_small \
    --restore-file $MUSICBERT_PATH \
    --max-tokens $((${TOKENS_PER_SAMPLE} * ${MAX_SENTENCES})) \
    --batch-size $MAX_SENTENCES \
    --max-positions $MAX_POSITIONS \
    --reset-optimizer --reset-dataloader --reset-meters \
    --task sentence_prediction_multilabel \
    --num-classes 256 \
    --optimizer adam --adam-betas '(0.9, 0.98)' --clip-norm 0.0 \
    --lr 5e-4 --lr-scheduler inverse_sqrt --warmup-updates 4000 \
    --dropout 0.3 --weight-decay 0.0001 \
    --criterion label_smoothed_cross_entropy --label-smoothing 0.1 \
    --max-tokens 4096 \
    --best-checkpoint-metric bleu --maximize-best-checkpoint-metric

