TOTAL_NUM_UPDATES=250000
WARMUP_UPDATES=50000
PEAK_LR=0.00005
TOKENS_PER_SAMPLE=8192
MAX_POSITIONS=8192
BATCH_SIZE=1
MAX_SENTENCES=1
MUSICBERT_PATH=checkpoints/checkpoint_last_musicbert_small.pt
HEAD_NAME=trans_head

CUDA_VISIBLE_DEVICES=0 fairseq-train \
    trans_data_bin/tokenized.txt-label \
    --arch transformer_iwslt_de_en --share-decoder-input-output-embed \
    --encoder_embed_path embed_tokens.txt\
    --encoder-embed-dim 768 \
    --optimizer adam --adam-betas '(0.9, 0.98)' --clip-norm 0.0 \
    --lr 5e-4 --lr-scheduler inverse_sqrt --warmup-updates 4000 \
    --dropout 0.3 --weight-decay 0.0001 \
    --criterion label_smoothed_cross_entropy --label-smoothing 0.1 \
    --max-tokens 4096 \
    --eval-bleu \
    --eval-bleu-args '{"beam": 5, "max_len_a": 1.2, "max_len_b": 10}' \
    --eval-bleu-detok moses \
    --eval-bleu-remove-bpe \
    --eval-bleu-print-samples \
    --best-checkpoint-metric bleu --maximize-best-checkpoint-metric

