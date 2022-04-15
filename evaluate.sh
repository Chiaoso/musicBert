fairseq-generate trans_data_bin/tokenized.txt-label \
    --path checkpoints/checkpoint_best.pt \
    --batch-size 128 --beam 5 --remove-bpe