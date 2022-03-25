#!/bin/bash
# 

PREFIX=trans
[[ -d "${PREFIX}_data_bin" ]] && { echo "output directory ${PREFIX}_data_bin already exists" ; exit 1; }
fairseq-preprocess \
    --source-lang txt \
    --target-lang label \
    --trainpref ${PREFIX}_data_raw/0/train \
    --validpref ${PREFIX}_data_raw/0/test \
    --destdir ${PREFIX}_data_bin/tokenized.txt-label \
    --srcdict ${PREFIX}_data_raw/0/dict.txt \
    --workers 24

# fairseq-preprocess \
#     --only-source \
#     --trainpref ${PREFIX}_data_raw/0/train.txt \
#     --validpref ${PREFIX}_data_raw/0/test.txt \
#     --destdir ${PREFIX}_data_bin/0/input0 \
#     --srcdict ${PREFIX}_data_raw/0/dict.txt \
#     --workers 24
# fairseq-preprocess \
#     --only-source \
#     --trainpref ${PREFIX}_data_raw/0/train.label \
#     --validpref ${PREFIX}_data_raw/0/test.label \
#     --destdir ${PREFIX}_data_bin/0/label \
#     --workers 24
# cp ${PREFIX}_data_raw/0/train.label ${PREFIX}_data_bin/0/label/train.label
# cp ${PREFIX}_data_raw/0/test.label ${PREFIX}_data_bin/0/label/valid.label