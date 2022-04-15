#!/bin/bash
# 

# # for translation
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


