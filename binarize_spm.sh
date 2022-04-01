#!/bin/bash
# 

# for sentence_prediction_multilable 
PREFIX=spm 
[[ -d "${PREFIX}_data_bin" ]] && { echo "output directory ${PREFIX}_data_bin already exists" ; exit 1; }
fairseq-preprocess \
    --only-source \
    --trainpref trans_data_raw/0/train.txt \
    --validpref trans_data_raw/0/test.txt \
    --destdir ${PREFIX}_data_bin/0/input0 \
    --srcdict trans_data_raw/0/dict.txt \
    --workers 24
fairseq-preprocess \
    --only-source \
    --trainpref trans_data_raw/0/train.label \
    --validpref TRANS_data_raw/0/test.label \
    --destdir ${PREFIX}_data_bin/0/label \
    --workers 24
cp trans_data_raw/0/train.label ${PREFIX}_data_bin/0/label/train.label
cp trans_data_raw/0/test.label ${PREFIX}_data_bin/0/label/valid.label