#!/bin/bash
rm -rf trans_data_raw/gen
rm -rf trans_data_bin
echo 40 | python3 gen_chord.py
cd trans_data_raw
rm -rf 0
cp -r gen 0
cp split.py 0
cd 0
python3 split.py
cd ../../
bash binarize_trans.sh
