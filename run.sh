#!/bin/sh
cd feats
rm *.feather
cd ../src

python 001_preprocess_no_agg.py
python 101_train_lgbm_no_agg.py

python 002_preprocess_agg.py
python 102_train_lgbm_agg.py
