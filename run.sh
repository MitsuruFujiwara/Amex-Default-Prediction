#!/bin/sh
cd feats
rm *.feather
cd ../src

python 001_preprocess.py

python 101_train_lgbm.py
