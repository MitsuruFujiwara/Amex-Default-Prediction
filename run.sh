#!/bin/sh
cd feats
rm *.feather
cd ../src

python 001_preprocess_no_agg.py
python 101_train_lgbm_no_agg.py

python 002_preprocess_agg.py
python 102_train_lgbm_agg.py
python 103_train_cb_agg.py
python 104_train_xgb_agg.py
python 201_ensemble.py

#kaggle competitions submit -c amex-default-prediction -f ../output/submission_lgbm_agg.csv -m "#x cv:xxx"