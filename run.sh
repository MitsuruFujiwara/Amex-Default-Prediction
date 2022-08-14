#!/bin/sh
cd feats
rm *.feather
cd ../src

python 001_preprocess_no_agg.py
python 101_train_lgbm_no_agg.py

python 002_preprocess_agg.py
python 003_preprocess_agg_last.py
python 004_preprocess_agg_mean.py
python 005_preprocess_agg_max.py
python 006_preprocess_agg_min.py

python 105_train_lgbm_agg_last.py
python 106_train_cb_agg_last.py
python 107_train_xgb_agg_last.py

python 108_train_lgbm_agg_mean.py
python 109_train_cb_agg_mean.py
python 110_train_xgb_agg_mean.py

python 111_train_lgbm_agg_max.py
python 112_train_cb_agg_max.py
python 113_train_xgb_agg_max.py

python 114_train_lgbm_agg_min.py
python 115_train_cb_agg_min.py
python 116_train_xgb_agg_min.py

python 201_ensemble.py

python 102_train_lgbm_agg.py
python 103_train_cb_agg.py
python 104_train_xgb_agg.py

#kaggle competitions submit -c amex-default-prediction -f ../output/submission_lgbm_agg.csv -m "#x cv:xxx"