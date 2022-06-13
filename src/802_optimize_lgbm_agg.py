
import gc
import json
import lightgbm
import numpy as np
import optuna
import pandas as pd
import sys
import warnings

from glob import glob
from sklearn.model_selection import StratifiedKFold
from tqdm import tqdm

from utils import NUM_FOLDS, FEATS_EXCLUDED, line_notify, to_json

#==============================================================================
# hyper parameter optimization by optuna
# https://github.com/pfnet/optuna/blob/master/examples/lightgbm_simple.py
#==============================================================================

warnings.filterwarnings('ignore')

# load datasets
CONFIGS = json.load(open('../configs/102_lgbm_agg.json'))

# load feathers
FILES = sorted(glob('../feats/f002_*.feather'))
DF = pd.concat([pd.read_feather(f) for f in tqdm(FILES)], axis=1)

# split train & test
TRAIN_DF = DF[~DF['is_test']]

del DF
gc.collect()

# use selected features
TRAIN_DF = TRAIN_DF[CONFIGS['features']]

FEATS = [f for f in TRAIN_DF.columns if f not in FEATS_EXCLUDED]

def objective(trial):
    lgbm_train = lightgbm.Dataset(TRAIN_DF[FEATS],
                                  TRAIN_DF['target'],
                                  free_raw_data=False
                                  )

    # ref: https://alphaimpact.jp/downloads/pydata20190927.pdf
    params = {'objective': 'binary',
              'metric': 'binary_logloss',
              'verbosity': -1,
              "learning_rate": 0.05,
#              'device': 'gpu',
              'boosting_type': 'gbdt',
              'max_depth': trial.suggest_int('max_depth', 3, 9),
              'num_leaves': trial.suggest_int('num_leaves', 2, 100),
              'subsample': trial.suggest_uniform('subsample', 0.5, 1.0),
              'subsample_freq': trial.suggest_int('subsample_freq', 1, 20),
              'colsample_bytree': trial.suggest_uniform('colsample_bytree', 0.01, 1.0),
              'min_child_samples': trial.suggest_int('min_child_samples', 1, 50),
              'min_child_weight': trial.suggest_loguniform('min_child_weight', 1e-3, 1e+1),
              'reg_lambda': trial.suggest_loguniform('reg_lambda', 1e-2, 1e+3),
              'reg_alpha': 0.0,
              'min_split_gain': 0.0
              }

    # folds
    folds = StratifiedKFold(n_splits=NUM_FOLDS, shuffle=True, random_state=47)

    # cv
    eval_dict = lightgbm.cv(params=params,
                            train_set=lgbm_train,
                            metrics=['binary_logloss'],
                            nfold=5,
                            folds=folds.split(TRAIN_DF[FEATS],groups=TRAIN_DF['target']),
                            num_boost_round=10000,
                            early_stopping_rounds=200,
                            verbose_eval=100,
                            seed=46,
                            )

    gc.collect()

    return min(eval_dict['binary_logloss-mean'])

if __name__ == '__main__':
    study = optuna.create_study()
    study.optimize(objective, n_trials=30)

    print('Number of finished trials: {}'.format(len(study.trials)))

    print('Best trial:')
    trial = study.best_trial

    print('  Value: {}'.format(trial.value))

    print('  Params: ')
    for key, value in trial.params.items():
        print('    {}: {}'.format(key, value))

    # save result
    hist_df = study.trials_dataframe()
    hist_df.to_csv("../output/optuna_result_lgbm_agg.csv")

    # save json
    CONFIGS['params'] = trial.params
    to_json(CONFIGS, '../configs/102_lgbm_agg.json')

    line_notify('{} finished. Value: {}'.format(sys.argv[0],trial.value))
