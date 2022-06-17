
import gc
import json
import lightgbm as lgb
import numpy as np
import pandas as pd
import sys
import warnings

from glob import glob
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from tqdm import tqdm

from utils import save_imp, amex_metric_mod, line_notify
from utils import NUM_FOLDS, FEATS_EXCLUDED

#==============================================================================
# Adversarial validation
#==============================================================================

warnings.filterwarnings('ignore')

configs = json.load(open('../configs/103_lgbm_agg_simple.json'))

feats_path = '../feats/f003_*.feather'

imp_path_png = '../imp/lgbm_importances_adversarial_validation.png'
imp_path_csv = '../imp/feature_importance_adversarial_validation.csv'

params = configs['params']

#params['device'] = 'gpu'
params['task'] = 'train'
params['boosting'] = 'gbdt'
params['objective'] = 'binary'
params['metric'] = 'auc'
params['learning_rate'] = 0.01
params['reg_alpha'] = 0.0
params['min_split_gain'] = 0.0
params['verbose'] = -1
#params['num_threads'] = -1
params['seed'] = 47
params['bagging_seed'] = 47
params['drop_seed'] = 47

def main():
    # load feathers
    files = sorted(glob(feats_path))
    df = pd.concat([pd.read_feather(f) for f in tqdm(files)], axis=1)

    # drop unused features
    df = df[configs['features']]

    # set target
    df['is_test'] = df['is_test'].astype(int)

    # Cross validation
    folds = StratifiedKFold(n_splits=NUM_FOLDS,shuffle=True,random_state=46)

    # Create arrays and dataframes to store results
    oof_preds = np.zeros(df.shape[0])

    # for importance
    imp_df = pd.DataFrame()

    # features to use
    feats = [f for f in df.columns if f not in FEATS_EXCLUDED]

    # k-fold
    for n_fold, (train_idx, valid_idx) in enumerate(folds.split(df[feats],df['is_test'])):
        train_x, train_y = df[feats].iloc[train_idx], df['is_test'].iloc[train_idx]
        valid_x, valid_y = df[feats].iloc[valid_idx], df['is_test'].iloc[valid_idx]

        # set data structure
        lgb_train = lgb.Dataset(train_x,
                                label=train_y,
                                free_raw_data=False)

        lgb_test = lgb.Dataset(valid_x,
                               label=valid_y,
                               free_raw_data=False)

        # train
        clf = lgb.train(
                        params,
                        lgb_train,
                        valid_sets=[lgb_train, lgb_test],
                        valid_names=['train', 'test'],
                        num_boost_round=10000,
                        early_stopping_rounds= 200,
                        verbose_eval=100
                        )

        # save predictions
        oof_preds[valid_idx] = clf.predict(valid_x,num_iteration=clf.best_iteration)

        # save importances
        fold_importance_df = pd.DataFrame()
        fold_importance_df['feature'] = feats
        fold_importance_df['importance'] = np.log1p(clf.feature_importance(importance_type='gain', iteration=clf.best_iteration))
        imp_df = pd.concat([imp_df, fold_importance_df], axis=0)

        # calc fold score
        fold_score = roc_auc_score(valid_y, oof_preds[valid_idx])

        print(f'Fold {n_fold+1} AUC: {fold_score}')

        del clf, train_x, train_y, valid_x, valid_y
        gc.collect()

    # Full score and LINE Notify
    full_score = round(roc_auc_score(df['is_test'], oof_preds),6)
    line_notify(f'Full AUC: {full_score}')

    # save importance
    save_imp(imp_df,imp_path_png,imp_path_csv)

    # LINE notify
    line_notify(f'{sys.argv[0]} done.')

if __name__ == '__main__':
    main()
