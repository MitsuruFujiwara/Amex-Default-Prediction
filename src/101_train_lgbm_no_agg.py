
import gc
import json
import lightgbm as lgb
import numpy as np
import pandas as pd
import sys
import warnings

from glob import glob
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import log_loss
from tqdm import tqdm

from utils import save_imp, line_notify
from utils import NUM_FOLDS, FEATS_EXCLUDED

#==============================================================================
# Train LightGBM
#==============================================================================

warnings.filterwarnings('ignore')

configs = json.load(open('../configs/101_lgbm_no_agg.json'))

feats_path = '../feats/f001_*.feather'

sub_path = '../output/submission_lgbm_no_agg.csv'
oof_path = '../output/oof_lgbm_no_agg.csv'

model_path = '../models/lgbm_no_agg_'

imp_path_png = '../imp/lgbm_importances_no_agg.png'
imp_path_csv = '../imp/feature_importance_lgbm_no_agg.csv'

params = configs['params']

#params['device'] = 'gpu'
params['task'] = 'train'
params['boosting'] = 'gbdt'
params['objective'] = 'binary'
params['metric'] = 'binary_logloss'
params['learning_rate'] = 0.05
params['reg_alpha'] = 0.0
params['min_split_gain'] = 0.0
params['verbose'] = -1
#params['num_threads'] = -1
params['seed'] = 42
params['bagging_seed'] = 42
params['drop_seed'] = 42

def main():
    # load feathers
    files = sorted(glob(feats_path))
    df = pd.concat([pd.read_feather(f) for f in tqdm(files)], axis=1)

    # drop unused features
    df = df[configs['features']]

    # split train & test
    train_df = df[~df['is_test']]
    test_df = df[df['is_test']]

    del df
    gc.collect()

    # Cross validation
    folds = StratifiedGroupKFold(n_splits=NUM_FOLDS,shuffle=True,random_state=42)

    # Create arrays and dataframes to store results
    oof_preds = np.zeros(train_df.shape[0])
    sub_preds = np.zeros(test_df.shape[0])

    # for importance
    imp_df = pd.DataFrame()

    # features to use
    feats = [f for f in train_df.columns if f not in FEATS_EXCLUDED]

    # k-fold
    for n_fold, (train_idx, valid_idx) in enumerate(folds.split(train_df[feats],train_df['target'],groups=train_df['customer_ID'])):
        train_x, train_y = train_df[feats].iloc[train_idx], train_df['target'].iloc[train_idx]
        valid_x, valid_y = train_df[feats].iloc[valid_idx], train_df['target'].iloc[valid_idx]

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
        sub_preds += clf.predict(test_df[feats],num_iteration=clf.best_iteration) / folds.n_splits

        # save importances
        fold_importance_df = pd.DataFrame()
        fold_importance_df['feature'] = feats
        fold_importance_df['importance'] = np.log1p(clf.feature_importance(importance_type='gain', iteration=clf.best_iteration))
        imp_df = pd.concat([imp_df, fold_importance_df], axis=0)

        # calc fold score
        fold_score = log_loss(valid_y,oof_preds[valid_idx])

        print(f'Fold {n_fold+1} logloss: {fold_score}')

        del clf, train_x, train_y, valid_x, valid_y
        gc.collect()

    # Full score and LINE Notify
    full_score = round(log_loss(train_df['target'], oof_preds),6)
    line_notify(f'Full logloss: {full_score}')

    # save importance
    save_imp(imp_df,imp_path_png,imp_path_csv)

    # save prediction
    train_df.loc[:,'pred_no_agg'] = oof_preds
    test_df.loc[:,'pred_no_agg'] = sub_preds

    # save csv
    train_df[['customer_ID','pred_no_agg']].to_csv(oof_path, index=False)
    test_df[['customer_ID','pred_no_agg']].to_csv(sub_path, index=False)

    # LINE notify
    line_notify(f'{sys.argv[0]} done.')

if __name__ == '__main__':
    main()
