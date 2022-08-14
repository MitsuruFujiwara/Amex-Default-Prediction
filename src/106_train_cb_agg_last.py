
import catboost as cb
import gc
import json
import numpy as np
import pandas as pd
import sys
import warnings

from glob import glob
from sklearn.model_selection import StratifiedKFold
from tqdm import tqdm

from utils import save_imp, amex_metric_mod, line_notify
from utils import AmexCatboostMetric, NUM_FOLDS, FEATS_EXCLUDED

#==============================================================================
# Train CatBoost last only
#==============================================================================

warnings.filterwarnings('ignore')

configs = json.load(open('../configs/103_lgbm_agg_last.json'))

feats_path = '../feats/f003_*.feather'

sub_path = '../output/submission_cb_agg_last.csv'
oof_path = '../output/oof_cb_agg_last.csv'

model_path = '../models/cb_agg_last_'

imp_path_png = '../imp/cb_importances_agg_last.png'
imp_path_csv = '../imp/feature_importance_cb_agg_last.csv'

params ={
        'max_depth': 7,    
        'od_type': 'Iter',        
        'l2_leaf_reg': 70,    
        'random_seed': 42,                    
        'loss_function': 'Logloss',
        'eval_metric': AmexCatboostMetric(),
        'learning_rate': 0.03,
        'early_stopping_rounds':1500,
        'verbose_eval':100,
        'train_dir':'../output/catboost_info',
        }

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
    folds = StratifiedKFold(n_splits=NUM_FOLDS,shuffle=True,random_state=42)

    # Create arrays and dataframes to store results
    oof_preds = np.zeros(train_df.shape[0])
    sub_preds = np.zeros(test_df.shape[0])

    # for importance
    imp_df = pd.DataFrame()

    # features to use
    feats = [f for f in train_df.columns if f not in FEATS_EXCLUDED]

    # k-fold
    for n_fold, (train_idx, valid_idx) in enumerate(folds.split(train_df[feats],train_df['target'])):
        train_x, train_y = train_df[feats].iloc[train_idx], train_df['target'].iloc[train_idx]
        valid_x, valid_y = train_df[feats].iloc[valid_idx], train_df['target'].iloc[valid_idx]

        # set data structure
        cb_train = cb.Pool(train_x,label=train_y)

        cb_test = cb.Pool(valid_x,label=valid_y)

        # change seed by folds
        params['random_seed'] = 42*(n_fold+1)

        # train
        clf = cb.train(
                       cb_train,
                       params,
                       num_boost_round=20500,
                       eval_set=cb_test
                       )

        # save model
        clf.save_model(f'{model_path}{n_fold}.txt')

        # save predictions
        oof_preds[valid_idx] = clf.predict(valid_x)
        sub_preds += clf.predict(test_df[feats]) / folds.n_splits

        # save importances
        fold_importance_df = pd.DataFrame()
        fold_importance_df['feature'] = feats
        fold_importance_df['importance'] = np.log1p(clf.feature_importances_)
        imp_df = pd.concat([imp_df, fold_importance_df], axis=0)

        # calc fold score
        fold_score = amex_metric_mod(valid_y,oof_preds[valid_idx])

        print(f'Fold {n_fold+1} kaggle metric: {fold_score}')

        del clf, train_x, train_y, valid_x, valid_y
        gc.collect()

    # Full score and LINE Notify
    full_score = round(amex_metric_mod(train_df['target'], oof_preds),6)
    line_notify(f'Full kaggle metric: {full_score}')

    # save importance
    save_imp(imp_df,imp_path_png,imp_path_csv)

    # save prediction
    train_df.loc[:,'prediction'] = oof_preds
    test_df.loc[:,'prediction'] = sub_preds

    # save csv
    train_df[['customer_ID','target','prediction']].to_csv(oof_path, index=False)
    test_df[['customer_ID','prediction']].to_csv(sub_path, index=False)

    # LINE notify
    line_notify(f'{sys.argv[0]} done.')

if __name__ == '__main__':
    main()
