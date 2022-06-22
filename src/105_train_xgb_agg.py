
import gc
import json
import numpy as np
import pandas as pd
import sys
import warnings
import xgboost as xgb

from glob import glob
from sklearn.model_selection import StratifiedKFold
from tqdm import tqdm

from utils import save_imp, amex_metric_mod, line_notify
from utils import NUM_FOLDS, FEATS_EXCLUDED

#==============================================================================
# Train XGBoost
#==============================================================================

warnings.filterwarnings('ignore')

configs = json.load(open('../configs/105_xgb_agg.json'))

feats_path = '../feats/f002_*.feather'

sub_path = '../output/submission_xgb_agg.csv'
oof_path = '../output/oof_xgb_agg.csv'

model_path = '../models/xgb_agg_'

imp_path_png = '../imp/xgb_importances_agg.png'
imp_path_csv = '../imp/feature_importance_xgb_agg.csv'

params = { 
          'max_depth':5,
          'learning_rate':0.01, 
          'subsample':0.8,
          'colsample_bytree':0.6, 
          'eval_metric':'logloss',
          'objective':'binary:logistic',
          'random_state':47
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
    folds = StratifiedKFold(n_splits=NUM_FOLDS,shuffle=True,random_state=46)

    # Create arrays and dataframes to store results
    oof_preds = np.zeros(train_df.shape[0])
    sub_preds = np.zeros(test_df.shape[0])

    # for importance
    imp_df = pd.DataFrame()

    # features to use
    feats = [f for f in train_df.columns if f not in FEATS_EXCLUDED]

    # dmatrix for test_df
    test_df_dmtrx = xgb.DMatrix(test_df[feats])

    # k-fold
    for n_fold, (train_idx, valid_idx) in enumerate(folds.split(train_df[feats],train_df['target'])):
        train_x, train_y = train_df[feats].iloc[train_idx], train_df['target'].iloc[train_idx]
        valid_x, valid_y = train_df[feats].iloc[valid_idx], train_df['target'].iloc[valid_idx]

        # set data structure
        xgb_train = xgb.DMatrix(train_x,label=train_y)

        xgb_test = xgb.DMatrix(valid_x,label=valid_y)

        # train
        clf = xgb.train(
                        params,
                        xgb_train,
                        num_boost_round=10000,
                        evals=[(xgb_train,'train'),(xgb_test,'test')],
                        early_stopping_rounds= 200,
                        verbose_eval=100
                        )

        # save model
        clf.save_model(f'{model_path}{n_fold}.txt')

        # save predictions
        oof_preds[valid_idx] = clf.predict(valid_x)
        sub_preds += clf.predict(test_df_dmtrx) / folds.n_splits

        # save importances
        fold_importance_df = pd.DataFrame.from_dict(clf.get_score(importance_type='gain'), orient='index', columns=['importance'])
        fold_importance_df["feature"] = fold_importance_df.index.tolist()
        fold_importance_df["fold"] = n_fold + 1
        feature_importance_df = pd.concat([feature_importance_df, fold_importance_df], axis=0)

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
