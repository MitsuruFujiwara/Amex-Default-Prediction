
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
from xgboost.callback import EarlyStopping

from utils import save_imp, amex_metric_mod, xgb_amex, line_notify
from utils import NUM_FOLDS, FEATS_EXCLUDED

#==============================================================================
# Train XGBoost
#==============================================================================

warnings.filterwarnings('ignore')

configs = json.load(open('../configs/102_lgbm_agg.json'))

feats_path = '../feats/f002_*.feather'

sub_path = '../output/submission_xgb_agg.csv'
oof_path = '../output/oof_xgb_agg.csv'

model_path = '../models/xgb_agg_'

imp_path_png = '../imp/xgb_importances_agg.png'
imp_path_csv = '../imp/feature_importance_xgb_agg.csv'

params = { 
          'max_depth':7,
          'learning_rate':0.03,
          'subsample':0.88,
          'colsample_bytree':0.5,
          'gamma': 1.5,
          'min_child_weight': 8,
          'lambda':70,
          'eval_metric':'logloss',
          'objective':'binary:logistic',
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

    # dmatrix for test_df
    test_df_dmtrx = xgb.DMatrix(test_df[feats])

    # k-fold
    for n_fold, (train_idx, valid_idx) in enumerate(folds.split(train_df[feats],train_df['target'])):
        train_x, train_y = train_df[feats].iloc[train_idx], train_df['target'].iloc[train_idx]
        valid_x, valid_y = train_df[feats].iloc[valid_idx], train_df['target'].iloc[valid_idx]

        # set data structure
        xgb_train = xgb.DMatrix(train_x,label=train_y)

        xgb_test = xgb.DMatrix(valid_x,label=valid_y)

        # change seed by fold
        params['random_state'] = 42*(n_fold+1)

        # specify early stopping
        early_stop = EarlyStopping(rounds=1500,
                                   metric_name='amex',
                                   data_name='test',
                                   maximize=True)

        # train
        clf = xgb.train(
                        params,
                        xgb_train,
                        num_boost_round=20500,
                        evals=[(xgb_train,'train'),(xgb_test,'test')],
                        feval=xgb_amex,
                        maximize=True,
                        callbacks=[early_stop],
                        verbose_eval=100
                        )

        # save model
        clf.save_model(f'{model_path}{n_fold}.txt')

        # save predictions
        oof_preds[valid_idx] = clf.predict(xgb_test)
        sub_preds += clf.predict(test_df_dmtrx) / folds.n_splits

        # save importances
        fold_importance_df = pd.DataFrame.from_dict(clf.get_score(importance_type='gain'), orient='index', columns=['importance'])
        fold_importance_df["feature"] = fold_importance_df.index.tolist()
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
