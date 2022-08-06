
import gc
import numpy as np
import pandas as pd
import sys

from sklearn.linear_model import Ridge

from utils import amex_metric_mod, line_notify

#==============================================================================
# Ensemble
#==============================================================================

sub_path = '../output/submission_ensemble.csv'
oof_path = '../output/oof_ensemble.csv'

def main():
    # load csv
    sub = pd.read_csv('../input/sample_submission.csv')
    oof = pd.read_csv('../input/train_labels.csv')

    sub_lgbm = pd.DataFrame()
    sub_cb = pd.DataFrame()
    sub_xgb = pd.DataFrame()

    oof_lgbm = pd.DataFrame()
    oof_cb = pd.DataFrame()
    oof_xgb = pd.DataFrame()

    sub_lgbm['customer_ID'] = sub['customer_ID']
    sub_cb['customer_ID'] = sub['customer_ID']
    sub_xgb['customer_ID'] = sub['customer_ID']

    oof_lgbm['customer_ID'] = oof['customer_ID']
    oof_cb['customer_ID'] = oof['customer_ID']
    oof_xgb['customer_ID'] = oof['customer_ID']

    sub_lgbm['prediction'] = np.zeros(sub.shape[0])
    sub_cb['prediction'] = np.zeros(sub.shape[0])
    sub_xgb['prediction'] = np.zeros(sub.shape[0])

    oof_lgbm['prediction'] = np.zeros(oof.shape[0])
    oof_cb['prediction'] = np.zeros(oof.shape[0])
    oof_xgb['prediction'] = np.zeros(oof.shape[0])

    for seed in [42, 52, 62]:
        sub_path_lgbm = f'../output/submission_lgbm_agg_{seed}.csv'
        sub_path_cb = f'../output/submission_cb_agg_{seed}.csv'
        sub_path_xgb = f'../output/submission_xgb_agg_{seed}.csv'

        oof_path_lgbm = f'../output/oof_lgbm_agg_{seed}.csv'
        oof_path_cb = f'../output/oof_cb_agg_{seed}.csv'
        oof_path_xgb = f'../output/oof_xgb_agg_{seed}.csv'

        sub_lgbm['prediction'] += pd.read_csv(sub_path_lgbm)['prediction']/3
        sub_cb['prediction'] += pd.read_csv(sub_path_cb)['prediction']/3
        sub_xgb['prediction'] += pd.read_csv(sub_path_xgb)['prediction']/3

        oof_lgbm['prediction'] += pd.read_csv(oof_path_lgbm)['prediction']/3
        oof_cb['prediction'] += pd.read_csv(oof_path_cb)['prediction']/3
        oof_xgb['prediction'] += pd.read_csv(oof_path_xgb)['prediction']/3

    # to rank
    sub_lgbm['prediction'] = sub_lgbm['prediction'].rank() / len(sub_lgbm)
    sub_cb['prediction'] = sub_cb['prediction'].rank() / len(sub_cb)
    sub_xgb['prediction'] = sub_xgb['prediction'].rank() / len(sub_xgb)

    oof_lgbm['prediction'] = oof_lgbm['prediction'].rank() / len(oof_lgbm)
    oof_cb['prediction'] = oof_cb['prediction'].rank() / len(oof_cb)
    oof_xgb['prediction'] = oof_xgb['prediction'].rank() / len(oof_xgb)

    # rename columns
    oof_lgbm.rename(columns={'prediction': 'prediction_lgbm'},inplace=True)
    oof_cb.rename(columns={'prediction': 'prediction_cb'},inplace=True)
    oof_xgb.rename(columns={'prediction': 'prediction_xgb'},inplace=True)

    # merge oof
    oof = oof.merge(oof_lgbm,on=['customer_ID','target'],how='left')
    oof = oof.merge(oof_cb,on=['customer_ID','target'],how='left')
    oof = oof.merge(oof_xgb,on=['customer_ID','target'],how='left')

    del oof_lgbm, oof_cb, oof_xgb
    gc.collect()

    # ridge regression
    reg = Ridge(alpha=1.0,fit_intercept=False,random_state=47)
    reg.fit(oof[['prediction_lgbm','prediction_cb','prediction_xgb']],oof['target'])

    # get weights
    w = reg.coef_ / sum(reg.coef_)
    print('weights: {}'.format(w))

    # calc prediction
    sub['prediction'] += w[0]*sub_lgbm['prediction']+w[1]*sub_cb['prediction']+w[2]*sub_xgb['prediction']
    oof['prediction'] = w[0]*oof['prediction_lgbm']+w[1]*oof['prediction_cb']+w[2]*oof['prediction_xgb']

    # save csv
    sub[['customer_ID','prediction']].to_csv(sub_path, index=False)

    # Full score and LINE Notify
    full_score = round(amex_metric_mod(oof['target'], oof['prediction']),6)

    # LINE notify
    line_notify(f'{sys.argv[0]} done. Full kaggle metric: {full_score}')

if __name__ == '__main__':
    main()