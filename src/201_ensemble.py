
import gc
import numpy as np
import pandas as pd
import sys

from utils import amex_metric_mod, line_notify

#==============================================================================
# Ensemble
#==============================================================================

sub_path = '../output/submission_ensemble.csv'
sub_path_lgbm = '../output/submission_lgbm_agg.csv'
sub_path_cb = '../output/submission_cb_agg.csv'
sub_path_xgb = '../output/submission_xgb_agg.csv'
sub_path_thedevastator = '../output/submission_thedevastator.csv'
sub_path_zb1373 = '../output/submission_zb1373.csv'

oof_path = '../output/oof_ensemble.csv'
oof_path_lgbm = '../output/oof_lgbm_agg.csv'
oof_path_cb = '../output/oof_cb_agg.csv'
oof_path_xgb = '../output/oof_xgb_agg.csv'

def main():
    # load csv
    oof = pd.read_csv('../input/train_labels.csv')
    sub = pd.read_csv('../input/sample_submission.csv')

    sub_lgbm = pd.read_csv(sub_path_lgbm)
    sub_cb = pd.read_csv(sub_path_cb)
    sub_xgb = pd.read_csv(sub_path_xgb)
    sub_thedevastator = pd.read_csv(sub_path_thedevastator)
    sub_zb1373 = pd.read_csv(sub_path_zb1373)

    oof_lgbm = pd.read_csv(oof_path_lgbm)
    oof_cb = pd.read_csv(oof_path_cb)
    oof_xgb = pd.read_csv(oof_path_xgb)

    # to rank
    sub_lgbm['prediction'] = sub_lgbm['prediction'].rank() / len(sub_lgbm)
    sub_cb['prediction'] = sub_cb['prediction'].rank() / len(sub_cb)
    sub_xgb['prediction'] = sub_xgb['prediction'].rank() / len(sub_xgb)
    sub_thedevastator['prediction'] = sub_thedevastator['prediction'].rank() / len(sub_thedevastator)
    sub_zb1373['prediction'] = sub_zb1373['prediction'].rank() / len(sub_zb1373)

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

    # check correlation
    print(np.corrcoef([sub_lgbm['prediction'],sub_cb['prediction'],sub_xgb['prediction']]))

    # weights
    w = [0.5,0.1,0.4]
    print('weights: {}'.format(w))

    # calc prediction
    sub['prediction'] += w[0]*sub_lgbm['prediction']+w[1]*sub_cb['prediction']+w[2]*sub_xgb['prediction']
    oof['prediction'] = w[0]*oof['prediction_lgbm']+w[1]*oof['prediction_cb']+w[2]*oof['prediction_xgb']

    # add thedevastator
    sub['prediction'] *= 0.4
    sub['prediction'] += 0.6 * sub_zb1373['prediction']

    # save csv
    sub[['customer_ID','prediction']].to_csv(sub_path, index=False)

    # Full score and LINE Notify
    full_score = round(amex_metric_mod(oof['target'], oof['prediction']),6)

    # LINE notify
    line_notify(f'{sys.argv[0]} done. Full kaggle metric: {full_score}')

if __name__ == '__main__':
    main()