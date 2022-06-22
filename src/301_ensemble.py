
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

oof_path = '../output/oof_ensemble.csv'
oof_path_lgbm = '../output/oof_lgbm_agg.csv'
oof_path_cb = '../output/oof_cb_agg.csv'

def main():
    # load csv
    oof = pd.read_csv('../input/train_labels.csv')
    sub = pd.read_csv('../input/sample_submission.csv')

    sub_lgbm = pd.read_csv(sub_path_lgbm)
    sub_cb = pd.read_csv(sub_path_cb)

    oof_lgbm = pd.read_csv(oof_path_lgbm)
    oof_cb = pd.read_csv(oof_path_cb)

    #TODO: logistic regression

    # calc prediction
    sub['prediction'] += 0.5*sub_lgbm['prediction']+0.5*sub_cb['prediction']
    oof['prediction'] = 0.5*oof_lgbm['prediction']+0.5*oof_cb['prediction']

    # save csv
    sub[['customer_ID','prediction']].to_csv(sub_path, index=False)

    # Full score and LINE Notify
    full_score = round(amex_metric_mod(oof['target'], oof['prediction']),6)

    # LINE notify
    line_notify(f'{sys.argv[0]} done. Full kaggle metric: {full_score}')

if __name__ == '__main__':
    main()