
import gc
import numpy as np
import pandas as pd
import sys

from scipy.optimize import minimize
from tqdm import tqdm

from utils import amex_metric_mod, line_notify

#==============================================================================
# Ensemble with kernel output
#==============================================================================

sub_path = '../output/submission_ensemble_kernel.csv'
sub_path_seed_avg = '../output/submission_ensemble_seed_avg.csv'
sub_path_lgbm_seed_avg = '../output/submission_lgbm_agg_seed_avg.csv'
sub_path_hikarutabata = '../output/submission_hikarutabata.csv'

def main():
    # load csv
    sub = pd.read_csv('../input/sample_submission.csv')
    oof = pd.read_csv('../input/train_labels.csv')

    sub_seed_avg = pd.read_csv(sub_path_seed_avg)
    sub_lgbm_seed_avg = pd.read_csv(sub_path_lgbm_seed_avg)
    sub_hikarutabata = pd.read_csv(sub_path_hikarutabata)

    # to rank
    sub_seed_avg['prediction'] = sub_seed_avg['prediction'].rank() / len(sub_seed_avg)
    sub_lgbm_seed_avg['prediction'] = sub_lgbm_seed_avg['prediction'].rank() / len(sub_lgbm_seed_avg)
    sub_hikarutabata['prediction'] = sub_hikarutabata['prediction'].rank() / len(sub_hikarutabata)

    # calc prediction
    sub['prediction'] += 0.4*sub_seed_avg['prediction']+0.6*sub_hikarutabata['prediction']

    # to rank
    sub['prediction'] = sub['prediction'].rank() / len(sub)

    # save csv
    sub[['customer_ID','prediction']].to_csv(sub_path, index=False)

    # LINE notify
    line_notify(f'{sys.argv[0]} done.')

if __name__ == '__main__':
    main()