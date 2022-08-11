
import gc
import numpy as np
import pandas as pd
import sys

from scipy.optimize import minimize
from tqdm import tqdm

from utils import amex_metric_mod, line_notify

#==============================================================================
# Ensemble by seed
#==============================================================================

sub_path = '../output/submission_ensemble_seed_avg.csv'
oof_path = '../output/oof_ensemble_seed_avg.csv'

sub_path_lgbm_seed_avg = '../output/submission_lgbm_agg_seed_avg.csv'
sub_path_cb_seed_avg = '../output/submission_cb_agg_seed_avg.csv'
sub_path_xgb_seed_avg = '../output/submission_xgb_agg_seed_avg.csv'

oof_path_lgbm_seed_avg = '../output/oof_lgbm_agg_seed_avg.csv'
oof_path_cb_seed_avg = '../output/oof_cb_agg_seed_avg.csv'
oof_path_xgb_seed_avg = '../output/oof_xgb_agg_seed_avg.csv'

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

    oof_lgbm['target'] = oof['target']
    oof_cb['target'] = oof['target']
    oof_xgb['target'] = oof['target']

    sub_lgbm['prediction'] = np.zeros(sub.shape[0])
    sub_cb['prediction'] = np.zeros(sub.shape[0])
    sub_xgb['prediction'] = np.zeros(sub.shape[0])

    oof_lgbm['prediction'] = np.zeros(oof.shape[0])
    oof_cb['prediction'] = np.zeros(oof.shape[0])
    oof_xgb['prediction'] = np.zeros(oof.shape[0])

    print('seed averaging...')
    for seed in tqdm([42, 52, 62]):
        # load csv
        sub_path_lgbm = f'../output/submission_lgbm_agg_{seed}.csv'
        sub_path_cb = f'../output/submission_cb_agg_{seed}.csv'
        sub_path_xgb = f'../output/submission_xgb_agg_{seed}.csv'

        oof_path_lgbm = f'../output/oof_lgbm_agg_{seed}.csv'
        oof_path_cb = f'../output/oof_cb_agg_{seed}.csv'
        oof_path_xgb = f'../output/oof_xgb_agg_{seed}.csv'

        tmp_sub_lgbm = pd.read_csv(sub_path_lgbm)
        tmp_sub_cb = pd.read_csv(sub_path_cb)
        tmp_sub_xgb = pd.read_csv(sub_path_xgb)

        tmp_oof_lgbm = pd.read_csv(oof_path_lgbm)
        tmp_oof_cb = pd.read_csv(oof_path_cb)
        tmp_oof_xgb = pd.read_csv(oof_path_xgb)

        # to rank
        tmp_sub_lgbm['prediction'] = tmp_sub_lgbm['prediction'].rank() / len(tmp_sub_lgbm)
        tmp_sub_cb['prediction'] = tmp_sub_cb['prediction'].rank() / len(tmp_sub_cb)
        tmp_sub_xgb['prediction'] = tmp_sub_xgb['prediction'].rank() / len(tmp_sub_xgb)

        tmp_oof_lgbm['prediction'] = tmp_oof_lgbm['prediction'].rank() / len(tmp_oof_lgbm)
        tmp_oof_cb['prediction'] = tmp_oof_cb['prediction'].rank() / len(tmp_oof_cb)
        tmp_oof_xgb['prediction'] = tmp_oof_xgb['prediction'].rank() / len(tmp_oof_xgb)
        
        # average predictions
        sub_lgbm['prediction'] += tmp_sub_lgbm['prediction'] / 3
        sub_cb['prediction'] += tmp_sub_cb['prediction'] / 3
        sub_xgb['prediction'] += tmp_sub_xgb['prediction'] / 3

        oof_lgbm['prediction'] += tmp_oof_lgbm['prediction'] / 3
        oof_cb['prediction'] += tmp_oof_cb['prediction'] / 3
        oof_xgb['prediction'] += tmp_oof_xgb['prediction'] / 3

    # to rank
    sub_lgbm['prediction'] = sub_lgbm['prediction'].rank() / len(sub_lgbm)
    sub_cb['prediction'] = sub_cb['prediction'].rank() / len(sub_cb)
    sub_xgb['prediction'] = sub_xgb['prediction'].rank() / len(sub_xgb)

    oof_lgbm['prediction'] = oof_lgbm['prediction'].rank() / len(oof_lgbm)
    oof_cb['prediction'] = oof_cb['prediction'].rank() / len(oof_cb)
    oof_xgb['prediction'] = oof_xgb['prediction'].rank() / len(oof_xgb)

    # calc full score
    full_score_lgbm = round(amex_metric_mod(oof_lgbm['target'], oof_lgbm['prediction']),6)
    full_score_cb = round(amex_metric_mod(oof_cb['target'], oof_cb['prediction']),6)
    full_score_xgb = round(amex_metric_mod(oof_xgb['target'], oof_xgb['prediction']),6)

    # LINE notify
    line_notify(f'Full kaggle metric lgbm: {full_score_lgbm}')
    line_notify(f'Full kaggle metric cb: {full_score_cb}')
    line_notify(f'Full kaggle metric xgb: {full_score_xgb}')

    # save csv
    sub_lgbm[['customer_ID','prediction']].to_csv(sub_path_lgbm_seed_avg, index=False)
    sub_cb[['customer_ID','prediction']].to_csv(sub_path_cb_seed_avg, index=False)
    sub_xgb[['customer_ID','prediction']].to_csv(sub_path_xgb_seed_avg, index=False)

    oof_lgbm[['customer_ID','prediction']].to_csv(oof_path_lgbm_seed_avg, index=False)
    oof_cb[['customer_ID','prediction']].to_csv(oof_path_cb_seed_avg, index=False)
    oof_xgb[['customer_ID','prediction']].to_csv(oof_path_xgb_seed_avg, index=False)

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

    # cols to use
    cols_pred = ['prediction_lgbm','prediction_cb','prediction_xgb']

    # objective function for scipy optimize
    def obj_func(weights):
        ''' scipy minimize will pass the weights as a numpy array '''
        final_prediction = 0
        for weight, c in zip(weights, cols_pred):
                final_prediction += weight*oof[c]

        return -amex_metric_mod(oof['target'], final_prediction)    

    # Optimization runs 100 times.
    lls = []
    wghts = []
    print('Optimization runs 100 times...')
    for i in tqdm(range(100)):
        starting_values = np.random.uniform(size=len(cols_pred))
        # cons are given as constraints.
        cons = ({'type':'eq','fun':lambda w: 1-sum(w)})
        bounds = [(0,1)]*len(cols_pred)
        
        res = minimize(obj_func, 
                       starting_values, 
                       constraints=cons,
                       bounds = bounds, 
                       method='SLSQP')

        lls.append(res['fun'])
        wghts.append(res['x'])

    # get weights
    bestSC = np.min(lls)
    w = wghts[np.argmin(lls)]
    print('\n Ensemble Score: {best_score:.7f}'.format(best_score=bestSC))
    print('weights: {}'.format(w))

    # calc prediction
    preds = [sub_lgbm, sub_cb, sub_xgb]

    oof['prediction'] = 0.0
    for i, (p, c) in enumerate(zip(preds,cols_pred)):
        sub['prediction'] += w[i]*p['prediction']
        oof['prediction'] += w[i]*oof[c]

    # save csv
    sub[['customer_ID','prediction']].to_csv(sub_path, index=False)

    # Full score and LINE Notify
    full_score = round(amex_metric_mod(oof['target'], oof['prediction']),6)

    # LINE notify
    line_notify(f'{sys.argv[0]} done. Full kaggle metric: {full_score}')

if __name__ == '__main__':
    main()