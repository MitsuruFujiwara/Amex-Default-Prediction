
import gc
import numpy as np
import pandas as pd
import sys

from scipy.optimize import minimize
from tqdm import tqdm

from utils import amex_metric_mod, line_notify

#==============================================================================
# Ensemble by aggregation types
#==============================================================================

sub_path = '../output/submission_ensemble_by_agg.csv'

sub_path_lgbm = '../output/submission_lgbm_agg.csv'
sub_path_cb = '../output/submission_cb_agg.csv'
sub_path_xgb = '../output/submission_xgb_agg.csv'

sub_path_lgbm_last = '../output/submission_lgbm_agg_last.csv'
sub_path_cb_last = '../output/submission_cb_agg_last.csv'
sub_path_xgb_last = '../output/submission_xgb_agg_last.csv'

sub_path_lgbm_mean = '../output/submission_lgbm_agg_mean.csv'
sub_path_cb_mean = '../output/submission_cb_agg_mean.csv'
sub_path_xgb_mean = '../output/submission_xgb_agg_mean.csv'

sub_path_lgbm_max = '../output/submission_lgbm_agg_max.csv'
sub_path_cb_max = '../output/submission_cb_agg_max.csv'
sub_path_xgb_max = '../output/submission_xgb_agg_max.csv'

sub_path_lgbm_min = '../output/submission_lgbm_agg_min.csv'
sub_path_cb_min = '../output/submission_cb_agg_min.csv'
sub_path_xgb_min = '../output/submission_xgb_agg_min.csv'

oof_path = '../output/oof_ensemble.csv'

oof_path_lgbm = '../output/oof_lgbm_agg.csv'
oof_path_cb = '../output/oof_cb_agg.csv'
oof_path_xgb = '../output/oof_xgb_agg.csv'

oof_path_lgbm_last = '../output/oof_lgbm_agg_last.csv'
oof_path_cb_last = '../output/oof_cb_agg_last.csv'
oof_path_xgb_last = '../output/oof_xgb_agg_last.csv'

oof_path_lgbm_mean = '../output/oof_lgbm_agg_mean.csv'
oof_path_cb_mean = '../output/oof_cb_agg_mean.csv'
oof_path_xgb_mean = '../output/oof_xgb_agg_mean.csv'

oof_path_lgbm_max = '../output/oof_lgbm_agg_max.csv'
oof_path_cb_max = '../output/oof_cb_agg_max.csv'
oof_path_xgb_max = '../output/oof_xgb_agg_max.csv'

oof_path_lgbm_min = '../output/oof_lgbm_agg_min.csv'
oof_path_cb_min = '../output/oof_cb_agg_min.csv'
oof_path_xgb_min = '../output/oof_xgb_agg_min.csv'

def main():
    # load csv
    print('load csv...')
    oof = pd.read_csv('../input/train_labels.csv')
    sub = pd.read_csv('../input/sample_submission.csv')

    sub_lgbm = pd.read_csv(sub_path_lgbm)
    sub_cb = pd.read_csv(sub_path_cb)
    sub_xgb = pd.read_csv(sub_path_xgb)

    sub_lgbm_last = pd.read_csv(sub_path_lgbm_last)
    sub_cb_last = pd.read_csv(sub_path_cb_last)
    sub_xgb_last = pd.read_csv(sub_path_xgb_last)

    sub_lgbm_mean = pd.read_csv(sub_path_lgbm_mean)
    sub_cb_mean = pd.read_csv(sub_path_cb_mean)
    sub_xgb_mean = pd.read_csv(sub_path_xgb_mean)

    sub_lgbm_max = pd.read_csv(sub_path_lgbm_max)
    sub_cb_max = pd.read_csv(sub_path_cb_max)
    sub_xgb_max = pd.read_csv(sub_path_xgb_max)

    sub_lgbm_min = pd.read_csv(sub_path_lgbm_min)
    sub_cb_min = pd.read_csv(sub_path_cb_min)
    sub_xgb_min = pd.read_csv(sub_path_xgb_min)

    oof_lgbm = pd.read_csv(oof_path_lgbm)
    oof_cb = pd.read_csv(oof_path_cb)
    oof_xgb = pd.read_csv(oof_path_xgb)

    oof_lgbm_last = pd.read_csv(oof_path_lgbm_last)
    oof_cb_last = pd.read_csv(oof_path_cb_last)
    oof_xgb_last = pd.read_csv(oof_path_xgb_last)

    oof_lgbm_mean = pd.read_csv(oof_path_lgbm_mean)
    oof_cb_mean = pd.read_csv(oof_path_cb_mean)
    oof_xgb_mean = pd.read_csv(oof_path_xgb_mean)

    oof_lgbm_max = pd.read_csv(oof_path_lgbm_max)
    oof_cb_max = pd.read_csv(oof_path_cb_max)
    oof_xgb_max = pd.read_csv(oof_path_xgb_max)

    oof_lgbm_min = pd.read_csv(oof_path_lgbm_min)
    oof_cb_min = pd.read_csv(oof_path_cb_min)
    oof_xgb_min = pd.read_csv(oof_path_xgb_min)

    # to rank
    print('to rank...')
    sub_lgbm['prediction'] = sub_lgbm['prediction'].rank() / len(sub_lgbm)
    sub_cb['prediction'] = sub_cb['prediction'].rank() / len(sub_cb)
    sub_xgb['prediction'] = sub_xgb['prediction'].rank() / len(sub_xgb)

    sub_lgbm_last['prediction'] = sub_lgbm_last['prediction'].rank() / len(sub_lgbm_last)
    sub_cb_last['prediction'] = sub_cb_last['prediction'].rank() / len(sub_cb_last)
    sub_xgb_last['prediction'] = sub_xgb_last['prediction'].rank() / len(sub_xgb_last)

    sub_lgbm_mean['prediction'] = sub_lgbm_mean['prediction'].rank() / len(sub_lgbm_mean)
    sub_cb_mean['prediction'] = sub_cb_mean['prediction'].rank() / len(sub_cb_mean)
    sub_xgb_mean['prediction'] = sub_xgb_mean['prediction'].rank() / len(sub_xgb_mean)

    sub_lgbm_max['prediction'] = sub_lgbm_max['prediction'].rank() / len(sub_lgbm_max)
    sub_cb_max['prediction'] = sub_cb_max['prediction'].rank() / len(sub_cb_max)
    sub_xgb_max['prediction'] = sub_xgb_max['prediction'].rank() / len(sub_xgb_max)

    sub_lgbm_min['prediction'] = sub_lgbm_min['prediction'].rank() / len(sub_lgbm_min)
    sub_cb_min['prediction'] = sub_cb_min['prediction'].rank() / len(sub_cb_min)
    sub_xgb_min['prediction'] = sub_xgb_min['prediction'].rank() / len(sub_xgb_min)

    oof_lgbm['prediction'] = oof_lgbm['prediction'].rank() / len(oof_lgbm)
    oof_cb['prediction'] = oof_cb['prediction'].rank() / len(oof_cb)
    oof_xgb['prediction'] = oof_xgb['prediction'].rank() / len(oof_xgb)

    oof_lgbm_last['prediction'] = oof_lgbm_last['prediction'].rank() / len(oof_lgbm_last)
    oof_cb_last['prediction'] = oof_cb_last['prediction'].rank() / len(oof_cb_last)
    oof_xgb_last['prediction'] = oof_xgb_last['prediction'].rank() / len(oof_xgb_last)

    oof_lgbm_mean['prediction'] = oof_lgbm_mean['prediction'].rank() / len(oof_lgbm_mean)
    oof_cb_mean['prediction'] = oof_cb_mean['prediction'].rank() / len(oof_cb_mean)
    oof_xgb_mean['prediction'] = oof_xgb_mean['prediction'].rank() / len(oof_xgb_mean)

    oof_lgbm_max['prediction'] = oof_lgbm_max['prediction'].rank() / len(oof_lgbm_max)
    oof_cb_max['prediction'] = oof_cb_max['prediction'].rank() / len(oof_cb_max)
    oof_xgb_max['prediction'] = oof_xgb_max['prediction'].rank() / len(oof_xgb_max)

    oof_lgbm_min['prediction'] = oof_lgbm_min['prediction'].rank() / len(oof_lgbm_min)
    oof_cb_min['prediction'] = oof_cb_min['prediction'].rank() / len(oof_cb_min)
    oof_xgb_min['prediction'] = oof_xgb_min['prediction'].rank() / len(oof_xgb_min)

    # rename columns
    oof_lgbm.rename(columns={'prediction': 'prediction_lgbm'},inplace=True)
    oof_cb.rename(columns={'prediction': 'prediction_cb'},inplace=True)
    oof_xgb.rename(columns={'prediction': 'prediction_xgb'},inplace=True)

    oof_lgbm_last.rename(columns={'prediction': 'prediction_lgbm_last'},inplace=True)
    oof_cb_last.rename(columns={'prediction': 'prediction_cb_last'},inplace=True)
    oof_xgb_last.rename(columns={'prediction': 'prediction_xgb_last'},inplace=True)

    oof_lgbm_mean.rename(columns={'prediction': 'prediction_lgbm_mean'},inplace=True)
    oof_cb_mean.rename(columns={'prediction': 'prediction_cb_mean'},inplace=True)
    oof_xgb_mean.rename(columns={'prediction': 'prediction_xgb_mean'},inplace=True)

    oof_lgbm_max.rename(columns={'prediction': 'prediction_lgbm_max'},inplace=True)
    oof_cb_max.rename(columns={'prediction': 'prediction_cb_max'},inplace=True)
    oof_xgb_max.rename(columns={'prediction': 'prediction_xgb_max'},inplace=True)

    oof_lgbm_min.rename(columns={'prediction': 'prediction_lgbm_min'},inplace=True)
    oof_cb_min.rename(columns={'prediction': 'prediction_cb_min'},inplace=True)
    oof_xgb_min.rename(columns={'prediction': 'prediction_xgb_min'},inplace=True)

    # merge oof
    oof = oof.merge(oof_lgbm,on=['customer_ID','target'],how='left')
    oof = oof.merge(oof_cb,on=['customer_ID','target'],how='left')
    oof = oof.merge(oof_xgb,on=['customer_ID','target'],how='left')

    oof = oof.merge(oof_lgbm_last,on=['customer_ID','target'],how='left')
    oof = oof.merge(oof_cb_last,on=['customer_ID','target'],how='left')
    oof = oof.merge(oof_xgb_last,on=['customer_ID','target'],how='left')

    oof = oof.merge(oof_lgbm_mean,on=['customer_ID','target'],how='left')
    oof = oof.merge(oof_cb_mean,on=['customer_ID','target'],how='left')
    oof = oof.merge(oof_xgb_mean,on=['customer_ID','target'],how='left')

    oof = oof.merge(oof_lgbm_max,on=['customer_ID','target'],how='left')
    oof = oof.merge(oof_cb_max,on=['customer_ID','target'],how='left')
    oof = oof.merge(oof_xgb_max,on=['customer_ID','target'],how='left')

    oof = oof.merge(oof_lgbm_min,on=['customer_ID','target'],how='left')
    oof = oof.merge(oof_cb_min,on=['customer_ID','target'],how='left')
    oof = oof.merge(oof_xgb_min,on=['customer_ID','target'],how='left')

    del oof_lgbm, oof_cb, oof_xgb
    del oof_lgbm_last, oof_cb_last, oof_xgb_last
    del oof_lgbm_mean, oof_cb_mean, oof_xgb_mean
    del oof_lgbm_max, oof_cb_max, oof_xgb_max
    del oof_lgbm_min, oof_cb_min, oof_xgb_min
    gc.collect()

    # ridge regression
    cols_pred = ['prediction_lgbm','prediction_cb','prediction_xgb',
                 'prediction_cb_last',
                 'prediction_cb_mean','prediction_xgb_mean',
                 'prediction_xgb_max',
                 ]

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
    preds = [sub_lgbm, sub_cb, sub_xgb,
             sub_cb_last,
             sub_cb_mean, sub_xgb_mean,
             sub_xgb_max,
             ]

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