
import gc
import json
import numpy as np
import pandas as pd
import sys

from utils import reduce_mem_usage, to_json, to_feature, line_notify

#==============================================================================
# preprocess
#==============================================================================

IS_DEBUG = False

data_types = json.load(open('../configs/002_data_types.json'))

def main():

    if IS_DEBUG:
        nrows=100000
    else:
        nrows=None

    # load csv
    train_df = pd.read_csv('../input/train_data.csv',nrows=nrows,dtype=data_types)
    train_labels = pd.read_csv('../input/train_labels.csv',nrows=nrows)
    test_df = pd.read_csv('../input/test_data.csv',nrows=nrows,dtype=data_types)
    sub = pd.read_csv('../input/sample_submission.csv')

    # to datetime
    train_df['S_2'] = pd.to_datetime(train_df['S_2'])
    test_df['S_2'] = pd.to_datetime(test_df['S_2'])

    # datetime features
    train_df['day'] = train_df['S_2'].dt.day
    train_df['month'] = train_df['S_2'].dt.month
    train_df['year'] = train_df['S_2'].dt.year
    train_df['seasonality'] = np.cos(np.pi*(train_df['S_2'].dt.dayofyear/366*2-1))

    test_df['day'] = test_df['S_2'].dt.day
    test_df['month'] = test_df['S_2'].dt.month
    test_df['year'] = test_df['S_2'].dt.year
    test_df['seasonality'] = np.cos(np.pi*(test_df['S_2'].dt.dayofyear/366*2-1))

    # reduce memory usage
    train_df = reduce_mem_usage(train_df)
    test_df = reduce_mem_usage(test_df)
    
    # grouped df
    train_df_grouped = train_df.groupby('customer_ID')
    test_df_grouped = test_df.groupby('customer_ID')

    # add features mean
    train_df_mean = train_df_grouped.mean().reset_index()
    test_df_mean = test_df_grouped.mean().reset_index()

    train_df_mean.columns = ['customer_ID']+[f'{c}_mean' for c in train_df_mean.columns if c not in ['customer_ID']]
    test_df_mean.columns = ['customer_ID']+[f'{c}_mean' for c in test_df_mean.columns if c not in ['customer_ID']]

    # merge mean
    train_df = train_labels.merge(train_df_mean,how='left',on='customer_ID')
    test_df = sub.merge(test_df_mean,how='left',on='customer_ID')

    del train_df_mean, test_df_mean
    gc.collect()

    # add features max
    train_df_max = train_df_grouped.max().reset_index()
    test_df_max = test_df_grouped.max().reset_index()

    train_df_max.columns = ['customer_ID']+[f'{c}_max' for c in train_df_max.columns if c not in ['customer_ID']]
    test_df_max.columns = ['customer_ID']+[f'{c}_max' for c in test_df_max.columns if c not in ['customer_ID']]

    # merge max
    train_df = train_df.merge(train_df_max,how='left',on='customer_ID')
    test_df = test_df.merge(test_df_max,how='left',on='customer_ID')

    del train_df_max, test_df_max
    gc.collect()

    # add features min
    train_df_min = train_df_grouped.min().reset_index()
    test_df_min = test_df_grouped.min().reset_index()

    train_df_min.columns = ['customer_ID']+[f'{c}_min' for c in train_df_min.columns if c not in ['customer_ID']]
    test_df_min.columns = ['customer_ID']+[f'{c}_min' for c in test_df_min.columns if c not in ['customer_ID']]

    # merge max
    train_df = train_df.merge(train_df_min,how='left',on='customer_ID')
    test_df = test_df.merge(test_df_min,how='left',on='customer_ID')

    del train_df_min, test_df_min
    gc.collect()

    # add features last
    train_df_last = train_df_grouped.nth(-1).reset_index()
    test_df_last = test_df_grouped.nth(-1).reset_index()

    train_df_last.columns = ['customer_ID']+[f'{c}_last' for c in train_df_last.columns if c not in ['customer_ID']]
    test_df_last.columns = ['customer_ID']+[f'{c}_last' for c in test_df_last.columns if c not in ['customer_ID']]

    # merge max
    train_df = train_df.merge(train_df_last,how='left',on='customer_ID')
    test_df = test_df.merge(test_df_last,how='left',on='customer_ID')

    del train_df_last, test_df_last
    gc.collect()

    del train_labels, sub, train_df_grouped, test_df_grouped
    gc.collect()    

    # drop prediction columns
    test_df.drop('prediction',axis=1,inplace=True)

    # add is_test flag
    train_df['is_test'] = False
    test_df['is_test'] = True

    # merge train & test
    df = train_df.append(test_df)

    del train_df, test_df
    gc.collect()

    # save as feather
    to_feature(df, '../feats/f001')

    # save feature name list
    features_json = {'features':df.columns.tolist()}
    to_json(features_json,'../configs/001_all_features.json')

    # LINE notify
    line_notify('{} done.'.format(sys.argv[0]))

if __name__ == '__main__':
    main()