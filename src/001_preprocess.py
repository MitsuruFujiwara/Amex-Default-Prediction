
import gc
import json
import numpy as np
import pandas as pd
import sys

from utils import reduce_mem_usage, to_json, to_feature, line_notify
from utils import CAT_COLS

#==============================================================================
# preprocess
#==============================================================================

IS_DEBUG = False

data_types = json.load(open('../configs/002_data_types.json'))

# get features
def get_features(df):

    # numeric columns
    NUM_COLS = [c for c in df.columns if c not in CAT_COLS+['customer_ID','S_2']]

    # aggregate
    df_num_agg = df.groupby("customer_ID")[NUM_COLS].agg(['mean', 'std', 'min', 'max', 'last'])
    df_cat_agg = df.groupby("customer_ID")[CAT_COLS].agg(['count', 'last', 'nunique'])

    # change column names
    df_num_agg.columns = ['_'.join(x) for x in df_num_agg.columns]
    df_cat_agg.columns = ['_'.join(x) for x in df_cat_agg.columns]

    # concat
    df = pd.concat([df_num_agg, df_cat_agg], axis=1)

    del df_num_agg, df_cat_agg
    gc.collect()

    return df

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

    # get features
    train_df = get_features(train_df)
    test_df = get_features(test_df)

    # add is test flag
    train_df['is_test'] = False
    test_df['is_test'] = True

    # add target
    train_df = train_labels.merge(train_df,how='left',on='customer_ID')
    test_df = sub.merge(test_df,how='left',on='customer_ID')

    # merge train & test
    df = pd.concat([train_df,test_df])

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