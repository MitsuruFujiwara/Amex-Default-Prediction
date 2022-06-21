
import gc
import json
import numpy as np
import pandas as pd
import sys

from utils import reduce_mem_usage, to_json, to_feature, line_notify
from utils import CAT_COLS

#==============================================================================
# preprocess aggregation
#==============================================================================

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

    # load csv
    train_df = pd.read_parquet('../input/train.parquet')
    train_labels = pd.read_csv('../input/train_labels.csv')
    test_df = pd.read_parquet('../input/test.parquet')
    test_labels = pd.read_csv('../input/sample_submission.csv')

    # to datetime
    train_df['S_2'] = pd.to_datetime(train_df['S_2'])
    test_df['S_2'] = pd.to_datetime(test_df['S_2'])

    # datetime features
    train_df['day'] = train_df['S_2'].dt.day
    train_df['seasonality'] = np.cos(np.pi*(train_df['S_2'].dt.day/31*2-1))

    test_df['day'] = test_df['S_2'].dt.day
    test_df['seasonality'] = np.cos(np.pi*(test_df['S_2'].dt.day/31*2-1))

    # drop unnecessary columns
    train_df.drop('S_2',axis=1,inplace=True)
    test_df.drop('S_2',axis=1,inplace=True)

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
    test_df = test_labels.merge(test_df,how='left',on='customer_ID')

    # merge train & test
    df = pd.concat([train_df,test_df])

    del train_df, test_df
    gc.collect()

    # save as feather
    to_feature(df, '../feats/f003')

    # save feature name list
    features_json = {'features':df.columns.tolist()}
    to_json(features_json,'../configs/003_all_features_agg_simple.json')

    # LINE notify
    line_notify('{} done.'.format(sys.argv[0]))

if __name__ == '__main__':
    main()