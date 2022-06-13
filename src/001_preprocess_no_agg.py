from cgi import test
import gc
import json
import numpy as np
import pandas as pd
import sys

from utils import reduce_mem_usage, to_json, to_feature, line_notify
from utils import CAT_COLS

#==============================================================================
# preprocess no aggregation
#==============================================================================

def main():

    # load csv
    train_df = pd.read_parquet('../input/train.parquet')
    train_labels = pd.read_csv('../input/train_labels.csv')
    test_df = pd.read_parquet('../input/test.parquet')
    sub = pd.read_csv('../input/sample_submission.csv')

    # merge target
    train_df = train_df.merge(train_labels,how='left',on='customer_ID')
    test_df = test_df.merge(sub,how='left',on='customer_ID')

    del train_labels, sub
    gc.collect()

    # drop prediction
    test_df.drop('prediction',axis=1,inplace=True)

    # reduce memory usage
    train_df = reduce_mem_usage(train_df)
    test_df = reduce_mem_usage(test_df)

    # add is test flag
    train_df['is_test'] = False
    test_df['is_test'] = True

    # merge train & test
    df = pd.concat([train_df,test_df])

    del train_df, test_df
    gc.collect()

    # to datetime
    df['S_2'] = pd.to_datetime(df['S_2'])

    # datetime features
    df['day'] = df['S_2'].dt.day
    df['month'] = df['S_2'].dt.month
    df['year'] = df['S_2'].dt.year
    df['seasonality'] = np.cos(np.pi*(df['S_2'].dt.dayofyear/366*2-1))

    # target encoding
    for c in CAT_COLS:
        df[f'{c}_te'] = df[c].map(df[[c,'target']].groupby(c).mean()['target'])

    # save as feather
    to_feature(df, '../feats/f001')

    # save feature name list
    features_json = {'features':df.columns.tolist()}
    to_json(features_json,'../configs/001_all_features_no_agg.json')

    # LINE notify
    line_notify('{} done.'.format(sys.argv[0]))

if __name__ == '__main__':
    main()