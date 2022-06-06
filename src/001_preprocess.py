
import gc
import numpy as np
import pandas as pd
import sys

from utils import reduce_mem_usage, to_json, to_feature, line_notify

#==============================================================================
# preprocess
#==============================================================================

is_debug = True

def main():

    if is_debug:
        train_df = pd.read_csv('../input/train_data.csv',nrows=100000)
        train_labels = pd.read_csv('../input/train_labels.csv',nrows=100000)
        test_df = pd.read_csv('../input/train_data.csv',nrows=100000)
        sub = pd.read_csv('../input/sample_submission.csv')
    else:
        train_df = pd.read_csv('../input/train_data.csv')
        train_labels = pd.read_csv('../input/train_labels.csv')
        test_df = pd.read_csv('../input/train_data.csv')
        sub = pd.read_csv('../input/sample_submission.csv')

    # to datetime
    train_df['S_2'] = pd.to_datetime(train_df['S_2'])
    test_df['S_2'] = pd.to_datetime(test_df['S_2'])

    # reduce memory usage
    train_df = reduce_mem_usage(train_df)
    test_df = reduce_mem_usage(test_df)
    
    # aggregate
    train_df = train_df.groupby('customer_ID').mean()
    test_df = test_df.groupby('customer_ID').mean()

    # merge target
    train_df = train_labels.merge(train_df,how='left',on='customer_ID')
    test_df = sub.merge(test_df,how='left',on='customer_ID')

    del train_labels, sub
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