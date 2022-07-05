from cgi import test
import gc
import json
import numpy as np
import pandas as pd
import sys

from tqdm import tqdm

from utils import to_json, to_feature, line_notify
from utils import CAT_COLS

#==============================================================================
# preprocess no aggregation
#==============================================================================

def main():

    # load csv
    train_df = pd.read_parquet('../input/train.parquet')
    train_labels = pd.read_csv('../input/train_labels.csv')
    test_df = pd.read_parquet('../input/test.parquet')

    # merge target
    train_df = train_df.merge(train_labels,how='left',on='customer_ID')

    # add is test flag
    train_df['is_test'] = False
    test_df['is_test'] = True

    # to datetime
    train_df['S_2'] = pd.to_datetime(train_df['S_2'])
    test_df['S_2'] = pd.to_datetime(test_df['S_2'])

    # days diff features
    train_df['days_diff'] = train_df[['customer_ID','S_2']].groupby('customer_ID').diff()['S_2'].dt.days
    test_df['days_diff'] = test_df[['customer_ID','S_2']].groupby('customer_ID').diff()['S_2'].dt.days

    # datetime features
    train_df['day'] = train_df['S_2'].dt.day
    train_df['month'] = train_df['S_2'].dt.month
    train_df['year'] = train_df['S_2'].dt.year
    train_df['seasonality_m'] = np.cos(np.pi*(train_df['S_2'].dt.day/31*2-1))
    train_df['seasonality_y'] = np.cos(np.pi*(train_df['S_2'].dt.dayofyear/366*2-1))

    test_df['day'] = test_df['S_2'].dt.day
    test_df['month'] = test_df['S_2'].dt.month
    test_df['year'] = test_df['S_2'].dt.year
    test_df['seasonality_m'] = np.cos(np.pi*(test_df['S_2'].dt.day/31*2-1))
    test_df['seasonality_y'] = np.cos(np.pi*(train_df['S_2'].dt.dayofyear/366*2-1))

    # target encoding
    print('target encoding...')
    for c in tqdm(CAT_COLS+['day','month','year']):
        dict_target = train_df[[c,'target']].groupby(c).mean()['target']
        train_df[f'{c}_te'] = train_df[c].map(dict_target)
        test_df[f'{c}_te'] = test_df[c].map(dict_target)

    # merge train & test
    df = pd.concat([train_df,test_df])

    del train_df, test_df
    gc.collect()

    # drop unnecessary columns
    df.drop('S_2',axis=1,inplace=True)

    # save as feather
    to_feature(df, '../feats/f001')

    # save feature name list
    features_json = {'features':df.columns.tolist()}
    to_json(features_json,'../configs/001_all_features_no_agg.json')

    # LINE notify
    line_notify('{} done.'.format(sys.argv[0]))

if __name__ == '__main__':
    main()