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

    # target encoding
    print('target encoding...')
    for c in tqdm(CAT_COLS):
        dict_target = train_df[[c,'target']].groupby(c).mean()['target']
        train_df[f'{c}_te'] = train_df[c].map(dict_target)
        test_df[f'{c}_te'] = test_df[c].map(dict_target)

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
    df['seasonality'] = np.cos(np.pi*(df['S_2'].dt.day/31*2-1))

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