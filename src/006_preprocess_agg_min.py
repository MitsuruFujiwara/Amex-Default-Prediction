
import gc
import json
import numpy as np
import pandas as pd
import sys

from tqdm import tqdm

from utils import to_json, to_feature, line_notify
from utils import CAT_COLS

#==============================================================================
# preprocess aggregation min only
#==============================================================================

# get features
def get_features(df):
    print('get features...')

    # numeric columns
    NUM_COLS = [c for c in df.columns if c not in CAT_COLS+['customer_ID','S_2']]

    # add diff features
    df_diff = df.groupby('customer_ID')[NUM_COLS].agg(['diff'])

    # change column names
    COLS_DIFF = [f'{x}_diff' for x in df_num_agg.columns]
    df_diff.columns = COLS_DIFF

    # merge diff
    df = df.merge(df_diff,on='customer_ID',how='left')

    del df_diff
    gc.collect()

    # aggregate
    print('aggregate...')
    df_num_agg = df.groupby("customer_ID")[NUM_COLS].agg(['min'])
    df_cat_agg = df.groupby("customer_ID")[CAT_COLS].agg(['min'])

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

    # merge target
    train_df = train_df.merge(train_labels,how='left',on='customer_ID')

    # to datetime
    train_df['S_2'] = pd.to_datetime(train_df['S_2'])
    test_df['S_2'] = pd.to_datetime(test_df['S_2'])

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
    test_df['seasonality_y'] = np.cos(np.pi*(test_df['S_2'].dt.dayofyear/366*2-1))

    # days diff features
    train_df['days_diff'] = train_df[['customer_ID','S_2']].groupby('customer_ID').diff()['S_2'].dt.days
    test_df['days_diff'] = test_df[['customer_ID','S_2']].groupby('customer_ID').diff()['S_2'].dt.days

    # target encoding
    print('target encoding...')
    for c in tqdm(CAT_COLS+['day','month','year']):
        dict_target = train_df[[c,'target']].groupby(c).mean()['target']
        train_df[f'{c}_te'] = train_df[c].map(dict_target)
        test_df[f'{c}_te'] = test_df[c].map(dict_target)

    # drop unnecessary columns
    train_df.drop(['S_2','target'],axis=1,inplace=True)
    test_df.drop(['S_2'],axis=1,inplace=True)

    # get features
    train_df = get_features(train_df)
    test_df = get_features(test_df)

    # add is test flag
    train_df['is_test'] = False
    test_df['is_test'] = True

    # add target
    train_df = train_labels.merge(train_df,how='left',on='customer_ID')
    test_df = test_labels.merge(test_df,how='left',on='customer_ID')

    print('load submission file...')
    sub_df = pd.read_csv('../output/submission_lgbm_no_agg.csv',dtype={'pred_no_agg':'float16'})

    print('add diff features...')
    sub_df['pred_no_agg_diff'] = sub_df[['customer_ID','pred_no_agg']].groupby('customer_ID').diff(1)['pred_no_agg']
    sub_df = sub_df.groupby("customer_ID")[['pred_no_agg','pred_no_agg_diff']].agg(['min'])

    # change column names
    sub_df.columns = ['_'.join(x) for x in sub_df.columns]

    test_df = test_df.merge(sub_df,how='left',on='customer_ID')

    del sub_df
    gc.collect()

    print('load oof file...')
    oof_df = pd.read_csv('../output/oof_lgbm_no_agg.csv',dtype={'pred_no_agg':'float16'})

    print('add diff features...')
    oof_df['pred_no_agg_diff'] = oof_df[['customer_ID','pred_no_agg']].groupby('customer_ID').diff(1)['pred_no_agg']

    oof_df = oof_df.groupby("customer_ID")[['pred_no_agg','pred_no_agg_diff']].agg(['min'])

    # change column names
    oof_df.columns = ['_'.join(x) for x in oof_df.columns]

    train_df = train_df.merge(oof_df,how='left',on='customer_ID')

    del oof_df
    gc.collect()

    # merge train & test
    df = pd.concat([train_df,test_df])

    del train_df, test_df
    gc.collect()

    # save as feather
    to_feature(df, '../feats/f006')

    # save feature name list
    features_json = {'features':df.columns.tolist()}
    to_json(features_json,'../configs/006_all_features_agg_min.json')

    # LINE notify
    line_notify('{} done.'.format(sys.argv[0]))

if __name__ == '__main__':
    main()