
import numpy as np
import pandas as pd
import sys

from utils import to_json, to_feature, line_notify

#==============================================================================
# preprocess
#==============================================================================

is_test = False

def main():

    if is_test:
        train_df = pd.read_csv('../input/train_data.csv',nrows=100000)
        train_labels = pd.read_csv('../input/train_labels.csv',nrows=100000)
        test_df = pd.read_csv('../input/train_data.csv',nrows=100000)
        sub = pd.read_csv('../input/sample_submission.csv')
    else:
        train_df = pd.read_csv('../input/train_data.csv')
        train_labels = pd.read_csv('../input/train_labels.csv')
        test_df = pd.read_csv('../input/train_data.csv')
        sub = pd.read_csv('../input/sample_submission.csv')

    
    # TODO


    # save as feather
    to_feature(df, '../feats/f001')

    # save feature name list
    features_json = {'features':df.columns.tolist()}
    to_json(features_json,'../configs/101_all_features_mens.json')

    # LINE notify
    line_notify('{} done.'.format(sys.argv[0]))

if __name__ == '__main__':
    main()