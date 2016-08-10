from __future__ import division
import os
import tensorflow as tf
import pandas as pd
from coin import Coin
from sklearn.cross_validation import train_test_split
import numpy as np

class CoinLabel:
    #class for managing coins
    #has the master cv list, creates train/test splits
    # and labels data
    #designed to work with a tf FIFO queue

    def __init__(self, img_folder, csv_file, coin_prop, label_col, test_pct = .2, random_state = 22):
        #coin prop is either 'img' or 'rad'
        if coin_prop not in ['img','rad']:
            print "coin_prop must be either 'img' or 'rad'"
            raise NameError
        if img_folder[-1] != '/':
            img_folder += '/'
        df = pd.read_csv(csv_file)[['ID',label_col]]
        df['file_names'] = map((lambda x:  str(x)),df['ID'])
        df = df[df['file_names'].isin(set(df['file_names']).intersection(set(os.listdir(img_folder))))]
        df['file_names'] = map(lambda x: img_folder + str(int(x)) + '/' + coin_prop +'.npy',df['file_names'])
        df.rename(columns={label_col: 'label'}, inplace=True)
        df.set_index('file_names', inplace = True)
        self.train_df, self.test_df = train_test_split(df['label'],
                                        test_size = test_pct,
                                        stratify = df['label'],
                                         random_state = random_state)
        self.label_dct = df['label'].to_dict()
        self.n_labels = len(self.train_df.unique())

    def get_class_weights(self):
        return [len(self.train_df)/len(self.train_df[self.train_df == value])
                                        for value in self.train_df.unique()]

    def read(self, file):
        #needs to be absolute file path
        return np.load(file),self.label_dct[file]


    def get_file_list(self, test = True):
        if test:
            return list(self.test_df.index.values)
        else:
            return list(self.train_df.index.values)
