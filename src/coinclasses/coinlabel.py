from __future__ import division
import os
import tensorflow as tf
import pandas as pd
from sklearn.cross_validation import train_test_split
import numpy as np
from random import shuffle

class CoinLabel:
    #class for managing coins
    #has the master cv list, creates train/test splits
    # and labels data
    #designed to work with a tf FIFO queue

    def __init__(self, img_folder, csv_file, coin_prop, label_col, test_pct = .2, random_state = 22):
        #coin prop is either 'img' or 'rad' or 'cr'
        if img_folder[-1] != '/':
            img_folder += '/'
        df = pd.read_csv(csv_file)
        if "name_lbl" in df.columns:
            self.n_names = len(df['name_lbl'].unique())
        if "grade_lbl" in df.columns:
            self.n_grades = len(df['grade_lbl'].unique())
        df['file_names'] = map((lambda x:  str(x)),df['ID'])
        df = df[df['file_names'].isin(set(df['file_names']).intersection(set(os.listdir(img_folder+'/'+coin_prop))))]
        df['file_names'] = map(lambda x: img_folder  +coin_prop+ '/' + str(int(x))  ,df['file_names'])
        df.rename(columns={label_col: 'label'}, inplace=True)
        df.set_index('file_names', inplace = True)
        self.train_df, self.test_df = train_test_split(df['label'],
                                        test_size = test_pct,
                                        stratify = df['label'],
                                         random_state = random_state)
        self.label_dct = df['label'].to_dict()
        self.n_labels = len(self.train_df.unique())

    def get_class_weights(self):
        return [len(self.train_df[self.train_df == value])/len(self.train_df) for value in np.sort(self.train_df.unique())]

    def read(self, file):
        #needs to be absolute file path
        return np.load(file),self.label_dct[file]

    def get_overfit_test_list(self):
        lst = []
        for value in self.train_df.unique()[:4]:
            lst.extend(self.train_df[self.train_df == value].index.values[:5])
        return lst

    def get_file_list(self, test = True):
        if test:
            return list(self.test_df.index.values)
        else:
            return list(self.train_df.index.values)

    def get_balanced_class_filelist(self, num_per_class, test = True):
        lst = []
        for label in self.train_df.unique():
            if test:
                lst += list(np.random.choice(self.test_df[self.test_df == label].index, num_per_class))
            else:
                lst += list(np.random.choice(self.train_df[self.train_df == label].index, num_per_class))
        shuffle(lst)
        return lst
