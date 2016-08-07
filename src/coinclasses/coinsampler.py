import os
import pandas as pd
import numpy as np
from coinclasses import Coin
import matplotlib.colors as colors
import numpy as np
import math
import scipy.ndimage
from scipy import misc
from sklearn.cross_validation import train_test_split

class CoinSampler:
    """container class which holds the master directory of coins and allows sampling of classes"""

    def __init__(self, csv_file=None, label_col=None, coin_folder=None, batch_size=None, seed = 0):
        if csv_file is not None:
            df = pd.read_csv(csv_file)
            self.IDlabel = df[['ID','grade_lbl']].values
            self.labels = np.unique(self.IDlabel[:,1])
            self._file = lambda ID: coin_folder + str(ID) + '.npy'
            self.batch_size = batch_size
            self.seed = seed

    def get_epoch(self, test_pct = None):
        # Creates a new coinsampler object which has the same number of coins in each
        # class label (equal to the minimum number in any class)
        cs = CoinSampler()
        cs.labels = self.labels
        cs._file = self._file
        cs.batch_size = self.batch_size
        sample_size = np.min([len(self.IDlabel[self.IDlabel[:,1]==label]) for label in self.labels])
        cs.IDlabel = np.zeros((sample_size*len(self.labels),2))
        for i, label in enumerate(self.labels):
            this_label = self.IDlabel[self.IDlabel[:,1]==label]
            cs.IDlabel[sample_size*i:sample_size*(i+1), :] = this_label[np.random.choice(range(len(this_label)),sample_size),:]
        return cs

    def batch_generator(self, coin_prop = 'img'):
        #makes a generator which creates an epic then gives out (batchsize-coins,labels, epoch) each next
        #keeps making epic everytime its called
        #ignores any batch that isn't correctly sized (the last in any epoch)
        #coin_prop allows you to get either coins, 'img', or 'rad'
        epoch = 0
        while True:
            epoch = self.get_epoch()
            shuffled = epoch.IDlabel[np.random.permutation(len(epoch.IDlabel)),:]
                for i in xrange(0,len(epoch.IDlabel),epoch.batch_size):
                X = epoch._get_coins(shuffled[i:min(len(epoch.IDlabel),i+epoch.batch_size),0].astype(int), coin_prop)
                y = shuffled[i:min(len(epoch.IDlabel),i+epoch.batch_size),1].astype(int)
                if  X.shape[0] == self.batch_size:
                    yield X,self._make_cat(y), epoch_num
            epoch_num += 1

    def get_test_split(self, test_size = .2):
        #takes test_size of the data and creates a new coinsample instance
        #removes old data from current instance
        #deletes test data from current IDlabel
        cs = CoinSampler(seed = self.seed)
        cs.labels = self.labels
        cs._file = self._file
        for i, label in enumerate(self.labels):
            this_label = self.IDlabel[self.IDlabel[:,1]==label]
            if i == 0:
                IDlabel_train,IDlabel_test = train_test_split(this_label, test_size = test_size, random_state = self.seed)
            else:
                train,test = train_test_split(this_label, test_size = test_size, random_state = self.seed)
                IDlabel_train = np.append(IDlabel_train, train,axis = 0)
                IDlabel_test = np.append(IDlabel_test, test,axis = 0)
        self.IDlabel = IDlabel_train
        cs.IDlabel = IDlabel_test
        return cs

    def get_all_coins(self, coin_prop = 'img'):
        X =  np.array([Coin().load(self._file(ID)) for ID in self.IDlabel[:,0]])
        return self._clean_coins(X), self._make_cat(self.IDlabel[:,1])

    def _get_coins(self, file_list, coin_prop = 'img'):
        X = []
        #doing it using a list to allow for better error management
        for f in file_list:
            try:
                if coin_prop == 'img':
                    X.append(Coin().load(self._file(f)).img)
                elif coin_prop == 'rad':
                    X.append(Coin().load(self._file(f)).rad)
                else:
                    X.append(Coin().load(self._file(f)))
                    return X
            except:
                print f
                raise
        #load coin.img for everything on the list
        X = filter(lambda coin: coin.shape == (90,90,3), X)
        #cut out missized coins
        X = np.concatenate(map(lambda x: x[np.newaxis,:,:,:],X))
        if len(X) == 0:
            print file_list
            raise ValueError
        return X

    def _make_cat(self ,Y):
        y_cat = np.zeros((len(Y),4))
        for i in range(4):
            y_cat[Y==i, i] = 1
        return y_cat
