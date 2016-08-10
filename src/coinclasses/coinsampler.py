from __future__ import division
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




class_weight = tf.constant([0.8781790, 0.06321889,0.01993216, 0.03165421 ,0.00701573 ])

weighted_logits = tf.mul(y, class_weight) # shape [batch_size, 5]
class CoinSampler:
    """container class which holds the master directory of coins and allows sampling of classes"""

    def __init__(self, coin_prop = None, csv_file=None, label_col=None, coin_folder=None, batch_size=None, seed = 0):
        if csv_file is not None:
            df = pd.read_csv(csv_file)
            self.IDlabel = df[['ID','grade_lbl']].values
            self.labels = np.unique(self.IDlabel[:,1])
            self.label_counts = [len(self.IDlabel[self.IDlabel[:,1]==label]) for label in self.labels]
            self.epoch_size = np.min(self.label_counts) * len(labels)
            self._file = lambda ID: coin_folder + str(ID) + '.npy'
            self.batch_size = batch_size
            self.seed = seed
            self.coin_prop = coin_prop

    def get_epoch(self, test_pct = None):
        # Creates a new coinsampler object which has the same number of coins in each
        # class label (equal to the minimum number in any class)
        cs = CoinSampler()
        cs.labels = self.labels
        cs._file = self._file
        cs.batch_size = self.batch_size
        cs.seed = self.seed
        sample_size = np.min([len(self.IDlabel[self.IDlabel[:,1]==label]) for label in self.labels])
        cs.IDlabel = np.zeros((sample_size*len(self.labels),2))
        for i, label in enumerate(self.labels):
            this_label = self.IDlabel[self.IDlabel[:,1]==label]
            cs.IDlabel[sample_size*i:sample_size*(i+1), :] = this_label[np.random.choice(range(len(this_label)),sample_size),:]
        return cs

    def batch_generator(self):
        #makes a generator which creates an epic then gives out (batchsize-coins,labels, epoch) each next
        #keeps making epic everytime its called
        #ignores any batch that isn't correctly sized (the last in any epoch)

        epoch = 0
        while True:
            epoch = self.get_epoch()
            shuffled = epoch.IDlabel[np.random.permutation(len(epoch.IDlabel)),:]
                for i in xrange(0,len(epoch.IDlabel),epoch.batch_size):
                X = epoch._get_coins(shuffled[i:min(len(epoch.IDlabel),i+epoch.batch_size),0].astype(int), self.coin_prop)
                y = shuffled[i:min(len(epoch.IDlabel),i+epoch.batch_size),1].astype(int)
                if  X.shape[0] == self.batch_size:
                    yield X,self._make_cat(y), epoch_num
            epoch_num += 1

    def get_test_split(self, test_size = .2, balance_classes = True):
        #takes test_size of the data and creates a new coinsample instance
        #removes old data from current instance
        #deletes test data from current IDlabel
        ratio = np.min([len(self.IDlabel[self.IDlabel[:,1]==label]) for label in self.labels])/[len(self.IDlabel[self.IDlabel[:,1]==label]) for label in self.labels]
        cs = self.get_epoch(seed = self.seed)
        cs.labels = self.labels
        cs._file = self._file
        cs.seed = self.seed
        for i, label in enumerate(self.labels):
            this_label = self.IDlabel[self.IDlabel[:,1]==label]
            if i == 0:
                IDlabel_train,IDlabel_test = train_test_split(this_label, test_size = test_size*ratio[i], random_state = self.seed)
            else:
                train,test = train_test_split(this_label, test_size = test_size*ratio[i], random_state = self.seed)
                IDlabel_train = np.append(IDlabel_train, train,axis = 0)
                IDlabel_test = np.append(IDlabel_test, test,axis = 0)
        self.IDlabel = IDlabel_train
        cs.IDlabel = IDlabel_test
        return cs

    def get_all_coins(self):
        X =  np.array([Coin().load(self._file(ID),self.coin_prop) for ID in self.IDlabel[:,0]])
        return self._clean_coins(X), self._make_cat(self.IDlabel[:,1])

    def _get_coins(self, file_list):
        X = []
        #doing it using a list to allow for better error management
        for f in file_list:
            try:
                X.append(Coin().load(self._file(f)),self.coin_prop)
                if self.coin_prop not in ['img', 'rad']:
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

    def _generate_image_and_label_batch(image, label, min_queue_examples,
                                        batch_size, shuffle):
      """Construct a queued batch of images and labels.
      Args:
        image: 3-D Tensor of [height, width, 3] of type.float32.
        label: 1-D Tensor of type.int32
        min_queue_examples: int32, minimum number of samples to retain
          in the queue that provides of batches of examples.
        batch_size: Number of images per batch.
        shuffle: boolean indicating whether to use a shuffling queue.
      Returns:
        images: Images. 4D tensor of [batch_size, height, width, 3] size.
        labels: Labels. 1D tensor of [batch_size] size.
      """
      # Create a queue that shuffles the examples, and then
      # read 'batch_size' images + labels from the example queue.
      num_preprocess_threads = 8
      if shuffle:
        images, label_batch = tf.train.shuffle_batch(
            [image, label],
            batch_size=batch_size,
            num_threads=num_preprocess_threads,
            capacity=min_queue_examples + 3 * batch_size,
            min_after_dequeue=min_queue_examples)
      else:
        images, label_batch = tf.train.batch(
            [image, label],
            batch_size=batch_size,
            num_threads=num_preprocess_threads,
            capacity=min_queue_examples + 3 * batch_size)

      # Display the training images in the visualizer.
      tf.image_summary('images', images)

      return images, tf.reshape(label_batch, [batch_size])
