from model.tfmodel import TFModel
from src.coinclasses.coinlabel import CoinLabel
from model.model5c3d import encode
import cPickle as pickle
import os

def main():
        tfm = TFModel(encode, 'test_saves')
        coinlabel = CoinLabel('/data2/processed/whole/', '/home/ubuntu/coin/data/IDlabel.csv',
                                'rad', 'grade_lbl')
        tfm.fit(coinlabel)
        tfm.evaluate(coinlabel)

if __name__ == '__main__':
    main()
