from model.tfmodel import TFModel
from src.coinclasses.coinlabel import CoinLabel
from model import model5c3d
import cPickle as pickle
import os

def main():
        tfm = TFModel(model5c3d.encode_rad, 'test_saves')
        coinlabel = CoinLabel('/data2/processed/whole/', '/home/ubuntu/coin/data/IDlabel.csv',
                                'rad', 'grade_lbl', random_state = model5c3d.SEED)
        tfm.fit0(coinlabel)
        # tfm.evaluate(coinlabel)

if __name__ == '__main__':
    main()
