from model.tfmodel import TFModel
from src.coinclasses.coinlabel import CoinLabel
from model import modelalex as model

def main():
        tfm = TFModel(model.encode_rad, 'data/rad_cropped_5c3d', batch_size = 10)
        coinlabel = CoinLabel('/data/images/', '/home/ubuntu/coin-ID/data/IDnamegrade.csv',
                                'rad', 'grade_lbl', random_state = model.SEED)
        test = tfm.fit(coinlabel, 100)

        # tfm.evaluate(coinlabel)

if __name__ == '__main__':
    main()
