from model.tfmodel import TFModel
from src.coinclasses.coinlabel import CoinLabel
from model import model3c2d as model

def main():
        tfm = TFModel(model.encode_rad, 'data/rad_cropped_3c2d_reg', batch_size = 25)
        coinlabel = CoinLabel('/data/images/', '/home/ubuntu/coin-OD/data/IDnamegrade.csv',
                                'rad', 'grade_lbl', random_state = model.SEED)
        test = tfm.fit(coinlabel, 100, load_save = False)
        # tfm.evaluate(coinlabel)

if __name__ == '__main__':
    main()
