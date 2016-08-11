from model.tfmodel import TFModel
from src.coinclasses.coinlabel import CoinLabel
from model import model3c2d

def main():
        tfm = TFModel(model3c2d.encode_rad, 'data/rad_cropped_3c2d')
        coinlabel = CoinLabel('/data/cropped/', '/home/ubuntu/coin-ID/data/IDnamegrade.csv',
                                'rad', 'grade_lbl', random_state = model3c2d.SEED)
        tfm.evaluate(coinlabel)

if __name__ == '__main__':
    main()
