from model.tfmodel import TFModel
from src.coinclasses.coinlabel import CoinLabel
from model import model3c2d

def main():
        tfm = TFModel(model3c2d.encode_rad, 'data/rad_cropped_3c2d')
        coinlabel = CoinLabel('/data2/processed/cropped/', '/home/ubuntu/coin/data/IDamegrade.csv',
                                'rad', 'name_lbl', random_state = model3c2d.SEED)
        test = tfm.fit(coinlabel, 100, grade = False)
        # tfm.evaluate(coinlabel)

if __name__ == '__main__':
    main()
