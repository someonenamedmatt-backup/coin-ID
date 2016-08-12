from model.tfmodel import TFModel
from src.coinclasses.coinlabel import CoinLabel
from model import model3c2d

def main():
        tfm = TFModel(model3c2d.encode_rad, 'rad_cropped_3c2d', batch_size = 25)
        coinlabel = CoinLabel('/data2/processed/cropped/', '/home/ubuntu/coin/data/IDnamegrade.csv',
                                'rad', 'grade_lbl', random_state = model3c2d.SEED)
        test = tfm.fit(coinlabel, 100, load_save = False, do = False)
        # tfm.evaluate(coinlabel)

if __name__ == '__main__':
    main()
