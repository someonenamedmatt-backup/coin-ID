from model.tfmodel import TFModel
from src.coinclasses.coinlabel import CoinLabel
from model import model5c3d

def main():
        tfm = TFModel(model5c3d.encode_rad, 'data/rad_cropped_5c3d')
        coinlabel = CoinLabel('/data2/processed/cropped/', '/home/ubuntu/coin-ID/data/IDamegrade.csv',
                                'rad', 'grade_lbl', random_state = model5c3d.SEED)
        test = tfm.fit(coinlabel, 100)

        # tfm.evaluate(coinlabel)

if __name__ == '__main__':
    main()
