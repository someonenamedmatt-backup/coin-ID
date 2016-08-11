from model.tfmodel import TFModel
from src.coinclasses.coinlabel import CoinLabel
from model import model3c2d


def main():
        tfm = TFModel(model3c2d.encode_img, 'data/img_cropped_3c2d')
        coinlabel = CoinLabel('/data/cropped/', '/home/ubuntu/coin-ID/data/IDnamegrade.csv',
                                'img', 'grade_lbl', random_state = model3c2d.SEED)
        test = tfm.fit(coinlabel, 100)
        # tfm.evaluate(coinlabel)

if __name__ == '__main__':
    main()
