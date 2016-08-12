from model.tfmodel import TFModel
from src.coinclasses.coinlabel import CoinLabel
from model import model3c2d as model

def main():
        tfm = TFModel(model.encode_rad, 'data/saves/m_rad_cr_logit', batch_size = 20)
        coinlabel = CoinLabel('/data/images/', '/home/ubuntu/coin-ID/data/IDnamegrade.csv',
                                'rad', 'grade_lbl', random_state = model.SEED)
        test = tfm.overt_fit_test(coinlabel, use_logit = True)
        # tfm.evaluate(coinlabel)

if __name__ == '__main__':
    main()
