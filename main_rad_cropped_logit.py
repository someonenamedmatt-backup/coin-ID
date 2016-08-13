from model.tfmodel import TFModel
from src.coinclasses.coinlabel import CoinLabel
from model import model3c2d as model

def main():
        tfm = TFModel(model.encode_rad, 'data/saves/m_rad_cr_logit_v1', batch_size = 100)
        coinlabel = CoinLabel('/data2/processed/cropped/', '/home/ubuntu/coin/data/IDnamegrade.csv',
                                'rad', 'grade_lbl', random_state = model.SEED)
        tfm.fit(coinlabel, use_logit = True, total_epochs = 50)
        # tfm.evaluate(coinlabel)

if __name__ == '__main__':
    main()
