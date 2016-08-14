from model.tfmodel import TFModel
from src.coinclasses.coinlabel import CoinLabel
from model import model3c2d as model

def main():
        tfm = TFModel(model.encode_img, 'data/saves/m_img_wh_logit_v1', batch_size = 100)
        coinlabel = CoinLabel('/data/images/', '/home/ubuntu/coin-ID/data/IDnamegrade.csv',
                                'rad', 'grade_lbl', random_state = model.SEED)
        tfm.fit(coinlabel, use_logit = True, total_epochs = 25, balance_classes = True, do = False)
        # tfm.evaluate(coinlabel)

if __name__ == '__main__':
    main()
