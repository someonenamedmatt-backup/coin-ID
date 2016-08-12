from model.tfmodel import TFModel
from src.coinclasses.coinlabel import CoinLabel
from model import model3c2d as model

def main():
        tfm = TFModel(model.encode_rad, 'data/saves/m_rad_cr_name_v0')
        coinlabel = CoinLabel('/data/images/', '/home/ubuntu/coin-ID/data/IDnamegrade.csv',
                                'rad', 'name_lbl', random_state = model.SEED)
        test = tfm.fit(coinlabel,  grade = False, total_epochs=25)
        # tfm.evaluate(coinlabel)

if __name__ == '__main__':
    main()
