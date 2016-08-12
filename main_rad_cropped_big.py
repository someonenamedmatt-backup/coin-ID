from model.tfmodel import TFModel
from src.coinclasses.coinlabel import CoinLabel
from model import model_network as model

def main():
        tfm = TFModel(model.encode_rad, 'data/saves/m_rad_cr_network_v0', batch_size = 10)
        coinlabel = CoinLabel('/data/images/', '/home/ubuntu/coin-ID/data/IDnamegrade.csv',
                                'rad', 'grade_lbl', random_state = model.SEED)
        test = tfm.fit(coinlabel, total_epochs = 25)

        # tfm.evaluate(coinlabel)

if __name__ == '__main__':
    main()
