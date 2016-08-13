from model.tfmodel import TFModel
from src.coinclasses.coinlabel import CoinLabel
from model import model3c2d as model

def main():
    tfm = TFModel(model.encode_img, 'data/saves/m_img_wh_v2', batch_size = 100)
    coinlabel = CoinLabel('/data/images/', '/home/ubuntu/coin-ID/data/IDnamegrade.csv',
                            'img', 'grade_lbl', random_state = model.SEED)
    tfm.fit(coinlabel, total_epochs = 25, load_save = False, balance_classes = True, do = False)

    # tfm.evaluate(coinlabel)

if __name__ == '__main__':
    main()
