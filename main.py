from model.tfmodel import TFModel
from src.coinclasses.coinlabel import CoinLabel
from model import model3c2d as model
import sys

def train_model(label, save_name, coin_prop ):
    coinlabel = CoinLabel('/data/'+coin_prop, '/home/ubuntu/coin-ID/data/IDlabel.csv',
                                            coin_prop, label, random_state = model.SEED)
    if coin_prop == 'rad':
        tfm =  TFModel(model.encode_rad, '/home/ubuntu/coin-ID/data/saves/' + save_name)
    else:
        tfm =  TFModel(model.encode_img, '/home/ubuntu/coin-ID/data/saves/' + save_name)
    return coinlabel, tfm


if __name__ == '__main__':
    if sys.argv[1] == '1':
        tfm, coinlabel = train_model('grade_lbl', 'm_rad_v' + sys.argv[2], 'rad')
        tfm.fit(coinlabel)
        tfm.evaluate(coinlabel)
    if sys.argv[1] == '2':
        tfm, coinlabel = train_model('grade_lbl', 'm_cr_v' + sys.argv[2], 'cr')
        tfm.fit(coinlabel)
        tfm.evaluate(coinlabel)
    if sys.argv[1] == '3':
        tfm, coinlabel = train_model('grade_lbl', 'm_cr_log_v' + sys.argv[2], 'cr')
        tfm.fit(coinlabel, use_logit = True)
        tfm.evaluate(coinlabel,use_logit = True)
    if sys.argv[1] == '4':
        tfm, coinlabel = train_model('grade_lbl', 'm_cr_no_do_v' + sys.argv[2], 'cr')
        tfm.fit(coinlabel)
        tfm.evaluate(coinlabel)
    if sys.argv[1] == '5':
        tfm, coinlabel = train_model('grade_lbl', 'm_cr_nobalance_v' + sys.argv[2], 'cr')
        tfm.fit(coinlabel, balance_classes = False)
        tfm.evaluate(coinlabel)
    if sys.argv[1] == '6':
        tfm, coinlabel = train_model('name_lbl', 'm_cr_name_v' + sys.argv[2])
        tfm.fit(coinlabel, grade = False)
        tfm.evaluate(coinlabel, grade = False)
