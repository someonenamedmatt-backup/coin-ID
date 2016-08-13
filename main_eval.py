from model.tfmodel import TFModel
from src.coinclasses.coinlabel import CoinLabel
from model import model3c2d as model
from model import model_network
import tensorflow as tf


def main():
        tfm_list = []
        coinlabel_lst = []
        tfm_list.append(TFModel(model.encode_rad, 'data/saves/m_rad_cr_of', batch_size = 20))
        coinlabel_lst.append(CoinLabel('/data2/processed/cropped/', '/home/ubuntu/coin/data/IDnamegrade.csv',
                                        'rad', 'grade_lbl', random_state = model.SEED))
        tfm_list.append(TFModel(model.encode_rad, 'data/saves/m_rad_cr_name_overfit', batch_size = 20))
        coinlabel_lst.append(CoinLabel('/data2/processed/cropped', '/home/ubuntu/coin/data/IDnamegrade.csv',
                                'rad', 'name_lbl', random_state = model.SEED))
        tfm_list.append(TFModel(model.encode_rad, 'data/saves/m_rad_cr_logit', batch_size = 20))
        coinlabel_lst.append(CoinLabel('/data2/processed/cropped', '/home/ubuntu/coin/data/IDnamegrade.csv',
                                    'rad', 'grade_lbl', random_state = model.SEED))
        tfm_list.append(TFModel(model_network.encode_rad, 'data/saves/m_rad_cr_network_overfit', batch_size = 10))
        coinlabel_lst.append(CoinLabel('/data2/processed/cropped', '/home/ubuntu/coin/data/IDnamegrade.csv',
                        'rad', 'grade_lbl', random_state = model_network.SEED))
        tfm_list.append(TFModel(model.encode_img, 'data/saves/m_img_cr_overfit', batch_size = 20))
        coinlabel_lst.append(CoinLabel('/data2/processed/whole', '/home/ubuntu/coin/data/IDnamegrade.csv',
                                'img', 'grade_lbl', random_state = model.SEED))
        tfm_list.append(TFModel(model.encode_img, 'data/saves/m_img_wh', batch_size = 20))
        coinlabel_lst.append(CoinLabel('/data2/processed/whole', '/home/ubuntu/coin/data/IDnamegrade.csv',
                                'img', 'grade_lbl', random_state = model.SEED))
        for tfm, cl in zip(tfm_list,coinlabel_lst,):
            model_name = tfm.save_dir.split('/')[-1]
            print "Precision of model {}: {}".format(model_name, tfm.evaluate(cl, over_fit_test = True))
            tf.reset_default_graph()



if __name__ == '__main__':
    main()
