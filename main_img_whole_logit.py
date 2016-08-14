from model.tfmodel import TFModel
from src.coinclasses.coinlabel import CoinLabel
from model import model3c2d as model
tfm = TFModel(model.encode_img, 'data/saves/m_img_wh_logit_v1', batch_size = 100)
coinlabel = CoinLabel('/data2/images', '/home/ubuntu/coin/data/IDnamegrade.csv',
                            'cr', 'grade_lbl', random_state = model.SEED)
# tfm.evaluate(coinlabel)
tfm.predict_images(['/data2/photograde/3CentNic-55o.jpg'],1.0)
# def main():
#         tfm = TFModel(model.encode_img, 'data/saves/m_img_wh_logit_v1', batch_size = 100)
#         coinlabel = CoinLabel('/data/images/', '/home/ubuntu/coin-ID/data/IDnamegrade.csv',
#                                 'rad', 'grade_lbl', random_state = model.SEED)
#         coinlabel = CoinLabel('/data2/images', '/home/ubuntu/coin/data/IDnamegrade.csv',
#                             'cr', 'grade_lbl', random_state = model.SEED)
#         tfm.fit(coinlabel, use_logit = True, total_epochs = 25, balance_classes = True, do = False)
#         # tfm.evaluate(coinlabel)
#         # tfm.predict_one('/data2/photograde/3CentNic-55o.jpg',1)
# if __name__ == '__main__':
#     main()
