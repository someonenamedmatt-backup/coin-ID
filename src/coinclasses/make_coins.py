import pandas as pd
from coin import Coin
import os
from multiprocessing import Manager
import multiprocessing as mp
import cPickle as pickle
import numpy as np

import time

bad_coins = Manager().list()
def make_coins_from_scratch():
    global df
    df = pd.read_csv('/home/ubuntu/coin/data/IDnamegrade.csv').set_index('ID')[['grade_lbl','name_lbl']]
    pool = mp.Pool(mp.cpu_count())
    rs = pool.map_async(make_coin_from_scratch,df.index.values)
    while not rs.ready():
        print("num left: {}".format(rs._number_left))
        time.sleep(1)
    pool.close()
    pool.join()

def make_coin_from_scratch(ID):
    try:
        f = '/data/whole/'+str(ID)+'.jpg'
        c = Coin().make_from_image(f)
        binarize_coin(c.img, ID).tofile('/data2/images/img/' + str(ID))
        binarize_coin(c.cr, ID).tofile('/data2/images/cr/' + str(ID))
        binarize_coin(c.rad, ID).tofile('/data2/images/rad/' + str(ID))
    except:
        bad_coins.append(ID)

def binarize_coin(img,ID):
    return np.append(img.flatten(),df.loc[ID].values)

#
# def make_coins():
#     try:
#         for typ in os.listdir(folder):
#             f = folder + '/' + typ
#             if typ in ['img', 'rad']:
#                 try:
#                     value = np.fromfile(f)
#                 except:
#                     bad_value.append(ID)
#                 np.append(value[:128*128*3],[grade_lbl,name_lbl]).tofile(f)
#             elif typ in ['img.npy', 'rad.npy']:
#                 value = np.load(f).flatten()
#                 np.append(value,[grade_dct[ID],name_dct[ID]]).tofile(f[:-4])
#     except:
#         print folder
#         bad_coins.append(folder)
#
#
# def make_coins(input_folder, output_folder, bad_coin_file, csv_file=None, find_circle=False, size=(128,128)):
#     #goes through a directory and makes a coin object for every file in input folder intersect csv_file (optional)
#     #find_circles and size let you choose if you want to crop the image using find_circles and what size you want
#     if not os.path.exists(output_folder):
#         os.makedirs(output_folder)
#     file_list = os.listdir(input_folder)
#     if csv_file is not None:
#         df = pd.read_csv(csv_file)
#         file_list = list(set(file_list).intersection(set(df['ID'].map(lambda id: str(id)+ '.jpg'))))
#     file_list = map(lambda id: (input_folder + id,find_circle,size, output_folder),file_list)
#     pool = mp.Pool(mp.cpu_count())
#     pool.map(make_coin,file_list)
#     pool.close()
#     pool.join()
#     print len(list(bad_coins))
#
# def make_coin((file, find_circle, size, output_folder)):
#     #takes file and options and makes a single coin in output folder
#     #takes them as a tuple for multiprocessing to work properly
#     if os.path.exists('output_folder+coin_name[:-4]'):
#         return True
#     try:
#         c = Coin().make_from_image(file, find_circle, size)
#
#         if type(c) == bool:
#             bad_coins.append(file.split('/')[-1][:-4])
#         else:
#             coin_name = file.split('/')[-1]
#         try:
#             c.save(output_folder+coin_name[:-4])
#         except:
#             bad_coins.append(file.split('/')[-1][:-4])
#     except:
#         bad_coins.append(file.split('/')[-1][:-4])
#
# def convert_coins(folder):
#     #goes through a folder and makes all the coins in it into the new coin format
#     coins = filter(lambda str: 'npy' in str, os.listdir(folder))
#     coins = map(lambda str: folder + str[:-4], coins)
#     pool = mp.Pool(mp.cpu_count())
#     pool.map(convert_coin,folder)
#     pool.close()
#     pool.join()
#     print len(list(bad_coins))
#
# def convert_coin(f):
#     try:
#         coin = Coin().load(f + '.npy')
#         os.mkdir(f)
#         np.save(f + '/img', coin.img)
#         np.save(f + '/rad', coin.rad)
#         os.remove(f + '.npy')
#     except:
#         bad_coins.append(f.split('/')[-1])
#         raise
#
#
# def clean_csv(csv_file, output_folder = None, save = False):
#     #takes the csv file from the scraper and cleans it up
#     #finds PCGA Grades and makes grade_lbl
#     #saves outputfile.csv and outputfile_full.csv
#     def get_numbers(str):
#         return [int(s) for s in str if s.isdigit()]
#     def hasNumbers(str):
#         return any(char.isdigit() for char in str)
#     def get_grade(lst):
#         for i,item in enumerate(lst[:-1]):
#             if 'Grade'in item:
#                 num_lst = get_numbers(lst[i+1])
#                 if len(num_lst)>2:
#                     return []
#                 else:
#                     try:
#                         return int(''.join(map(str,num_lst)))
#                     except:
#                         pass
#         return []
#     def make_label(grade):
#         if grade <= 20:
#             return 0
#         elif grade <= 50:
#             return 1
#         elif grade <= 62:
#             return 2
#         else:
#             return 3
#     def get_name(lst):
#         for item in lst:
#             if item in ['Two Cents', 'Three Cents', 'Twenty Cent Pieces', 'Colonial', 'America the Beautiful 2010-Now']:
#                 return item
#             elif '(' in item and ')' in item and 'Proof' not in item and 'See more' not in item:
#                 return item
#         return 'None'
#     df = pd.read_csv(csv_file)
#     df['categories']=map(lambda str: eval(str.replace(';',',')),df['Ebay Categories'])
#     df['attributes']=map(lambda str: eval(str.replace(';',',')),df['Attributes'])
#     #Filter out collections
#     df = df[map(lambda lst: 'Collections' not in lst[2]  ,df['categories'])]
#     df = df[map(lambda lst: 'World' not in lst[1]  ,df['categories'])]
#     df['Grade']=map(get_grade,df['attributes'])
#     df = df[map(lambda thing: type(thing)==int ,df['Grade'])]
#     df = df[df['Grade']!=0]
#     df = df[(~df['Grade'].isin(non_grades))&(df['Grade']<=70)]
#     df['grade_lbl'] = df['Grade'].map(make_label)
#     df = df[df['Ebay Categories'].isin(filter(lambda str: 'Errors' not in str, df['Ebay Categories']))]
#     df['coin_name'] = df['categories'].map(get_name)
#     name_dct = {x:i for i,x  in enumerate(df['coin_name'].unique())}
#     df['name_lbl'] = df['coin_name'].map(name_dct.get)
#     grade_dct = {x: df[df['Grade'] == x ]['grade_lbl'].unique()[0] for x in df['Grade'].unique()}
#     if save:
#         df[['ID','grade_lbl']].to_csv(output_folder+'IDgrade.csv')
#         df[['ID','name_lbl']].to_csv(output_folder+'IDname.csv')
#         df[['ID','name_lbl','grade_lbl']].to_csv(output_folder+'IDnamegrade.csv')
#         pickle.dump(name_dct, open(output_folder+'name_dct.pkl','wb'))
#         pickle.dump(grade_dct, open(output_folder+'grade_dct.pkl','wb'))
#     return df
#
# def convert_coins_bin():
#     #goes through a folder and makes all the coins in it into the new coin format
#     coins1 = map(lambda str: '/data2/processed/cropped/' + str + '/', os.listdir('/data2/processed/cropped'))
#     coins2 = map(lambda str: '/data2/processed/whole/' + str + '/', os.listdir('/data2/processed/whole'))
#     coins = coins1 + coins2
#     pool = mp.Pool(mp.cpu_count())
#     pool.map(convert_coin_bin, coins)
#     pool.close()
#     pool.join()
#
# def convert_coin_bin(f):
#     try:
#         if os.path.exists(f+'img'):
#             np.load(f + 'img.npy').tofile(f+'img')
#             np.load(f + 'rad.npy').tofile(f+'rad')
#             os.remove(f + 'img.npy')
#             os.remove(f + 'rad.npy')
#     except:
#         pass
#
# def add_labels():
#     df = pd.read_csv('/home/ubuntu/coin/data/IDnamegrade.csv')
#     df['cropped_folders'] = df.ID.map(lambda ID: '/data2/processed/cropped/' + str(ID))
#     df['whole_folders'] = df.ID.map(lambda ID: '/data2/processed/whole/' + str(ID))
#     pool = mp.Pool(mp.cpu_count())
#     pool.map(add_label,list(df[['ID','cropped_folders','grade_lbl','name_lbl']].values))
#     pool.map(add_label,list(df[['ID','whole_folders','grade_lbl','name_lbl']].values))
#     pool.close()
#     pool.join()
#
# bad_value = Manager().list()
# not_in_dict = Manager().list()
#
# def add_label((ID, folder, grade_lbl, name_lbl)):
#     df = pd.read_csv('/home/ubuntu/coin/data/IDnamegrade.csv').set_index('ID')
#     try:
#         for typ in os.listdir(folder):
#             f = folder + '/' + typ
#             if typ in ['img', 'rad']:
#                 try:
#                     value = np.fromfile(f)
#                 except:
#                     bad_value.append(ID)
#                 np.append(value[:128*128*3],[grade_lbl,name_lbl]).tofile(f)
#             elif typ in ['img.npy', 'rad.npy']:
#                 value = np.load(f).flatten()
#                 np.append(value,[grade_dct[ID],name_dct[ID]]).tofile(f[:-4])
#     except:
#         print folder
#         bad_coins.append(folder)
#
# def filter_coins():
# #goes through and deletes any coins without an entry in IDnamegrade
#     del_convert = set(bad_value).union(set(bad_coins))
#     whole_coins = set(map(lambda word:int(word), set(os.listdir('/data2/processed/whole'))))
#     cropped_coins = set(map(lambda word:int(word),set(os.listdir('/data2/processed/cropped'))))
#     coins_to_keep = whole_coins.intersection(cropped_coins)
#     IDnamegrade = pd.read_csv('/home/ubuntu/coin/data/IDnamegrade.csv')
#     IDname = pd.read_csv('/home/ubuntu/coin/data/IDname.csv')
#     IDgrade = pd.read_csv('/home/ubuntu/coin/data/IDgrade.csv')
#     csv_coins = set(pd.read_csv('/home/ubuntu/coin/data/IDnamegrade.csv')['ID'])
#     coins_to_keep = coins_to_keep.intersection(csv_coins)
#     whole_del = (whole_coins - coins_to_keep).union(del_convert)
#     cropped_del = (cropped_coins - coins_to_keep).union(del_convert)
#     IDnamegrade = IDnamegrade[IDnamegrade.ID.isin(coins_to_keep)]
#     IDnamegrade[['ID','name_lbl','grade_lbl']].to_csv('/home/ubuntu/coin/data/IDnamegrade.csv')
#     IDname = IDname[IDname.ID.isin(coins_to_keep)]
#     IDname[['ID','name_lbl']].to_csv('/home/ubuntu/coin/data/IDname.csv')
#     IDgrade = IDgrade[IDgrade.ID.isin(coins_to_keep)]
#     IDgrade[['ID','grade_lbl']].to_csv('/home/ubuntu/coin/data/IDgrade.csv')
# for i, ID in enumerate(whole_del):
#     if i%10000 == 0:
#         print i
#     os.system('rm -r /data2/processed/whole/'+ str(ID))
#     for i, ID in enumerate(cropped_del):
#         if i%10000 == 0:
#             print i
#         os.system('rm -r /data2/processed/cropped/'+ str(ID))
# def clean_coins():
#     whole_coins = set(map(lambda word:int(word), set(os.listdir('/data2/processed/whole'))))
#     cropped_coins = set(map(lambda word:int(word),set(os.listdir('/data2/processed/cropped'))))
#     coins_to_keep = whole_coins.intersection(cropped_coins)
#     IDnamegrade = pd.read_csv('/home/ubuntu/coin/data/IDnamegrade.csv')
#     IDname = pd.read_csv('/home/ubuntu/coin/data/IDname.csv')
#     IDgrade = pd.read_csv('/home/ubuntu/coin/data/IDgrade.csv')
#     csv_coins = set(pd.read_csv('/home/ubuntu/coin/data/IDnamegrade.csv')['ID'])
#     coins_to_keep = coins_to_keep.intersection(csv_coins)
#
#     for i,f in enumerate(coins_to_keep):
#         if i % 10000 == 0:
#             print i
#         x, y = '/data2/processed/whole/' + str(f)+'/img', '/data2/processed/whole/' + str(f) + '/rad'
#         if len(np.fromfile(x))!= 128*128*3+2 or len(np.fromfile(y))!= 128*128*3+2:
#             print f
#             os.system('rm -r /data2/processed/whole/' + str(f))
#         x, y = '/data2/processed/cropped/' + str(f)+'/img', '/data2/processed/cropped/' + str(f) + '/rad'
#         if len(np.fromfile(x))!= 128*128*3+2 or len(np.fromfile(y))!= 128*128*3+2:
#             print f
#             os.system('rm -r /data2/processed/cropped/' + str(f))
#

if __name__ == '__main__':
    pass
