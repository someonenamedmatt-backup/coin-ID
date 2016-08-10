import pandas as pd
from coin import Coin
import os
from multiprocessing import Manager
import multiprocessing as mp
import cPickle as pickle
import numpy as np
bad_coins = Manager().list()

def make_coins(input_folder, output_folder, bad_coin_file, csv_file=None, find_circle=False, size=(128,128)):
    #goes through a directory and makes a coin object for every file in input folder intersect csv_file (optional)
    #find_circles and size let you choose if you want to crop the image using find_circles and what size you want
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    file_list = os.listdir(input_folder)
    if csv_file is not None:
        df = pd.read_csv(csv_file)
        file_list = list(set(file_list).intersection(set(df['ID'].map(lambda id: str(id)+ '.jpg'))))
    file_list = map(lambda id: (input_folder + id,find_circle,size, output_folder),file_list)
    pool = mp.Pool(mp.cpu_count())
    pool.map(make_coin,file_list)
    pool.close()
    pool.join()
    print len(list(bad_coins))

def make_coin((file, find_circle, size, output_folder)):
    #takes file and options and makes a single coin in output folder
    #takes them as a tuple for multiprocessing to work properly
    if os.path.exists('output_folder+coin_name[:-4]'):
        return True
    try:
        c = Coin().make_from_image(file, find_circle, size)

        if type(c) == bool:
            bad_coins.append(file.split('/')[-1][:-4])
        else:
            coin_name = file.split('/')[-1]
        try:
            c.save(output_folder+coin_name[:-4])
        except:
            bad_coins.append(file.split('/')[-1][:-4])
    except:
        bad_coins.append(file.split('/')[-1][:-4])

def convert_coins(folder):
    #goes through a folder and makes all the coins in it into the new coin format
    coins = filter(lambda str: 'npy' in str, os.listdir(folder))
    coins = map(lambda str: folder + str[:-4], coins)
    pool = mp.Pool(mp.cpu_count())
    pool.map(convert_coin,folder)
    pool.close()
    pool.join()
    print len(list(bad_coins))

def convert_coin(f):
    try:
        coin = Coin().load(f + '.npy')
        os.mkdir(f)
        np.save(f + '/img', coin.img)
        np.save(f + '/rad', coin.rad)
        os.remove(f + '.npy')
    except:
        bad_coins.append(f.split('/')[-1])
        raise


def clean_csv(csv_file, output_file):
    #takes the csv file from the scraper and cleans it up
    #finds PCGA Grades and makes grade_lbl
    #saves outputfile.csv and outputfile_full.csv
    def get_numbers(str):
        return [int(s) for s in str if s.isdigit()]
    def hasNumbers(str):
        return any(char.isdigit() for char in str)
    def get_grade(lst):
        for i,item in enumerate(lst[:-1]):
            if 'Grade'in item:
                num_lst = get_numbers(lst[i+1])
                if len(num_lst)>2:
                    return []
                else:
                    try:
                        return int(''.join(map(str,num_lst)))
                    except:
                        pass
        return []
    def make_label(grade):
        if grade <= 20:
            return 0
        elif grade <= 50:
            return 1
        elif grade <= 62:
            return 2
        else:
            return 3

    non_grades = [5,7,16]
    df = pd.read_csv(csv_file, index_col = 0)
    df['categories']=map(lambda str: eval(str.replace(';',',')),df['Ebay Categories'])
    df['attributes']=map(lambda str: eval(str.replace(';',',')),df['Attributes'])
    #Filter out collections
    df = df[map(lambda lst: 'Collections' not in lst[2]  ,df['categories'])]
    df = df[map(lambda lst: 'World' not in lst[1]  ,df['categories'])]
    df['Grade']=map(get_grade,df['attributes'])
    df = df[map(lambda thing: type(thing)==int ,df['Grade'])]
    df = df[df['Grade']!=0]
    df = df[(~df['Grade'].isin(non_grades))&(df['Grade']<=70)]
    df['grade_lbl'] = df['Grade'].map(make_label)
    # bad_coin_list = map(lambda str: int(str), filter(lambda str: len(str) > 0, list(bad_coins)))
    # df=df[not df['ID'].isin(set(bad_coin_list))]
    df[['ID','grade_lbl']].to_csv(output_file+'.csv')
    df[['ID','grade_lbl']].to_csv(output_file+'full.csv')

def convert_coins_bin():
    #goes through a folder and makes all the coins in it into the new coin format
    coins1 = map(lambda str: '/data2/processed/cropped/' + str + '/', os.listdir('/data2/processed/cropped'))
    coins2 = map(lambda str: '/data2/processed/whole/' + str + '/', os.listdir('/data2/processed/whole'))
    coins = coins1 + coins2
    pool = mp.Pool(mp.cpu_count())
    pool.map(convert_coin_bin, coins)
    pool.close()
    pool.join()

def convert_coin_bin(f):
    try:
        if os.path.exists(f+'img'):
            np.load(f + 'img.npy').tofile(f+'img')
            np.load(f + 'rad.npy').tofile(f+'rad')
            os.remove(f + 'img.npy')
            os.remove(f + 'rad.npy')
    except:
        pass

def add_label():
    df = pd.read_csv('/home/ubuntu/coin/data/IDlabel.csv').set_index('ID')['grade_lbl'].to_dict()
    for subdir, dirs, files in os.walk('/data2/processed'):
        for name in files:
            if name in ['img', 'rad']:
                label = df[int(subdir.split('/')[-1])]
                value = np.fromfile(os.path.join(subdir,name))
                if len(value) == 128*128*3:
                    np.append(value,label).tofile(name)
            elif name in ['img.npy', 'rad.npy']:
                label = df[int(subdir.split('/')[-1])]
                value = np.load(os.path.join(subdir,name)).flatten()
                np.append(value,label).tofile(name[:-4])




if __name__ == '__main__':
    pass
