import urllib
import urllib2
import sys
import os
import pandas as pd
import threading
from bs4 import BeautifulSoup
import numpy as np
import requests
import multiprocessing as mp
ebay_jargon = '?_fsrp=1&_trkparms=pageci%3D0a207eed-6535-467e-a540-1a0f99e619a0%2Cparentrq%3D4d9966fc1560abd824e3a2c1ffffeeee%2Ciid%3D1%2Cobjectid%3D173593&_dcat=173593&Certification=PCGS&rt=nc&_pgn='
coin_start_page = 'http://www.ebay.com/rpp/coins-us?_fsrp=2'

d = mp.Manager().list()
#### Not using below right now
search_terms = ['MS%252070', 'MS%252069', 'MS%252068', 'MS%252067',
'MS%252066', 'MS%252065', 'MS%252064', 'MS%252063',
'MS%252062', 'MS%252061', 'MS%252060', 'PR%252070',
 'PR%252069', 'PR%252068', 'PR%252067', 'PR%252066',
  'PR%252065', 'PR%252064', 'PR%252063', 'PR%252062',
   'PR%252061', 'AU%252058', 'AU%252055', 'AU%252053',
    'AU%252050', 'XF%252045', 'XF%252040', 'VF%252035',
     'VF%252030', 'VF%252025', 'VF%252020', 'F%252015',
     'F%252012', 'VG%252010', 'VG%25208', 'G%25206', 'G%25204',
      'AG%25203', 'FA%25202', 'P%25201']

def get_starting_pages():
    #searches starting at coin_start_page for coin subcategories
    #Makes a list of each cateogories page (limited to PCGA certed coins)
    searches = []
    soup = BeautifulSoup(requests.get(coin_start_page).text,'html.parser')
    coin_pages = soup.findAll('li', {'data-node-id':True})[1:-7]
    coin_pages = [item.find('a').get('href') for item in coin_pages]
    for i, url in enumerate(coin_pages):
        if i in [3,8,12,14, 15]:
            searches.append(url)
        else:
            subsoup = BeautifulSoup(requests.get(url).text,'html.parser')
            subcoin = subsoup.findAll('div', {'class':'cat-link'})[:-1]
            subcoin = [item.find('a').get('href') for item in subcoin]
            searches.extend(subcoin)
    return searches

def build_list_of_links(link):
     page = requests.get(link).text
     soup = BeautifulSoup(page,'html.parser')
     list_of_links = []
     for item in soup.findAll('a', {'class':'vip'}):
         list_of_links.append(item.get('href'))
     return(list_of_links)


def make_url_generator(starting_link):
    i = 0
    link = lambda i: starting_link+ebay_jargon + str(i)
    while True:
        i += 1
        yield link(i)

def search_one_page(url):
    link_generator = make_url_generator(url)
    link = link_generator.next()
    temp_link_list = build_list_of_links(link)
    link_list = temp_link_list[:]
    while  len(temp_link_list) >0:
        link = link_generator.next()
        temp_link_list = build_list_of_links(link)
        link_list.extend(temp_link_list)
        if len(link_list)>10000:
            break
    print "Finished {}".format(url)
    d.extend(link_list)

def make_url_csv():
    print "Making URL List"
    search_terms =  get_starting_pages()
    jobs = []
    for term in search_terms:
        thread = threading.Thread(name=term, target=search_one_page, args=(term, ))
        jobs.append(thread)
        thread.start()
    for j in jobs:
        j.join()
    print "Completed"
    df = pd.DataFrame(list(d),columns = ['url'])
    df.to_csv('/home/ubuntu/coin/urls.csv',encoder='utf-8')
    return df



if __name__ == '__main__':
    pass
