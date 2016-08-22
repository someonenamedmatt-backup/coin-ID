'''
Call make_url_csv to run
It works by using the ebay page for "us coins"
and breaks it down by subcategories then finding the search results for all subcategories
'''

import pandas as pd
import threading
from bs4 import BeautifulSoup
import numpy as np
import requests
import multiprocessing as mp

coin_start_page = 'http://www.ebay.com/sch/US-Coins/253/bn_1848949/i.html'
ebay_jargon = '?_fsrp=1&_trkparms=pageci%3D9b814cfc-b615-4dbe-a518-a90728a15a9e%2Cparentrq%3Db3a55ed81560a860e4136df0fffc7d78%2Ciid%3D1%2Cobjectid%3D173591&_skc=50&rt=nc&_pgn=2'
d = mp.Manager().list()

def make_url_csv(filename):
    '''
    Creates a csv file of URLs which list ebay listings
    INPUT: filename (full path) to save the output
    OUTPUT: saves a csv of all listings
    '''
    print "Making URL List"
    search_terms =  _get_starting_pages()
    jobs = []
    for term in search_terms:
        thread = threading.Thread(name=term, target=_search_one_page, args=(term, ))
        jobs.append(thread)
        thread.start()
    for j in jobs:
        j.join()
    print "Completed"
    df = pd.DataFrame(list(d), columns = ['url'])
    df.to_csv(filename, encoder='utf-8')
    return df


def _get_starting_pages():
    '''
    Searches through all subcategories to get the base urls of types of coins
    INPUT: NONE
    OUTPUT: A list of strings each of which is the start of a url that is iterable with pages
    '''
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

def _build_list_of_links(link):
    '''
    Finds all the ebay listing from a given page
    INPUT: url of an ebay search page
    OUTPUT: List containing the urls of all ebay listings from a given search page
    '''
     page = requests.get(link).text
     soup = BeautifulSoup(page,'html.parser')
     list_of_links = []
     for item in soup.findAll('a', {'class':'vip'}):
         list_of_links.append(item.get('href'))
     return(list_of_links)


def _make_url_generator(starting_link):
    '''
    Creates a generator of additional pages of ebay listings
    INPUT: string containing the start of a category of ebay listings
    OUTPUT: generator which iterates through pages of the search
    '''
    i = 0
    link = lambda i: starting_link+ebay_jargon + str(i)
    while True:
        i += 1
        yield link(i)

def _search_one_page(url):
    '''
    Finds all the URLS for a given base search
    INPUT: base url (string), multiprocessing list in namespace
    OUTPUT: extends multiprocessed list of strings
    '''
    link_generator = _make_url_generator(url)
    link = link_generator.next()
    temp_link_list = _build_list_of_links(link)
    link_list = temp_link_list[:]
    while  len(temp_link_list) >0:
        link = link_generator.next()
        temp_link_list = _build_list_of_links(link)
        link_list.extend(temp_link_list)
        if len(link_list)>10000:
            break
    print "Finished {}".format(url)
    d.extend(link_list)





if __name__ == '__main__':
    pass
