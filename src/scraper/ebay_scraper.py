from find_circles import CircleImage
import urllib
import pandas as pd
import urllib2
import sys
import os
import threading
from bs4 import BeautifulSoup
import numpy as np
import requests
import multiprocessing as mp
import re
from make_url_list import make_url_csv

d = mp.Manager().list()
k = 50

def clean_link_df(df):
    url_lst = filter(lambda str: 'http' in str, df['url'])
    return list(set(url_lst))

def get_listing_info(link):
    soup = BeautifulSoup(requests.get(link).text, 'html.parser')
    try:
        title = soup.find('h1').text
        categories = soup.findAll(text=True, attrs = { 'itemprop':'name'})
        #find ebay classification of coin type
        categories = str([item.text for item in categories]).replace(',',';')
        # use semicolons for lists that will be one column in df (for now)
        table = soup.find( attrs = {'class':'section'})
        #other ebay attributes
        attributes = [re.sub('\s+', '',col.text) for row in table.findAll('tr') for col in row.findAll('td')]
        attributes = str(attributes).replace(',',';')
    except:
        print "{} can't find the table".format(link)
        raise
    try:
        imgurl_list = get_img_url_from_soup(soup)
    except:
        print "{} has no images".format(link)
        raise
    return [title,categories,attributes],imgurl_list

def get_img_url_from_soup(soup):
    img_mess = soup.findAll(attrs = {'class':"tdThumb"})
    urls = map(lambda soup_tag: soup_tag.find('img')['src'],img_mess)
    resized_urls = map(lambda url: url[:url.find('s-l')]+'s-l300.jpg',urls)
    return resized_urls

def write_listing_info(link, id_num):
    coin_info, imgurl_list = get_listing_info(link)
    for i in range(len(imgurl_list)):
        d.append([id_num+1000000*i]+coin_info)
        write_img_concurrent(imgurl_list,id_num)

def download_and_crop((url,img_num)):
    whole_img_name = '/data/whole/'+str(img_num)+'.jpg'
    cropped_img_name = '/data/cropped/'+str(img_num)+'.jpg'
    urllib.urlretrieve(url,whole_img_name)
    circle = CircleImage(whole_img_name)
    try:
        circle.find_circle()
    except ValueError:
        print url
        raise
    circle.write_cropped(cropped_img_name)


def write_img_concurrent(imgurl_list,id_num):
    jobs = []
    for i, url in enumerate(imgurl_list):
        img_num = 1000000*i+id_num
        thread = threading.Thread(name=i, target=download_and_crop, args=((url,img_num), ))
        jobs.append(thread)
        thread.start()
    for j in jobs:
        j.join()

def write_k_concurrent(start_index, content,k):
    #scrapes the first k elements of content
    #writes the title of the page and the idnum (start_index + i)
    #saves the images to a folder labeleld wit the id_num
    jobs = []
    for i in range(min(len(content),k)):
        link = content[i]
        write_1 = lambda i: write_listing_info(link, i+start_index)
        thread = threading.Thread(name=i, target=write_1, args=(i, ))
        jobs.append(thread)
        thread.start()
    for j in jobs:
        j.join()
    if start_index%1000 == 0:
        print "At file number {}".format(start_index)

def make_k_bunch(lst):
    #takes a list and returns a list of tuples
    #each tuple contains the starting index of the list
    # and a list each with k elements
    return [(i,lst[i:i+k]) for i in xrange(0,len(lst),k)]

def write(lst):
    write_k_concurrent(lst[0],lst[1],k)

def scrape(links):
    pool = mp.Pool(mp.cpu_count())
    k_bunched_links = make_k_bunch(links)
    pool.map(write,k_bunched_links)
    pool.close()
    pool.join()


if __name__ == '__main__':
    pass
    df = pd.read_csv('urls.csv')
    links_to_scrape = clean_link_df(df)
    scrape(links_to_scrape)
    df = pd.DataFrame(list(d),columns = ['ID','Name', 'Ebay Categories', 'Attributes'] )
    df.to_csv('ebay_coin_data_v3.csv',encoder='utf-8')
