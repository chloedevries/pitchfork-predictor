import numpy as np
import requests
from bs4 import BeautifulSoup
import pandas as pd
import datetime
import scrapefork

from scrapefork import *

# function taken from https://github.com/evanm31/p4k-scraper/blob/master/data/scrapefork.py, edits made
def gather_links(pages, startPage):
    pageList = [] #list of album review pages
    linkList = [] #list of album links
    for x in range(startPage,(startPage+pages)): #check the first n pages after the requested one
        pageList.append(requests.get("https://pitchfork.com/reviews/albums/?page=" + str(x))) #add each page to list
    for page in pageList:
        soup = BeautifulSoup(page.content, 'html.parser') #parse its contents
        links = soup.find_all(class_="review__link") #gather its links (in raw html)
        for link in links: #for each link
            linkList.append(link.get('href')) #append only the link itself
    return linkList

def pull_a_lot_of_reviews(pages, chunk_size, fileLocation, fileName,
                       page_start = 1):
    cols = ['artist', 'album', 'score', 'genre', 'review', 'best', 'date']
    master_df = pd.DataFrame(columns = cols)
    page_count = chunk_size
    for i in np.arange(chunk_size, pages + 1, chunk_size):
        data = gather(page_count, page_start, 'data', 'pitchfork_reviews')
        master_df = master_df.append(data)
        page_start += chunk_size
        print(f'chunk {i}/{int(pages)} complete')

    master_df.to_csv(f'{fileLocation}/{fileName}.csv')
    master_df = master_df.reset_index()
    return master_df