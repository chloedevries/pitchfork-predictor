import requests
from bs4 import BeautifulSoup
import pandas as pd
import re
import json
import pitchfork_predictor.scrapefork

# function from https://github.com/evanm31/p4k-scraper/blob/master/data/scrapefork.py, edits made
def gather_links(pages, startPage):
    pageList = [] #list of album review pages
    linkList = [] #list of album links
    for x in range(startPage,(startPage+pages)): #check the first n pages after the requested one
        pageList.append(requests
                        .get("https://pitchfork.com/features/?page="
                             + str(x))) #add each page to list
    for page in pageList:
        soup = BeautifulSoup(page.content, 'html.parser') #parse its contents
        links = soup.find_all(class_="title-link module__title-link") #gather its links (in raw html)
        for link in links: #for each link
            url = link.get('href')
            url = re.sub('^(.*?)pitchfork.com', '', url)
            linkList.append(url) #append only the link itself
    return linkList

def gather_features(pages, start_page):
    linkList = gather_links(pages, start_page)
    results = pd.DataFrame()
    for link in linkList:
        if str(link)[0:10] == '/features/':
            page = requests.get(f'https://pitchfork.com{link}')
            soup = BeautifulSoup(page.content, 'html.parser')
            # save article text
            sents = [element.text for element in soup.find_all('p')]
            article_string = " ".join(sents)
            for x in soup.find_all('script'):
                t = x.get_text()
                if t[0:11] == 'window.App=':
                    json_text = json.loads(t[11:-1])
                    id = list(json_text['context']['dispatcher']['stores'] \
                                  ['FeaturesStore']['items'].keys())[0]
                    info_dict = json_text['context']['dispatcher']['stores'] \
                                         ['FeaturesStore']['items'][id]
                    # collect article info
                    pub_date = info_dict['pubDate']
                    article_type = info_dict['subChannel']
                    special_feature = info_dict['specialFeature']
                    title = info_dict['title']
                    # add row to df for each artist tagged in article
                    for item in json_text['context']['dispatcher']['stores']['FeaturesStore'] \
                                         ['items'][id]['artists']:
                        artist = [item.get('display_name')]
                        results = results.append(
                                    pd.DataFrame({'artist': artist,
                                                  'pub_date': pub_date,
                                                  'article_type': article_type,
                                                  'title': title,
                                                  'special_feature':special_feature,
                                                  'article':article_string}))

    return results

def pull_a_lot_of_features(pages, chunk_size, fileLocation, fileName,
                            page_start = 1):
    cols = ['artist', 'pub_date', 'article_type', 'title', 'special_feature', 'article']
    master_df = pd.DataFrame(columns = cols)
    page_count = chunk_size
    for i in np.arange(chunk_size, pages + 1, chunk_size):
        data = gather_features(page_count, page_start)
        master_df = master_df.append(data)
        page_start += chunk_size
        print(f'chunk {i}/{int(pages)} complete')

    master_df.to_csv(f'{fileLocation}/{fileName}.csv')
    master_df = master_df.reset_index()
    return master_df
