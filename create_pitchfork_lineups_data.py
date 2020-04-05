import numpy as np
import requests
from bs4 import BeautifulSoup
import pandas as pd
import datetime

# cleans & creates df from raw text file

def parse_lineup_data_string(lineups_raw):
    lineups_raw = lineups_raw.splitlines()
    for line in lineups_raw:
        if line == '':
            lineups_raw.remove(line)

    lineups_dict = {}
    for line in lineups_raw:
        year, artists = line.split(': ')
        lineups_dict[year] = artists

    for key, item in lineups_dict.items():
        artist_list = item.split(',')
        lineups_dict[key] = artist_list

    lineups_df = pd.DataFrame(columns = ['year', 'artist'])
    for key, item in lineups_dict.items():
        for artist in item:
            df_row = pd.DataFrame({'year':[key], 'artist':[artist]})
            lineups_df = lineups_df.append(df_row)

    return lineups_df

def count_previous_appearances(lineups):
    lineups['ct'] = 1
    lineups.sort_values(['artist_clean', 'fest_date'], inplace=True)
    lineups['previous_fest_count'] = lineups \
                                    .groupby('artist_clean').ct.cumsum()
    lineups = lineups.drop('ct', axis = 1)
    lineups['played_previous_fest'] = lineups.previous_fest_count > 1
    return lineups

def get_lineups_data(filepath):
    with open(filepath, 'r') as file:
        lineups_raw = file.read()

    lineups_df = parse_lineup_data_string(lineups_raw)
    lineups_df['year'] = lineups_df.year.astype('int')
    lineups_df.artist = lineups_df.artist.str.strip(' ')
    lineups_df['fest_date'] = lineups_df['year'] \
                .apply(lambda x: datetime.datetime(int(x), 7, 15))
    lineups_df['artist_clean'] = lineups_df.artist \
                    .apply(lambda x: x.lower().replace('&', 'and'))

    lineups_df = lineups_df.reset_index()
    lineups_df = lineups_df.drop('index', axis=1)
    lineups_df = count_previous_appearances(lineups_df)

    return lineups_df
