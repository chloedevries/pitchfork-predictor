import pandas as pd
import numpy as np

def clean_reviews_df(reviews_df):
    reviews = reviews_df.copy()

    reviews = reviews.drop('Unnamed: 0', axis=1)
    reviews.drop(reviews.loc[reviews.artist == '503 Service Unavailable'] \
                    .index, inplace=True)

    reviews['date_clean'] = np.where(reviews.date == '18 hrs ago', 'February 19 2020', reviews.date)
    reviews['date_clean'] = pd.to_datetime(reviews.date_clean)
    reviews = reviews.drop('date', axis = 1)

    reviews['artist'] = reviews['artist'].apply(
                            lambda x: str(x).rstrip().lstrip())
    reviews['artist_clean'] = reviews.artist.apply(lambda x: x.lower().replace('&', 'and'))

    reviews['score'] = reviews['score'].astype('float')
    reviews.review.fillna('', inplace=True)
    return reviews

def clean_features_df(features_df):
    features = features_df.copy()
    features = features.drop('Unnamed: 0', axis = 1)
    features = features.drop('feature_type', axis = 1)

    features['artist'] = features['artist'].apply(
                            lambda x: str(x).rstrip().lstrip())
    features['artist_clean'] = features.artist.apply(lambda x: str(x).lower().replace('&', 'and'))

    features['date_clean'] = pd.to_datetime(pd.to_datetime(features.pub_date) \
                                      .dt.strftime('%m/%d/%Y'))
    features = features.drop('pub_date', axis = 1)
    features = features.join(pd.get_dummies(features.article_type))

    return features

def gather_cumulative_review_data_by_artist(reviews):
    reviews_cumul = reviews.copy()

    # add cumulative review count
    reviews_cumul['ct'] = 1
    reviews_cumul = reviews_cumul.sort_values(['artist_clean', 'date_clean']) \
        .reset_index(drop=True)
    reviews_cumul['previous_reviews_count'] = reviews_cumul \
        .groupby('artist_clean').ct.cumsum()

    # add cumulative average score
    cumul_scores = reviews_cumul.groupby('artist_clean') \
        ['score'].expanding().mean()

    reviews_cumul = reviews_cumul.join(
        cumul_scores.reset_index().drop('artist_clean', axis=1) \
            .rename(columns={'level_1': 'index',
                             'score': 'avg_cumul_score'}) \
            .set_index('index')
    )

    reviews_cumul = reviews_cumul.drop('ct', axis=1)
    return reviews_cumul


# additional reviews_df features
def add_genre_dummies(df):
    return df.join(pd.get_dummies(df['genre']))


def add_performance_mention(df):
    df['performance_mention'] = \
        df['review'].str.contains('performance') \
        | df['review'].str.contains('sold-out') \
        | df['review'].str.contains('stage show')
    return df


def get_chicago_bands(df):
    df['chicago_based'] = pd.notnull(df['review']) \
                          & df['review'].str.contains('Chicago')
    return df


def get_reissues(df):
    df['reissue'] = pd.notnull(df['review']) \
                    & df['review'].str.contains('reissue')
    return df


def add_review_features(df):
    df_plus = df.copy()
    df_plus = add_genre_dummies(df_plus)
    df_plus = add_performance_mention(df_plus)
    df_plus = get_chicago_bands(df_plus)
    df_plus = get_reissues(df_plus)
    return df_plus


# create dataset for each year and join reviews to lineups
def group_cumul_reviews_by_year(reviews):
    reviews_dfs = {}
    for year in range(2006, 2021):
        fest_date = datetime.datetime(int(year), 7, 15)
        announcement_date = datetime.datetime(int(year), 3, 1)
        # filter reviews to those before announcement of given
        df = reviews.loc[reviews.date_clean <= announcement_date]
        # dedup by artist (so we have only the most recent review)
        df = df.sort_values('date_clean') \
            .drop_duplicates(subset='artist_clean',
                             keep="last")
        # add days-to-festival variable
        df['fest_date'] = fest_date
        df['days_to_fest'] = \
            (df['fest_date'] - df['date_clean']) / pd.Timedelta('1 day')
        df['announcement_date'] = announcement_date
        df['days_to_announcement'] = \
            (df['announcement_date'] - df['date_clean']) \
            / pd.Timedelta('1 day')
        reviews_dfs[year] = df
    return reviews_dfs


def group_articles_by_year(articles):
    articles_dfs = {}
    article_types = articles.article_type.unique().tolist()
    article_types.remove(np.nan)

    for year in range(2006, 2020):
        fest_date = datetime.datetime(int(year), 7, 15)
        announcement_date = datetime.datetime(int(year), 3, 1)
        # filter features to those before announcement of given festival
        df = articles.loc[pd.to_datetime(articles.date_clean)
                          <= announcement_date]
        # add days-to-festival variable
        df['fest_date'] = fest_date
        df['days_to_fest'] = \
            (df['fest_date'] - df['date_clean']) / pd.Timedelta('1 day')

        df['announcement_date'] = announcement_date
        df['days_to_announcement'] = \
            (df['announcement_date'] - df['date_clean']) \
            / pd.Timedelta('1 day')
        df = df.loc[df.days_to_announcement <= 365]

        # aggregate to artist level with information about
        # every review within the past year
        df_agg = df.sort_values('date_clean') \
            .groupby(['artist_clean', 'fest_date']).agg({
            'date_clean': 'max', 'days_to_announcement': 'min',
            'special_feature': 'max', 'artist': 'count',
            'title': 'last', 'article': 'last'}) \
            .join(
            df[['artist_clean'] + article_types] \
                .groupby('artist_clean').max()
        ).reset_index()
        df_agg = df_agg.rename(columns={'artist': 'article_count'})

        df_agg = df_agg.set_index('title').join(
            pd.DataFrame(
                df_agg.groupby('title').artist_clean.count()
            ).rename(columns
                     ={'artist_clean': 'artist_count'})
        ).reset_index()

        articles_dfs[year] = df_agg
    return articles_dfs


def transform_days_column_to_months(df, column):
    months_to = round(df[column] / 30, 0)
    return months_to


def normalize_variable(df, column):
    min_val = df[column].min()
    max_val = df[column].max()
    normed_col = ((df[column] - min_val) / (max_val - min_val))

    return normed_col


def join_year_dfs(dfs_dict):
    entire_df = pd.DataFrame()
    for year in range(2006, 2020):
        entire_df = entire_df.append(dfs_dict[year])
    return entire_df


def join_lineups(df, lineups_df, how):
    lineups_df['played_fest'] = True
    joined_df = df.set_index(['artist_clean', 'fest_date']) \
        .join(lineups.set_index(['artist_clean', 'fest_date']) \
                  [['played_previous_fest', 'played_fest']],
              how=how, lsuffix='_word_vec').reset_index()
    joined_df['played_fest'].fillna(False, inplace=True)
    joined_df['played_previous_fest'].fillna(False, inplace=True)
    return joined_df


def format_next_layer_df(df, model, input_cols):
    next_df = df.copy()
    next_df['prob'] = \
        pd.DataFrame(
            model.predict_proba(
                df[input_cols])
        )[1]

    return next_df[['artist_clean', 'fest_date', 'prob']]