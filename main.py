import numpy as np
import requests
from bs4 import BeautifulSoup
import pandas as pd
import datetime
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score, confusion_matrix
from sklearn.ensemble import RandomForestClassifier

from pitchfork_predictor.create_pitchfork_lineups_data import *
from pitchfork_predictor.cleaning_functions import *
from pitchfork_predictor.other_utils import *

def predict_by_top_probabilities(probability_preds, full_df):
    percent_true = full_df.played_fest.mean()
    number_true = pd.DataFrame(probability_preds)[1].shape[0] * percent_true
    true_prediction_cutoff = pd.DataFrame(probability_preds)[1].sort_values().tail(int(number_true)).min()

    reviews_preds_by_prob = (pd.DataFrame(probability_preds)[1] >= true_prediction_cutoff)
    return reviews_preds_by_prob

# import & clean datasets
# lineups
filepath = 'pitchfork_predictor/data/raw_p4k_lineups.txt'
lineups = get_lineups_data(filepath)

# reviews
reviews_df = pd.read_csv('pitchfork_predictor/data/pitchfork_reviews.csv')
reviews = clean_reviews_df(reviews_df)

# features
articles_df = pd.read_csv('pitchfork_predictor/data/pitchfork_features_with_article_txt.csv')
articles = clean_features_df(articles_df)
print('data uploaded')


# first, train the reviews model
# formatting -
reviews_cumul = gather_cumulative_review_data_by_artist(reviews)
reviews_cumul_final = add_review_features(reviews_cumul)
reviews_dfs = group_cumul_reviews_by_year(reviews_cumul_final)

# normalize numeric variables
reviews_full_df = join_year_dfs(reviews_dfs)

reviews_full_df['score_norm'] \
    = normalize_variable(reviews_full_df, 'score')

reviews_full_df['avg_cumul_score_norm'] \
    = normalize_variable(reviews_full_df, 'avg_cumul_score')

reviews_full_df['days_to_announcement_norm'] \
    = normalize_variable(reviews_full_df, 'days_to_announcement')

# interaction terms
reviews_full_df['score_x_days'] = reviews_full_df.score_norm \
                                  * reviews_full_df.days_to_announcement_norm
reviews_full_df['reissue_x_days'] = reviews_full_df.reissue \
                                    * reviews_full_df.days_to_announcement_norm
reviews_full_df['cumul_score_x_days'] = reviews_full_df.avg_cumul_score_norm \
                                        * reviews_full_df.days_to_announcement_norm
reviews_full_df['chicago_x_days'] = reviews_full_df.chicago_based \
                                    * reviews_full_df.days_to_announcement_norm
reviews_full_df['performance_x_days'] = reviews_full_df.performance_mention \
                                        * reviews_full_df.days_to_announcement_norm
reviews_full_df['best_x_days'] = reviews_full_df.best \
                                 * reviews_full_df.days_to_announcement_norm
reviews_full_df['chicago_x_score'] = reviews_full_df.chicago_based \
                                     * reviews_full_df.score_norm
reviews_full_df['performance_x_score'] = reviews_full_df.performance_mention \
                                         * reviews_full_df.score_norm
reviews_full_df['chicago_x_performance'] = reviews_full_df.chicago_based \
                                           * reviews_full_df.performance_mention
print('reviews data munged')

# tf-idf vectorization of reviews texts
from sklearn.feature_extraction.text import TfidfVectorizer

tfidf_vectorizer = TfidfVectorizer(stop_words = 'english', min_df=100)

X_reviews = tfidf_vectorizer.fit_transform(reviews_full_df['review']) \
                             .toarray()

# X_reviews.tofile("saved_features/x_reviews.txt")
# X_reviews = np.fromfile("saved_features/x_reviews.txt")

reviews_frequency_matrix = pd.DataFrame(X_reviews,
                                        columns = tfidf_vectorizer.get_feature_names())
reviews_frequency_matrix.to_csv("pitchfork_predictor/saved_features/reviews_frequency_matrix_mindf100.csv")
print('tf-df done')
print(reviews_frequency_matrix.shape)

#reviews_frequency_matrix = pd.read_csv("pitchfork_predictor/saved_features/reviews_frequency_matrix_mindf100.csv")
#print("loaded frequency matrix")
#print(reviews_frequency_matrix.shape)

# add new tf-idf features to df
reviews_full_full_df = reviews_full_df.reset_index(drop=True) \
                            .join(reviews_frequency_matrix,
                                  lsuffix='_og')

# join in y var
reviews_model_df = join_lineups(reviews_full_full_df, lineups, 'left')


# train reviews sub-model
reviews_inputs = ['score_norm', 'avg_cumul_score_norm',
                  'best', 'previous_reviews_count',
                  'performance_mention', 'chicago_based', 'reissue',
                  'days_to_announcement_norm',
                  'Rock', 'Rap', 'Jazz', 'Experimental', 'Pop/R&B',
                  'Electronic', 'Metal', 'Global', 'Folk/Country',
                  'score_x_days', 'reissue_x_days',
                  'cumul_score_x_days', 'chicago_x_days',
                  'performance_x_days', 'best_x_days',
                  'chicago_x_score', 'performance_x_score',
                  'chicago_x_performance'] \
                    + reviews_frequency_matrix.columns.tolist()
                    #+ tfidf_vectorizer.get_feature_names()

output = ['played_fest']

from sklearn.model_selection import StratifiedShuffleSplit

X = reviews_model_df[reviews_inputs]
y = reviews_model_df[output]

sss = StratifiedShuffleSplit(n_splits=2, test_size=0.33, random_state=42)
sss.split(X, y)

for train_index, test_index in sss.split(X, y):
    X_train, X_test = pd.DataFrame(np.asarray(X)[train_index], columns = reviews_inputs), \
                            pd.DataFrame(np.asarray(X)[test_index], columns = reviews_inputs)
    y_train, y_test = pd.DataFrame(np.asarray(y)[train_index], columns = output), \
                            pd.DataFrame(np.asarray(y)[test_index], columns = output)

print("created training and testing groups")

reviews_model = RandomForestClassifier()
reviews_model.fit(X_train, y_train)
print("fit model")

# look at basic performance
reviews_training_preds = reviews_model.predict(X_train)
reviews_training_prob_preds = reviews_model.predict_proba(X_train)

print("training dataset:")
print("from model:")
print_performance_metrics(y_train,
                          reviews_training_preds)
print("by probabilities:")
reviews_training_preds_by_prob = predict_by_top_probabilities(reviews_training_prob_preds, reviews_model_df)
print_performance_metrics(y_train, reviews_training_preds_by_prob)

reviews_preds = reviews_model.predict(X_test)
reviews_prob_preds = reviews_model.predict_proba(X_test)

print("test dataset:")
print("from model:")
print_performance_metrics(y_test, reviews_preds)
print("by probabilities:")
reviews_preds_by_prob = predict_by_top_probabilities(reviews_prob_preds, reviews_model_df)
print_performance_metrics(y_test, reviews_preds_by_prob)


import sys
sys.exit('break')


# next, the articles model
# formatting -
articles_dfs = group_articles_by_year(articles)

# join into full df
articles_full_df = join_year_dfs(articles_dfs)
# normalize date var
articles_full_df['days_to_announcement_norm'] \
    = normalize_variable(articles_full_df, 'days_to_announcement')

# tf-idf vectorization of articles texts
tfidf_vectorizer = TfidfVectorizer(stop_words = 'english')

X_article = tfidf_vectorizer.fit_transform(articles_full_df['article']).toarray()

articles_frequency_matrix = pd.DataFrame(X_article, columns
                                = tfidf_vectorizer.get_feature_names())
articles_frequency_matrix.to_csv('saved_features/articles_frequency_matrix.csv')

articles_full_full_df = articles_full_df.reset_index(drop=True) \
                    .join(articles_frequency_matrix, lsuffix='_article')

# join y col
articles_model_df = join_lineups(articles_full_full_df,
                                 lineups, 'left')

# train articles sub-model
articles_training_set = articles_model_df.loc[
                        articles_model_df.fest_date != '2019-07-15']
articles_test_set = articles_model_df.loc[
                        articles_model_df.fest_date == '2019-07-15']

articles_inputs = ['months_to_announcement_norm', 'artist_count',
                   'Interview', 'Moodboard', 'Rising', 'Song by Song',
                   '5-10-15-20', 'Longform', 'Profile', 'Lists & Guides',
                   'Photo Gallery', 'Podcast', 'Family Matters',
                   'Overtones', 'Festival Report', 'Cover Story',
                   'Afterword', 'Situation Critical', 'Director\'s Cut'] \
                    + tfidf_vectorizer.get_feature_names()

output = ['played_fest']

X_train = articles_training_set[articles_inputs]
y_train = articles_training_set[output].astype('int')
X_test = articles_test_set[articles_inputs]
y_test = articles_test_set[output].astype('int')

articles_model = RandomForestClassifier()
articles_model.fit(X_train, y_train)

# basic performance
articles_training_preds = articles_model.predict(X_train)
print("training dataset:")
print_performance_metrics(y_train, articles_training_preds)

articles_preds = articles_model.predict(X_test)
print("test dataset:")
print_performance_metrics(y_test, articles_preds)


# next, output dataframes for next model layer
reviews_layer_output = format_next_layer_df(reviews_model_df,
                                            reviews_model,
                                            reviews_inputs)

articles_layer_output = format_next_layer_df(articles_model_df,
                                             articles_model,
                                             articles_inputs)

# then join them together & join in lineups
model_df = reviews_layer_output \
                .set_index(['artist_clean', 'fest_date']) \
            .join(articles_layer_output
                      .set_index(['artist_clean', 'fest_date']),
                  how = 'outer', lsuffix = '_reviews',
                  rsuffix = '_articles').reset_index()

full_model_df = join_lineups(model_df, lineups, 'outer')
full_model_df.fillna(0, inplace=True)
full_model_df['review_x_article'] = full_model_df.prob_reviews \
                                        * full_model_df.prob_articles

# final model
training_set = full_model_df.loc[
                    full_model_df.fest_date != '2019-07-15']
test_set = full_model_df.loc[
                    full_model_df.fest_date == '2019-07-15']

inputs = ['prob_reviews', 'prob_articles', 'played_previous_fest',
          'review_x_article']
ycol = ['played_fest']

full_model = RandomForestClassifier()
full_model.fit(training_set[inputs], training_set[ycol])

preds = full_model.predict(test_set[inputs])
prob_preds = full_model.predict_proba(test_set[inputs])
print_performance_metrics(test_set[ycol], preds)