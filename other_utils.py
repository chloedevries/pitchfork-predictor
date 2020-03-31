import pandas as pd
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score, confusion_matrix

def formatted_confusion_matrix(actual, predicted):
    return pd.DataFrame(
        confusion_matrix(actual, predicted),
        columns=['predicted false', 'predicted true'],
        index=['actual false', 'actual true'])

def print_performance_metrics(actual, predicted):
    print('f1: ', f1_score(actual, predicted).round(3))
    print('accuracy: ', accuracy_score(actual, predicted).round(3))
    print('precision: ', precision_score(actual, predicted).round(3))
    print('recall: ', recall_score(actual, predicted).round(3))
    print(formatted_confusion_matrix(actual, predicted))