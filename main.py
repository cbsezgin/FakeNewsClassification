from utils import read_dataframe, text_stat, clean_data
from constants import *
from preprocessing import preparing_data

news = read_dataframe("data/train.csv")
test = read_dataframe("data/test.csv")
submit = read_dataframe("data/submit.csv")
test["label"] = submit.label

text_stats = text_stat(news, "text")
title_stats = text_stat(news, "title")

df = news.copy()

df = clean_data(df, cols=remove_cols)
df_test = clean_data(test, cols=remove_cols)

X_train,y_train = preparing_data(df, text_features=text_features)
X_test, y_test = preparing_data(df_test, text_features=text_features)





