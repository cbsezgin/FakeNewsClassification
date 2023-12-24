from utils import read_dataframe, text_stat, clean_data
from constants import *
from preprocessing import preparing_data
from tokenizer import build_tokenizer, save_tokenizer
from embedding import build_embeddings
from models import *

from keras.preprocessing.sequence import pad_sequences


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

tokenizer, word_index = build_tokenizer(X_train)
save_tokenizer(tokenizer)

train_text_seq = tokenizer.texts_to_sequences(X_train)
test_text_seq = tokenizer.texts_to_sequences(X_test)

train_text_pad = pad_sequences(train_text_seq, maxlen=max_text_length)
test_text_pad = pad_sequences(test_text_seq, maxlen=max_text_length)

embedding_layer = build_embeddings(tokenizer)

model_rnn = build_RNN_model(embedding_layer)
model_rnn, history = train_model(model_rnn, train_text_pad, y_train, test_text_pad, y_test)
store_model(model_rnn, filepath="/output/model_rnn", filename="model_rnn")
score = model_rnn.evaluate(X_test, y_test, verbose=0)

