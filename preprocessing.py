import nltk
import re

from collections import Counter
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn.preprocessing import LabelEncoder

ps = PorterStemmer()
stop_words = stopwords.words('english')
stop_words_dict = Counter(stop_words)


def merge_features(df, text_features, col="news"):
    df[col] = df[text_features].agg(' '.join, axis=1)
    return df


def clean_text(text):
    text = str(text).replace(r'http[\w:/\.]+', ' ')
    text = str(text).replace(r'[^\.\w\s]', ' ')
    text = str(text).replace('[^a-zA-Z]', ' ')
    text = str(text).replace(r'\s\s+', ' ')
    text = text.lower().strip()
    return text


def nltk_preprocessing(text):
    text = ' '.join([word for word in text.split() if word not in stop_words_dict])
    return text


def process_labels(labels):
    if labels.dtype == 'object':
        label_encoder = LabelEncoder()
        labels = label_encoder.fit_transform(labels)
    return labels


def preparing_data(df, text_features):
    data = df.copy()
    data = merge_features(data, text_features)
    data['news'] = data['news'].apply(clean_text)
    data['news'] = data['news'].apply(nltk_preprocessing)
    X = data['news']
    y = data.label
    if y.dtype == 'object':
        y = process_labels(y)
    return X,y
