from constants import *
from keras.preprocessing.text import Tokenizer


def build_tokenizer(df, num_words=vocab_size):
    if num_words is None:
        tokenizer = Tokenizer(oov_token=oov_token)
    else:
        tokenizer = Tokenizer(oov_token=oov_token, num_words=num_words)

