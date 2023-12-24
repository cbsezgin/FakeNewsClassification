from constants import *
from keras.preprocessing.text import Tokenizer

import io
import json


def build_tokenizer(df, num_words=vocab_size):
    if num_words is None:
        tokenizer = Tokenizer(oov_token=oov_token)
    else:
        tokenizer = Tokenizer(oov_token=oov_token, num_words=num_words)

    tokenizer.fit_on_texts(df)
    word_index = tokenizer.word_index
    return tokenizer, word_index


def save_tokenizer(tokenizer, num_words=vocab_size, model_dir='/output/models', filename=None):
    if filename is None:
        filepath = model_dir + 'tokenizer_' + str(num_words) + '.json'
    else:
        filepath = model_dir + filename

    with io.open(filepath, 'w', encoding='utf-8') as outfile:
        outfile.write(json.dumps(tokenizer.to_json(), ensure_ascii=False))
    outfile.close()





