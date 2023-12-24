import pandas as pd
import numpy as np

from constants import *

from keras.preprocessing.text import one_hot
from keras.preprocessing.sequence import pad_sequences
from keras.layers import Embedding


def read_glove(path=glove_file_path):
    word_vectors = pd.read_table(path, sep='\s', header=None, engine="python", encoding="iso-8859-1")
    word_vectors.set_index(0, inplace=True)
    return word_vectors


def glove_embedding(tokenizer):
    embedding_index = read_glove()
    embedding_matrix = np.zeros((vocab_size, embedding_dim))

    index_n_word = [(i, tokenizer.index_word[i]) for i in range(1, len(embedding_matrix))
                    if tokenizer.index_word[i] in embedding_index.index]

    idx, word = zip(*index_n_word)
    embedding_matrix[idx, :] = embedding_index.loc[word, :].values
    return embedding_matrix


def one_hot_embedding(tokenizer):
    one_hot_vector = [one_hot(words, (len(tokenizer.word_counts)+1)) for words in tokenizer.word_index.keys()]
    embedded_docs = pad_sequences(one_hot_vector, maxlen=max_text_length, padding="pre")
    return embedded_docs


def build_embeddings(tokenizer):
    if embedding_type == "glove":
        embedding_matrix = glove_embedding(tokenizer)
        embedding_layer = Embedding(input_dim=vocab_size, output_dim=embedding_dim, input_length=max_text_length,
                                   weights=[embedding_matrix], trainable=False)
    else:
        embedding_matrix = one_hot_embedding(tokenizer)
        embedding_layer = Embedding(input_dim=vocab_size, output_dim=embedding_dim, input_length=max_text_length,
                                   weights=[embedding_matrix], trainable=False)

    return embedding_layer
