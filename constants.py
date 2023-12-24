remove_cols = ["id", "author"]
text_features = ["title", "text"]

vocab_size = 100000
oov_token = "<OOV>"

max_text_length = 100

embedding_dim = 100
glove_file = "/data/glove/"
glove_file_path = glove_file + "glove.6B." + str(embedding_dim) + "d.txt"
embedding_type = "glove"

hidden_layer = 32
lstm_size = 50
batch_size = 32
epochs = 30
