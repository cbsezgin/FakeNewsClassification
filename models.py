from constants import *

from keras.models import Sequential
from keras.layers import SimpleRNN, Dropout, Dense, LSTM, GRU


def build_RNN_model(input_shape):
    model = Sequential()
    model.add(input_shape)
    model.add(SimpleRNN(100))
    model.add(Dropout(0.3))
    model.add(Dense(hidden_layer, activation='relu'))
    model.add(Dropout(0.3))
    model.add(Dense(1, activation="sigmoid"))
    return model


def build_LSTM_model(input_shape):
    model = Sequential()
    model.add(input_shape)
    model.add(LSTM(lstm_size))
    model.add(Dropout(0.2))
    model.add(Dense(32, activation="relu"))
    model.add(Dropout(0.2))
    model.add(Dense(1, activation="sigmoid"))
    return model


def build_GRU_model(input_shape):
    model = Sequential()
    model.add(input_shape)
    model.add(GRU(100))
    model.add(Dropout(0.3))
    model.add(Dense(hidden_layer, activation="relu"))
    model.add(Dropout(0.3))
    model.add(Dense(1, activation="sigmoid"))
    return model


def train_model(model, X_train, y_train, X_test, y_test):
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    history = model.fit(X_train,y_train,batch_size=batch_size, epochs=epochs, validation_data= [X_test, y_test])
    return model, history


def store_model(model, filepath, filename):
    model_json = model.to_json()
    with open(filepath+filename+'.json', "w") as json_file:
        json_file.write(model_json)
    model.save_weights(filepath+filename+'.h5')



