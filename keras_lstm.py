#one hot encode版本
import numpy as np
import time
import csv
from keras.layers.core import Dense, Activation, Dropout
from keras.layers.recurrent import LSTM, GRU
from keras.models import Sequential, load_model
from keras.layers import Bidirectional, Embedding
from keras.preprocessing import sequence
import collections
import random
import pickle
from f1score import f1

np.set_printoptions(threshold=np.inf)

def model_init(max_len, embedding_size, hidden_units, use_dropout, 
                dropout_rate = 0.1):
    vocab_size = 10000

    model = Sequential()

    model.add(Embedding(vocab_size, embedding_size, input_length=max_len))

    model.add(Bidirectional(LSTM(hidden_units)))

    if use_dropout:
        model.add(Dropout(dropout_rate))
    
    model.add(Dense(hidden_units, activation='relu'))

    if use_dropout:
        model.add(Dropout(dropout_rate))

    #model.add(Dense(9))
    #model.add(Activation("sigmod"))
    #model.compile(optimizer = "adam", loss = "binary_crossentropy", metrics=[f1])
    model.add(Dense(28))
    
    model.add(Activation("softmax"))

    model.compile(optimizer = "adam", loss = "categorical_crossentropy", metrics=['accuracy'])
    model.summary()
    return model


def rnn_network(epochs, batch_size, embedding_size, hidden_units, use_dropout):
    
    DATA_PATH = "./data.pkl"
    dataset = pickle.load(open(DATA_PATH, "rb"))
    #x_train, y_train, x_test, y_test, max_len = load_data()
    x_train = dataset["x_train"]
    y_train = dataset["y_train"]
    x_test = dataset["x_test"]
    y_test = dataset["y_test_trans"]
    max_len = dataset["maxlen"]
    word_index = dataset["word_idx"]
    #print(x_test)

    print ('\nData Loaded. Compiling...\n')

    model = model_init(max_len, embedding_size, hidden_units, use_dropout)
    #metrics = Metrics('./best_model.h5')
    model.fit(
        x_train, y_train,
        batch_size=batch_size, epochs= epochs,
        validation_data=(x_test, y_test)
        #validation_split=0.1
    )

    score, acc = model.evaluate(x_test, y_test, batch_size=batch_size)
    print("\nTest score: %.3f, accuracy: %.3f" % (score, acc))
    model.save('emotion_model.h5') # save model
    print("model save success")

if __name__ == '__main__' :
    rnn_network(
        epochs = 20,
        batch_size = 8,
        embedding_size = 64,
        hidden_units = 64,
        use_dropout = True
    )

