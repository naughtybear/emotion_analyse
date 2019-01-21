#one hot encode版本
import numpy as np
import time
from keras.layers.core import Dense, Activation, Dropout
from keras.layers.recurrent import LSTM, GRU
from keras.models import Sequential, load_model
from keras.layers import Bidirectional, Embedding
from keras.preprocessing import sequence
import collections
import random
import csv

np.set_printoptions(threshold=np.inf)

def load_data(train_x_path = './train_x.csv', train_y_path = './train_y.csv',
             test_x_path = './test_x.csv', test_y_path = 'test_y.csv'):
    x_train = []
    y_train = []
    x_test = []
    y_test = []
    
    with open(train_x_path) as f:
        data= csv.reader(f, delimiter = ',')
        for line in data:
            x_train.append(line)

    with open(train_y_path) as f:
        data= csv.reader(f, delimiter = ',')
        for line in data:
            y_train.append(line)
    
    with open(test_x_path) as f:
        data= csv.reader(f, delimiter = ',')
        for line in data:
            x_test.append(line)
    
    with open(test_y_path) as f:
        data= csv.reader(f, delimiter = ',')
        for line in data:
            y_test.append(line)

    #print(y_train)
    x_train = np.array(x_train)
    y_train = np.array(y_train)
    y_train = y_train.astype(np.int)
    x_test = np.array(x_test)
    y_test = np.array(y_test)
    y_test = y_test.astype(np.int)
    x_train = np.reshape(x_train, (-1, 9,1))
    x_test = np.reshape(x_test, (-1, 9,1))
    print(x_test.shape)
    
    return x_train, y_train, x_test, y_test

def model_init():
    hidden_layer_size = 64
    model = Sequential()

    model.add(Dense(hidden_layer_size, input_shape = (9,1,)))

    model.add(Dense(hidden_layer_size))
    
    #model.add(Dense(hidden_layer_size, activation='relu'))

    model.add(Dense(6))

    model.add(Activation("softmax"))

    start = time.time()
    model.compile(optimizer = "adam", loss = "categorical_crossentropy", metrics=["accuracy"])
    print ("Compilation Time : ", time.time() - start)
    return model


def rnn_network():
    epochs = 10
    batch_size = 2
    x_train, y_train, x_test, y_test = load_data()
    print ('\nData Loaded. Compiling...\n')

    model = model_init()
    model.fit(
        x_train, y_train,
        batch_size=batch_size, epochs= epochs,
        validation_data=(x_test, y_test)
        #validation_split=0.1
    )

    score, acc = model.evaluate(x_test, y_test, batch_size=batch_size)
    print("\nTest score: %.3f, accuracy: %.3f" % (score, acc))
    print(model.predict(x_test).shape)
    for i in range(28):
        print(x_test[i])
        print(model.predict(x_test)[i]) 

if __name__ == '__main__' :
    #load_data()
    #x_train, y_train, x_test, y_test = load_data()
    #print(x_train)
    rnn_network()
