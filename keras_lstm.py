#one hot encode版本
import numpy as np
import time
import csv
from keras.layers.core import Dense, Activation, Dropout
from keras.layers.recurrent import LSTM, GRU
from keras.models import Sequential, load_model
from keras.layers import Embedding
from keras.preprocessing import sequence
import collections

np.set_printoptions(threshold=np.inf)

def load_data(train_path = './train_data.csv', test_path = './test_data.csv', data_path = './all_data.csv'):
    x_train = []
    y_train = []
    x_test = []
    y_test = []
    train_cnt = 0
    test_cnt = 0
    max_len = 0
    word_freq = collections.Counter()
    print("---load training data---")
    with open(train_path) as f:
        data = csv.reader(f, delimiter = ',')
        for line in data:
            #print("len:", len(line))
            if len(line[0]) > max_len:
                max_len = len(line[0])
            tmp_line = list(line[0])
            x_train.append(tmp_line)
            y_train.append(line[1:])
            for word in tmp_line:
                word_freq[word] += 1
            train_cnt += 1
    print('len of word_freq:',max_len)
    
    print("---load testing data---")
    with open(test_path) as f:
        data = csv.reader(f, delimiter = ',')
        for line in data:
            x_test.append(list(line[0]))
            y_test.append(line[1:])
            test_cnt += 1
    
    #x_train = np.array(x_train)
    y_train = np.array(y_train)
    y_train = y_train.astype(np.int)
    #x_test = np.array(x_test)
    y_test = np.array(y_test)
    y_test = y_test.astype(np.int)
    
    word_index = {x[0]: i+4 for i, x in enumerate(word_freq.most_common(len(word_freq)+3))}
    word_index["PAD"] = 0
    word_index["UNK"] = 1
    word_index["EOS"] = 2
    word_index["GO"] = 3
    #index2word = {v:k for k, v in word_index.items()}
    
    x_train_out = np.empty(train_cnt, dtype = list)
    x_test_out = np.empty(test_cnt, dtype = list)

    for i, line in enumerate(x_train):
        seq = []
        seq.append(word_index["GO"])
        for word in line:
            if word in word_index:
                seq.append(word_index[word])
            else:
                seq.append(word_index["UNK"])
        seq.append(word_index["EOS"])
        x_train_out[i] = seq
    x_train_out = sequence.pad_sequences(x_train_out, maxlen= max_len+2)

    for i, line in enumerate(x_test):
        seq = []
        seq.append(word_index["GO"])
        for word in line:
            if word in word_index:
                seq.append(word_index[word])
            else:
                seq.append(word_index["UNK"])
        seq.append(word_index["EOS"])
        x_test_out[i] = seq
    x_test_out = sequence.pad_sequences(x_test_out, maxlen= max_len+2)
    
    return x_train_out, y_train, x_test_out, y_test, max_len+2

def model_init(max_len):
    vocab_size = 10000
    embedding_size = 64
    hidden_layer_size = 64
    model = Sequential()

    model.add(Embedding(vocab_size, embedding_size))

    model.add(LSTM(hidden_layer_size, dropout = 0.1))

    model.add(Dense(6))

    model.add(Activation("relu"))

    start = time.time()
    model.compile(optimizer = "adam", loss = "binary_crossentropy", metrics=["accuracy"])
    print ("Compilation Time : ", time.time() - start)
    return model


def rnn_network():
    epochs = 8
    batch_size = 16
    x_train, y_train, x_test, y_test, max_len = load_data()
    print ('\nData Loaded. Compiling...\n')

    model = model_init(max_len)
    model.fit(
        x_train, y_train,
        batch_size=batch_size, epochs= epochs,
        validation_data=(x_test, y_test)
        #validation_split=0.1
    )

    score, acc = model.evaluate(x_test, y_test, batch_size=batch_size)
    print("\nTest score: %.3f, accuracy: %.3f" % (score, acc))


if __name__ == '__main__' :
    #x_train, y_train, x_test, y_test, max_len = load_data()
    #print(x_train)
    rnn_network()
