import numpy as np
from keras.preprocessing import sequence
import csv
import random
import collections
import pickle

def pretreat(train_path = './train_data.csv', test_path = './test_data.csv'):
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
    
    y_train = np.array(y_train)
    y_train = y_train.astype(np.int)
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

    randnum = random.randint(0,100)
    np.random.seed(randnum)
    np.random.shuffle(x_train_out)
    np.random.seed(randnum)
    np.random.shuffle(y_train)

    #print(x_train_out.shape)
    #return x_train_out, y_train, x_test_out, y_test, max_len+2
    print("pickle dump")
    with open("data.pkl", "wb") as f:
        pickle.dump({
            "x_train": x_train_out,
            "y_train": y_train,
            "x_test": x_test_out,
            "y_test": y_test,
            "word_idx": word_index,
            "maxlen": max_len+2
        }, file = f)


if __name__ == '__main__':
    pretreat()