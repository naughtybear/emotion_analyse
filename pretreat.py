import numpy as np
from keras.preprocessing import sequence
import csv
import random
import collections
import pickle

def pretreat(train_path = './data/train_data.csv', test_path = './data/test_data.csv', label_encode = True, delete_zero = False, zero_num = 100, noise_num = 1):
    '''
    input: 
        train_path : train data 路徑
        test_path : test data 路徑
        delete_zero : 是否刪除結果全部都是零的項目， 若為True， 預設則會留下前100個， 其他全部刪除
        zero_num : 可以更改delete_zero要留下的全零項目的個數， 若delete項目設為false， 則不用管這項
        label_encode : 是否將label做one-hot encoding 
        noise_num : 每個句子要產生幾個加入雜訊的句子

    output:
        輸出為一個pickle檔供機器學習模型讀取
    '''
    # initialize
    x_train = []
    y_train = []
    x_test = []
    y_test = []
    y_zero = ['0', '0', '0', '0', '0', '0', '0', '0', '0']
    train_cnt = 0
    test_cnt = 0
    max_len = 0
    count_zero = 0
    word_freq = collections.Counter() # 計算每個字在訓練資料中出現幾次， 方便做成字典

    # 讀取訓練資料
    print("---load training data---")
    with open(train_path) as f:
        data = csv.reader(f, delimiter = ',')
        for line in data:
            # 限制全部都是0的資料個數，
            if delete_zero and line[1:]== y_zero and count_zero < zero_num:   
                count_zero += 1
            elif delete_zero and line[1:]== y_zero and count_zero == zero_num:
                continue
            
            if len(line[0]) > max_len:
                max_len = len(line[0])
            tmp_line = list(line[0])
            x_train.append(tmp_line)
            y_train.append(line[1:])
            for word in tmp_line:
                word_freq[word] += 1
            train_cnt += 1
    #print('len of word_freq:',max_len)
    print(f"train count:{train_cnt}")
    
    # 讀取測試資料
    print("---load testing data---")
    with open(test_path) as f:
        data = csv.reader(f, delimiter = ',')
        for line in data:
            x_test.append(list(line[0]))
            y_test.append(line[1:])
            test_cnt += 1
    print(f"test count:{test_cnt}")

    # 轉為np array
    y_train = np.array(y_train).astype(np.int)
    y_test = np.array(y_test).astype(np.int)

    # 將label做onehot encoding, 例如 0,0,0,0,0,0,0,0 為0, 0,0,0,0,0,0,0,1 為1, 0,0,0,0,0,0,1,1 為2, 將出現過的排序都做encode
    y_train_trans = []
    y_test_trans = []
    out_idx = []
    count = 0

    if label_encode:
        for label in y_train:
            flag = False
            for idx, value in enumerate(out_idx):
                if np.array_equal(value, label):
                    flag = True
                    y_train_trans.append(idx)
                    continue
            if flag:
                continue
            out_idx.append(label)
            y_train_trans.append(count)
            count += 1
        
        for i in range(len(y_train_trans)):
            tmp = np.zeros(count+1, dtype=np.int)
            tmp[y_train_trans[i]] = 1
            y_train_trans[i] = tmp
        y_train_trans = np.asarray(y_train_trans)
        y_train = y_train_trans 

        for label in y_test:
            flag = False
            for idx, value in enumerate(out_idx):
                if np.array_equal(value, label):
                    flag = True
                    y_test_trans.append(idx)
                    continue
            if flag:
                continue
            y_test_trans.append(count)

        for i in range(len(y_test_trans)):
            tmp = np.zeros(count+1, dtype=np.int)
            tmp[y_test_trans[i]] = 1
            y_test_trans[i] = tmp
        y_test_trans = np.asarray(y_test_trans) # 將y_test_trans另外存，因為測試時會需要看原本的

    # 將每個字做one-hot encoding, 從文字轉成數字
    word_index = {x[0]: i+4 for i, x in enumerate(word_freq.most_common(len(word_freq)+3))}
    word_index["PAD"] = 0
    word_index["UNK"] = 1
    word_index["EOS"] = 2
    word_index["GO"] = 3
    #index2word = {v:k for k, v in word_index.items()}
    
    x_train_out = np.empty(train_cnt, dtype = list)
    x_test_out = np.empty(test_cnt, dtype = list)
    # train data word to index
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
    print(f"原始個數: {len(x_train_out)}")
    #print(type(x_train_out))
    
    count = 0
    if noise_num > 1:
        for i, line in enumerate(x_train):
            count += 1
            #if y_train[i][-1] == 1:
            for _ in range(noise_num):
                seq = []
                seq.append(word_index["GO"])
                tmp_idx = 0
                rand_num = random.sample(range(1, max_len), max_len - len(line))
                for k in range(max_len):
                    if k in rand_num:
                        seq.append(random.randint(4, len(word_index)-1))
                    else:
                        seq.append(word_index[line[tmp_idx]])
                        tmp_idx += 1

                seq.append(word_index["EOS"])
                x_train_out = np.append(x_train_out, np.asarray(seq).reshape(1, -1), axis=0)
                y_train = np.append(y_train, y_train[i].reshape(1, -1), axis=0)

    print(f"x加入noise後個數: {len(x_train_out)}")
    print(f"y加入noise後個數: {len(y_train)}")
    print(f"確認: {count}")

    #test data word to index
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

    # 將資料隨機排序
    randnum = random.randint(0,100)
    np.random.seed(randnum)
    np.random.shuffle(x_train_out)
    np.random.seed(randnum)
    np.random.shuffle(y_train)

    print("pickle dump")
    with open("data.pkl", "wb") as f:
        pickle.dump({
            "x_train": x_train_out,
            "y_train": y_train,
            "x_test": x_test_out,
            "y_test_trans": y_test_trans,
            "y_test": y_test,
            "word_idx": word_index,
            "maxlen": max_len+2,
            "out_idx": out_idx
        }, file = f)


if __name__ == '__main__':
    pretreat(train_path = './data/train_v3.csv',
             test_path = './data/test_v3.csv',
             delete_zero = True,
             label_encode = True,
             noise_num= 10)
