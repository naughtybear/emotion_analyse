from keras.layers.core import Dense, Activation, Dropout
from keras.layers.recurrent import LSTM, GRU
from keras.models import Sequential, load_model
from keras.layers import Bidirectional, Embedding
from keras.preprocessing import sequence
import numpy as np
import pickle

np.set_printoptions(threshold=np.inf)

def find_two_max(input_list):
    length = len(input_list)
    max_idx1 = 0
    max_idx2 = 0
    max1 = 0
    max2 = 0
    for i in range(length):
        if input_list[i] > max1:
            max2 = max1
            max_idx2 = max_idx1
            max1 = input_list[i]
            max_idx1 = i
        
        elif input_list[i] > max2:
            max2 = input_list[i]
            max_idx2 = i
    
    return max_idx1, max_idx2

def test(model_path = "emotion_model.h5"):
    
    model = load_model(model_path)
    DATA_PATH = "./data.pkl"
    dataset = pickle.load(open(DATA_PATH, "rb"))
    x_test = dataset["x_test"]
    y_test = dataset["y_test"]
    max_len = dataset["maxlen"]
    word_index = dataset["word_idx"]

    lable_type = ["極端化", "災難性思考", "讀心術", "「應該」與「必須」的陳述", "個人化"] 
    lable_emotion = ["憂鬱情緒", "焦慮情緒", "正向情緒"]
    
    for i in range(15):
        tmp = x_test[i].reshape((1,-1))
        predict_result = model.predict(tmp)
        #print(predict_result)
        predict_result = np.split(predict_result[0], [5,8])
        sen_type_pre = predict_result[0]
        sen_emotion_pre = predict_result[1]
        sen_pos_pre = predict_result[2]
        true_result = np.split(y_test[i], [5,8])
        sen_type_true = true_result[0]
        sen_emotion_true = true_result[1]
        sen_pos_true = true_result[2]
        #print(sen_type_true)

        print("\n", i)
        print("認知偏誤類型:")
        pos1, pos2 = find_two_max(sen_type_pre)
        pos3, pos4 = find_two_max(sen_type_true)
        print("\t預測：", lable_type[pos1], sen_type_pre[pos1], "\t真實:", lable_type[pos3], sen_type_true[pos3])
        print("\t預測：", lable_type[pos2], sen_type_pre[pos2], "\t真實:", lable_type[pos4], sen_type_true[pos4])
        print("\n情緒類型:")
        pos1, pos2 = find_two_max(sen_emotion_pre)
        pos3, pos4 = find_two_max(sen_emotion_true)
        print("\t預測：", lable_emotion[pos1], sen_emotion_pre[pos1], "\t真實:", lable_emotion[pos3], sen_emotion_true[pos3])
        print("\t預測：", lable_emotion[pos2], sen_emotion_pre[pos2], "\t真實:", lable_emotion[pos4], sen_emotion_true[pos4])
        print("\n是否為治療性對話:")
        if sen_pos_pre[0] > 0.5:
            print("\t預測：是", sen_pos_pre[0])
        else:
            print("\t預測：否", sen_pos_pre[0])
        if sen_pos_true[0] > 0.5:
            print("\t真實：是", sen_pos_true[0])
        else:
            print("\t真實：否", sen_pos_true[0])
        
        
        #print(i)
        #print("predict:", model.predict(tmp))
        #print("true:", y_test[i])
    
    
    '''
    while True:
        sentence = input(">>")
        if sentence == "quit":
            exit(0)
        seq = []
        seqs = []
        seq.append(word_index['GO'])
        for word in sentence:
            if word in word_index:
                seq.append(word_index[word])
            else:
                seq.append(word_index["UNK"])
        seq.append(word_index["EOS"])
        seqs.append(seq)
        seqs = sequence.pad_sequences(seqs, maxlen=max_len)

        print(seqs[0])
        print (model.predict(seqs))
        print (model.predict_classes(seqs))
    '''

if __name__ == "__main__":
    test()