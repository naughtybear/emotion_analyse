from keras.layers.core import Dense, Activation, Dropout
from keras.layers.recurrent import LSTM, GRU
from keras.models import Sequential, load_model
from keras.layers import Bidirectional, Embedding
from keras.preprocessing import sequence
import numpy as np
import pickle
from f1score import f1

np.set_printoptions(threshold=np.inf)
np.set_printoptions(precision=8,suppress=True)

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
    
    model = load_model(model_path, custom_objects={"f1":f1})
    DATA_PATH = "./data.pkl"
    dataset = pickle.load(open(DATA_PATH, "rb"))
    x_test = dataset["x_test"]
    y_test = dataset["y_test"]
    max_len = dataset["maxlen"]
    word_index = dataset["word_idx"]
    out_index = dataset["out_idx"]

    lable_type = ["極端化", "災難性思考", "讀心術", "「應該」與「必須」的陳述", "個人化"] 
    lable_emotion = ["憂鬱情緒", "焦慮情緒", "正向情緒"]
    
    
    cognitive_bias = np.zeros(4)
    emotion = np.zeros(4)
    conversation = np.zeros(4)
    for i in range(676):
        idx = -1
        tmp = x_test[i].reshape((1,-1))
        predict_result = model.predict(tmp)
        idx = np.argmax(predict_result[0])
        predict_result = out_index[idx]

        #print(idx)
        #print(predict_result)
        predict_result = np.split(predict_result, [5,8])
        sen_type_pre = predict_result[0]
        sen_emotion_pre = predict_result[1]
        sen_pos_pre = predict_result[2]
        true_result = np.split(y_test[i], [5,8])
        sen_type_true = true_result[0]
        sen_emotion_true = true_result[1]
        sen_pos_true = true_result[2]
        
        # 0: true postive, 1: false positive. 2: false negative, 3: true negative
        '''
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
        '''

        #print(cognitive_bias)
        for i in range(len(sen_type_pre)):
            if sen_type_true[i] == 1 and sen_type_pre[i] > 0.5:
                cognitive_bias[0] += 1
            elif sen_type_true[i] == 1 and sen_type_pre[i] < 0.5: 
                cognitive_bias[1] += 1
            elif sen_type_true[i] == 0 and sen_type_pre[i] > 0.5: 
                cognitive_bias[2] += 1
            elif sen_type_true[i] == 0 and sen_type_pre[i] < 0.5: 
                cognitive_bias[3] += 1
        
        for i in range(len(sen_emotion_pre)):
            if sen_emotion_true[i] == 1 and sen_emotion_pre[i] > 0.5:
                emotion[0] += 1
            elif sen_emotion_true[i] == 1 and sen_emotion_pre[i] < 0.5: 
                emotion[1] += 1
            elif sen_emotion_true[i] == 0 and sen_emotion_pre[i] > 0.5: 
                emotion[2] += 1
            elif sen_emotion_true[i] == 0 and sen_emotion_pre[i] < 0.5: 
                emotion[3] += 1
        
        for i in range(len(sen_pos_true)):
            if sen_pos_true[i] == 1 and sen_pos_pre[i] > 0.5:
                conversation[0] += 1
            elif sen_pos_true[i] == 1 and sen_pos_pre[i] < 0.5: 
                conversation[1] += 1
            elif sen_pos_true[i] == 0 and sen_pos_pre[i] > 0.5: 
                conversation[2] += 1
            elif sen_pos_true[i] == 0 and sen_pos_pre[i] < 0.5: 
                conversation[3] += 1
        
    #print(cognitive_bias)
    print("認知偏誤類型:")
    print(f"TP:{cognitive_bias[0]} FP:{cognitive_bias[1]} FN:{cognitive_bias[2]} TN:{cognitive_bias[3]}")
    print(f"acc:{(cognitive_bias[0]+cognitive_bias[3])/np.sum(cognitive_bias)} recall:{cognitive_bias[0]/(cognitive_bias[0]+cognitive_bias[2])} precision:{cognitive_bias[0]/(cognitive_bias[0]+cognitive_bias[1])} F1:{2*cognitive_bias[0]/(2*cognitive_bias[0]+cognitive_bias[1]+cognitive_bias[2])}")
    print("\n情緒類型:")
    print(f"TP:{emotion[0]} FP:{emotion[1]} FN:{emotion[2]} TN:{emotion[3]}")
    print(f"acc:{(emotion[0]+emotion[3])/np.sum(emotion)} recall:{emotion[0]/(emotion[0]+emotion[2])} precision:{emotion[0]/(emotion[0]+emotion[1])} F1:{2*emotion[0]/(2*emotion[0]+emotion[1]+emotion[2])}")
    print("\n是否為治療性對話:")
    print(f"TP:{conversation[0]} FP:{conversation[1]} FN:{conversation[2]} TN:{conversation[3]}")
    print(f"acc:{(conversation[0]+conversation[3])/np.sum(conversation)} recall:{conversation[0]/(conversation[0]+conversation[2])} precision:{conversation[0]/(conversation[0]+conversation[1])} F1:{2*conversation[0]/(2*conversation[0]+conversation[1]+conversation[2])}")
    
    '''
    count = 0
    while True:
        sentence = input()
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
        
        #print(seqs[0])
        #print (model.predict(seqs))
        #print (model.predict_classes(seqs))
        predict_result = model.predict(seqs)
        #print(predict_result)
        predict_result = np.split(predict_result[0], [5,8])
        sen_type_pre = predict_result[0]
        sen_emotion_pre = predict_result[1]
        sen_pos_pre = predict_result[2]
        pos1, pos2 = find_two_max(sen_type_pre)
        print("認知偏誤類型:")
        print("\t預測：", lable_type[pos1], sen_type_pre[pos1])
        print("\t預測：", lable_type[pos2], sen_type_pre[pos2])
        print("\n情緒類型:")
        pos1, pos2 = find_two_max(sen_emotion_pre)
        print("\t預測：", lable_emotion[pos1], sen_emotion_pre[pos1])
        print("\t預測：", lable_emotion[pos2], sen_emotion_pre[pos2])
        print("\n是否為治療性對話:")
        if sen_pos_pre[0] > 0.5:
            print("\t預測：是", sen_pos_pre[0])
        else:
            print("\t預測：否", sen_pos_pre[0])
        
        #print(count)
        predict_result = model.predict(seqs)[0]
        #print("認知偏誤類型:")
        print(f"極端化:\t{predict_result[0]:.6f}")
        print(f"災難性思考:\t{predict_result[1]:.6f}")
        print(f"讀心術:\t{predict_result[2]:.6f}")
        print(f"「應該」與「必須」的陳述:\t{predict_result[3]:.6f}")
        print(f"個人化:\t{predict_result[4]:.6f}")
        #print("情緒類型:")
        print(f"憂鬱情緒:\t{predict_result[5]:.6f}")
        print(f"焦慮情緒:\t{predict_result[6]:.6f}")
        print(f"正向情緒:\t{predict_result[7]:.6f}")
        #print(f"是否為治療性對話:")
        print(f"是:\t{predict_result[8]:.6f}")
        print(f"否:\t{1 - predict_result[8]:.6f}")
        #count += 1
    '''

if __name__ == "__main__":
    test()