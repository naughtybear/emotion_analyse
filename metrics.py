from sklearn.metrics import f1_score, recall_score, precision_score
from keras.callbacks import Callback

def boolMap(arr):
    if arr > 0.5:
        return 1
    else:
        return 0


class Metrics(Callback):
    def __init__(self, filepath):
        self.file_path = filepath

    def on_train_begin(self, logs=None):
        self.val_f1s = []
        self.best_val_f1 = 0
        self.val_recalls = []
        self.val_precisions = []

    def on_epoch_end(self, epoch, logs=None):
        val_predict = list(map(boolMap, self.model.predict([self.validation_data[0], self.validation_data[1]])))
        val_targ = self.validation_data[2]
        _val_f1 = f1_score(val_targ, val_predict)
        _val_recall = recall_score(val_targ, val_predict)
        _val_precision = precision_score(val_targ, val_predict)
        self.val_f1s.append(_val_f1)
        self.val_recalls.append(_val_recall)
        self.val_precisions.append(_val_precision)
        print(_val_f1, _val_precision, _val_recall)
        print("max f1")
        print(max(self.val_f1s))
        if _val_f1 > self.best_val_f1:
            self.model.save_weights(self.file_path, overwrite=True)
            self.best_val_f1 = _val_f1
            print("best f1: {}".format(self.best_val_f1))
        else:
            print("val f1: {}, but not the best f1".format(_val_f1))
        return