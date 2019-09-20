from torch.utils.data import Dataset
import pickle

class Sentance_Dataset(Dataset):
    def __init__(self, data_file, is_train):
        
        with open(data_file, "rb") as f:
            dataset = pickle.load()
            if is_train:
                self.x = dataset["x_train"]
                self.y = dataset["y_train"]
            else:
                self.x = dataset["x_test"]
                self.y = dataset["y_test"]

    def __len__(self):
        return len(self.x)
    
    def __getitem__(self, index):
        return self.x[index], self.y[index]