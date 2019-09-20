import csv
import random

def shuffle_data(input_path, train_path, test_path):

    with open(input_path, newline='') as csvfile:
        rows = csv.reader(csvfile)
        rows = list(rows)
        random.shuffle(rows)
        with open(train_path, 'w', newline='') as train_data:
            writer = csv.writer(train_data)
            writer.writerows(rows[:600])

        with open(test_path, 'w', newline='') as test_data:
            writer = csv.writer(test_data)
            writer.writerows(rows[600:])
        
if __name__ == "__main__":
    shuffle_data('./data/data_v2.csv', './data/train_v2.csv', './data/test_v2.csv')
            