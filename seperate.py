import random

def seperate_data(file_path = "./all_data_new.csv"):
    with open(file_path, "r") as csvfile:

        li = csvfile.readlines()
        random.shuffle(li)
        count = len(li)
        #print(count)

        with open("./test_data_new.csv", "w") as outfile:
            for i in range(15):
                outfile.write(li[i])

        with open("./train_data_new.csv", "w") as outfile:
            for i in range(15, count):
                outfile.write(li[i])

if __name__ == '__main__':
    seperate_data()