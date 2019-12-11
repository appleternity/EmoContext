import pandas as pd
import numpy as np
import json
import os
import pickle
from config import *
from nltk import word_tokenize
from collections import Counter
import csv

# TODO: 
# (1) twitter tokenizer
# (2) Analyzing <UNK> ratio in dev/test data
def load_data(filename, dir_path=data_path, redo=False):
    target_file = os.path.join(dir_path, "processed_"+filename)
    if os.path.isfile(target_file) and not redo:
        with open(target_file, 'rb') as infile:
            data = pickle.load(infile)
        return data

    with open(os.path.join(dir_path, filename), 'r', encoding='utf-8') as infile:
        data = [line.strip().split("\t") for line in infile]
        data = data[1:]
        data = pd.DataFrame(data, columns=["id", "1", "2", "3", "emotion"])
        data = data.drop("id", axis=1)
        data["label"] = data["emotion"].apply(lambda x: label_mapping[x])
        my_tokenize = lambda x: word_tokenize(x.lower())
        data["1_tokenized"] = data["1"].apply(my_tokenize)
        data["2_tokenized"] = data["2"].apply(my_tokenize)
        data["3_tokenized"] = data["3"].apply(my_tokenize)

    with open(target_file, 'wb') as outfile:
        pickle.dump(data, outfile)

    return data

def load_test_data(filename, dir_path=data_path, redo=False):
    target_file = os.path.join(dir_path, "processed_"+filename)
    if os.path.isfile(target_file) and not redo:
        with open(target_file, 'rb') as infile:
            data = pickle.load(infile)
        return data

    with open(os.path.join(dir_path, filename.replace(".txt", "withoutlabels.txt")), 'r', encoding='utf-8') as infile:
        data = [line.strip().split("\t") for line in infile]
        data = data[1:]
        data = pd.DataFrame(data, columns=["id", "1", "2", "3"])
        my_tokenize = lambda x: word_tokenize(x.lower())
        data["1_tokenized"] = data["1"].apply(my_tokenize)
        data["2_tokenized"] = data["2"].apply(my_tokenize)
        data["3_tokenized"] = data["3"].apply(my_tokenize)

    with open(target_file, 'wb') as outfile:
        pickle.dump(data, outfile)

    return data

def load_twitter_data(num=None):
    filename = "/apple_data/workspace/emoContext/training.1600000.processed.noemoticon.csv"
    #data = pd.read_csv(filename, header=None, encoding='ascii')
    #print(data.columns)

    with open(filename, 'r', encoding='utf-8', errors="ignore") as infile:
        reader = csv.reader(infile)
        x = []
        y = []
        for row in reader:
            y.append(int(row[0]))
            x.append(row[-1])
        y = [yy if yy!=4 else 1 for yy in y]

    # sample
    if num is not None:
        index = np.random.permutation(len(x))
        x = [x[i] for i in index[:num]]
        y = [y[i] for i in index[:num]]

    # x
    length = np.array([len(xx) for xx in x])
    print("twitter length", length.mean(), length.std(), np.max(length), np.min(length))

    # y
    y_uni = np.unique(y)
    print("# uni y", y_uni.shape)

    return x, y

def word_count(filename, dir_path=data_path):
    target_file = os.path.join(dir_path, "processed_"+filename)
    
    with open(target_file, 'rb') as infile:
        data = pickle.load(infile)

    counter = Counter()
    #print(data["1_tokenized"])

    counter.update(word for sent in data["1_tokenized"] for word in sent)
    counter.update(word for sent in data["2_tokenized"] for word in sent)
    counter.update(word for sent in data["3_tokenized"] for word in sent)
    #print(counter)
    print(len(counter))

    token_num = sum([
        sum(len(sent) for sent in data["1_tokenized"]),
        sum(len(sent) for sent in data["1_tokenized"]),
        sum(len(sent) for sent in data["1_tokenized"]),
    ])
    print(token_num)

def main():
    load_twitter_data()
    quit()

    data = load_data("train.txt", redo=True)
    #print(data)

    data = load_data("dev.txt", redo=True)
    #print(data)

    word_count("train.txt")

if __name__ == "__main__":
    main()
