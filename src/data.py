import pandas as pd
import numpy as np
import json
import os
import pickle
from config import *
from nltk import word_tokenize

def load_data(filename, dir_path=data_path):
    target_file = os.path.join(dir_path, "processed_"+filename)
    if os.path.isfile(target_file):
        with open(target_file, 'rb') as infile:
            data = pickle.load(infile)
        return data

    with open(os.path.join(dir_path, filename), 'r', encoding='utf-8') as infile:
        data = [line.strip().split("\t") for line in infile]
        data = data[1:]
        data = pd.DataFrame(data, columns=["id", "1", "2", "3", "emotion"])
        data = data.drop("id", axis=1)
        data["label"] = data["emotion"].apply(lambda x: label_mapping[x])
        data["1_tokenized"] = data["1"].apply(word_tokenize)
        data["2_tokenized"] = data["2"].apply(word_tokenize)
        data["3_tokenized"] = data["3"].apply(word_tokenize)
    
    with open(target_file, 'wb') as outfile:
        pickle.dump(data, outfile)

    return data

def main():
    data = load_data("train.txt")
    print(data)

    data = load_data("dev.txt")
    print(data)

if __name__ == "__main__":
    main()