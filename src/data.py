import pandas as pd
import numpy as np
import json
import os
import pickle
from config import *
from nltk import word_tokenize
from collections import Counter
import csv
import sentencepiece as spm
import re

# TODO: 
# (1) twitter tokenizer
# (2) Analyzing <UNK> ratio in dev/test data
def load_data(filename, dir_path=data_path, redo=False):
    target_file = os.path.join(dir_path, "processed_"+filename)
    if os.path.isfile(target_file) and not redo:
        with open(target_file, 'rb') as infile:
            data = pickle.load(infile)
        return data
    
    sp = spm.SentencePieceProcessor()
    sp_name = "m_2000"
    sp.Load(sp_name+".model")

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
        
        # sentence piece
        for x in ["1", "2", "3"]:
            data["{}_{}".format(x, sp_name)] = data["1"].apply(sp.EncodeAsPieces)

    print(data.columns)
    with open(target_file, 'wb') as outfile:
        pickle.dump(data, outfile)

    return data

def load_test_data(filename, dir_path=data_path, redo=False):
    target_file = os.path.join(dir_path, "processed_"+filename)
    if os.path.isfile(target_file) and not redo:
        with open(target_file, 'rb') as infile:
            data = pickle.load(infile)
        return data
    
    sp = spm.SentencePieceProcessor()
    sp_name = "m_2000"
    sp.Load(sp_name+".model")

    with open(os.path.join(dir_path, filename.replace(".txt", "withoutlabels.txt")), 'r', encoding='utf-8') as infile:
        data = [line.strip().split("\t") for line in infile]
        data = data[1:]
        data = pd.DataFrame(data, columns=["id", "1", "2", "3"])
        my_tokenize = lambda x: word_tokenize(x.lower())
        data["1_tokenized"] = data["1"].apply(my_tokenize)
        data["2_tokenized"] = data["2"].apply(my_tokenize)
        data["3_tokenized"] = data["3"].apply(my_tokenize)
        
        # sentence piece
        for x in ["1", "2", "3"]:
            data["{}_{}".format(x, sp_name)] = data["1"].apply(sp.EncodeAsPieces)

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
    
    sp = spm.SentencePieceProcessor()
    sp_name = "m_2000"
    sp.Load(sp_name+".model")
    x = [sp.EncodeAsPieces(xx) for xx in x]

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

def train_sentence_piece():
    # extract data
    """
    train = load_data("train.txt")
    valid = load_data("dev.txt")
    test = load_data("test.txt")
    twitter = load_twitter_data()
    pattern = re.compile(r"@[\w\d]+")

    with open(os.path.join(data_path, "all_text.txt"), 'w', encoding='utf-8') as outfile:
        for data in [train, valid, test]:
            for x in ["1", "2", "3"]:
                for text in data[x]:
                    outfile.write(text + "\n")

        for text in twitter[0]:
            text = pattern.sub("", text).strip()
            outfile.write(text + "\n")
    """

    # run sentence piece
    spm.SentencePieceTrainer.Train('--input={} --model_prefix=m_2000 --vocab_size=2000'.format(
        os.path.join(data_path, "all_text.txt")
    ))

def test_sentence_piece():
    sp = spm.SentencePieceProcessor()
    sp.Load("m_2000.model")

    sentences = [
        "This is a test",
        "How r u?",
    ]
    
    for sent in sentences:
        r = sp.EncodeAsPieces(sent)
        print(r)

def main():
    #train_sentence_piece()
    """
    test_sentence_piece()
    quit()


    load_twitter_data()
    quit()
    """

    data = load_data("train.txt", redo=True)
    #print(data)

    data = load_data("dev.txt", redo=True)
    #print(data)

    data = load_test_data("test.txt", redo=True)

    word_count("train.txt")

if __name__ == "__main__":
    main()
