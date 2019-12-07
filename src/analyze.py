import json
from config import *
from data import load_data, load_test_data
from collections import Counter
import numpy as np

def word_distribution():
    # load data
    get_data = lambda d: [d["1"], d["2"], d["3"]] # character
    #get_data = lambda d: [d["1_tokenized"], d["2_tokenized"], d["3_tokenized"]] # word
    
    data = load_data("train.txt")
    x_train = get_data(data)

    data = load_data("dev.txt")
    x_valid = get_data(data)

    data = load_test_data("test.txt")
    x_test = get_data(data)

    # freq
    freq_train = Counter(
        w
        for x in x_train
        for row in x
        for w in row
    )
    freq_valid = Counter(
        w
        for x in x_valid
        for row in x
        for w in row
    )
    freq_test = Counter(
        w
        for x in x_test
        for row in x
        for w in row
    )

    # statistic
    print("Train All", sum(freq_train.values()))
    print("Valid All", sum(freq_valid.values()))

    # overlap
    print("=====================================")
    for min_freq in [0, 5, 10]:
        print("min_freq = ", min_freq)
        available_chars = {k:v for k, v in freq_train.items() if v >= min_freq} 
        not_in_valid = {k:v for k, v in freq_valid.items() if k not in available_chars}
        not_in_test  = {k:v for k, v in freq_test.items() if k not in available_chars}
        
        print("Train # unique chars = {}, # chars = {}".format(len(available_chars), sum(available_chars.values())))
        print("Valid Not in Train -- # unique chars = {}, # chars = {} | # available chars = {}".format(
            len(not_in_valid), 
            sum(not_in_valid.values()), 
            sum(freq_valid.values()) - sum(not_in_valid.values()), 
        ))
        print("Test Not in Train -- # unique chars = {}, # chars = {} | # available chars = {}".format(
            len(not_in_test), 
            sum(not_in_test.values()), 
            sum(freq_test.values()) - sum(not_in_test.values()), 
        ))
        print()

def length_analyze():
    # load data
    #get_data = lambda d: [d["1"], d["2"], d["3"]] # character
    get_data = lambda d: [d["1_tokenized"], d["2_tokenized"], d["3_tokenized"]] # word
    
    data = load_data("train.txt")
    x_train = get_data(data)

    data = load_data("dev.txt")
    x_valid = get_data(data)

    data = load_test_data("test.txt")
    x_test = get_data(data)

    l_train = np.array([len(row) for x in x_train for row in x])
    l_valid = np.array([len(row) for x in x_valid for row in x])
    l_test = np.array([len(row) for x in x_test for row in x])

    for length, name in zip([l_train, l_valid, l_test], ["train", "valid", "test"]):
        print("============================")
        print("length max", np.max(length))
        print("length min", np.min(length))
        print("length mean", length.mean())
        print("length std", length.std())

def main():
    #word_distribution()
    length_analyze()

if __name__ == "__main__":
    main()
