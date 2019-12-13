import pandas as pd
import numpy as np
import json
import os
import pickle
from config import *
from nltk import word_tokenize
from collections import Counter
import collections
import torch
import nltk
import tensorflow as tf

from bert import tokenization
#from bert.tokenization import convert_to_unicode

nltk.download('punkt')
from pytorch_pretrained_bert import BertTokenizer

# TODO: 
# (1) twitter tokenizer
# (2) Analyzing <UNK> ratio in dev/test data
def load_data(filename, dir_path=data_path, redo=False):
    vocab = createBERTVocab()
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
        # tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        # print('kavyaaaaaaa')
        # for token in tokenizer.vocab.keys():
          # print(token)

        my_tokenize = lambda x:constructTokens(x.lower(), vocab)
        # print('hereeee')
        # print(data["1"])
        data["1_tokenized_wordpiece"] = data["1"].apply(my_tokenize)
        # print(data["1_tokenized"])
        data["2_tokenized_wordpiece"] = data["2"].apply(my_tokenize)
        data["3_tokenized_wordpiece"] = data["3"].apply(my_tokenize)

    with open(target_file, 'wb') as outfile:
        pickle.dump(data, outfile)

    return data

def load_test_data(filename, dir_path=data_path, redo=False):
    vocab = createBERTVocab()    
    target_file = os.path.join(dir_path, "processed_"+filename)
    if os.path.isfile(target_file) and not redo:
        with open(target_file, 'rb') as infile:
            data = pickle.load(infile)
        return data

    with open(os.path.join(dir_path, filename.replace(".txt", "withoutlabels.txt")), 'r', encoding='utf-8') as infile:
        data = [line.strip().split("\t") for line in infile]
        data = data[1:]
        data = pd.DataFrame(data, columns=["id", "1", "2", "3"])
        # tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')        
        my_tokenize = lambda x:constructTokens(x.lower(), vocab)
        # constructTokens(data["1"])
        # print('hereeee')
        # print(data["1"])
        data["1_tokenized"] = data["1"].apply(my_tokenize)
        # print(data["1_tokenized"])
        data["2_tokenized"] = data["2"].apply(my_tokenize)
        data["3_tokenized"] = data["3"].apply(my_tokenize)

    with open(target_file, 'wb') as outfile:
        pickle.dump(data, outfile)

    return data

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
    # print(len(counter))

    token_num = sum([
        sum(len(sent) for sent in data["1_tokenized"]),
        sum(len(sent) for sent in data["1_tokenized"]),
        sum(len(sent) for sent in data["1_tokenized"]),
    ])
    # print(token_num)

def createBERTVocab():
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    with open("vocabulary.txt", 'w') as f:
    
    # For each token...
      for token in tokenizer.vocab.keys():
        
        # Write it out and escape any unicode characters.            
        f.write(token + '\n')

  
    vocab = collections.OrderedDict()
    index = 0
    with tf.gfile.GFile("vocabulary.txt", "r") as reader:
      while True:
        token = convert_to_unicode(reader.readline())
        if not token:
          break
        token = token.strip()
        vocab[token] = index
        index += 1
    return vocab

def whitespace_tokenize(text):
  text = text.strip()
  if not text:
    return []
  tokens = text.split()
  return tokens
    
def constructTokens(text, vocab):
    # print('Innnnnnnnnnn')
    # print(File)
    text = convert_to_unicode(text)

    output_tokens = []
    for token in whitespace_tokenize(text):
      chars = list(token)
      if len(chars) > 200:
        output_tokens.append("[UNK]")
        continue

      is_bad = False
      start = 0
      sub_tokens = []
      while start < len(chars):
        end = len(chars)
        cur_substr = None
        while start < end:
          substr = "".join(chars[start:end])
          if start > 0:
            substr = "##" + substr
          if substr in vocab:
            cur_substr = substr
            break
          end -= 1
        if cur_substr is None:
          is_bad = True
          break
        sub_tokens.append(cur_substr)
        start = end

      if is_bad:
        output_tokens.append("[UNK]")
      else:
        output_tokens.extend(sub_tokens)
    return output_tokens

def main():
    # print('In mainnnnn')
    # print(vocab)
    data = load_data("train.txt", redo=True)
    #print(data)

    data = load_data("dev.txt", redo=True)
    #print(data)

    word_count("train.txt")

if __name__ == "__main__":
    main()
