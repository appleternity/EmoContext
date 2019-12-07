import h5py
import json
import os
import numpy as np
from nltk.tokenize import word_tokenize

glove_path = "/home/appleternity/corpus/glove"

class Glove:
    def __init__(self, version="glove.6B.300d"):
        self.word_dictionary, self.matrix = Glove.load_glove(version)
   
    @staticmethod
    def load_glove(version):
        with open(os.path.join(glove_path, "{}.json".format(version)), 'r', encoding='utf-8') as infile:
            word_dictionary = json.load(infile)

        with h5py.File(os.path.join(glove_path, "{}.h5".format(version))) as infile:
            matrix = np.empty(infile["matrix"].shape, dtype=np.float32)
            infile["matrix"].read_direct(matrix)

        return word_dictionary, matrix

    def get_vector(self, word):
        return self.matrix[self.word_dictionary.get(word.lower(), 0)]

    def get_vector_over_text(self, text, mode="sum"):
        tokens = word_tokenize(text)
        vecs = np.vstack([
            self.matrix[self.word_dictionary.get(t.lower(), 0)] for t in tokens    
        ])
        if mode == "sum":
            return np.sum(vecs, axis=0)
        elif mode == "average":
            return np.mean(vecs, axis=0)

