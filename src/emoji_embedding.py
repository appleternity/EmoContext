import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.contrib.eager.python import tfe
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation, Dropout, Conv2D, MaxPool2D, Flatten
import numpy as np
import json
from zipfile import ZipFile
import cv2
from config import *
import os
import pickle
from scipy.spatial.distance import cdist

tf.enable_eager_execution()

class EmojiCNN(tf.keras.Model):
    def __init__(self, emoji_img, **kwargs):
        super(EmojiCNN, self).__init__()

        # parameters
        self.batch_size = kwargs.get("batch_size", 128)
        self.output_size = kwargs.get("output_size", 4)

        # model
        self.img_emb = tfe.Variable(emoji_img, name="imgs")
        self.conv_1 = Conv2D(32, (5, 5), activation="relu") 
        self.max_pool_1 = MaxPool2D((2, 2))
        self.conv_2 = Conv2D(32, (5, 5), activation="relu") 
        self.max_pool_2 = MaxPool2D((4, 4))
        self.conv_3 = Conv2D(32, (5, 5), activation="relu") 
        self.max_pool_3 = MaxPool2D((4, 4))
        self.flatten = Flatten()
        self.output_layer = Dense(self.output_size, activation="softmax")

    def call(self, x, training=False):
        imgs = tf.nn.embedding_lookup(self.img_emb, x)
        imgs = tf.reshape(imgs, (-1, 128, 128, 3))

        rep = self.conv_1(imgs)
        rep = self.max_pool_1(rep)
        rep = self.conv_2(rep)
        rep = self.max_pool_2(rep)
        rep = self.conv_3(rep)
        rep = self.max_pool_3(rep)
        flat = self.flatten(rep)
        output = self.output_layer(flat)
        return output

    def get_vector(self, x):
        imgs = tf.nn.embedding_lookup(self.img_emb, x)
        imgs = tf.reshape(imgs, (-1, 128, 128, 3))

        rep = self.conv_1(imgs)
        rep = self.max_pool_1(rep)
        rep = self.conv_2(rep)
        rep = self.max_pool_2(rep)
        rep = self.conv_3(rep)
        rep = self.max_pool_3(rep)
        flat = self.flatten(rep)
        return flat

class Embedding:
    def __init__(self, vector, info):
        self.vector = vector
        self.info = info

    def most_similar(self, x, num=10):
        vec = self.vector[x, :].reshape([1, -1])
        d = cdist(vec, self.vector, "cosine").reshape([-1, ])
        d = np.argsort(d)
        
        return [self.info[dd] for dd in d[:num]]

def build_emoji_data():
    dir_path = os.path.join(data_path, "Emoji_Downloaded")
    filenames = os.listdir(dir_path)
    
    data = []
    for filename in filenames:
        path = os.path.join(dir_path, filename)
        img = cv2.imread(path).astype(np.float32) / 255
        img = img.reshape((1, -1))
        data.append(img)

    data = np.vstack(data)
    print(data.shape)

    # save file
    with open(os.path.join(data_path, "emoji_img_data.pkl"), 'wb') as outfile:
        pickle.dump(data, outfile)

def vertorize(y):
    if type(y) != np.ndarray:
        y = np.array(y)
    uni = np.unique(y)
    one_hot = np.zeros([y.shape[0], uni.shape[0]])
    for i, yy in enumerate(y):
        one_hot[i, yy] = 1
    return one_hot

def train():
    # load data
    with open(os.path.join(data_path, "emoji_img_data.pkl"), 'rb') as infile:
        emoji_img = pickle.load(infile)
    
    with open(os.path.join(data_path, "emoji_dataset.json"), 'r', encoding='utf-8') as infile:
        data = json.load(infile)
        x = np.array([d[0] for d in data])
        y = np.array([d[1] for d in data])
        y = vertorize(y)
    
    batch_size = 128

    model = EmojiCNN(
        batch_size = 128,
        output_size = 4,
        emoji_img = emoji_img
    )

    model.compile(
        optimizer='adam',
        loss="categorical_crossentropy",
        metrics=['acc']
    )

    history = model.fit(
        x,
        y,
        epochs=1000, 
        batch_size=batch_size,
    )

    # get vector
    test_x = np.array([i for i in range(0, emoji_img.shape[0])])
    output = model.get_vector(test_x).numpy()
    print(output.shape)

    with open(os.path.join(data_path, "emoji_embedding.pkl"), 'wb') as outfile:
        pickle.dump(output, outfile)

def test():
    with open(os.path.join(data_path, "emoji_embedding.pkl"), 'rb') as infile:
        vector = pickle.load(infile)
    print(vector.shape)

    with open(os.path.join(data_path, "char.json"), 'r', encoding='utf-8') as infile:
        char_dict = json.load(infile)
        char_info = sorted(char_dict.values(), key=lambda x: x["index"])

    embedding = Embedding(vector, char_info)
    res = embedding.most_similar(0)
    print(res)

def main():
    #build_emoji_data()
    #train()
    test()

if __name__ == "__main__":
    main()


