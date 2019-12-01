import numpy as np
import pandas as pd
import os
import json
import tensorflow as tf
import tensorflow.contrib.eager as tfe
from data import *
from config import *
import pickle

# CUDA SETTING
#tf.enable_eager_execution()
#tf.executing_eagerly()

# TODO:
# (1) multi-class prediction: predict next-next sales

class ConversationLSTM(tf.keras.Model):
    def __init__(self, **kwargs):
        print(kwargs)
        super(ConversationLSTM, self).__init__()

        # parameter setting
        self.size_output = 4
        self.size_hidden      = kwargs.get("hidden_size", 128)
        self.batch_size       = kwargs.get("batch_size", 256)
        self.input_keep_rate  = kwargs.get("input_keep_rate", 0.8)
        self.output_keep_rate = kwargs.get("output_keep_rate", 0.8)
        self.state_keep_rate  = kwargs.get("state_keep_rate", 0.8)
        self.stack_num        = kwargs.get("stack_num", 3)
        self.vocab_size       = kwargs.get("vocab_size", 1024)
        self.word_embedding   = kwargs.get("word_embedding", None)
        self.train_embedding  = kwargs.get("train_embedding", True)
        self.residual         = kwargs.get("residual", False)

        self.is_training = True

        # building model
        if self.word_embedding:
            self.word_embedding = tfe.Variable(tf.convert_to_tensor(self.word_embedding), name="word_embedding")
        else:
            self.word_embedding = tfe.Variable(tf.random.normal([self.vocab_size, self.size_hidden]), name="word_embedding")

        self.lstm_layers = [
            tf.contrib.cudnn_rnn.CudnnCompatibleLSTMCell(num_units=self.size_hidden)
            for _ in range(self.stack_num)
        ]
        if self.residual:
            self.cell = tf.nn.rnn_cell.MultiRNNCell([
                tf.contrib.rnn.ResidualWrapper(lstm_cell)    
                for i, lstm_cell in enumerate(self.lstm_layers)
            ], name="multi_rnn_cell")
            self.dropout_cell = tf.nn.rnn_cell.MultiRNNCell([
                tf.contrib.rnn.ResidualWrapper(
                    tf.contrib.rnn.DropoutWrapper(
                        lstm_cell,
                        input_keep_prob=self.input_keep_rate, 
                        output_keep_prob=self.output_keep_rate, 
                        state_keep_prob=self.state_keep_rate,
                        variational_recurrent=True,
                        #input_size=self.size_input if i == 0 else self.size_hidden,
                        input_size=self.size_hidden,
                        dtype=tf.float32
                    )
                )
                for i, lstm_cell in enumerate(self.lstm_layers)
            ], name="dropout_cell")
        else:
            self.cell = tf.nn.rnn_cell.MultiRNNCell(self.lstm_layers)
            self.dropout_cell = tf.nn.rnn_cell.MultiRNNCell([
                tf.contrib.rnn.DropoutWrapper(
                    lstm_cell,
                    input_keep_prob=self.input_keep_rate, 
                    output_keep_prob=self.output_keep_rate, 
                    state_keep_prob=self.state_keep_rate,
                    variational_recurrent=True,
                    #input_size=self.size_input if i == 0 else self.size_hidden,
                    input_size=self.size_hidden,
                    dtype=tf.float32
                )
                for i, lstm_cell in enumerate(self.lstm_layers)
            ])

        self.batch_norm_1 = tf.keras.layers.BatchNormalization()
        self.batch_norm_2 = tf.keras.layers.BatchNormalization()
        self.batch_norm_3 = tf.keras.layers.BatchNormalization()

        self.output_dense_1 = tf.keras.layers.Dense(self.size_hidden*2, name="output_1")
        self.output_dense_2 = tf.keras.layers.Dense(self.size_hidden//5, name="output_2")
        self.output_dense_3 = tf.keras.layers.Dense(self.size_output, name="output_3")
            
    def call(self, text_1, text_2, text_3):
        # embedding
        embedding_1 = tf.nn.embedding_lookup(self.word_embedding, text_1)
        embedding_2 = tf.nn.embedding_lookup(self.word_embedding, text_2)
        embedding_3 = tf.nn.embedding_lookup(self.word_embedding, text_3)
        
        # lstm 
        init_state = self.cell.zero_state(text_1.shape[0], tf.float32)
        if self.is_training:
            rnn_output_1, state = tf.nn.dynamic_rnn(self.dropout_cell, embedding_1, initial_state=init_state)
            rnn_output_2, state = tf.nn.dynamic_rnn(self.dropout_cell, embedding_2, initial_state=init_state)
            rnn_output_3, state = tf.nn.dynamic_rnn(self.dropout_cell, embedding_3, initial_state=init_state)
        else:
            rnn_output_1, state = tf.nn.dynamic_rnn(self.cell, embedding_1, initial_state=init_state)
            rnn_output_2, state = tf.nn.dynamic_rnn(self.cell, embedding_2, initial_state=init_state)
            rnn_output_3, state = tf.nn.dynamic_rnn(self.cell, embedding_3, initial_state=init_state)

        representation = tf.concat([rnn_output_1[:, -1, :], rnn_output_2[:, -1, :], rnn_output_3[:, -1, :]], axis=1)
        representation = self.batch_norm_1(representation)
        representation = tf.nn.selu(self.output_dense_1(representation))
        representation = self.batch_norm_2(representation)
        representation = tf.nn.selu(self.output_dense_2(representation))
        representation = self.batch_norm_3(representation)
        output = self.output_dense_3(representation)

        return output

    def eval(self):
        self.is_training = False

    def train(self):
        self.is_training = True


