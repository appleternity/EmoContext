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
        self.weight           = kwargs.get("weight", 0.1)
        self.f1_weight        = tf.convert_to_tensor([1.0, 1.0, 1.0, self.weight], dtype=tf.float32)

        self.is_training = True
        self.loss = self.weighted_cross_entropy_loss

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
        # cut length
        s = tf.reduce_sum(text_1, axis=0)
        text_1 = tf.boolean_mask(text_1, s>0, axis=1)
        s = tf.reduce_sum(text_2, axis=0)
        text_2 = tf.boolean_mask(text_2, s>0, axis=1)
        s = tf.reduce_sum(text_3, axis=0)
        text_3 = tf.boolean_mask(text_3, s>0, axis=1)

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

    def weighted_cross_entropy_loss(self, y_true, y_pred, weights=False):
        if weights:
            l = tf.nn.softmax_cross_entropy_with_logits_v2(logits=y_pred, labels=y_true)
            w = tf.ones([y_true.shape[0]], dtype=tf.float32) - y_true[:, 3]*self.weight
            return tf.reduce_mean(l*w)
        else:
            l = tf.nn.softmax_cross_entropy_with_logits_v2(logits=y_pred, labels=y_true)
            return tf.reduce_mean(l)

    def f1_loss(self, y_true, y_pred, weights=False):
        pass

    def f1_loss_macro(self, y_pred, y_true, eps=1e-10, weights=False):
        y_pred = tf.nn.sigmoid(y_pred)
        tp = tf.reduce_sum(y_pred * y_true, axis=0)
        fp = tf.reduce_sum(y_pred * (1-y_true), axis=0)
        fn = tf.reduce_sum((1-y_pred) * y_true, axis=0)
    
        p = tp / (tp + fp + eps)
        r = tp / (tp + fn + eps)
        f1 = 2*p*r / (p+r+eps)
        #if weights:
        #    f1 = f1 * self.f1_weight
        return 1 - tf.reduce_mean(f1)

class ConversationBiLSTM(tf.keras.Model):
    def __init__(self, **kwargs):
        print(kwargs)
        super(ConversationBiLSTM, self).__init__()

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
        self.weight           = kwargs.get("weight", 0.1)
        self.f1_weight        = tf.convert_to_tensor([1.0, 1.0, 1.0, self.weight], dtype=tf.float32)

        self.is_training = True
        self.loss = self.weighted_cross_entropy_loss

        # building model
        if self.word_embedding:
            self.word_embedding = tfe.Variable(tf.convert_to_tensor(self.word_embedding), name="word_embedding")
        else:
            self.word_embedding = tfe.Variable(tf.random.normal([self.vocab_size, self.size_hidden]), name="word_embedding")

        self.lstm_layers = [
            tf.contrib.cudnn_rnn.CudnnCompatibleLSTMCell(num_units=self.size_hidden)
            for _ in range(self.stack_num)
        ]
        self.lstm_layers_back = [
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

            self.cell_back = tf.nn.rnn_cell.MultiRNNCell(self.lstm_layers_back)
            self.dropout_cell_back = tf.nn.rnn_cell.MultiRNNCell([
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
                for i, lstm_cell in enumerate(self.lstm_layers_back)
            ])

        self.batch_norm_1 = tf.keras.layers.BatchNormalization()
        self.batch_norm_2 = tf.keras.layers.BatchNormalization()
        self.batch_norm_3 = tf.keras.layers.BatchNormalization()

        self.output_dense_1 = tf.keras.layers.Dense(self.size_hidden*2, name="output_1")
        self.output_dense_2 = tf.keras.layers.Dense(self.size_hidden//5, name="output_2")
        self.output_dense_3 = tf.keras.layers.Dense(self.size_output, name="output_3")
            
    def call(self, text_1, text_2, text_3):
        # cut length
        s = tf.reduce_sum(text_1, axis=0)
        text_1 = tf.boolean_mask(text_1, s>0, axis=1)
        s = tf.reduce_sum(text_2, axis=0)
        text_2 = tf.boolean_mask(text_2, s>0, axis=1)
        s = tf.reduce_sum(text_3, axis=0)
        text_3 = tf.boolean_mask(text_3, s>0, axis=1)

        # embedding
        embedding_1 = tf.nn.embedding_lookup(self.word_embedding, text_1)
        embedding_2 = tf.nn.embedding_lookup(self.word_embedding, text_2)
        embedding_3 = tf.nn.embedding_lookup(self.word_embedding, text_3)
        
        # lstm 
        init_state = self.cell.zero_state(text_1.shape[0], tf.float32)
        if self.is_training:
            ((rnn_fw_1, rnn_bw_1), _) = tf.nn.bidirectional_dynamic_rnn(self.dropout_cell, self.dropout_cell_back, embedding_1, dtype=tf.float32)
            ((rnn_fw_2, rnn_bw_2), _) = tf.nn.bidirectional_dynamic_rnn(self.dropout_cell, self.dropout_cell_back, embedding_2, dtype=tf.float32)
            ((rnn_fw_3, rnn_bw_3), _) = tf.nn.bidirectional_dynamic_rnn(self.dropout_cell, self.dropout_cell_back, embedding_3, dtype=tf.float32)
        else:
            ((rnn_fw_1, rnn_bw_1), _) = tf.nn.bidirectional_dynamic_rnn(self.cell, self.cell_back, embedding_1, dtype=tf.float32)
            ((rnn_fw_2, rnn_bw_2), _) = tf.nn.bidirectional_dynamic_rnn(self.cell, self.cell_back, embedding_2, dtype=tf.float32)
            ((rnn_fw_3, rnn_bw_3), _) = tf.nn.bidirectional_dynamic_rnn(self.cell, self.cell_back, embedding_3, dtype=tf.float32)

        #representation = tf.concat([rnn_output_1[:, -1, :], rnn_output_2[:, -1, :], rnn_output_3[:, -1, :]], axis=1)
        representation = tf.concat([
            rnn_fw_1[:, -1, :], rnn_bw_1[:, 0, :], 
            rnn_fw_2[:, -1, :], rnn_bw_2[:, 0, :], 
            rnn_fw_3[:, -1, :], rnn_bw_3[:, 0, :]], axis=1)
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

    def weighted_cross_entropy_loss(self, y_true, y_pred, weights=False):
        if weights:
            l = tf.nn.softmax_cross_entropy_with_logits_v2(logits=y_pred, labels=y_true)
            w = tf.ones([y_true.shape[0]], dtype=tf.float32) - y_true[:, 3]*self.weight
            return tf.reduce_mean(l*w)
        else:
            l = tf.nn.softmax_cross_entropy_with_logits_v2(logits=y_pred, labels=y_true)
            return tf.reduce_mean(l)

    def f1_loss(self, y_true, y_pred, weights=False):
        pass

    def f1_loss_macro(self, y_pred, y_true, eps=1e-10, weights=False):
        y_pred = tf.nn.sigmoid(y_pred)
        tp = tf.reduce_sum(y_pred * y_true, axis=0)
        fp = tf.reduce_sum(y_pred * (1-y_true), axis=0)
        fn = tf.reduce_sum((1-y_pred) * y_true, axis=0)
    
        p = tp / (tp + fp + eps)
        r = tp / (tp + fn + eps)
        f1 = 2*p*r / (p+r+eps)
        #if weights:
        #    f1 = f1 * self.f1_weight
        return 1 - tf.reduce_mean(f1)

