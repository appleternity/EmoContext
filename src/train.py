from data import load_data, load_twitter_data
from model import ConversationLSTM, ConversationBiLSTM, ConversationCNNLSTM
import tensorflow as tf
import tensorflow.contrib.eager as tfe
import numpy as np
import os
from config import *
from sklearn.metrics import *
import json

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
tf.enable_eager_execution(config=config)
tf.executing_eagerly()
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# TODO:
# v (1) Evaluation Metric (F1)
# (2) Test Data
# (3) Emoji Embedding
# (4) Existing Embedding, such as GloVe
# v (5) Bins Training for acceleration
# v (6) Check data (lower?)
# v (7) Weight
# (8) character-based

def metric(y_true, y_pred):
    return [
        precision_recall_fscore_support(y_true, y_pred, average="micro", labels=[0, 1, 2]),
        precision_recall_fscore_support(y_true, y_pred, average="macro", labels=[0, 1, 2]),
    ]

def my_metric(y_true, y_pred, eps=0.00000000001):
    matrix = np.zeros([4, 4], dtype=np.int32)
    # build confusion matrix
    for t, p in zip(y_true, y_pred): 
        matrix[p, t] += 1
    print(matrix)

    # compute tp, fp, fn
    tp = np.sum(matrix[i, i] for i in range(0, 3))
    fp = np.sum(matrix) - tp - np.sum(matrix[-1, :])
    fn = np.sum(matrix) - tp - np.sum(matrix[:, -1])

    # compute scores
    precision = tp / (tp+fp+eps)
    recall = tp / (tp+fn+eps)
    f1 = 2 * precision * recall / (precision + recall + eps)
    
    return precision, recall, f1

class EarlyStop:
    def __init__(self, mode="max", history=5):
        if mode == "max":
            self.best_func = np.max
            self.best_val = -np.inf 
            self.comp = lambda x, y: x >= y
        elif mode == "min":
            self.best_func = np.min
            self.best_val = np.inf 
            self.comp = lambda x, y: x <= y
        else:
            print("Please use 'max' or 'min' for mode.")
            quit()
        
        self.history_num = history
        self.history = np.zeros((self.history_num, ))
        self.total_num = 0

    def check(self, score):
        self.history[self.total_num % self.history_num] = score
        self.total_num += 1
        current_best_val = self.best_func(self.history)
        
        if self.total_num <= self.history_num:
            return False

        if self.comp(current_best_val, self.best_val):
            self.best_val = current_best_val
            return False
        else:
            return True

class MyCount:
    def __init__(self, start=0):
        self.num = start

    def count(self):
        temp = self.num
        self.num += 1
        return temp

class Trainer:
    def __init__(self):
        #self.learning_rate = 0.0003
        self.learning_rate = 0.001
        #self.learning_rate = 0.05
        #self.epoch_num = 1000
        self.epoch_num = 40
        self.output_size = 4
        self.history_num = 40

        # model hyper-parameter
        self.stack_num = 1
        self.cnn_num = 3
        self.batch_size = 64
        self.hidden_size = 300
        self.input_keep_rate = 0.80
        self.output_keep_rate = 0.80
        self.state_keep_rate = 0.80
        self.weight = 0.3

        self.residual = False
        self.character = True
        #self.bidirectional = True
        self.model_type = "cnn-lstm" # lstm / bi-lstm / cnn-lstm
        self.model_list = {
            "lstm": ConversationLSTM,
            "bi-lstm": ConversationBiLSTM,
            "cnn-lstm": ConversationCNNLSTM
        }
        self.outside_data = False

        self.optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate)
        self.loss = lambda y_pred, y_true: tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=y_pred, labels=y_true))

    def save_history(self, filename, history):
        history["train_metric"] = np.array(history["train_metric"]).tolist()
        history["valid_metric"] = np.array(history["valid_metric"]).tolist()

        with open(os.path.join(history_path, filename), 'w', encoding='utf-8') as outfile:
            json.dump(history, outfile, indent=4)

    def output_test(self, filename):
        pass
    
    def build_training_data(self, x_train, y_train):
        train_data = tf.data.Dataset.from_tensor_slices(
                (x_train[0], x_train[1], x_train[2], y_train)
            ).shuffle(
                buffer_size=200
            ).batch(batch_size=self.batch_size)
        return train_data

    def process_data(self, x, y, word_dictionary):
        x = self.turn_index(x, word_dictionary)
        max_len = max(len(row) for x in x for row in x)
        x = self.padding(x, max_len)
        y = self.vectorize(y)
        return x, y

    def accuracy_function(self, yhat, true_y):
        correct_prediction = tf.equal(tf.argmax(yhat, 1), tf.argmax(true_y, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        return accuracy

    def train(self):
        ## set model parameter
        model_name = "{}lstm_layer{}_dim{}_w{}".format(
                self.model_type, self.stack_num, self.hidden_size, self.weight
            )
        if self.model_type == "cnn-lstm":
            model_name += "_cnn{}".format(self.cnn_num)
        try:
            os.mkdir(os.path.join(model_path, model_name))
        except FileExistsError:
            pass

        ## build data
        data = load_data("train.txt")
        if self.character:
            x_train = [data["1"], data["2"], data["3"]]
        else:
            x_train = [data["1_tokenized"], data["2_tokenized"], data["3_tokenized"]]
        y_train = data.label.values
        word_mapping = self.build_dictionary(x_train)
        print("len(word) = {}".format(len(word_mapping)))

        data = load_data("dev.txt")
        if self.character:
            x_valid = [data["1"], data["2"], data["3"]]
        else:
            x_valid = [data["1_tokenized"], data["2_tokenized"], data["3_tokenized"]]
            
        y_valid = data.label.values
        
        #data = load_data("test.txt")
        #x_test = [data["1_tokenized"], data["2_tokenized"], data["3_tokenized"]]
        #y_test = data.label.values

        x_train, y_train = self.process_data(x_train, y_train, word_mapping)
        x_valid, y_valid = self.process_data(x_valid, y_valid, word_mapping)
        #x_test, y_test   = self.process_data(x_test, y_test, word_mapping)

        ## print data shape
        for setting, (x, y) in zip(
            ["train, valid, text"], 
            [[x_train, y_train], [x_valid, y_valid]],
            #[[x_train, y_train], [x_valid, y_valid], [x_test, y_test]],
        ):
            print(setting)
            print(" x = ", x[0].shape, " y = ", y.shape)

        ## build dataset
        train_data = tf.data.Dataset.from_tensor_slices(tuple(x_train + [y_train]))
        valid_data = tf.data.Dataset.from_tensor_slices(
                tuple(x_valid + [y_valid])
            ).batch(batch_size=self.batch_size)
        #test_data = tf.data.Dataset.from_tensor_slices(
        #        x_test
        #    ).batch(batch_size=self.batch_size)

        ############################
        # Twitter data
        x_twitter, y_twitter = load_twitter_data(num=500000) 
        
        unk_index = word_mapping["<UNK>"]
        x_twitter = [[word_mapping.get(w, unk_index) for w in row] for row in x_twitter]
        max_len = 150
        new_data = np.zeros((len(x_twitter), max_len), dtype=np.int32)
        for i, row in enumerate(x_twitter):
            length = min(len(row), max_len)
            new_data[i, max_len-length:] = row[:length]
        y_twitter = self.vectorize(y_twitter)
        print(new_data)
        twitter_data = tf.data.Dataset.from_tensor_slices((new_data, y_twitter))
        twitter_steps = int(new_data.shape[0] / self.batch_size)
        print(twitter_steps)

        total_steps = int(x_train[0].shape[0] / self.batch_size)

        ModelType = self.model_list[self.model_type]
        model = ModelType(
            hidden_size = self.hidden_size,
            batch_size = self.batch_size,
            stack_num = self.stack_num,
            input_keep_rate = self.input_keep_rate,
            output_keep_rate = self.output_keep_rate,
            state_keep_rate = self.state_keep_rate,
            residual = self.residual,
            weight = self.weight,
            vocab_size = len(word_mapping),
            cnn_num = self.cnn_num,
        )

        # save dictionary & important information
        with open(os.path.join(model_path, model_name, "info.json"), 'w', encoding='utf-8')  as outfile:
            json.dump(word_mapping, outfile, indent=4)

        ## train
        history = {
            "train":[],
            "validation":[],
            "train_metric": [],
            "valid_metric": [],
            "epoch":[],
        }
        prediction_history = []
        stopper = EarlyStop(mode="max", history=self.history_num)
        for epoch in range(self.epoch_num):
            model.train()
            
            if self.outside_data and epoch % 3 == 0:
                # twitter
                loss_array = []
                acc_array = []
                current_twitter_data = twitter_data.shuffle(x_train[0].shape[0]).batch(batch_size=self.batch_size)
                print("\nTraining Embedding")
                for step, (x, y) in enumerate(tfe.Iterator(current_twitter_data)):
                    with tf.GradientTape() as tape:
                        predicted = model.encode(x)
                        current_loss = model.loss(y_pred=predicted, y_true=y, weights=False)
                        grads = tape.gradient(current_loss, model.variables)
                        self.optimizer.apply_gradients(zip(grads, model.variables))
                        current_acc = self.accuracy_function(predicted, y)

                    loss_array.append(current_loss.numpy())
                    acc_array.append(current_acc.numpy())

                    if step % 1 == 0:
                        loss = round(np.hstack(loss_array).mean(), 5)
                        acc = round(np.hstack(acc_array).mean(), 5)
                        print("\x1b[2K\rEpoch:{} [{}%] loss={:.5f} acc={:.5f}".format(epoch, round(step/twitter_steps*100, 2), loss, acc), end="")

            # normal data
            current_train_data = train_data.shuffle(x_train[0].shape[0]).batch(batch_size=self.batch_size)
            loss_array = []
            acc_array = []
            y_true = []
            y_pred = []
            print("\nEmoContext Training")
            for step, (x1, x2, x3, y) in enumerate(tfe.Iterator(current_train_data), 1):
                with tf.GradientTape() as tape:
                    predicted = model(x1, x2, x3)
                    current_loss = model.loss(y_pred=predicted, y_true=y, weights=True)
                    grads = tape.gradient(current_loss, model.variables)
                    self.optimizer.apply_gradients(zip(grads, model.variables))
                    current_acc = self.accuracy_function(predicted, y)

                loss_array.append(current_loss.numpy())
                acc_array.append(current_acc.numpy())
                y_true.extend(tf.argmax(y, axis=1).numpy().tolist())
                y_pred.extend(tf.argmax(predicted, axis=1).numpy().tolist())

                if step % 1 == 0:
                    loss = round(np.hstack(loss_array).mean(), 5)
                    acc = round(np.hstack(acc_array).mean(), 5)
                    print("\x1b[2K\rEpoch:{} [{}%] loss={:.5f} acc={:.5f}".format(epoch, round(step/total_steps*100, 2), loss, acc), end="")

            loss = round(np.hstack(loss_array).mean(), 5)
            acc = round(np.hstack(acc_array).mean(), 5)
            res = metric(y_true, y_pred)
            res = [float(res[0][0]), float(res[0][1]), float(res[0][2]), float(res[1][0]), float(res[1][1]), float(res[1][2])]
            print("\nMicro: p = {:.4f} r = {:.4f} f = {:.4f} | Macro: p = {:.4f} r = {:.4f} f = {:.4f}".format(*res))
            #my_res = my_metric(y_true, y_pred)
            #print(my_res)
            history["train"].append(float(acc))
            history["epoch"].append(epoch)
            history["train_metric"].append(res)

            # validation
            loss_array = []
            acc_array = []
            y_true = []
            y_pred = []
            model.eval()
            for step, (x1, x2, x3, y) in enumerate(tfe.Iterator(valid_data), 1):
                predicted = model(x1, x2, x3)
                current_loss = model.loss(y_pred=predicted, y_true=y, weights=True)
                current_acc = self.accuracy_function(predicted, y)

                loss_array.append(current_loss.numpy())
                acc_array.append(current_acc.numpy())
                y_true.extend(tf.argmax(y, axis=1).numpy().tolist())
                y_pred.extend(tf.argmax(predicted, axis=1).numpy().tolist())

            loss = round(np.hstack(loss_array).mean(), 5)
            acc = round(np.hstack(acc_array).mean(), 5)
            res = metric(y_true, y_pred)
            res = [float(res[0][0]), float(res[0][1]), float(res[0][2]), float(res[1][0]), float(res[1][1]), float(res[1][2])]
            history["validation"].append(float(acc))
            history["valid_metric"].append(res)
            print("\nValid loss={:.5f} acc={:.5f}".format(loss, acc))
            print("Micro: p = {:.4f} r = {:.4f} f = {:.4f} | Macro: p = {:.4f} r = {:.4f} f = {:.4f}".format(*res))
            #my_res = my_metric(y_true, y_pred)
            #print(my_res)

            self.save_history(model_name + ".json", history)

            # check early stopping
            if stopper.check(acc):
                print("Early Stopping at Epoch = ", epoch)
                break

            # save model
            if epoch % 1 == 0:
                tfe.Saver(model.variables).save(
                    os.path.join(model_path, model_name, "checkpoint"),
                    global_step=epoch,
                )
        
        # save model
        tfe.Saver(model.variables).save(
            os.path.join(model_path, model_name, "checkpoint"),
            global_step=epoch,
        )
        
        # output prediction history
        self.save_history(model_name + ".json", history)
        #prediction_history = np.vstack(prediction_history)
        #with open(os.path.join(prediction_history_path, model_name+".pkl"), 'wb') as outfile:
        #    pickle.dump(prediction_history, outfile)

        ## test
        model.eval()
        result_pred = []
        for step, (x1, x2, x3) in enumerate(tfe.Iterator(test_data), 1):
            predicted = model(x1, x2, x3)
            pred_res = tf.argmax(predicted)
            result_pred.append(pred_res)
        result_pred = np.hstack(result_pred)
        save_result(model_name + ".pkl", result_red)

    def build_dictionary(self, data_list, min_freq=5, verbose=True):
        # 0: <PAD>, 1: <UNK>
        freq = {}
        for data in data_list:
            for row in data:
                for word in row:
                    freq[word] = freq.get(word, 0) + 1

        counter = MyCount(start=2)
        freq_filtered = {
            word : counter.count()
            for word, count in freq.items()
            if count > min_freq
        }
        freq_filtered["<PAD>"] = 0
        freq_filtered["<UNK>"] = 1
        if verbose:
            print("len(freq_filtered) = {}, freq = {}".format(len(freq_filtered), len(freq)))
        return freq_filtered

    def turn_index(self, data_list, word_dictionary):
        unk_index = word_dictionary["<UNK>"]
        new_data = [
            [
                [word_dictionary.get(word, unk_index) for word in row]
                for row in data    
            ]
            for data in data_list        
        ]
        return new_data

    def padding(self, data_list, pad_num=50, pad_index=0):
        new_data_list = []
        for data in data_list:
            new_data = np.ones((len(data), pad_num), dtype=np.int32) * pad_index
            for i, row in enumerate(data):
                length = min(len(row), pad_num)
                new_data[i, pad_num-length:] = row[:length]
            new_data_list.append(new_data)
        return new_data_list

    def vectorize(self, y):
        if type(y) != np.ndarray:
            y = np.array(y)
        uni = np.unique(y)
        one_hot = np.zeros([y.shape[0], uni.shape[0]], dtype=np.float32)
        for i, yy in enumerate(y):
            one_hot[i, yy] = 1
        return one_hot

def main():
    # build trainer
    trainer = Trainer()
    trainer.train()

if __name__ == "__main__":
    main()
