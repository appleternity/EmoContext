from data import load_data
from model import ConversationLSTM
import tensorflow as tf
import tensorflow.contrib.eager as tfe
import numpy as np
import os
from config import *
from sklearn.metrics import *
tf.enable_eager_execution()
tf.executing_eagerly()
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# TODO:
# (1) Evaluation Metric (F1)
# (2) Test Data
# (3) Emoji Embedding
# (4) Existing Embedding, such as GloVe
# (5) Bins Training for acceleration
# (6) Check data (lower?)

def metric(y_true, y_pred):
    return [
        precision_recall_fscore_support(y_true, y_pred, average="micro", labels=[0, 1, 2]),
        precision_recall_fscore_support(y_true, y_pred, average="micro", labels=[0, 1, 2, 3]),
        precision_recall_fscore_support(y_true, y_pred, average="macro", labels=[0, 1, 2]),
        precision_recall_fscore_support(y_true, y_pred, average="macro", labels=[0, 1, 2, 3]),
    ]

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
        self.learning_rate = 0.0001
        #self.learning_rate = 0.05
        #self.epoch_num = 1000
        self.epoch_num = 100
        self.output_size = 4
        self.history_num = 5

        # model hyper-parameter
        self.batch_size = 512
        self.hidden_size = 128
        self.stack_num = 5
        self.input_keep_rate = 0.80
        self.output_keep_rate = 0.80
        self.state_keep_rate = 0.80

        self.residual = False

        self.optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate)
        self.loss = lambda y_pred, y_true: tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=y_pred, labels=y_true))

    def save_history(self, filename, history):
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

    def process_data(x, y, word_dictionary):
        x = self.turn_index(x, word_mapping)
        max_len = max(len(row) for x in x for row in x)
        x = self.padding(x, max_len)
        y = self.vectorize(y)

    def accuracy_function(self, yhat, true_y):
        correct_prediction = tf.equal(tf.argmax(yhat, 1), tf.argmax(true_y, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        return accuracy

    def train(self):
        ## set model parameter
        model = ConversationLSTM(
            hidden_size = self.hidden_size,
            batch_size = self.batch_size,
            stack_num = self.stack_num,
            input_keep_rate = self.input_keep_rate,
            output_keep_rate = self.output_keep_rate,
            state_keep_rate = self.state_keep_rate,
            residual = self.residual,
        )

        model_name = "lstm_layer{}_dim{}".format(
                self.stack_num, self.hidden_size,
                "" if self.residual is False else "_res"
            )
        try:
            os.mkdir(os.path.join(model_path, model_name))
        except FileExistsError:
            pass

        ## build data
        data = load_data("train.txt") 
        x_train = [data["1_tokenized"], data["2_tokenized"], data["3_tokenized"]]
        y_train = data.label.values
        word_mapping = self.build_dictionary(x_train)
        print("len(word) = {}".format(len(word_mapping)))

        data = load_data("valid.txt")
        x_valid = [data["1_tokenized"], data["2_tokenized"], data["3_tokenized"]]
        y_valid = data.label.values
        
        data = load_data("test.txt")
        x_test = [data["1_tokenized"], data["2_tokenized"], data["3_tokenized"]]
        y_test = data.label.values

        x_train, y_train = self.process_data(x_train, y_train, word_mapping)
        x_valid, y_valid = self.process_data(x_valid, y_valid, word_mapping)
        x_test, y_test   = self.process_data(x_test, y_test, word_mapping)

        ## print data shape
        for setting, (x, y) in zip(
            ["train, valid, text"], 
            [[x_train, y_train], [x_valid, y_valid], [x_test, y_test]]
        ):
            print(setting)
            print(" x = ", x.shape, " y = ", y.shape)

        ## build dataset
        train_data = tf.data.Dataset.from_tensor_slices(x_train + [y_train])
        valid_data = tf.data.Dataset.from_tensor_slices(
                x_valid + [y_valid]
            ).batch(batch_size=self.batch_size)
        test_data = tf.data.Dataset.from_tensor_slices(
                x_test
            ).batch(batch_size=self.batch_size)

        total_steps = int(x_train[0].shape[0] / self.batch_size)

        ## train
        history = {
            "train":[],
            "validation":[],
            "epoch":[],
        }
        prediction_history = []
        stopper = EarlyStop(mode="max", history=self.history_num)
        for epoch in range(self.epoch_num):
            model.train()
            current_train_data = train_data.shuffle(x_train.shape[0]).batch(batch_size=self.batch_size)
            loss_array = []
            acc_array = []
            for step, (x1, x2, x3, y) in enumerate(tfe.Iterator(current_train_data), 1):
                with tf.GradientTape() as tape:
                    predicted = model(x1, x2, x3)
                    current_loss = self.loss(predicted, y)
                    grads = tape.gradient(current_loss, model.variables)
                    self.optimizer.apply_gradients(zip(grads, model.variables))
                    current_acc = self.accuracy_function(predicted, y)

                loss_array.append(current_loss.numpy())
                acc_array.append(current_acc.numpy())

                if step % 1 == 0:
                    loss = round(np.hstack(loss_array).mean(), 5)
                    acc = round(np.hstack(acc_array).mean(), 5)
                    print("\x1b[2K\rEpoch:{} [{}%] loss={:.5f} acc={:.5f}".format(epoch, round(step/total_steps*100, 2), loss, acc), end="")

            loss = round(np.hstack(loss_array).mean(), 5)
            acc = round(np.hstack(acc_array).mean(), 5)
            history["train"].append(acc)
            history["epoch"].append(epoch)

            # validation
            loss_array = []
            acc_array = []
            model.eval()
            for step, (x1, x2, x3, y) in enumerate(tfe.Iterator(valid_data), 1):
                predicted = model(x1, x2, x3)
                current_loss = self.loss(predicted, y)
                current_acc = self.accuracy_function(predicted, y)

                loss_array.append(current_loss.numpy())
                acc_array.append(current_acc.numpy())

            loss = round(np.hstack(loss_array).mean(), 5)
            acc = round(np.hstack(acc_array).mean(), 5)
            history["validation"].append(acc)
            print("\nValid loss={:.5f} acc={:.5f}".format(loss, acc))

            # check early stopping
            if stopper.check(acc):
                print("Early Stopping at Epoch = ", epoch)
                break

            # save model
            if epoch % 10 == 0:
                tfe.Saver(model.variables).save(
                    os.path.join(model_path, model_name, "checkpoint"),
                    global_step=epoch,
                )
        
        # save model
        tfe.Saver(model.variables).save(
            os.path.join(model_path, model_name, "checkpoint"),
            global_step=epoch,
        )
        
        """
        # output prediction history
        prediction_history = np.vstack(prediction_history)
        with open(os.path.join(prediction_history_path, model_name+".pkl"), 'wb') as outfile:
            pickle.dump(prediction_history, outfile)

        ## test
        model.eval()
        result_pred = []
        for step, (x, shop_ids, seq_len, month_seq, year_seq) in enumerate(tfe.Iterator(test_data), 1):
            if self.year:
                predicted, predicted_next_list = model(shop_ids, x, seq_len, month_seq, year_seq)
            else:
                predicted, predicted_next_list = model(shop_ids, x, seq_len, month_seq)
            result_pred.append(predicted.numpy())
        y_pred = np.vstack(result_pred)[:, -1, :]
        save_result(model_name + ".pkl", y_pred)
        self.save_history(model_name + ".json", history)
        """

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
