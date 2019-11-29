from data import load_data
from model import ConversationLSTM
import tensorflow as tf
import tensorflow.contrib.eager as tfe
tf.enable_eager_execution()
tf.executing_eagerly()

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

class Trainer:
    def __init__(self):
        #self.learning_rate = 0.0003
        self.learning_rate = 0.001
        #self.learning_rate = 0.05
        #self.epoch_num = 1000
        self.epoch_num = 1000
        self.output_size = 22170
        self.history_num = 30

        # model hyper-parameter
        self.batch_size = 30
        self.hidden_size = 1500
        self.num_multi_task = 2
        self.stack_num = 3
        self.input_keep_rate = 0.80
        self.output_keep_rate = 0.80
        self.state_keep_rate = 0.80

        #self.data_mode = "all" # "train"
        self.data_mode = "train"
        self.weighted = True
        self.year = False
        self.residual = False

        self.optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate)
        #self.loss = tf.losses.mean_squared_error
        self.loss = tf.keras.losses.Huber(delta=0.5, reduction=tf.losses.Reduction.NONE)

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

    def train(self):
        ## set model parameter
        model = TimeLSTM(
            hidden_size = self.hidden_size,
            num_multi_task = self.num_multi_task,
            batch_size = self.batch_size,
            stack_num = self.stack_num,
            input_keep_rate = self.input_keep_rate,
            output_keep_rate = self.output_keep_rate,
            state_keep_rate = self.state_keep_rate,
            residual = self.residual,
        )

        model_name = "lstm_layer{}_dim{}_multi{}_vr_{}{}{}{}".format(
                self.stack_num, self.hidden_size, self.num_multi_task, self.data_mode,
                "" if self.weighted is False else "_weighted",
                "" if self.year is False else "_year",
                "" if self.residual is False else "_res"
            )
        try:
            os.mkdir(os.path.join(model_path, model_name))
        except:
            pass

        ## build data
        data = load_time_sequence_matrix()
        data = data.astype(np.float32)
        print("data.shape = ", data.shape)
        data = np.clip(data, 0, 20)

        if self.data_mode == "train":
            x_train = data[0:32, :]
            y_train = data[1:33, :]
        elif self.data_mode == "all":
            x_train = data[0:33, :]
            y_train = data[1:34, :]
        else:
            print("Please enter a valid data_mode value ['train', 'all']")
            quit()

        x_valid = data[0:33, :]
        y_valid = data[1:34, :]
        x_test = data[0:34, :]

        x_train, y_train = build_lstm_data(x_train, y_train, verbose=True) 
        x_valid, y_valid = build_lstm_data(x_valid, y_valid)
        x_test, _ = build_lstm_data(x_test)

        ## build dataset
        train_data = tf.data.Dataset.from_tensor_slices(tuple(list(x_train) + [y_train]))
        valid_data = tf.data.Dataset.from_tensor_slices(
                tuple(list(x_valid) + [y_valid])
            ).batch(batch_size=self.batch_size)
        test_data = tf.data.Dataset.from_tensor_slices(
                x_test
            ).batch(batch_size=self.batch_size)

        total_steps = int(x_train[0].shape[0] / self.batch_size)

        # load mask for testing
        item_ids = load_testing_items()  
        mask = np.ones(self.output_size, dtype=np.float32) * 0.05
        mask[item_ids] = 1.0
        mask = tf.convert_to_tensor(mask)
        mask = tf.reshape(mask, [1, 1, -1])
        print(mask)
        if self.weighted == True:
            def loss_function(y_true, y_pred): 
                raw_loss = self.loss(y_true, y_pred)
                return tf.reduce_sum(tf.losses.compute_weighted_loss(raw_loss, weights=mask))
        else:
            def loss_function(y_true, y_pred):
                return tf.reduce_mean(self.loss(y_true, y_pred))

        ## train
        history = {
            "validation":[],
            "epoch":[],
            "prediction":[],
        }
        prediction_history = []
        stopper = EarlyStop(mode="min", history=self.history_num)
        for epoch in range(self.epoch_num):
            model.train()
            current_train_data = train_data.shuffle(200).batch(batch_size=self.batch_size)
            loss_array = []
            all_loss_array = []
            for step, (x, shop_ids, seq_len, month_seq, year_seq, y) in enumerate(tfe.Iterator(current_train_data), 1):
                with tf.GradientTape() as tape:
                    if self.year:
                        predicted, predicted_next_list = model(shop_ids, x, seq_len, month_seq, year_seq)
                    else:
                        predicted, predicted_next_list = model(shop_ids, x, seq_len, month_seq)
                    current_loss = loss_function(y, predicted)

                    # multi-task
                    next_loss = sum([
                        loss_function(y[:, i:, :], predicted_next[:, :-i, :])
                        for i, predicted_next in enumerate(predicted_next_list, 1)
                    ])
                    
                    all_loss = (self.num_multi_task+1)*current_loss + next_loss

                    grads = tape.gradient(all_loss, model.variables)
                    self.optimizer.apply_gradients(zip(grads, model.variables))
                    
                    loss_array.append(current_loss.numpy())
                    all_loss_array.append(all_loss.numpy())

                if step % 1 == 0:
                    loss = round(np.hstack(loss_array).mean(), 5)
                    all_loss = round(np.hstack(all_loss_array).mean(), 5)
                    print("\x1b[2K\rEpoch:{} [{}%] loss={:.5f} all_loss={:.5f}".format(epoch, round(step/total_steps*100, 2), loss, all_loss), end="")

            # validation
            if epoch % 1 == 0:
                print("\nvalidation ", end="")
                model.eval()
                result_pred = []
                result_true = []
                for step, (x, shop_ids, seq_len, month_seq, year_seq, y) in enumerate(tfe.Iterator(valid_data), 1):
                    if self.year:
                        predicted, predicted_next_list = model(shop_ids, x, seq_len, month_seq, year_seq)
                    else:
                        predicted, predicted_next_list = model(shop_ids, x, seq_len, month_seq)
                    result_pred.append(predicted.numpy())
                    result_true.append(y.numpy())
                
                y_true = np.vstack(result_true)[:, -1, :]
                y_pred = np.vstack(result_pred)[:, -1, :]
                score = self.evaluation(y_true, y_pred)
                print("score = ", score)
                history["validation"].append(score)
                history["epoch"].append(epoch)
                prediction_history.append(y_pred.reshape([1, -1]))

                # check early stopping
                if stopper.check(score):
                    print("Early Stopping at Epoch = ", epoch)
                    break

            # save model
            if epoch % 100 == 0:
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

def main():
    # build model
    
    # build trainer
    trainer = Trainer()
    trainer.train()
    #model_name = "lstm_early_stop_cleaning_month_dim300_ed_vr_all"
    #trainer.output_test(filename=model_name+".pkl")

if __name__ == "__main__":
    main()
