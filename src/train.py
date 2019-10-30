from data import load_data
from model import ConversationLSTM
import tensorflow as tf
import tensorflow.contrib.eager as tfe
tf.enable_eager_execution()
tf.executing_eagerly()

class Trainer:
    def __init__(self):
        self.batch_size = 5
        self.learning_rate = 0.0003
        self.epoch_num = 700
        self.output_size = 22170

        self.optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate)
        self.loss = tf.losses.mean_squared_error # hinge

    def evaluation(self, y_true, y_pred):
        # clipping into range [0, 20]
        y_true = np.clip(y_true, 0, 20)
        y_pred = np.clip(y_pred, 0, 20)
    
        y_diff = y_true - y_pred
        se = np.square(y_diff)
        mse = se.mean()
        rmse = np.power(mse, 0.5)
        return rmse

    def save_history(self, filename, history):
        with open(os.path.join(history_path, filename), 'w', encoding='utf-8') as outfile:
            json.dump(history, outfile, indent=4)

    def output_test(self, filename):
        testing_data = load_testing()
        res = load_result(filename).reshape((-1, ))
        print(res.shape)
        print(testing_data)

        index_list = testing_data[:, 1]*22170 + testing_data[:, 2]
        result = res[index_list].reshape((-1, 1))
        result = np.clip(result, 0, 20)

        print(result.shape)
        result = np.hstack([testing_data[:, 0].reshape((-1, 1)), result])
        print(result.shape)
        
        table = pd.DataFrame(result, columns=["ID", "item_cnt_month"])
        table.ID = table.ID.astype(int)
        table.to_csv(
            os.path.join(submission_path, "lstm.csv"),
            index=False
        )
    
    def build_training_data(self, x_train, y_train):
        train_data = tf.data.Dataset.from_tensor_slices(
                (x_train[0], x_train[1], x_train[2], y_train)
            ).shuffle(
                buffer_size=32
            ).batch(batch_size=self.batch_size)
        return train_data

    def train(self, model):
        ## build data
        data = load_time_sequence_matrix()
        data = data.astype(np.float32)
        print("data.shape = ", data.shape)

        x_train = data[0:32, :]
        y_train = data[1:33, :]
        x_valid = data[0:33, :]
        y_valid = data[1:34, :]
        x_test = data[0:34, :]
        
        x_train, y_train = build_lstm_data(x_train, y_train) 
        x_valid, y_valid = build_lstm_data(x_valid, y_valid)
        x_test, _ = build_lstm_data(x_test)

        ## build dataset
        train_data = tf.data.Dataset.from_tensor_slices(
                (x_train[0], x_train[1], x_train[2], y_train)
            ).batch(batch_size=self.batch_size)

        valid_data = tf.data.Dataset.from_tensor_slices(
                (x_valid[0], x_valid[1], x_valid[2], y_valid)
            ).batch(batch_size=self.batch_size)

        test_data = tf.data.Dataset.from_tensor_slices(
                x_test
            ).batch(batch_size=self.batch_size)

        total_steps = int(x_train[0].shape[0] / self.batch_size)

        ## train
        history = {
            "validation":[],
            "epoch":[],
        }
        for epoch in range(self.epoch_num):
            current_train_data = train_data.shuffle(32)
            loss_array = []
            for step, (x, shop_ids, seq_len, y) in enumerate(tfe.Iterator(current_train_data), 1):
                with tf.GradientTape() as tape:
                    predicted = model.predict_train(shop_ids, x, seq_len)
                    current_loss = self.loss(y, predicted)

                    grads = tape.gradient(current_loss, model.get_variable())
                    self.optimizer.apply_gradients(zip(grads, model.variables))
                    
                    loss_array.append(current_loss.numpy())

                if step % 2 == 0:
                    loss = round(np.hstack(loss_array).mean(), 5)
                    print("\x1b[2K\rEpoch:{} [{}%] loss={} ".format(epoch, round(step/total_steps*100, 2), loss), end="")

            # validation
            if epoch % 1 == 0:
                print("\nvalidation ", end="")
                result_pred = []
                result_true = []
                for step, (x, shop_ids, seq_len, y) in enumerate(tfe.Iterator(valid_data), 1):
                    predicted = model.predict(shop_ids, x, seq_len)
                    result_pred.append(predicted.numpy())
                    result_true.append(y.numpy())
                
                y_true = np.vstack(result_true)[:, -1, :]
                y_pred = np.vstack(result_pred)[:, -1, :]
                score = self.evaluation(y_true, y_pred)
                print("score = ", score)
                history["validation"].append(score)
                history["epoch"].append(epoch)

        ## test
        result_pred = []
        for step, (x, shop_ids, seq_len) in enumerate(tfe.Iterator(test_data), 1):
            predicted = model.predict(shop_ids, x, seq_len)
            result_pred.append(predicted.numpy())
        y_pred = np.vstack(result_pred)[:, -1, :]
        save_result("lstm_2000.pkl", y_pred)
        self.save_history("lstm_2000.json", history)


def main():
    pass

if __name__ == "__main__":
    main()
