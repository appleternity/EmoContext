import tensorflow as tf
import tensorflow.contrib.eager as tfe
tf.enable_eager_execution()
tf.executing_eagerly()

class ConversationLSTM:
    def __init__(self):
       	# parameter setting
        #self.size_input = 22170
        self.size_hidden = 300
        self.size_output = 4
        self.shop_num = 60
        self.batch_size = 5
        self.keep_rate = 0.8
        self.num_lstm = 3
        self.vocab_size = 300

        # building model
        self.word_embedding = tfe.Variable(tf.random.normal(self.vocab_size, self.self.size_hidden))

        self.lstm_layers = [tf.contrib.cudnn_rnn.CudnnCompatibleLSTMCell(num_units=self.size_hidden) for _ in range(self.num_lstm)]
        self.cell = tf.nn.rnn_cell.MultiRNNCell(self.lstm_layers)
        self.dropout_cell = tf.nn.rnn_cell.MultiRNNCell([
            tf.contrib.rnn.DropoutWrapper(
                lstm_cell,
                input_keep_prob=self.keep_rate,
                output_keep_prob=self.keep_rate,
                state_keep_prob=self.keep_rate,
            )
            for lstm_cell in self.lstm_layers
        ])

        self.w = tfe.Variable(tf.random.normal([self.size_input, self.size_hidden]))
        self.b = tfe.Variable(tf.random.normal([1, self.size_hidden]))
        
        self.output_w = tfe.Variable(tf.random.normal([self.size_hidden, self.size_output]))
        self.output_b = tfe.Variable(tf.random.normal([1, self.size_output]))

        #self.variables = [self.cell.variables, self.w, self.b, self.shop_adaption_w, self.output_w, self.output_b]
        self.variables = None   

    def get_variable(self):
        self.variables = self.cell.variables + [self.w, self.b, self.shop_adaption_w, self.output_w, self.output_b]
        self.get_variable = self.get_variable_static
        return self.variables

    def get_variable_static(self):
        return self.variables

    def predict_train(self, turn_1, turn_2, turn_3):
        # turns: [batch_size, seq_len]
        emb_1 = tf.nn.embedding_lookup(self.word_embedding, turn_1)
        emb_2 = tf.nn.embedding_lookup(self.word_embedding, turn_2)
        emb_3 = tf.nn.embedding_lookup(self.word_embedding, turn_3)

        init_state = self.cell.zero_state(self.batch_size, tf.float32)
        rep_1, _ = tf.nn.dynamic_rnn(self.dropout_cell, emb_1, initial_state=init_state)
        rep_2, _ = tf.nn.dynamic_rnn(self.dropout_cell, emb_2, initial_state=init_state)
        rep_3, _ = tf.nn.dynamic_rnn(self.dropout_cell, emb_3, initial_state=init_state)

        print(rep_1.shape)
        print(rep_2.shape)
        print(rep_3.shape)
        quit()
        tf.concat((rep_1, rep_2, rep_3), axis=1)

    def predict(self, turn_1, turn_2, turn_3, seq_1, seq_2, seq_3):
        pass

def main():
    pass

if __name__ == "__main__":
    main()
