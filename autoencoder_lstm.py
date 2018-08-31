import tensorflow as tf
from tensorflow.contrib.rnn import LSTMCell, LayerNormBasicLSTMCell
from tensorflow.nn.rnn_cell import MultiRNNCell
from tensorflow.contrib.layers import xavier_initializer


class AutoEncoder_LSTM(object):
    def __init__(self, layer1_hidden, layer2_hidden, layer3_hidden, lstm_hidden, dropout_keep_prob, learning_rate,
                 sequence_length, feature_num, num_layers):
        self.layer1_hidden = layer1_hidden
        self.layer2_hidden = layer2_hidden
        self.layer3_hidden = layer3_hidden
        self.lstm_hidden = lstm_hidden
        self.dropout_keep_prob = dropout_keep_prob
        self.learning_rate = learning_rate
        self.sequence_length = sequence_length
        self.feature_number = feature_num
        self.num_layers = num_layers

        self.sequence_lengths = tf.placeholder(tf.int32, shape=[None], name='sequence_lengths')
        self.x = tf.placeholder(tf.float32, shape=[None, self.sequence_length], name='auto_encoder')
        self.y = tf.placeholder(tf.float32, shape=[None, self.sequence_length], name='lstm_label')

        self.x = tf.reshape(self.x, [-1, self.sequence_length, self.feature_number])
        self.y = tf.reshape(self.y, [-1, self.sequence_length, 1])

        with tf.variable_scope('encoder'):
            encoder_cell_l1 = LayerNormBasicLSTMCell(self.layer1_hidden)
            encoder_cell_l2 = LayerNormBasicLSTMCell(self.layer2_hidden)
            encoder_cell_l3 = LayerNormBasicLSTMCell(self.layer3_hidden)
            encoder_cells = MultiRNNCell([encoder_cell_l1, encoder_cell_l2, encoder_cell_l3])

            encoder_output, context = tf.nn.dynamic_rnn(
                cell=encoder_cells,
                inputs=self.x,
                sequence_length=self.sequence_lengths,
                dtype=tf.float32,
                scope='encoder_layer'
            )

        with tf.variable_scope('decoder'):
            decoder_cell_layer1 = LayerNormBasicLSTMCell(self.layer1_hidden)
            decoder_cell_layer2 = LayerNormBasicLSTMCell(self.layer2_hidden)
            decoder_cell_layer3 = LayerNormBasicLSTMCell(self.layer3_hidden)
            decoder_cells = MultiRNNCell([decoder_cell_layer1, decoder_cell_layer2, decoder_cell_layer3])

            dec_inputs = tf.zeros(tf.shape(self.x), dtype=tf.float32)

            decoder_output, _ = tf.nn.dynamic_rnn(
                cell=decoder_cells,
                inputs=dec_inputs,
                sequence_length=self.sequence_lengths,
                initial_state=context,
                dtype=tf.float32,
                scope='decoder_layer'
            )

        # print(encoder_output)  shape = [batch_size, sequence_length, layer3_hidden]

        with tf.variable_scope('LSTM_Layer'):
            cells = []
            for i in range(len(self.num_layers)):
                cell = LayerNormBasicLSTMCell(self.num_layers[i])
                cells.append(cell)

            lstm_cells = MultiRNNCell(cells)

            lstm_output, _ = tf.nn.dynamic_rnn(
                cell=lstm_cells,
                inputs=encoder_output,
                sequence_length=self.sequence_lengths,
                dtype=tf.float32,
                scope='lstm_layer'
            )

            lstm_output = tf.nn.dropout(lstm_output, self.dropout_keep_prob)

        # print(lstm_output)      shape = [batch_size, sequence_length, lstm_output_size]

        with tf.variable_scope('DNN'):
            W = tf.get_variable(name='W',
                                shape=[self.num_layers[-1], 1],
                                initializer=xavier_initializer(),
                                dtype=tf.float32)

            b = tf.get_variable(name='b',
                                shape=[1],
                                initializer=tf.zeros_initializer(),
                                dtype=tf.float32)

            output_shape = tf.shape(lstm_output)
            output = tf.reshape(lstm_output, [-1, output_shape[-1]])
            output = tf.matmul(output, W) + b

            self.logits = tf.reshape(output, [-1, output_shape[1], 1])  # shape = [batch_size, sequence_length, 1]

        with tf.variable_scope('compiler'):
            self.loss = tf.losses.mean_squared_error(labels=self.y, predictions=self.logits)
            self.global_step = tf.Variable(tf.constant(0), name='global_step', trainable=False)
            optim = tf.train.AdamOptimizer(learning_rate=self.learning_rate)
            self.train_op = optim.minimize(self.loss)


if __name__ == '__main__':
    epoch = 50
    model = AutoEncoder_LSTM(128, 64, 32, 64, 0.5, 1e-4, 3, 1, [32, 32, 32])
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for i in range(epoch):
            loss, _ = sess.run([model.loss, model.train_op], feed_dict={model.x: [[[3], [2], [1]], [[1], [2], [3]]],
                                                                    model.sequence_lengths: [3, 3],
                                                                    model.y: [[[1], [2], [1]], [[3], [2], [1]]]})


            print('Epoch: {}\tLoss: {}'.format(str(i + 1), loss))

        print(sess.run(model.logits, feed_dict={model.x: [[[2], [2], [1]]],
                                    model.sequence_lengths: [3],
                                    model.y: [[[1], [2], [1]]]}))