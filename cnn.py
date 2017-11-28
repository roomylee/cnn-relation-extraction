import tensorflow as tf


class ConvNet:
    def __init__(self, input_x, input_y, dropout_keep_prob):
        self.input_x = input_x
        self.input_y = input_y
        self.output = None
        self.dropout_keep_prob = dropout_keep_prob
        self.build_model()

    def build_model(self):
        image = tf.reshape(self.input_x, [-1, 64, 554, 1])

        pooled_outputs = []
        filter_sizes = [2, 3, 4, 5]
        for filter_size in filter_sizes:
            with tf.name_scope('conv_layer%s' % filter_size):
                W_conv = ConvNet.weight_variable([filter_size, 554, 1, 128])
                b_conv = ConvNet.bias_variable([128])

                h_conv = tf.nn.relu(ConvNet.conv2d(image, W_conv) + b_conv)
                h_pool = ConvNet.max_pool_1D(h_conv, filter_size)

                pooled_outputs.append(h_pool)

        h_pool1 = tf.concat(pooled_outputs, axis=0)
        h_pool1_flat = tf.reshape(h_pool1, [-1, 128*4])  # -1은 batch size를 유지하는 것.

        with tf.name_scope('fully_connected_layer'):
            W_fc = ConvNet.weight_variable([128*4, 256])
            b_fc = ConvNet.bias_variable([256])

            h_fc1 = tf.nn.relu(tf.matmul(h_pool1_flat, W_fc) + b_fc)

        # dropout
        with tf.name_scope('dropout'):
            h_fc1_drop = tf.nn.dropout(h_fc1, self.dropout_keep_prob)

        with tf.name_scope('soft_max_layer'):
            # readout layer for deep net
            W_fc2 = ConvNet.weight_variable([256, 19])
            b_fc2 = ConvNet.bias_variable([19])
            self.output = tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)

    # truncated normal distribution에 기반해서 랜덤한 값으로 초기화
    def weight_variable(shape):
        # tf.truncated_normal:
        # Outputs random values from a truncated normal distribution.
        # values whose magnitude is more than 2 standard deviations from the mean are dropped and re-picked.
        initial = tf.truncated_normal(shape, stddev=0.1)
        return tf.Variable(initial)

    # 0.1로 초기화
    def bias_variable(shape):
        initial = tf.constant(0.1, shape=shape)
        return tf.Variable(initial)

    # convolution & max pooling
    def conv2d(x, W):
        return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='VALID')

    def max_pool_1D(x, filter_size):
        return tf.nn.max_pool(x, ksize=[1, 64-filter_size+1, 1, 1], strides=[1, 1, 1, 1], padding='VALID')
