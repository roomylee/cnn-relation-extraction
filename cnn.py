import tensorflow as tf


class ConvNet:
    def __init__(self, input_x, input_y, dropout_keep_prob):
        self.input_x = input_x
        self.input_y = input_y
        self.output = None
        self.dropout_keep_prob = dropout_keep_prob
        self.build_model()

    def build_model(self):
        image = tf.reshape(self.input_x, [-1, 97, 52, 1])
        with tf.name_scope('conv_layer1'):
            W_conv1 = ConvNet.weight_variable([2, 52, 1, 128])
            b_conv1 = ConvNet.bias_variable([128])

            h_conv1 = tf.nn.relu(ConvNet.conv2d(image, W_conv1) + b_conv1)
            h_pool1 = ConvNet.max_pool_1D(h_conv1)

        with tf.name_scope('fully_connected_layer'):
            W_fc1 = ConvNet.weight_variable([128, 64])
            b_fc1 = ConvNet.bias_variable([64])

            h_pool1_flat = tf.reshape(h_pool1, [-1, 128]) # -1은 batch size를 유지하는 것.
            h_fc1 = tf.nn.relu(tf.matmul(h_pool1_flat, W_fc1) + b_fc1)

        # dropout
        with tf.name_scope('dropout'):
            h_fc1_drop = tf.nn.dropout(h_fc1, self.dropout_keep_prob)

        with tf.name_scope('soft_max_layer'):
            # readout layer for deep net
            W_fc2 = ConvNet.weight_variable([64, 19])
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

    def max_pool_1D(x):
        return tf.nn.max_pool(x, ksize=[1, 97-2+1, 1, 1], strides=[1, 1, 1, 1], padding='VALID')
