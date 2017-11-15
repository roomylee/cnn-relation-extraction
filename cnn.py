import tensorflow as tf


class ConvNet:
    def __init__(self, input_x, input_y, dropout_keep_prob):
        self.input_x = input_x
        self.input_y = input_y
        self.dropout_keep_prob = dropout_keep_prob

        self.build_model()

    def build_model(self):
        image = tf.reshape(self.input_x, [-1, 28, 28, 1])
        with tf.name_scope('conv_layer1'):
            # [5,5,1,32]: 5x5 convolution patch, 1 input channel, 32 output channel.
            # MNIST의 pixel은 0/1로 표현되는 1개의 벡터이므로 1 input channel임.
            # CIFAR-10 같이 color인 경우에는 RGB 3개의 벡터로 표현되므로 3 input channel일 것이다.
            # Shape을 아래와 같이 넣으면 넣은 그대로 5x5x1x32의 텐서를 생성함.
            W_conv1 = ConvNet.weight_variable([5, 5, 1, 32])
            b_conv1 = ConvNet.bias_variable([32])
            # 최종적으로 32개의 output channel에 대해 각각 5x5의 convolution patch(filter) weight와 1개의 bias를 갖게 됨.

            h_conv1 = tf.nn.relu(ConvNet.conv2d(image, W_conv1) + b_conv1)
            # print (h_conv1.get_shape())
            # => (40000, 28, 28, 32)
            h_pool1 = ConvNet.max_pool_2x2(h_conv1)
            # print (h_pool1.get_shape())
            # => (40000, 14, 14, 32)

        with tf.name_scope('conv_layer2'):
            # second convolutional layer
            # channels (features) : 32 => 64
            # 5x5x32x64 짜리 weights.
            W_conv2 = ConvNet.weight_variable([5, 5, 32, 64])
            b_conv2 = ConvNet.bias_variable([64])

            h_conv2 = tf.nn.relu(ConvNet.conv2d(h_pool1, W_conv2) + b_conv2)
            # print (h_conv2.get_shape()) # => (40000, 14,14, 64)
            h_pool2 = ConvNet.max_pool_2x2(h_conv2)
            # print (h_pool2.get_shape()) # => (40000, 7, 7, 64)

        with tf.name_scope('fully_connected_layer'):
            # densely connected layer (fully connected layer)
            # 7*7*64는 h_pool2의 output (7*7의 reduced image * 64개의 채널). 1024는 fc layer의 뉴런 수.
            W_fc1 = ConvNet.weight_variable([7 * 7 * 64, 1024])
            b_fc1 = ConvNet.bias_variable([1024])

            # (40000, 7, 7, 64) => (40000, 3136)
            h_pool2_flat = tf.reshape(h_pool2, [-1, 7 * 7 * 64]) # -1은 batch size를 유지하는 것.
            h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)
            # print (h_fc1.get_shape()) # => (40000, 1024)

        # dropout
        with tf.name_scope('dropout'):
            h_fc1_drop = tf.nn.dropout(h_fc1, self.dropout_keep_prob)

        with tf.name_scope('soft_max_layer'):
            # readout layer for deep net
            W_fc2 = ConvNet.weight_variable([1024, 10])
            b_fc2 = ConvNet.bias_variable([10])
            self.output = tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)
            # print (y.get_shape()) # => (40000, 10)

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
    # vanila version of CNN
    # x (아래 함수들에서) : A 4-D `Tensor` with shape `[batch, height, width, channels]`
    def conv2d(x, W):
        # stride = 1, zero padding은 input과 output의 size가 같도록.
        return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

    def max_pool_2x2(x):
        # pooling
        # [[0,3],
        #  [4,2]] => 4

        # [[0,1],
        #  [1,1]] => 1
        return tf.nn.max_pool(x, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')
