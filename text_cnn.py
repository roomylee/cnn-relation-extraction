import tensorflow as tf


class TextCNN:
    def __init__(self, sequence_length, num_classes,
                 text_vocab_size, text_embedding_size, pos_vocab_size, pos_embedding_size,
                 filter_sizes, num_filters, l2_reg_lambda=0.0):

        # Placeholders for input, output and dropout
        self.input_text = tf.placeholder(tf.int32, shape=[None, sequence_length], name='input_text')
        self.input_pos1 = tf.placeholder(tf.int32, shape=[None, sequence_length], name='input_pos1')
        self.input_pos2 = tf.placeholder(tf.int32, shape=[None, sequence_length], name='input_pos2')
        self.input_y = tf.placeholder(tf.float32, shape=[None, num_classes], name='input_y')
        self.dropout_keep_prob = tf.placeholder(tf.float32, name='dropout_keep_prob')

        # Keeping track of l2 regularization loss (optional)
        l2_loss = tf.constant(0.0)

        # Embedding layer
        with tf.device('/cpu:0'), tf.name_scope("text-embedding"):
            self.W_text = tf.Variable(tf.random_uniform([text_vocab_size, text_embedding_size], -1.0, 1.0), name="W_text")
            self.text_embedded_chars = tf.nn.embedding_lookup(self.W_text, self.input_text)
            self.text_embedded_chars_expanded = tf.expand_dims(self.text_embedded_chars, -1)
        with tf.device('/cpu:0'), tf.name_scope("position-embedding"):
            self.W_position = tf.Variable(tf.random_uniform([pos_vocab_size, pos_embedding_size], -1.0, 1.0), name="W_position")
            self.pos1_embedded_chars = tf.nn.embedding_lookup(self.W_position, self.input_pos1)
            self.pos1_embedded_chars_expanded = tf.expand_dims(self.pos1_embedded_chars, -1)
            self.pos2_embedded_chars = tf.nn.embedding_lookup(self.W_position, self.input_pos2)
            self.pos2_embedded_chars_expanded = tf.expand_dims(self.pos2_embedded_chars, -1)

        self.embedded_chars_expanded = tf.concat([self.text_embedded_chars_expanded,
                                                  self.pos1_embedded_chars_expanded,
                                                  self.pos2_embedded_chars_expanded], 2)

        embedding_size = text_embedding_size + 2*pos_embedding_size

        # Create a convolution + maxpool layer for each filter size
        pooled_outputs = []
        for i, filter_size in enumerate(filter_sizes):
            with tf.name_scope("conv-maxpool-%s" % filter_size):
                # Convolution Layer
                filter_shape = [filter_size, embedding_size, 1, num_filters]
                W = tf.Variable(tf.truncated_normal(filter_shape, stddev=0.1), name="W")
                b = tf.Variable(tf.constant(0.1, shape=[num_filters]), name="b")
                conv = tf.nn.conv2d(self.embedded_chars_expanded, W, strides=[1, 1, 1, 1], padding="VALID", name="conv")
                # Apply nonlinearity
                h = tf.nn.relu(tf.nn.bias_add(conv, b), name="relu")
                # Maxpooling over the outputs
                pooled = tf.nn.max_pool(h, ksize=[1, sequence_length - filter_size + 1, 1, 1], strides=[1, 1, 1, 1], padding='VALID', name="pool")
                pooled_outputs.append(pooled)

        # Combine all the pooled features
        num_filters_total = num_filters * len(filter_sizes)
        self.h_pool = tf.concat(pooled_outputs, 3)
        self.h_pool_flat = tf.reshape(self.h_pool, [-1, num_filters_total])

        # Add dropout
        with tf.name_scope("dropout"):
            self.h_drop = tf.nn.dropout(self.h_pool_flat, self.dropout_keep_prob)

        # Final scores and predictions
        with tf.name_scope("output"):
            W = tf.get_variable("W", shape=[num_filters_total, num_classes], initializer=tf.contrib.layers.xavier_initializer())
            b = tf.Variable(tf.constant(0.1, shape=[num_classes]), name="b")
            l2_loss += tf.nn.l2_loss(W)
            l2_loss += tf.nn.l2_loss(b)
            self.logits = tf.nn.xw_plus_b(self.h_drop, W, b, name="logits")
            self.predictions = tf.argmax(self.logits, 1, name="predictions")

        # Calculate mean cross-entropy loss
        with tf.name_scope("loss"):
            losses = tf.nn.softmax_cross_entropy_with_logits(logits=self.logits, labels=self.input_y)
            self.loss = tf.reduce_mean(losses) + l2_reg_lambda * l2_loss

        # Accuracy
        with tf.name_scope("accuracy"):
            correct_predictions = tf.equal(self.predictions, tf.argmax(self.input_y, 1))
            self.accuracy = tf.reduce_mean(tf.cast(correct_predictions, tf.float32), name="accuracy")
