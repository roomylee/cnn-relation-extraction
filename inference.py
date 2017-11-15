import tensorflow as tf
from cnn import ConvNet
import data_helpers
import numpy as np


def inference(test_images):
    # evaluation
    input_x = tf.placeholder(tf.float32, shape=[None, 28 * 28], name='input_x')
    input_y = tf.placeholder(tf.float32, shape=[None, 10], name='input_y')
    keep_prob = tf.placeholder(tf.float32, name='dropout_keep_prob')

    model = ConvNet(input_x, input_y, keep_prob)

    saver = tf.train.Saver()

    sess = tf.Session()
    saver.restore(sess, tf.train.latest_checkpoint('saved'))

    with sess.as_default():
        # prediction function
        with tf.name_scope('predict'):
            # [0.1, 0.9, 0.2, 0.1, 0.1 0.3, 0.5, 0.1, 0.2, 0.3] => 1
            predict = tf.argmax(model.output, 1)

        feed_dict = {model.input_x: test_images,
                     model.dropout_keep_prob: 1.0}
        predictions = sess.run(predict, feed_dict=feed_dict)
        return predictions


def main():
    x = data_helpers.load_test_data('data/test.csv')
    a = inference(x)
    print(a)
    np.savetxt('submission_softmax.csv',
               np.c_[range(1, len(x) + 1), a],
               delimiter=',',
               header='ImageId,Label',
               comments='',
               fmt='%d')

if __name__ == "__main__":
    main()




