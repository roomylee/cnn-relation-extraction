import tensorflow as tf
from cnn import ConvNet
import data_helpers
from tensorflow.examples.tutorials.mnist import input_data



def evaluate(eval_images, eval_labels):
    # evaluation
    input_x = tf.placeholder(tf.float32, shape=[None, 28 * 28], name='input_x')
    input_y = tf.placeholder(tf.float32, shape=[None, 10], name='input_y')
    keep_prob = tf.placeholder(tf.float32, name='dropout_keep_prob')

    model = ConvNet(input_x, input_y, keep_prob)

    saver = tf.train.Saver()

    sess = tf.Session()
    saver.restore(sess, tf.train.latest_checkpoint('saved'))

    with sess.as_default():
        with tf.name_scope('accuracy'):
            correct_prediction = tf.equal(tf.argmax(model.output, 1), tf.argmax(model.input_y, 1))
            accuracy = tf.reduce_mean(tf.cast(correct_prediction, 'float'))

        feed_dict = {model.input_x: eval_images,
                     model.input_y: eval_labels,
                     model.dropout_keep_prob: 1.0}
        acc = sess.run(accuracy, feed_dict=feed_dict)
        return acc


def main():
    mnist = input_data.read_data_sets('MNIST_data', one_hot=True)
    a = evaluate(mnist.test.images, mnist.test.labels)
    print(a)

if __name__ == '__main__':
    main()
    #tf.app.run()
