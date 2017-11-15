import tensorflow as tf
import datetime
from cnn import ConvNet
import data_helpers


def train(batch_size=50,
          epochs=2500,
          learning_rate=1e-4,
          dropout_keep_prob=0.5):
    input_x = tf.placeholder(tf.float32, shape=[None, 28*28], name='input_x')
    input_y = tf.placeholder(tf.float32, shape=[None, 10], name='input_y')
    keep_prob = tf.placeholder(tf.float32, name='dropout_keep_prob')

    model = ConvNet(input_x, input_y, keep_prob)

    output = model.output

    # cost function
    with tf.name_scope("cross_entropy"):
        cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=output, labels=input_y))
    # optimisation function
    with tf.name_scope("optimizer"):
        optimizer = tf.train.AdamOptimizer(learning_rate).minimize(cost)
    # Get accuracy of model
    with tf.name_scope("accuracy"):
        correct_prediction = tf.equal(tf.argmax(input_y, 1), tf.argmax(output, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    # start TensorFlow session
    saver = tf.train.Saver()
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

    display_step = 1
    train_images, train_labels = data_helpers.load_data_and_labels('data/train.csv')
    with sess.as_default():
        batches = data_helpers.batch_iter(list(zip(train_images, train_labels)), batch_size, epochs)
        for step, batch in enumerate(batches):
            batch_xs, batch_ys = zip(*batch)
            feed_dict = {input_x: batch_xs,
                         input_y: batch_ys,
                         keep_prob: dropout_keep_prob}
            loss, _ = sess.run([cost, optimizer], feed_dict=feed_dict)
            if step % display_step == 0 or (step + 1) == epochs:
                time_str = datetime.datetime.now().isoformat()
                saver.save(sess, "saved/model.ckpt", step)
                acc = sess.run(accuracy, feed_dict={input_x: batch_xs, input_y: batch_ys, keep_prob: 1.0})
                print("{}: step {}, loss {:g}, acc {:g}".format(time_str, step, loss, acc))
                if step % (display_step * 10) == 0 and step:
                    display_step *= 10


def main():
    train(epochs=1)

if __name__ == "__main__":
    main()
    #tf.app.run()
