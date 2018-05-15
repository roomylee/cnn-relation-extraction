import tensorflow as tf
import numpy as np
import os
import datetime
import time
from text_cnn import TextCNN
import data_helpers
from sklearn.metrics import f1_score
import warnings
import sklearn.exceptions
warnings.filterwarnings("ignore", category=sklearn.exceptions.UndefinedMetricWarning)

# Parameters
# ==================================================

# Data loading params
tf.flags.DEFINE_string("train_dir", "SemEval2010_task8_all_data/SemEval2010_task8_training/TRAIN_FILE.TXT", "Path of train data")
tf.flags.DEFINE_float("dev_sample_percentage", .1, "Percentage of the training data to use for validation")
tf.flags.DEFINE_integer("max_sentence_length", 100, "Max sentence length in train(98)/test(70) data (Default: 100)")

# Model Hyperparameters
tf.flags.DEFINE_string("word2vec", None, "Word2vec file with pre-trained embeddings")
tf.flags.DEFINE_integer("text_embedding_dim", 300, "Dimensionality of word embedding (Default: 300)")
tf.flags.DEFINE_integer("position_embedding_dim", 100, "Dimensionality of position embedding (Default: 100)")
tf.flags.DEFINE_string("filter_sizes", "2,3,4,5", "Comma-separated filter sizes (Default: 2,3,4,5)")
tf.flags.DEFINE_integer("num_filters", 128, "Number of filters per filter size (Default: 128)")
tf.flags.DEFINE_float("dropout_keep_prob", 0.5, "Dropout keep probability (Default: 0.5)")
tf.flags.DEFINE_float("l2_reg_lambda", 3.0, "L2 regularization lambda (Default: 3.0)")

# Training parameters
tf.flags.DEFINE_integer("batch_size", 64, "Batch Size (Default: 64)")
tf.flags.DEFINE_integer("num_epochs", 100, "Number of training epochs (Default: 100)")
tf.flags.DEFINE_integer("display_every", 10, "Number of iterations to display training info.")
tf.flags.DEFINE_integer("evaluate_every", 100, "Evaluate model on dev set after this many steps")
tf.flags.DEFINE_integer("checkpoint_every", 100, "Save model after this many steps")
tf.flags.DEFINE_integer("num_checkpoints", 5, "Number of checkpoints to store")
tf.flags.DEFINE_float("learning_rate", 1e-3, "Which learning rate to start with. (Default: 1e-3)")

# Misc Parameters
tf.flags.DEFINE_boolean("allow_soft_placement", True, "Allow device soft device placement")
tf.flags.DEFINE_boolean("log_device_placement", False, "Log placement of ops on devices")


FLAGS = tf.flags.FLAGS
FLAGS._parse_flags()
print("\nParameters:")
for attr, value in sorted(FLAGS.__flags.items()):
    print("{} = {}".format(attr.upper(), value))
print("")


def train():
    with tf.device('/cpu:0'):
        x_text, pos1, pos2, y = data_helpers.load_data_and_labels(FLAGS.train_dir)

    # Build vocabulary
    # Example: x_text[3] = "A misty <e1>ridge</e1> uprises from the <e2>surge</e2>."
    # ['a misty ridge uprises from the surge <UNK> <UNK> ... <UNK>']
    # =>
    # [27 39 40 41 42  1 43  0  0 ... 0]
    # dimension = FLAGS.max_sentence_length
    text_vocab_processor = tf.contrib.learn.preprocessing.VocabularyProcessor(FLAGS.max_sentence_length)
    text_vec = np.array(list(text_vocab_processor.fit_transform(x_text)))
    print("Text Vocabulary Size: {:d}".format(len(text_vocab_processor.vocabulary_)))

    # Example: pos1[3] = [-2 -1  0  1  2   3   4 999 999 999 ... 999]
    # [95 96 97 98 99 100 101 999 999 999 ... 999]
    # =>
    # [11 12 13 14 15  16  21  17  17  17 ...  17]
    # dimension = MAX_SENTENCE_LENGTH
    pos_vocab_processor = tf.contrib.learn.preprocessing.VocabularyProcessor(FLAGS.max_sentence_length)
    pos_vocab_processor.fit(pos1 + pos2)
    pos1_vec = np.array(list(pos_vocab_processor.transform(pos1)))
    pos2_vec = np.array(list(pos_vocab_processor.transform(pos2)))
    print("Position Vocabulary Size: {:d}".format(len(pos_vocab_processor.vocabulary_)))

    x = np.array([list(i) for i in zip(text_vec, pos1_vec, pos2_vec)])

    print("x = {0}".format(x.shape))
    print("y = {0}".format(y.shape))
    print("")

    # Randomly shuffle data
    np.random.seed(10)
    shuffle_indices = np.random.permutation(np.arange(len(y)))
    x_shuffled = x[shuffle_indices]
    y_shuffled = y[shuffle_indices]

    # Split train/test set
    # TODO: This is very crude, should use cross-validation
    dev_sample_index = -1 * int(FLAGS.dev_sample_percentage * float(len(y)))
    x_train, x_dev = x_shuffled[:dev_sample_index], x_shuffled[dev_sample_index:]
    x_dev = np.array(x_dev).transpose((1, 0, 2))
    y_train, y_dev = y_shuffled[:dev_sample_index], y_shuffled[dev_sample_index:]
    print("Train/Dev split: {:d}/{:d}\n".format(len(y_train), len(y_dev)))

    with tf.Graph().as_default():
        session_conf = tf.ConfigProto(
            allow_soft_placement=FLAGS.allow_soft_placement,
            log_device_placement=FLAGS.log_device_placement)
        sess = tf.Session(config=session_conf)
        with sess.as_default():
            cnn = TextCNN(
                sequence_length=x_train.shape[2],
                num_classes=y_train.shape[1],
                text_vocab_size=len(text_vocab_processor.vocabulary_),
                text_embedding_size=FLAGS.text_embedding_dim,
                pos_vocab_size=len(pos_vocab_processor.vocabulary_),
                pos_embedding_size=FLAGS.position_embedding_dim,
                filter_sizes=list(map(int, FLAGS.filter_sizes.split(","))),
                num_filters=FLAGS.num_filters,
                l2_reg_lambda=FLAGS.l2_reg_lambda)

            # Define Training procedure
            global_step = tf.Variable(0, name="global_step", trainable=False)
            train_op = tf.train.AdamOptimizer(FLAGS.learning_rate).minimize(cnn.loss, global_step=global_step)

            # Output directory for models and summaries
            timestamp = str(int(time.time()))
            out_dir = os.path.abspath(os.path.join(os.path.curdir, "runs", timestamp))
            print("Writing to {}\n".format(out_dir))

            # Summaries for loss and accuracy
            loss_summary = tf.summary.scalar("loss", cnn.loss)
            acc_summary = tf.summary.scalar("accuracy", cnn.accuracy)

            # Train Summaries
            train_summary_op = tf.summary.merge([loss_summary, acc_summary])
            train_summary_dir = os.path.join(out_dir, "summaries", "train")
            train_summary_writer = tf.summary.FileWriter(train_summary_dir, sess.graph)

            # Dev summaries
            dev_summary_op = tf.summary.merge([loss_summary, acc_summary])
            dev_summary_dir = os.path.join(out_dir, "summaries", "dev")
            dev_summary_writer = tf.summary.FileWriter(dev_summary_dir, sess.graph)

            # Checkpoint directory. Tensorflow assumes this directory already exists so we need to create it
            checkpoint_dir = os.path.abspath(os.path.join(out_dir, "checkpoints"))
            checkpoint_prefix = os.path.join(checkpoint_dir, "model")
            if not os.path.exists(checkpoint_dir):
                os.makedirs(checkpoint_dir)
            saver = tf.train.Saver(tf.global_variables(), max_to_keep=FLAGS.num_checkpoints)

            # Write vocabulary
            text_vocab_processor.save(os.path.join(out_dir, "text_vocab"))
            pos_vocab_processor.save(os.path.join(out_dir, "position_vocab"))

            # Initialize all variables
            sess.run(tf.global_variables_initializer())

            # Pre-trained word2vec
            if FLAGS.word2vec:
                # initial matrix with random uniform
                initW = np.random.uniform(-0.25, 0.25, (len(text_vocab_processor.vocabulary_), FLAGS.text_embedding_dim))
                # load any vectors from the word2vec
                print("Load word2vec file {0}".format(FLAGS.word2vec))
                with open(FLAGS.word2vec, "rb") as f:
                    header = f.readline()
                    vocab_size, layer1_size = map(int, header.split())
                    binary_len = np.dtype('float32').itemsize * layer1_size
                    for line in range(vocab_size):
                        word = []
                        while True:
                            ch = f.read(1).decode('latin-1')
                            if ch == ' ':
                                word = ''.join(word)
                                break
                            if ch != '\n':
                                word.append(ch)
                        idx = text_vocab_processor.vocabulary_.get(word)
                        if idx != 0:
                            initW[idx] = np.fromstring(f.read(binary_len), dtype='float32')
                        else:
                            f.read(binary_len)
                sess.run(cnn.W_text.assign(initW))
                print("Success to load pre-trained word2vec model!\n")

            # Generate batches
            batches = data_helpers.batch_iter(
                list(zip(x_train, y_train)), FLAGS.batch_size, FLAGS.num_epochs)
            # Training loop. For each batch...
            for batch in batches:
                x_batch, y_batch = zip(*batch)
                x_batch = np.array(x_batch).transpose((1, 0, 2))

                # Train
                feed_dict = {
                    cnn.input_text: x_batch[0],
                    cnn.input_pos1: x_batch[1],
                    cnn.input_pos2: x_batch[2],
                    cnn.input_y: y_batch,
                    cnn.dropout_keep_prob: FLAGS.dropout_keep_prob
                }
                _, step, summaries, loss, accuracy = sess.run(
                    [train_op, global_step, train_summary_op, cnn.loss, cnn.accuracy], feed_dict)
                train_summary_writer.add_summary(summaries, step)

                # Training log display
                if step % FLAGS.display_every == 0:
                    time_str = datetime.datetime.now().isoformat()
                    print("{}: step {}, loss {:g}, acc {:g}".format(time_str, step, loss, accuracy))

                # Evaluation
                if step % FLAGS.evaluate_every == 0:
                    print("\nEvaluation:")
                    feed_dict = {
                        cnn.input_text: x_dev[0],
                        cnn.input_pos1: x_dev[1],
                        cnn.input_pos2: x_dev[2],
                        cnn.input_y: y_dev,
                        cnn.dropout_keep_prob: 1.0
                    }
                    summaries, loss, accuracy, predictions = sess.run(
                        [dev_summary_op, cnn.loss, cnn.accuracy, cnn.predictions], feed_dict)
                    dev_summary_writer.add_summary(summaries, step)

                    time_str = datetime.datetime.now().isoformat()
                    print("{}: step {}, loss {:g}, acc {:g}".format(time_str, step, loss, accuracy))
                    print("(2*9+1)-Way Macro-Average F1 Score (excluding Other): {:g}\n".format(
                        f1_score(np.argmax(y_dev, axis=1), predictions, labels=np.array(range(1, 19)), average="macro")))

                # Model checkpoint
                if step % FLAGS.checkpoint_every == 0:
                    path = saver.save(sess, checkpoint_prefix, global_step=step)
                    print("Saved model checkpoint to {}\n".format(path))


def main(_):
    train()


if __name__ == "__main__":
    tf.app.run()
