import tensorflow as tf
import numpy as np
import os
import subprocess

import data_helpers
import utils
from configure import FLAGS


def eval():
    with tf.device('/cpu:0'):
        x_text, y, pos1, pos2 = data_helpers.load_data_and_labels(FLAGS.test_path)

    # Map data into vocabulary
    text_path = os.path.join(FLAGS.checkpoint_dir, "..", "text_vocab")
    text_vocab_processor = tf.contrib.learn.preprocessing.VocabularyProcessor.restore(text_path)
    x = np.array(list(text_vocab_processor.transform(x_text)))

    # Map data into position
    position_path = os.path.join(FLAGS.checkpoint_dir, "..", "pos_vocab")
    position_vocab_processor = tf.contrib.learn.preprocessing.VocabularyProcessor.restore(position_path)
    p1 = np.array(list(position_vocab_processor.transform(pos1)))
    p2 = np.array(list(position_vocab_processor.transform(pos2)))

    checkpoint_file = tf.train.latest_checkpoint(FLAGS.checkpoint_dir)

    graph = tf.Graph()
    with graph.as_default():
        session_conf = tf.ConfigProto(
            allow_soft_placement=FLAGS.allow_soft_placement,
            log_device_placement=FLAGS.log_device_placement)
        session_conf.gpu_options.allow_growth = FLAGS.gpu_allow_growth
        sess = tf.Session(config=session_conf)
        with sess.as_default():
            # Load the saved meta graph and restore variables
            saver = tf.train.import_meta_graph("{}.meta".format(checkpoint_file))
            saver.restore(sess, checkpoint_file)

            # Get the placeholders from the graph by name
            input_text = graph.get_operation_by_name("input_text").outputs[0]
            input_p1 = graph.get_operation_by_name("input_p1").outputs[0]
            input_p2 = graph.get_operation_by_name("input_p2").outputs[0]
            # input_y = graph.get_operation_by_name("input_y").outputs[0]
            dropout_keep_prob = graph.get_operation_by_name("dropout_keep_prob").outputs[0]

            # Tensors we want to evaluate
            predictions = graph.get_operation_by_name("output/predictions").outputs[0]

            # Generate batches for one epoch
            batches = data_helpers.batch_iter(list(zip(x, p1, p2)), FLAGS.batch_size, 1, shuffle=False)

            # Collect the predictions here
            preds = []
            for batch in batches:
                x_batch, p1_batch, p2_batch = zip(*batch)
                pred = sess.run(predictions, {input_text: x_batch,
                                              input_p1: p1_batch,
                                              input_p2: p2_batch,
                                              dropout_keep_prob: 1.0})
                preds.append(pred)
            preds = np.concatenate(preds)
            truths = np.argmax(y, axis=1)

            prediction_path = os.path.join(FLAGS.checkpoint_dir, "..", "predictions.txt")
            truth_path = os.path.join(FLAGS.checkpoint_dir, "..", "ground_truths.txt")
            prediction_file = open(prediction_path, 'w')
            truth_file = open(truth_path, 'w')
            for i in range(len(preds)):
                prediction_file.write("{}\t{}\n".format(i, utils.label2class[preds[i]]))
                truth_file.write("{}\t{}\n".format(i, utils.label2class[truths[i]]))
            prediction_file.close()
            truth_file.close()

            perl_path = os.path.join(os.path.curdir,
                                     "SemEval2010_task8_all_data",
                                     "SemEval2010_task8_scorer-v1.2",
                                     "semeval2010_task8_scorer-v1.2.pl")
            process = subprocess.Popen(["perl", perl_path, prediction_path, truth_path], stdout=subprocess.PIPE)
            for line in str(process.communicate()[0].decode("utf-8")).split("\\n"):
                print(line)


def main(_):
    eval()


if __name__ == "__main__":
    tf.app.run()