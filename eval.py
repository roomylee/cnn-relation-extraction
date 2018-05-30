import tensorflow as tf
import numpy as np
import os
import data_helpers
from sklearn.metrics import f1_score
import warnings
import sklearn.exceptions
warnings.filterwarnings("ignore", category=sklearn.exceptions.UndefinedMetricWarning)


# Parameters
# ==================================================

# Data loading params
tf.flags.DEFINE_string("eval_dir", "SemEval2010_task8_all_data/SemEval2010_task8_testing_keys/TEST_FILE_FULL.TXT", "Path of evaluation data")
tf.flags.DEFINE_string("output_dir", "result/prediction.txt", "Path of prediction for evaluation data")
tf.flags.DEFINE_string("target_dir", "result/answer.txt", "Path of target(answer) file for evaluation data")

# Eval Parameters
tf.flags.DEFINE_integer("batch_size", 64, "Batch Size (Default: 64)")
tf.flags.DEFINE_string("checkpoint_dir", "", "Checkpoint directory from training run")

# Misc Parameters
tf.flags.DEFINE_boolean("allow_soft_placement", True, "Allow device soft device placement")
tf.flags.DEFINE_boolean("log_device_placement", False, "Log placement of ops on devices")


FLAGS = tf.flags.FLAGS
FLAGS._parse_flags()
print("\nParameters:")
for attr, value in sorted(FLAGS.__flags.items()):
    print("{} = {}".format(attr.upper(), value))
print("")


def eval():
    with tf.device('/cpu:0'):
        x_text, pos1, pos2, y = data_helpers.load_data_and_labels(FLAGS.eval_dir)

    # Map data into vocabulary
    text_path = os.path.join(FLAGS.checkpoint_dir, "..", "text_vocab")
    text_vocab_processor = tf.contrib.learn.preprocessing.VocabularyProcessor.restore(text_path)
    text_vec = np.array(list(text_vocab_processor.transform(x_text)))

    # Map data into position
    position_path = os.path.join(FLAGS.checkpoint_dir, "..", "position_vocab")
    position_vocab_processor = tf.contrib.learn.preprocessing.VocabularyProcessor.restore(position_path)
    pos1_vec = np.array(list(position_vocab_processor.transform(pos1)))
    pos2_vec = np.array(list(position_vocab_processor.transform(pos2)))

    x_eval = np.array([list(i) for i in zip(text_vec, pos1_vec, pos2_vec)])
    y_eval = np.argmax(y, axis=1)

    checkpoint_file = tf.train.latest_checkpoint(FLAGS.checkpoint_dir)

    graph = tf.Graph()
    with graph.as_default():
        session_conf = tf.ConfigProto(
            allow_soft_placement=FLAGS.allow_soft_placement,
            log_device_placement=FLAGS.log_device_placement)
        sess = tf.Session(config=session_conf)
        with sess.as_default():
            # Load the saved meta graph and restore variables
            saver = tf.train.import_meta_graph("{}.meta".format(checkpoint_file))
            saver.restore(sess, checkpoint_file)

            # Get the placeholders from the graph by name
            input_text = graph.get_operation_by_name("input_text").outputs[0]
            input_pos1 = graph.get_operation_by_name("input_pos1").outputs[0]
            input_pos2 = graph.get_operation_by_name("input_pos2").outputs[0]
            # input_y = graph.get_operation_by_name("input_y").outputs[0]
            dropout_keep_prob = graph.get_operation_by_name("dropout_keep_prob").outputs[0]

            # Tensors we want to evaluate
            predictions = graph.get_operation_by_name("output/predictions").outputs[0]

            # Generate batches for one epoch
            batches = data_helpers.batch_iter(list(x_eval), FLAGS.batch_size, 1, shuffle=False)

            # Collect the predictions here
            all_predictions = []
            for x_eval_batch in batches:
                x_batch = np.array(x_eval_batch).transpose((1, 0, 2))
                batch_predictions = sess.run(predictions, {input_text: x_batch[0],
                                                           input_pos1: x_batch[1],
                                                           input_pos2: x_batch[2],
                                                           dropout_keep_prob: 1.0})
                all_predictions = np.concatenate([all_predictions, batch_predictions])

            labelsMapping = {0: 'Other',
                             1: 'Message-Topic(e1,e2)', 2: 'Message-Topic(e2,e1)',
                             3: 'Product-Producer(e1,e2)', 4: 'Product-Producer(e2,e1)',
                             5: 'Instrument-Agency(e1,e2)', 6: 'Instrument-Agency(e2,e1)',
                             7: 'Entity-Destination(e1,e2)', 8: 'Entity-Destination(e2,e1)',
                             9: 'Cause-Effect(e1,e2)', 10: 'Cause-Effect(e2,e1)',
                             11: 'Component-Whole(e1,e2)', 12: 'Component-Whole(e2,e1)',
                             13: 'Entity-Origin(e1,e2)', 14: 'Entity-Origin(e2,e1)',
                             15: 'Member-Collection(e1,e2)', 16: 'Member-Collection(e2,e1)',
                             17: 'Content-Container(e1,e2)', 18: 'Content-Container(e2,e1)'}
            output_file = open(FLAGS.output_dir, 'w')
            target_file = open(FLAGS.target_dir, 'w')
            for i in range(len(all_predictions)):
                output_file.write("{}\t{}\n".format(i, labelsMapping[all_predictions[i]]))
                target_file.write("{}\t{}\n".format(i, labelsMapping[y_eval[i]]))
            output_file.close()
            target_file.close()

            correct_predictions = float(sum(all_predictions == y_eval))
            print("\nTotal number of test examples: {}".format(len(y_eval)))
            print("Accuracy: {:g}".format(correct_predictions / float(len(y_eval))))
            print("(2*9+1)-Way Macro-Average F1 Score (excluding Other): {:g}".format(
                f1_score(y_eval, all_predictions, labels=np.array(range(1, 19)), average="macro")))


def main(_):
    eval()


if __name__ == "__main__":
    tf.app.run()