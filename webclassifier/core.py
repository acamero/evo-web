import helper as hlp
import web_nn as nn
import tensorflow as tf
import argparse
import os.path
import sys
import time


# DEFAULT VALUES
FLAGS = None
# System configuration
DEF_PATH_TO_DATA = "webshot_data/path_to_data.txt"
DEF_LOG_DIR = "logs"
DEF_RANDOM_SEED = 10
DEF_REPORT_CYCLE = 10
DEF_CHECKPOINT = 10
DEF_INTRA_THREADS = 0
DEF_INTER_THREADS = 0

# Data configuration
DEF_TEST_SET_PERC = 0.2 # x% of the data is going to be used for testing
DEF_BATCH_SIZE = 20
DEF_TRAIN_CYCLES = 20

# NN configuration
DEF_HIDDEN1_UNITS = 16
DEF_HIDDEN2_UNITS = 32
DEF_LINEAR_UNITS = 256

# Optimizer configuration
DEF_LEARNING_RATE = 0.001


def print_variables():
    total_parameters = 0
    for variable in tf.trainable_variables():
        # shape is an array of tf.Dimension
        shape = variable.get_shape()
        print(shape)
        print(len(shape))
        variable_parametes = 1
        for dim in shape:
            print(dim)
            variable_parametes *= dim.value
        print(variable_parametes)
        total_parameters += variable_parametes
    print("Total number of training variables %d" % total_parameters)


def placeholder_inputs(batch_size, image_width, image_height, num_channels, num_classes):
    images_placeholder = tf.placeholder(tf.float32, [batch_size, image_width * image_height * num_channels])
    labels_placeholder = tf.placeholder(tf.float32, [batch_size, num_classes])
    keep_prob_placeholder = tf.placeholder(tf.float32)
    return images_placeholder, labels_placeholder, keep_prob_placeholder


def fill_feed_dict(sess, images_queue, labels_queue, keep_prob, keep_prob_pl, images_pl, labels_pl):
    images_feed, labels_feed = sess.run([images_queue, labels_queue])
    feed_dict = {
        images_pl: images_feed,
        labels_pl: labels_feed,
        keep_prob_pl: keep_prob
    }
    batch_size = images_feed.shape[0]
    return feed_dict, batch_size


def do_eval(sess, times, eval_correct, images_placeholder, labels_placeholder, images_queue, labels_queue, keep_prob_placeholder, keep_prob):
    avg = 0  # Counts the number of correct predictions.
    size = 0
    for i in xrange(times):
        feed_dict, batch_size = fill_feed_dict(sess,
                images_queue, 
                labels_queue,
                keep_prob,
                keep_prob_placeholder,
                images_placeholder,
                labels_placeholder)
        val = sess.run(eval_correct, feed_dict=feed_dict)
        avg = ((size * avg) + (val * batch_size)) / (size + batch_size)
        size += batch_size
    print('  Num examples: %d  Accuracy: %0.04f' % (size, avg))
    #


def run_training():    
    # Tell TensorFlow that the model will be built into the default Graph.
    with tf.Graph().as_default():
        tf.set_random_seed(FLAGS.random_seed)
        reader = hlp.ImageFolderReader(FLAGS.input_data_dir, FLAGS.test_set_perc, FLAGS.random_seed)
        train_eval_cycles = (reader.set_size - reader.test_set_size) // FLAGS.batch_size
        test_eval_cycles = reader.test_set_size // FLAGS.batch_size
        train_images_in, train_labels_in = reader.get_input_queue(
            FLAGS.batch_size, #batch_size
            None, # epochs
            True, # one_hot
            True) # train
        test_images_in, test_labels_in = reader.get_input_queue(
            FLAGS.batch_size, 
            None, # epochs
            True, # one_hot
            False) # train
        # Generate placeholders for the images and labels.
        images_placeholder, labels_placeholder, keep_p_placeholder = placeholder_inputs(
                None, # batch size
                reader.image_width, 
                reader.image_height, 
                reader.num_channels,
                reader.max_label)
        # Build a Graph that computes predictions from the inference model.
        web_nn = nn.inference(
                images_placeholder, 
                keep_p_placeholder,
                reader.max_label, 
                reader.image_height, 
                reader.image_width, 
                reader.num_channels, 
                FLAGS.hidden1,
                FLAGS.hidden2,
                FLAGS.linear)
        # Add to the Graph the Ops for loss calculation.
        loss = nn.loss(web_nn, labels_placeholder)
        # Add to the Graph the Ops that calculate and apply gradients.
        train_op = nn.training(loss, FLAGS.learning_rate)
        # Add the Op to compare the prediction to the labels during evaluation.
        eval_correct = nn.evaluation(web_nn, labels_placeholder)
        # Build the summary Tensor based on the TF collection of Summaries.
        summary = tf.summary.merge_all()
        # Add the variable initializer Op.
        init = tf.global_variables_initializer()
        if FLAGS.print_variables:
            print_variables()
        if not FLAGS.execute:
            exit()
        # Create a saver for writing training checkpoints.
        saver = tf.train.Saver()
        # Create a session for running Ops on the Graph.
        with tf.Session(
                config=tf.ConfigProto(
                    inter_op_parallelism_threads=FLAGS.inter_threads, 
                    intra_op_parallelism_threads=FLAGS.intra_threads)) as sess: 
            # Instantiate a SummaryWriter to output summaries and the Graph.
            summary_writer = tf.summary.FileWriter(FLAGS.log_dir, sess.graph)
            # And then after everything is built:
            # Run the Op to initialize the variables.
            sess.run(init)
            coord = tf.train.Coordinator()
            threads = tf.train.start_queue_runners(sess=sess, coord=coord)
            # Start the training loop.
            for step in xrange(FLAGS.max_steps):
                start_time = time.time()
                feed_dict,_ = fill_feed_dict(
                        sess, 
                        train_images_in, 
                        train_labels_in, 
                        0.5, # keep_prob
                        keep_p_placeholder,
                        images_placeholder,
                        labels_placeholder)
                # Run one step of the model.  The return values are the activations
                # from the `train_op` (which is discarded) and the `loss` Op.  To
                # inspect the values of your Ops or variables, you may include them    
                # in the list passed to sess.run() and the value tensors will be
                # returned in the tuple from the call.
                _, loss_value = sess.run([train_op, loss], feed_dict=feed_dict)
                duration = time.time() - start_time
                # Write the summaries and print an overview fairly often.
                if step % FLAGS.report_cycle == 0:
                    # Print status to stdout.
                    print('Step %d: loss = %.2f (%.3f sec)' % (step, loss_value, duration))    
                    # Update the events file.    
                    summary_str = sess.run(summary, feed_dict=feed_dict)
                    summary_writer.add_summary(summary_str, step)
                    summary_writer.flush()  
                # Save a checkpoint and evaluate the model periodically.
                if (step + 1) % FLAGS.checkpoint == 0 or (step + 1) == FLAGS.max_steps:
                    checkpoint_file = os.path.join(FLAGS.log_dir, 'model.ckpt')
                    saver.save(sess, checkpoint_file, global_step=step)
                    # Evaluate against the training set.
                    print('Training Data Eval:')
                    do_eval(sess,
                        train_eval_cycles, 
                        eval_correct,
                        images_placeholder,
                        labels_placeholder,
                        train_images_in, 
                        train_labels_in, 
                        keep_p_placeholder,
                        1.0) 
                    # Evaluate against the test set.                    
                    print('Test Data Eval:')
                    do_eval(sess,
                        test_eval_cycles, 
                        eval_correct,
                        images_placeholder,
                        labels_placeholder,
                        test_images_in, 
                        test_labels_in, 
                        keep_p_placeholder,
                        1.0) 
                # if
            # for
            coord.request_stop()
            coord.join(threads)
            sess.close()
        # with
    # 
# def


def main(_):
    if tf.gfile.Exists(FLAGS.log_dir):
        tf.gfile.DeleteRecursively(FLAGS.log_dir)
    tf.gfile.MakeDirs(FLAGS.log_dir)
    run_training()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
          '--test_set_perc',
          type=float,
          default=DEF_TEST_SET_PERC,
          help='Relative size of the test set.'
    )
    parser.add_argument(
          '--report_cycle',
          type=int,
          default=DEF_REPORT_CYCLE,
          help='Number of steps to run trainer.'
    )
    parser.add_argument(
          '--checkpoint',
          type=int,
          default=DEF_CHECKPOINT,
          help='Number of steps to run trainer.'
    )
    parser.add_argument(
          '--learning_rate',
          type=float,
          default=DEF_LEARNING_RATE,
          help='Initial learning rate.'
    )
    parser.add_argument(
          '--max_steps',
          type=int,
          default=DEF_TRAIN_CYCLES,
          help='Number of steps to run trainer.'
    )
    parser.add_argument(
          '--hidden1',
          type=int,
          default=DEF_HIDDEN1_UNITS,
          help='Number of units in hidden layer 1.'
    )
    parser.add_argument(
          '--hidden2',
          type=int,
          default=DEF_HIDDEN2_UNITS,
          help='Number of units in hidden layer 2.'
    )
    parser.add_argument(
          '--linear',
          type=int,
          default=DEF_LINEAR_UNITS,
          help='Number of units in linear layer.'
    )
    parser.add_argument(
          '--batch_size',
          type=int,
          default=DEF_BATCH_SIZE,
          help='Batch size.  Must divide evenly into the dataset sizes.'
    )
    parser.add_argument(
          '--input_data_dir',
          type=str,
          default=DEF_PATH_TO_DATA,
          help='Directory to put the input data.'
    )
    parser.add_argument(
          '--log_dir',
          type=str,
          default=DEF_LOG_DIR,
          help='Directory to put the log data.'
    )
    parser.add_argument(
          '--execute',
          default=False,
          help='If true, execute the training.',
          action='store_true'
    )
    parser.add_argument(
          '--print_variables',
          default=False,
          help='If true, print variables.',
          action='store_true'
    )
    parser.add_argument(
          '--inter_threads',
          type=int,
          default=DEF_INTER_THREADS,
          help='Number of threads for blocking nodes (0 means that the system picks an appropriate number).'
    )
    parser.add_argument(
          '--intra_threads',
          type=int,
          default=DEF_INTRA_THREADS,
          help='Number of threads per individual Op (0 means that the system picks an appropriate number).'
    )
    parser.add_argument(
          '--random_seed',
          type=int,
          default=DEF_RANDOM_SEED,
          help='Random seed.'
    )
 
    FLAGS, unparsed = parser.parse_known_args()
    tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
#


