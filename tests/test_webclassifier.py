import tensorflow as tf
import random

FILENAME = "webshot_data/path_to_data.txt"
ONE_HOT = False

def _read_labeled_image_list(image_list_file):
    f = open(image_list_file, 'r')
    filenames = []
    labels = []
    for line in f:
        filename, label = line[:-1].split('\t')
        filenames.append(filename)
        labels.append(int(label))
    return filenames, labels

def _read_images_from_disk(input_queue):
    label = input_queue[1]
    file_contents = tf.read_file(input_queue[0])
    example = tf.image.decode_png(file_contents, channels=1)
    return example, label

image_height = int(768/10)
image_width = int(1024/10)
resize_method = tf.image.ResizeMethod.BILINEAR
seed = 0
image_list, label_list = _read_labeled_image_list(FILENAME)
max_label = max(label_list)
images = tf.convert_to_tensor(image_list, dtype=tf.string)
labels = tf.convert_to_tensor(label_list, dtype=tf.int32)
set_size = len(image_list)
partitions = [0] * set_size
test_set_size = int(set_size * 0.2)
partitions[:test_set_size] = [1] * test_set_size
random.seed(seed)
random.shuffle(partitions)
train_images, test_images = tf.dynamic_partition(images, partitions, 2)
train_labels, test_labels = tf.dynamic_partition(labels, partitions, 2)
train_input_queue = tf.train.slice_input_producer(
                                        [train_images, train_labels],
                                        num_epochs=None,
                                        shuffle=True,
                                        seed=seed)
test_input_queue = tf.train.slice_input_producer(
                                        [test_images, test_labels],
                                        num_epochs=1,
                                        shuffle=True,
                                        seed=seed)
train_image, train_label = _read_images_from_disk(train_input_queue)
test_image, test_label = _read_images_from_disk(test_input_queue)
if ONE_HOT:
    train_label = tf.one_hot(train_label, max_label)
    test_label = tf.one_hot(test_label, max_label)
train_image = tf.image.resize_images(train_image, [image_height, image_width], method=resize_method)
test_image = tf.image.resize_images(test_image, [image_height, image_width], method=resize_method)
train_image = tf.reshape(train_image, [-1])
test_image = tf.reshape(test_image, [-1])
train_image_batch, train_label_batch = tf.train.batch( 
                                    [train_image, train_label],
                                    batch_size=10)
test_image_batch, test_label_batch = tf.train.batch(
                                    [test_image, test_label],
                                    batch_size=10)


def evaluate(sess, coord, queue):
    print("Evaluate")
    try:
        while not coord.should_stop():
            print( sess.run(queue) )
    except Exception, e:
        print("Evaluation exception")
    finally:
        print("Finally evaluation")
    #

with tf.Session(config=tf.ConfigProto(inter_op_parallelism_threads=1, intra_op_parallelism_threads=1)) as sess:  
    # initialize the variables
    sess.run(tf.local_variables_initializer())
    #sess.run(tf.global_variables_initializer())  
    # initialize the queue threads to start to shovel data
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)
    try:
        for j in xrange(3):
            for i in range(5):
                print("Train cycle %d" % i)
                print( sess.run(train_label_batch) )            
            #
            evaluate(sess, coord, test_label_batch)
    except Exception, e:
        coord.request_stop(e)
        print("Exception")
    finally:
        # When done, ask the threads to stop.
        coord.request_stop()
        coord.join(threads)
        print("Finally")
    #
    sess.close()
