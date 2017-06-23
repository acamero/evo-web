import tensorflow as tf
import random

class ImageFolderReader(object):

    scale_factor = 5
    image_width = int(1024 / scale_factor)
    image_height = int(768 / scale_factor)
    num_channels = 1
    resize_method = tf.image.ResizeMethod.BILINEAR

    def __init__(self, filename, test_set_perc, seed = 0):
        self.seed = seed
        random.seed(self.seed)
        self.filename = filename
        # Reads pfathes of images together with their labels
        image_list, label_list = self._read_labeled_image_list(filename)
        self.max_label = max(label_list) + 1
        self.images = tf.convert_to_tensor(image_list, dtype=tf.string)
        self.labels = tf.convert_to_tensor(label_list, dtype=tf.int32)
        self.set_size = len(image_list)
        self._partition(test_set_perc)
        


    def _partition(self, test_set_perc):
        # create a partition vector
        self.partitions = [0] * self.set_size
        self.test_set_size = int(self.set_size * test_set_perc)
        self.partitions[:self.test_set_size] = [1] * self.test_set_size
        random.shuffle(self.partitions)
        # partition our data into a test and train set according to our partition vector
        self.train_images, self.test_images = tf.dynamic_partition(self.images, self.partitions, 2)
        self.train_labels, self.test_labels = tf.dynamic_partition(self.labels, self.partitions, 2)


    def _read_labeled_image_list(self, image_list_file):
        """Reads a .txt file containing pathes and labeles
        Args:
           image_list_file: a .txt file with one /path/to/image per line
           label: optionally, if set label will be pasted after each line
        Returns:
           List with all filenames in file image_list_file
        """
        f = open(image_list_file, 'r')
        filenames = []
        labels = []
        for line in f:
            filename, label = line[:-1].split('\t')
            filenames.append(filename)
            labels.append(int(label))
        return filenames, labels

    def _read_images_from_disk(self, input_queue):
        """Consumes a single filename and label as a ' '-delimited string.
        Args:
          filename_and_label_tensor: A scalar string tensor.
        Returns:
          Two tensors: the decoded image, and the string label.
        """
        label = input_queue[1]
        file_contents = tf.read_file(input_queue[0])
        example = tf.image.decode_png(file_contents, channels=self.num_channels)
        return example, label


    def _get_queue(self, images_partition, labels_partitions, epochs):
        """Input queue from images and labels using partitions
        Args:
            images_partition: tensor list of images
            labels_partitions: tensor list of labels
            epochs: slice input epochs
        Returns:
            input_queue: the desired input_queues
        """  
        # create input queues
        queue = tf.train.slice_input_producer(
                                        [images_partition, labels_partitions],
                                        num_epochs=epochs,
                                        shuffle=True)
        return queue


    def get_input_queue(self, batch_size, epochs, one_hot, train=True):
        if train:
            queue = self._get_queue(self.train_images, self.train_labels, epochs)
        else:
            queue = self._get_queue(self.test_images, self.test_labels, epochs)
        image, label = self._read_images_from_disk(queue)
        # [0,n] ->  [k_0 .. k_n], k in [0,1]
        if one_hot:
            label = tf.one_hot(label, self.max_label)
        # TODO: add preprocessing if needed...
        image = tf.image.resize_images(image, [self.image_height, self.image_width], method=self.resize_method)
        image = tf.reshape(image, [-1])
        # collect batches of images before processing
        image_batch, label_batch = tf.train.batch( 
                                    [image, label],
                                    batch_size=batch_size
                                    #,num_threads=1
                                    )
        return image_batch, label_batch


# class ImageFolderReader
