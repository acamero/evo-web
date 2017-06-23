import tensorflow as tf


def conv2d(x,W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')


def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')


def inference(images, keep_prob, num_classes, image_height, image_width, num_channels, hidden1_units, hidden2_units, linear_units):
    # Hidden 1
    with tf.name_scope('hidden1'):
        weights = tf.Variable(
            tf.truncated_normal([5, 5, num_channels, hidden1_units], stddev=0.1),
            name='weights')  
        biases = tf.Variable(
            tf.constant(0.1, shape=[hidden1_units]),
            name='biases')
        x_images = tf.reshape(images,[-1, image_height, image_width, num_channels])
        conv = tf.nn.relu(conv2d(x_images, weights) + biases)
        hidden1 = max_pool_2x2(conv)
    # Hidden 2
    with tf.name_scope('hidden2'):
        weights = tf.Variable(
            tf.truncated_normal([5, 5, hidden1_units, hidden2_units], stddev=0.1),
            name='weights')  
        biases = tf.Variable(
            tf.constant(0.1, shape=[hidden2_units]),
            name='biases')
        conv = tf.nn.relu(conv2d(hidden1, weights) + biases)
        hidden2 = max_pool_2x2(conv)
    # Linear
    with tf.name_scope('linear'):
        t_size = reduce(lambda x,y: (x if x else 1) * (y if y else 1), hidden2.get_shape().as_list())
        weigths = tf.Variable(
            tf.truncated_normal([t_size, linear_units], stddev=0.1),
            name='weights') 
        biases = tf.Variable(
            tf.constant(0.1, shape=[linear_units]),
            name='biases') 
        flatten = tf.reshape(hidden2, [-1, t_size])
        linear = tf.nn.relu(tf.matmul(flatten, weigths) + biases)
    # Output
    with tf.name_scope('output'):
        dropout = tf.nn.dropout(linear, keep_prob)
        weights = tf.Variable(
            tf.truncated_normal([linear_units, num_classes], stddev=0.1),
            name='weights')
        biases = tf.Variable(
            tf.constant(0.1, shape=[num_classes]),
            name='biases')
        output = tf.matmul(dropout, weights) + biases
    return output


def loss(inf_classes, labels):
    #epsilon = tf.constant( 0.0001, shape=labels.get_shape().as_list()[0] )
    #logits = epsilon + inf_classes
    #return tf.reduce_mean( tf.nn.softmax_cross_entropy_with_logits(labels=labels, logits=logits), name='xentropy_mean')
    #return tf.reduce_mean( tf.nn.sigmoid_cross_entropy_with_logits(labels=labels, logits=inf_classes), name='xentropy_mean')
    return tf.reduce_mean( tf.nn.softmax_cross_entropy_with_logits(labels=labels, logits=inf_classes), name='xentropy_mean')


def training(loss, learning_rate):
    # Add a scalar summary for the snapshot loss.
    tf.summary.scalar('loss', loss)
    # Create the gradient descent optimizer with the given learning rate.
    optimizer = tf.train.GradientDescentOptimizer(learning_rate)
    # Create a variable to track the global step.
    global_step = tf.Variable(0, name='global_step', trainable=False)
    # Use the optimizer to apply the gradients that minimize the loss
    # (and also increment the global step counter) as a single training step.
    train_op = optimizer.minimize(loss, global_step=global_step)
    return train_op


def evaluation(inf_classes, labels):
    # For a classifier model, we can use the in_top_k Op.
    # It returns a bool tensor with shape [batch_size] that is true for
    # the examples where the label is in the top k (here k=1)
    # of all logits for that example.
    #correct = tf.nn.in_top_k(inf_classes, labels, 1)
    # Return the number of true entries.
    #return tf.reduce_sum(tf.cast(correct, tf.int32))
    correct_prediction = tf.equal(tf.argmax(inf_classes,1),tf.argmax(labels,1))
    # accuracy
    return tf.reduce_mean(tf.cast(correct_prediction,tf.float32))



