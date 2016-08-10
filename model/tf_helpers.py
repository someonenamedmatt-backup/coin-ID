import tensorflow as tf
import pandas as pd
from sklearn.cross_validation import train_test_split
import numpy as np
import os

def pad_radian(layer, pad_width, pad_dist):
    """ Add padding to radian layer to continue back to rotate back around """
    s = layer.get_shape()
    sl = tf.slice(layer, [0,0,0,0],[-1,-1,pad_dist,-1])
    l = tf.concat(2,[layer,sl])
    l = tf.pad(l,[[0,0],[0,pad_width],[0,0],[0,0]],"CONSTANT")
    return (l)

def get_radian_conv(name, input, width, height, dim, pad=True,reuse=False, pool=True):
    """ Get convolution to find features in a convolution """
    with tf.variable_scope(name, reuse=reuse) as scope:
        kernel = tf.get_variable("weights", shape=[width, height, input.get_shape()[3].value, dim], initializer=tf.contrib.layers.xavier_initializer())
        if pad:
            input = pad_radian(input,width-1, height - 1)
        conv = tf.nn.conv2d(input, kernel, [1,1,1,1], padding='VALID')
        b = tf.get_variable("bias",shape=[dim])
        conv = tf.nn.bias_add(conv, b)
        conv = tf.nn.relu(conv)
        _activation_summary(conv)
    return (conv)

def get_radian_pool(input):
    pool = tf.nn.max_pool(input, [1,2,2,1],[1,2,2,1],'VALID')
    return pool

def get_dense_layer_relu(name, input, dim, reuse=False):
    with tf.variable_scope(name, reuse=reuse):
        input_ = tf.reshape(input, [input.get_shape()[0].value,-1])
        w = tf.get_variable("w", shape=[input_.get_shape()[1].value,dim], initializer=tf.contrib.layers.xavier_initializer() )
        b = tf.get_variable("b", shape=[dim])
        output = tf.nn.relu(tf.matmul(input_,w) + b)
        _activation_summary(output)
    return output


def get_dense_layer(name, input, dim, reuse=False):
    with tf.variable_scope(name, reuse=reuse):
        input_ = tf.reshape(input, [input.get_shape()[0].value,-1])
        w = tf.get_variable("w", shape=[input_.get_shape()[1].value,dim], initializer=tf.contrib.layers.xavier_initializer() )
        b = tf.get_variable("b", shape=[dim])
        output = tf.nn.softmax(tf.matmul(input_,w) + b)
        _activation_summary(output)
    return output

def _activation_summary(x):
  """Helper to create summaries for activations.
  Creates a summary that provides a histogram of activations.
  Creates a summary that measure the sparsity of activations.
  Args:
    x: Tensor
  Returns:
    nothing
  """
  # Remove 'tower_[0-9]/' from the name in case this is a multi-GPU training
  # session. This helps the clarity of presentation on tensorboard.
  tf.histogram_summary( x.op.name + '/activations', x)
  tf.scalar_summary(x.op.name + '/sparsity', tf.nn.zero_fraction(x))

def _add_loss_summaries(total_loss):
    """Add summaries for losses
    Generates moving average for all losses and associated summaries for
    visualizing the performance of the network.
    Args:
    total_loss: Total loss from loss().
    Returns:
    loss_averages_op: op for generating moving averages of losses.
    """
    # Compute the moving average of all individual losses and the total loss.
    loss_averages = tf.train.ExponentialMovingAverage(0.9, name='avg')
    losses = tf.get_collection('losses')
    loss_averages_op = loss_averages.apply(losses + [total_loss])

    # Attach a scalar summary to all individual losses and the total loss; do the
    # same for the averaged version of the losses.
    for l in losses + [total_loss]:
        # Name each loss as '(raw)' and name the moving average version of the loss
        # as the original loss name.
        tf.scalar_summary(l.op.name +' (raw)', l)
        tf.scalar_summary(l.op.name, loss_averages.average(l))
    return loss_averages_op

def loss(logits, labels):
    """Add L2Loss to all the trainable variables.
    Add summary for "Loss" and "Loss/avg".
    Args:
    logits: Logits from inference().
    labels: Labels from distorted_inputs or inputs(). 1-D tensor
        of shape [batch_size]
    Returns:
    Loss tensor of type float.
    """
    # Calculate the average cross entropy loss across the batch.
    labels = tf.cast(labels, tf.int64)
    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
      logits, labels, name='cross_entropy_per_example')
    cross_entropy_mean = tf.reduce_mean(cross_entropy, name='cross_entropy')
    tf.add_to_collection('losses', cross_entropy_mean)

    # The total loss is defined as the cross entropy loss plus all of the weight
    # decay terms (L2 loss).
    return tf.add_n(tf.get_collection('losses'), name='total_loss')

def get_file_list(folder, csv_file, test_pct, label_col):
    #returns a list of files, their labels,
    #and if they're in the training or test set
    print folder
    if folder[-1] != '/':
        folder += '/'
    print csv_file
    df = pd.read_csv(csv_file)[['ID',label_col]]

    print df.head()
    df['file_names'] = map((lambda x:  str(x)),df['ID'])
    df = df[df['file_names'].isin(set(df['file_names']).intersection(set(os.listdir(folder))))]
    df['file_names'] = map(lambda x: folder + str(int(x)) + '/',df['file_names'])
    print df.head()

    return train_test_split(df[['file_names',label_col]], test_size = test_pct, stratify = df[label_col], random_state = 22)

def add_loss_summaries(total_loss):
  """Add summaries for losses in CIFAR-10 model.
  Generates moving average for all losses and associated summaries for
  visualizing the performance of the network.
  Args:
    total_loss: Total loss from loss().
  Returns:
    loss_averages_op: op for generating moving averages of losses.
  """
  # Compute the moving average of all individual losses and the total loss.
  loss_averages = tf.train.ExponentialMovingAverage(0.9, name='avg')
  losses = tf.get_collection('losses')
  loss_averages_op = loss_averages.apply(losses + [total_loss])

  # Attach a scalar summary to all individual losses and the total loss; do the
  # same for the averaged version of the losses.
  for l in losses + [total_loss]:
    # Name each loss as '(raw)' and name the moving average version of the loss
    # as the original loss name.
    tf.scalar_summary(l.op.name +' (raw)', l)
    tf.scalar_summary(l.op.name, loss_averages.average(l))

  return loss_averages_op
