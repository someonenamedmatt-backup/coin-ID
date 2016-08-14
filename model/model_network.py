import tensorflow as tf
import tf_helpers

SEED = 22

def encode_rad(input, n_labels, do=True, weight_decay = 0):
    #do is dropouts, true for training, false for testing
    l = tf_helpers.get_radian_conv("conv1",input,width = 5,height = 5, dim = 192, stride = 1)
    l = tf_helpers.get_radian_conv("conv2",l,width = 1,height = 1, dim = 160, stride = 1)
    l = tf_helpers.get_radian_conv("conv3",l,width = 1,height = 1, dim = 96, stride = 1)
    l = tf_helpers.get_radian_pool(l,1, ksize = 3, stride = 2)
    if do:
       l = tf.nn.dropout(l,.5)
    l = tf_helpers.get_radian_conv("conv4",l,width = 5,height = 5, dim = 192, stride = 1)
    l = tf_helpers.get_radian_conv("conv5",l,width = 1,height = 1, dim = 192, stride = 1)
    l = tf_helpers.get_radian_conv("conv6",l,width = 1,height = 1, dim = 192, stride = 1)
    l = tf.nn.avg_pool(l, [1,3,3,1],[1,2,2,1],'VALID', name = 'avg_pool1')
    if do:
       l = tf.nn.dropout(l,.5)
    l = tf_helpers.get_radian_conv("conv7",l,width = 3,height = 3, dim = 192, stride = 1)
    l = tf_helpers.get_radian_conv("conv8",l,width = 1,height = 1, dim = 192, stride = 1)
    l = tf_helpers.get_radian_conv("conv9",l,width = 1,height = 1, dim = n_labels, stride = 1)
    l = tf.nn.avg_pool(l, [1,8,8,1],[1,1,1,1],'VALID', name = 'avg_pool2')
    l = tf.contrib.layers.flatten(l)
    l = tf_helpers.get_softmax_linear_layer("softmax_linear",l,n_labels)
    return l

def encode_img(input, n_labels, do=True, batch_size = 100, weight_decay = .004):
    l = tf_helpers.get_conv("conv1",input,width = 5,height = 5, dim = 192, stride = 1)
    l = tf_helpers.get_conv("conv2",l,width = 1,height = 1, dim = 160, stride = 1)
    l = tf_helpers.get_conv("conv3",l,width = 1,height = 1, dim = 96, stride = 1)
    l = tf_helpers.get_pool(l,1, ksize = 3, stride = 2)
    if do:
       l = tf.nn.dropout(l,.5)
    l = tf_helpers.get_conv("conv4",l,width = 5,height = 5, dim = 192, stride = 1)
    l = tf_helpers.get_conv("conv5",l,width = 1,height = 1, dim = 192, stride = 1)
    l = tf_helpers.get_conv("conv6",l,width = 1,height = 1, dim = 192, stride = 1)
    l = tf.nn.avg_pool(l, [1,3,3,1],[1,2,2,1],'VALID', name = 'avg_pool1')
    if do:
       l = tf.nn.dropout(l,.5)
    l = tf_helpers.get_conv("conv7",l,width = 3,height = 3, dim = 192, stride = 1)
    l = tf_helpers.get_conv("conv8",l,width = 1,height = 1, dim = 192, stride = 1)
    l = tf_helpers.get_conv("conv9",l,width = 1,height = 1, dim = n_labels, stride = 1)
    l = tf.nn.avg_pool(l, [1,8,8,1],[1,1,1,1],'VALID', name = 'avg_pool2')
    l = tf.contrib.layers.flatten(l)
    # softmax, i.e. softmax(WX + b)
    with tf.variable_scope('softmax_linear') as scope:
        weights = _variable_with_weight_decay('weights', [192, n_labels],
                                              stddev=1/192.0, wd=None)
        biases = _variable_on_cpu('biases', [n_labels],
                                  tf.constant_initializer(0.0))
        softmax_linear = tf.add(tf.matmul(l, weights), biases, name=scope.name)
        tf_helpers._activation_summary(softmax_linear)
    return softmax_linear
