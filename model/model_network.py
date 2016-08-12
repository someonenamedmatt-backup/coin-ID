import tensorflow as tf
import tf_helpers

SEED = 22

def encode_rad(input, n_labels, do=True):
    #do is dropouts, true for training, false for testing
    l = tf_helpers.get_radian_conv("conv1",input,width = 5,height = 5, dim = 192, stride = 1)
    
    l = tf_helpers.get_radian_conv("conv3",l,width = 1,height = 1, dim = 96, stride = 1)
    l = tf_helpers.get_radian_pool(l,1, ksize = 3, strides = 2)
    if do:
       l = tf.nn.dropout(l,.5)
    l = tf_helpers.get_radian_conv("conv4",l,width = 5,height = 5, dim = 192, stride = 1)

    l = tf_helpers.get_radian_conv("conv6",l,width = 1,height = 1, dim = 192, stride = 1)
    l = tf.nn.avg_pool(l, [1,3,3,1],[1,2,2,1],'VALID', name = 'avg_pool1')
    if do:
       l = tf.nn.dropout(l,.5)
    l = tf_helpers.get_radian_conv("conv7",l,width = 3,height = 3, dim = 192, stride = 1)

    l = tf_helpers.get_radian_conv("conv9",l,width = 1,height = 1, dim = n_labels, stride = 1)
    l = tf.nn.avg_pool(l, [1,8,8,1],[1,1,1,1],'VALID', name = 'avg_pool2')
    l = tf.contrib.layers.flatten(l)
    return l
