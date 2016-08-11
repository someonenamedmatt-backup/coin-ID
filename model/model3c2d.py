import tensorflow as tf
import tf_helpers

SEED = 22

def encode_rad(input, n_labels, do=True):
    #do is dropouts, true for training, false for testing
    l = tf_helpers.get_radian_conv("conv1",input,3,3,32)
    l = tf_helpers.get_radian_pool(l,1)
    l = tf_helpers.get_radian_conv("conv2",l,3,3,64)
    l = tf_helpers.get_radian_pool(l,2)
    l = tf_helpers.get_radian_conv("conv3",l,3,3,128)
    l = tf_helpers.get_radian_pool(l,3)
    if do:
       l = tf.nn.dropout(l,.5)
    l = tf_helpers.get_dense_layer_relu("dense2",l,64)
    if do:
       l = tf.nn.dropout(l,.5)
    l = tf_helpers.get_softmax_linear_layer("softmax_linear",l,n_labels)
    return l

def encode_img(input, n_labels, do=True):
    l = tf_helpers.get_conv("conv1",input,3,3,32)
    l = tf_helpers.get_pool_and_lrn(l,1)
    l = tf_helpers.get_conv("conv2",input,3,3,64)
    l = tf_helpers.get_pool_and_lrn(l,2)
    l = tf_helpers.get_conv("conv3",input,3,3,128)
    l = tf_helpers.get_pool_and_lrn(l,3)
    if do:
       l = tf.nn.dropout(l,.5)
    l = tf_helpers.get_dense_layer_relu("dense2",l,64)
    if do:
       l = tf.nn.dropout(l,.5)
    l = tf_helpers.get_softmax_linear_layer("softmax_linear",l,n_labels)
    return l
