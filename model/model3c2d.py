import tensorflow as tf
import tf_helpers

SEED = 22

def encode_rad(input, n_labels, do=True, weight_decay = 0.04):
    #do is dropouts, true for training, false for testing
    l = tf_helpers.get_radian_conv("conv1",input,width = 3,height = 3, dim = 32)
    l = tf_helpers.get_radian_pool(l,1, ksize = 2)
    l = tf_helpers.get_radian_conv("conv2",l,width = 3,height = 3, dim = 64, stride = 1)
    l = tf_helpers.get_radian_conv("conv3",l,width = 3,height = 3, dim = 64, stride = 1)
    l = tf_helpers.get_radian_pool(l,2, ksize = 2)
    l = tf_helpers.get_dense_layer_relu("dense1",l,256, weight_decay)
    if do:
       l = tf.nn.dropout(l,.5)
    l = tf_helpers.get_softmax_linear_layer("softmax_linear",l,n_labels)
    return l


def encode_img(input, n_labels, do=True, weight_decay = 0.04):
    #do is dropouts, true for training, false for testing
    l = tf_helpers.get_conv("conv1",input,width = 3,height = 3, dim = 32)
    l = tf_helpers.get_pool_and_lrn(l,1, ksize = 2)
    l = tf_helpers.get_conv("conv2",l,width = 3,height = 3, dim = 64, stride = 1)
    l = tf_helpers.get_conv("conv3",l,width = 3,height = 3, dim = 64, stride = 1)
    l = tf_helpers.get_pool_and_lrn(l,2, ksize = 2)
    l = tf_helpers.get_dense_layer_relu("dense1",l,256, weight_decay)
    if do:
       l = tf.nn.dropout(l,.5)
    l = tf_helpers.get_softmax_linear_layer("softmax_linear",l,n_labels)
    return l
