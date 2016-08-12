import tensorflow as tf
import tf_helpers

SEED = 22

def encode_rad(input, n_labels, do=True):
    #do is dropouts, true for training, false for testing
    l = tf_helpers.get_radian_conv("conv1",input,width = 11,height = 11, dim = 48, stride = 4)
    l = tf_helpers.get_pool_and_lrn(l, 1, ksize=3, strides=2)

    l = tf_helpers.get_radian_conv("conv2",input,width = 5,height = 5, dim = 128, stride = 1)
    l = tf_helpers.get_pool_and_lrn(l, 2, ksize=3, strides=2)

    l = tf_helpers.get_radian_conv("conv3",input,width = 3,height = 3, dim = 128, stride = 1)    
    l = tf_helpers.get_radian_conv("conv5",input,width = 3,height = 3, dim = 96, stride = 1)
    l = tf_helpers.get_pool_and_lrn(l, 3, ksize=3, strides=2)

    l = tf_helpers.get_dense_layer_relu("dense1",l,400)
    if do:
       l = tf.nn.dropout(l,.5)
    l = tf_helpers.get_dense_layer_relu("dense2",l,400)
    if do:
       l = tf.nn.dropout(l,.5)
    l = tf_helpers.get_softmax_linear_layer("softmax_linear",l,n_labels)
    return l
