import tensorflow as tf
import numpy as np
import data
import sys
import scipy.misc
import scipy.ndimage
import matplotlib.pyplot as plt
import os
from coinsampler import CoinSampler
import csv
from collections import deque

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
    return output


def get_dense_layer(name, input, dim, reuse=False):
    with tf.variable_scope(name, reuse=reuse):
        input_ = tf.reshape(input, [input.get_shape()[0].value,-1])
        w = tf.get_variable("w", shape=[input_.get_shape()[1].value,dim], initializer=tf.contrib.layers.xavier_initializer() )
        b = tf.get_variable("b", shape=[dim])
        output = tf.nn.softmax(tf.matmul(input_,w) + b)

    return output

def encode(input, n_labels, do=True):
    #do is dropouts, true for training, false for testing
    l = get_radian_conv("cl1",input,3,3,32)
    l = get_radian_conv("cl2",l,3,3,32)
    l = get_radian_pool(l)
    l = get_radian_conv("cl3",l,3,3,64)
    l = get_radian_conv("cl4",l,3,3,64)
    l = get_radian_pool(l)
    l = get_radian_conv("cl5",l,3,3,128)
    l = get_radian_conv("cl6",l,3,3,128)
    l = get_radian_pool(l)
    l = get_radian_conv("cl7",l,3,3,256)
    l = get_radian_conv("cl8",l,3,3,256)
    l = get_radian_pool(l)
    l = get_radian_conv("cl9",l,3,3,512)
    l = get_radian_conv("cl10",l,3,3,512)
    l = get_radian_pool(l)
    l = get_radian_conv("cl11",l,2,2,1024)
    l = get_radian_conv("cl12",l,2,2,1024)
    l = get_radian_pool(l)

    l = get_dense_layer_relu("d1",l,1024)
    if do:
       l = tf.nn.dropout(l,.5)
    l = get_dense_layer_relu("d2",l,1024)
    if do:
       l = tf.nn.dropout(l,.5)
    l = get_dense_layer("d3",l,n_labels)
    return l

def test_model(cs,model_title, coin_prop = 'rad'):
    tf.reset_default_graph()
    n_labels = len(cs.labels)
    input_tensor = tf.placeholder(tf.float32, [len(cs.IDlabel),90,90,3])
    val = encode(input_tensor, n_labels, do = False)

    saver = tf.train.Saver()

    with tf.Session() as sess:
        tf.initialize_all_variables().run()
        saver.restore(sess, 'tf_saves/' + model_title+'.ckpt')
        x,y = cs.get_all_coins(coin_prop)
        v = accuracy.eval(feed_dict={input_tensor: x, labels: y})
        print("test accuracy %g"%v)
    return(v)

def train_model(cs,ts, model_title, iters=50000, coin_prop = 'rad'):
    tf.reset_default_graph()
    n_labels = len(cs.labels)
    #number of output classes
    batch_gen = cs.batch_generator(coin_prop)

    input_tensor = tf.placeholder(tf.float32, [BATCH_LEN,90,90,3])
    labels = tf.placeholder(tf.float32,[BATCH_LEN,n_labels])
    predict_proba = encode(input_tensor, n_labels)
    cost = tf.reduce_mean(-tf.reduce_sum(labels * tf.log(predict_proba), reduction_indices=[1]))
    train = tf.train.AdamOptimizer(0.001).minimize(cost)

    correct_prediction = tf.equal(tf.argmax(labels,1), tf.argmax(predict_proba,1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    saver = tf.train.Saver()
    with tf.Session() as sess:
        epoch_num = 0

        sess.run(tf.initialize_all_variables())
        stop_point = 0
        if os.path.exists('tf_saves/' +model_title + '.ckpt'):
            with open('tf_saves/' +model_title + '.csv', 'r') as f:
                stop_point = int(deque(csv.reader(f), 1)[0][0])
            saver.restore(sess,'tf_saves/'+model_title+'.ckpt')
        else:
            stop_point = 0

        for b in range(iters-stop_point):
            x, y = batch_gen.next()
            _, loss_value, prediction  = sess.run([train, cost, predict_proba], feed_dict={input_tensor: x, labels: y})
            if b%250 == 0:
                train_accuracy = accuracy.eval(feed_dict={input_tensor: x, labels: y})
                with open('tf_saves/' +model_title + '.csv', 'a') as f:
                    writer = csv.writer(f)
                    writer.writerow([b+stop_point, train_accuracy])
                saver.save(sess, 'tf_saves/'+model_title+'.ckpt')
            if (b+stop_point)/len(cs.IDlabel) > epoch_num:
                epoch_num+=1
                x_test,y_test = ts.get_all_coins(coin_prop)
                v = accuracy.eval(feed_dict={input_tensor: x_test, labels: y_test})
                with open('tf_saves/' +model_title + '-epoch-accuracy.csv', 'a') as f:
                    writer = csv.writer(f)
                    writer.writerow([epoch_num, v])




SEED = 22
BATCH_LEN = 30
