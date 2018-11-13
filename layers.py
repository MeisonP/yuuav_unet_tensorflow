# coding:utf-8
# masom: Unet-tensorflow version
# 2018/11/5

import tensorflow as tf
from tensorflow.contrib.layers.python.layers import layers as tf_ctb_layers
import numpy as np
from config import *


###################################layers
def dense(input_, neural, name):
    dense = tf.layers.dense(input_, neural)
    logging.info("layer {0}, [{1}]".format(name, dense.shape))
    return dense


def conv_relu(input_, ksize, filter_num, name):
    _, h, w, d = input_.shape
    filter_shape = (ksize, ksize, input_.get_shape()[-1].value, filter_num)
    filter_ = tf.Variable(np.zeros(filter_shape, dtype=np.float32))
    bias = tf.Variable(np.zeros(filter_num, dtype=np.float32))

    conv = tf.nn.conv2d(input_, filter_, strides=[1, 1, 1, 1], padding="SAME")
    conv = tf.nn.bias_add(conv, bias)
    if batch_normalization:
        btn = tf_ctb_layers.batch_norm(conv, scale=True)
        output = tf.nn.relu(btn)
    else:
        output = tf.nn.relu(conv)
    logging.info("layer {0}, filter{1}, output{2}".format(name, filter_shape, output.shape))
    return output


def pool(input_, ksize, type_, name):
    if type_ == "max":
        pooling = tf.nn.max_pool(input_, [1, ksize, ksize, 1], strides=[1, ksize, ksize, 1], padding='SAME')
    else:
         pooling = tf.nn.avg_pool(input_, [1, ksize, ksize, 1], strides=[1, ksize, ksize, 1], padding='SAME')

    logging.info("layer {0}, [{1}]".format(name, pooling.shape))
    return pooling


def dropout(input_, keep_prob_, name):
    dropout_ = tf.nn.dropout(input_, keep_prob_)
    logging.info("layer {0}, [{1}]".format(name, dropout_.shape))
    return dropout_


def deconv(input_, filter_num, factor, name):
    _, h, w, d = input_.shape
    #  factor: Integer, upsampling factor

    filter_shape = (h, w, d, filter_num)
    filter_ = tf.Variable(np.zeros(filter_shape, dtype=np.float32))

    bias_ = tf.Variable(np.zeros(filter_num, dtype=np.float32))

    output_shape_ = tf.stack([batch_size, h * factor, w * factor, d])  # tf.stack() 矩阵拼接函数

    deconv_ = tf.nn.conv2d_transpose(input_, filter_, output_shape=output_shape_,
                                     strides=[1, factor, factor, 1], padding="SAME")
    deconv_ = tf.nn.bias_add(deconv_, bias_)

    if batch_normalization:
        btn = tf_ctb_layers.batch_norm(deconv_, scale=True)
        output = tf.nn.relu(btn)
    else:
        output = tf.nn.relu(deconv_)
    logging.info("layer {0}, [{1}]".format(name, output.shape))
    return output


def devconv_upsampling():
    pass

# ############################################loss & acc


def segment_loss(labels_, net_output):   # cross_entropy
    cross_entropy = tf.nn.softmax_cross_entropy_with_logits_v2(labels=labels_, logits=net_output)
    segment_loss_ = cross_entropy
    tf.summary.scalar("segment_loss", segment_loss_)

    return segment_loss_


def l2_regular():  # l2 Regularization
    weights = [var for var in tf.trainable_variables() if var.name.endswith('weights:0')]
    logging.info("{}".format(weights))
    l2_regular_ = tf.add_n(tf.nn.l2_loss(w) for w in weights)
    tf.summary.scalar("l2_regular", l2_regular)
    return l2_regular_


def acc(logits, labels_):
    # reshape
    # labels = tf.reshape(tf.to_int64(labels_), [-1, 1])
    labels = tf.reshape(tf.argmax(labels_, axis=1), [-1, 1])
    predicted_annots = tf.reshape(tf.argmax(logits, axis=1), [-1, 1])

    # cal
    correct_predictions = tf.equal(predicted_annots, labels)
    # type trans
    seg_acc = tf.reduce_mean(tf.cast(correct_predictions, tf.float32))
    return seg_acc




