# coding:utf-8
"""module, define a unet network (supplying some-layer method and net method)

2018/12/04
tensorflow ==1.11
python ==2.7.15

Note:
    #  factor: Integer, upsampling factor
    # tf.stack() 矩阵拼接函数
    # reshape(-1, num), the -1 mean that
        simply means that it is an unknown dimension and we want numpy to figure it out.

    # logits output， -1 of reshape means that the axis is unknowing and will be  computed, (x, class_num).

    the first dimension of net input shape is the batch size, as the data feed into the neural network each time
    is a batch, not a single images, that why the input dimension is four (batch_size, h,w, d)


    # the filter size of last layer :conv2d is [1, 1, class_num]

    the net output is logits

"""


import tensorflow as tf
from tensorflow.contrib.layers.python.layers import layers as tf_ctb_layers
from config import *
import numpy as np


def dense(input_, neural, name):
    """full connect layer
    """

    dense = tf.layers.dense(input_, neural)
    logging.info("layer {0}, [{1}]".format(name, dense.shape))
    return dense


def conv_relu(input_, ksize, filter_num, name, activation=True):
    """ convolutional layer, with specific activation func and  batch_normal
    """

    with tf.variable_scope(name):
        if activation is True:
            _, h, w, d = input_.shape
            filter_shape = (ksize, ksize, input_.get_shape()[-1].value, filter_num)

            # filter_ = tf.Variable(np.zeros(filter_shape, dtype=np.float32))
            filter_ = tf.get_variable('weights', filter_shape, tf.float32)

            # bias = tf.Variable(np.zeros(filter_num, dtype=np.float32))
            bias = tf.get_variable('bias', filter_num, dtype=tf.float32)

            conv = tf.nn.conv2d(input_, filter_, strides=[1, 1, 1, 1], padding="SAME")
            conv = tf.nn.bias_add(conv, bias)
            if batch_normalization:
                btn = tf_ctb_layers.batch_norm(conv, scale=True)
                output = tf.nn.relu(btn)
            else:
                output = tf.nn.relu(conv)

        else:
            filter_shape = [ksize, ksize, input_.get_shape()[-1].value, filter_num]

            filter_ = tf.get_variable("weights", filter_shape, tf.float32)
            # filter_ = tf.Variable(np.zeros(filter_shape, dtype=np.float32))

            output = tf.nn.conv2d(input_, filter_,
                                  strides=[1, 1, 1, 1],
                                  padding="SAME")
    logging.info("layer {0}, filter{1}, output{2}".format(name, filter_shape, output.shape))
    return output


def pool(input_, ksize, type_, name):
    """ pooling layer
    """

    with tf.name_scope(name):

        if type_ == "max":
            pooling = tf.nn.max_pool(input_, [1, ksize, ksize, 1], strides=[1, ksize, ksize, 1], padding='SAME')
        else:
            pooling = tf.nn.avg_pool(input_, [1, ksize, ksize, 1], strides=[1, ksize, ksize, 1], padding='SAME')

    logging.info("layer {0}, {1}".format(name, pooling.shape))
    return pooling


def dropout(input_, keep_prob_, name):
    """ dropout layer
    """

    with tf.name_scope(name):
        dropout_ = tf.nn.dropout(input_, keep_prob_)
        logging.info("layer {0}, {1}".format(name, dropout_.shape))
    return dropout_


def deconv(input_, filter_num, factor, name):
    """ de-convolutional layer, tf.nn.using conv2d_transpose()
    """

    with tf.variable_scope(name):

        batch_size_, h, w, d = input_.shape

        filter_shape = (h, w, d, filter_num)

        # filter_ = tf.Variable(np.zeros(filter_shape, dtype=np.float32))
        filter_ = tf.get_variable('weights', filter_shape, tf.float32)
        # bias_ = tf.Variable(np.zeros(filter_num, dtype=np.float32))
        bias_ = tf.get_variable('bias', filter_num)

        output_shape_ = tf.stack([batch_size_, h * factor, w * factor, d])

        deconv_ = tf.nn.conv2d_transpose(input_, filter_, output_shape=output_shape_,
                                         strides=[1, factor, factor, 1], padding="SAME")
        deconv_ = tf.nn.bias_add(deconv_, bias_)

        if batch_normalization:
            btn = tf_ctb_layers.batch_norm(deconv_, scale=True)
            output = tf.nn.relu(btn)
        else:
            output = tf.nn.relu(deconv_)

    logging.info("layer {0}, {1}".format(name, output.shape))
    return output


def concat(input_a, input_b, name_, axis_=3):
    """ concat two input layer
    """
    with tf.name_scope("concat"):
        concat_ = tf.concat([input_a, input_b], axis=axis_, name=name_)

    return concat_


def unet(input_):

    inputs = input_
    logging.info("the input shape: {}".format(inputs.shape))
    net = {}

    # #############conv
    # block 1
    net['conv1_1'] = conv_relu(input_=inputs, ksize=3, filter_num=filters, name="conv1_1")
    net['conv1_2'] = conv_relu(net['conv1_1'], 3, filters, "conv1_2")
    net['pool1'] = pool(net['conv1_2'], ksize=2, type_=max, name='pool1')

    # block 2
    net['conv2_1'] = conv_relu(net['pool1'], 3, filters * 2, "conv2_1")
    net['conv2_2'] = conv_relu(net['conv2_1'], 3, filters * 2, "conv2_2")
    net['pool2]'] = pool(net['conv2_2'], 2, max, 'pool2')

    # block 3
    net['conv3_1'] = conv_relu(net['pool2]'], 3, filters * 4, "conv3_1")
    net['conv3_2'] = conv_relu(net['conv3_1'], 3, filters * 4, "conv3_2")
    net['pool3'] = pool(net['conv3_2'], 2, max, 'pool3')
    net['dropout3'] = dropout(net['pool3'], keep_prob, name='dropout3')

    # block 4
    net['conv4_1'] = conv_relu(net['dropout3'], 3, filters * 8, "conv4_1")
    net['conv4_2'] = conv_relu(net['conv4_1'], 3, filters * 8, "conv4_2")
    net['pool4'] = pool(net['conv4_2'], 2, max, 'pool4')
    net['dropout4'] = dropout(net['pool4'], keep_prob, name='dropout4')


    # block 5
    net['conv5_1'] = conv_relu(net['dropout4'], 3, filters * 16, "conv5_1")
    net['conv5_2'] = conv_relu(net['conv5_1'], 3, filters * 16, "conv5_2")
    net['dropout5'] = dropout(net['conv5_2'], keep_prob, name='dropout5')

    # #############deconv
    # block 6
    net['upsample6'] = deconv(net['dropout5'], filters * 16, 2, "upsample6")
    net['concat6'] = concat(net['upsample6'], net['conv4_2'], axis_=3, name_='concat6')

    net['conv6_1'] = conv_relu(net['concat6'], 3, filters * 8, "conv6_1")
    net['conv6_2'] = conv_relu(net['conv6_1'], 3, filters * 8, "conv6_2")
    net['dropout6'] = dropout(net['conv6_2'], keep_prob, name='dropout6')

    # block 7
    net['upsample7'] = deconv(net['dropout6'], filters * 8, 2, "upsample7")
    net['concat7'] = concat(net['upsample7'], net['conv3_2'], axis_=3, name_='concat7')

    net['conv7_1'] = conv_relu(net['concat7'], 3, filters * 4, "conv7_1")
    net['conv7_2'] = conv_relu(net['conv7_1'], 3, filters * 4, "conv7_2")
    net['dropout7'] = dropout(net['conv7_2'], keep_prob, name='dropout7')

    # block 8
    net['upsample8'] = deconv(net['dropout7'], filters * 4, 2, "upsample8")
    net['concat8'] = concat(net['upsample8'], net['conv2_2'], axis_=3, name_='concat8')

    net['conv8_1'] = conv_relu(net['concat8'], 3, filters * 2, "conv8_1")
    net['conv8_2'] = conv_relu(net['conv8_1'], 3, filters * 2, "conv8_2")

    # block 9
    net['upsample9'] = deconv(net['conv8_2'], filters * 2, 2, "upsample9")
    net['concat9'] = concat(net['upsample9'], net['conv1_2'], axis_=3, name_='concat9')

    net['conv9_1'] = conv_relu(net['concat9'], 3, filters, "conv9_1")
    net['conv9_2'] = conv_relu(net['conv9_1'], 3, filters, "conv9_2")

    # block 10
    # the filter 3 is mean the image channel num

    net['conv10'] = conv_relu(net['conv9_2'], 1, num_classes, "conv10", False)
    # net['conv10'] = conv_relu(net['conv9_2'], 1, 3, "conv10", False)

    with tf.variable_scope('net_output'):
        net['output'] = tf.reshape(net['conv10'], (-1, num_classes), name='logits')
    # net['output'] = tf.reshape(net['conv10'], (-1, 3))  # the 3 is the rgb channel

    logging.info("the model output shape: {}".format(net["output"].shape))

    return net



