# coding:utf-8
# masom: Unet-tensorflow version
# 2018/11/5

import tensorflow as tf
import numpy as np
import layers
from config import class_num, keep_prob, logging


def net(input_):
    inputs = input_
    logging.info("the input shape: {}".format(inputs.shape))
    net = {}

    # #############conv
    # block 1
    net['conv1_1'] = layers.conv_relu(input_=inputs, ksize=3, filter_num=64, name="conv1_1")
    net['conv1_2'] = layers.conv_relu(net['conv1_1'], 3, 64, "conv1_2")
    net['pool1'] = layers.pool(net['conv1_2'], ksize=2, type_=max, name='pool1')

    # block 2
    net['conv2_1'] = layers.conv_relu(net['pool1'], 3, 128, "conv2_1")
    net['conv2_2'] = layers.conv_relu(net['conv2_1'], 3, 128, "conv2_2")
    net['pool2]'] = layers.pool(net['conv2_2'], 2, max, 'pool2')

    # block 3
    net['conv3_1'] = layers.conv_relu(net['pool2]'], 3, 256, "conv3_1")
    net['conv3_2'] = layers.conv_relu(net['conv3_1'], 3, 256, "conv3_2")
    net['pool3'] = layers.pool(net['conv3_2'], 2, max, 'pool3')
    net['dropout3'] = layers.dropout(net['pool3'], keep_prob, name='dropout3')

    # block 4
    net['conv4_1'] = layers.conv_relu(net['pool3'], 3, 512, "conv4_1")
    net['conv4_2'] = layers.conv_relu(net['conv4_1'], 3, 512, "conv4_2")
    net['pool4'] = layers.pool(net['conv4_2'], 2, max, 'pool4')
    net['dropout4'] = layers.dropout(net['pool4'], keep_prob, name='dropout4')

    # block 5
    net['conv5_1'] = layers.conv_relu(net['dropout4'], 3, 1024, "conv5_1")
    net['conv5_2'] = layers.conv_relu(net['conv5_1'], 3, 1024, "conv5_2")
    net['dropout5']=layers.dropout(net['conv5_2'], keep_prob, name='dropout5')

    # #############deconv
    # block 6
    net['upsample6'] = layers.deconv(net['dropout5'], 1024, 2, "upsample6")
    net['concat6'] = tf.concat([net['upsample6'], net['conv4_2']],axis=3,name='concat6')

    net['conv6_1'] = layers.conv_relu(net['concat6'], 3, 512, "conv6_1")
    net['conv6_2'] = layers.conv_relu(net['conv6_1'], 3, 512, "conv6_2")
    net['dropout6'] = layers.dropout(net['conv6_2'], keep_prob, name='dropout6')

    # block 7
    net['upsample7'] = layers.deconv(net['dropout6'], 512, 2, "upsample7")
    net['concat7'] = tf.concat([net['upsample7'], net['conv3_2']], axis=3, name='concat7')

    net['conv7_1'] = layers.conv_relu(net['concat7'], 3, 256, "conv7_1")
    net['conv7_2'] = layers.conv_relu(net['conv7_1'], 3, 256, "conv7_2")
    net['dropout7'] = layers.dropout(net['conv7_2'], keep_prob, name='dropout7')

    # block 8
    net['upsample8'] = layers.deconv(net['dropout7'], 256, 2, "upsample8")
    net['concat8'] = tf.concat([net['upsample8'], net['conv2_2']], axis=3, name='concat8')

    net['conv8_1'] = layers.conv_relu(net['concat8'], 3, 128, "conv8_1")
    net['conv8_2'] = layers.conv_relu(net['conv8_1'], 3, 128, "conv8_2")

    # block 9
    net['upsample9'] = layers.deconv(net['conv8_2'], 128, 2, "upsample9")
    net['concat9'] = tf.concat([net['upsample9'], net['conv1_2']], axis=3, name='concat9')

    net['conv9_1'] = layers.conv_relu(net['concat9'], 3, 64, "conv9_1")
    net['conv9_2'] = layers.conv_relu(net['conv9_1'], 3, 64, "conv9_2")

    # block 10
    # the filter 3 is mean the image channel num
    net['conv10'] = tf.nn.conv2d(net['conv9_2'], filter=tf.Variable(np.zeros([1, 1, 64, 3], dtype=np.float32)),
                                 strides=[1, 1, 1, 1], padding="SAME", name='conv10')
    logging.info("the layer conv10 shape: {}".format(net["conv10"].shape))

    # logits output， -1 of reshape means that the axis is unknowing and will be  computed, (x, class_num).
    net['output'] = tf.reshape(net['conv10'], (-1, class_num))

    logging.info("the model output shape: {}".format(net["output"].shape))
    return net



