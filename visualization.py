# coding: utf-8
""" module prepared for deploy.py ;  visualizing the network output,

2018/12/04
python ==2.7.15

Note:
    the net['output'] is a tensor with shape (BS*h*w, num_classes) without softmax,
    this module do not cover the sofmax preocess

    mainly, contain two step:
    1. from   softmax process of net['output'] --> label_2d format matrix
    2. label_2d matrix --> 3channel bgr_image using colormap


    ！！！the output of network is (BS*h*w, num_class) with the value float 0 to 1 such as 0.01

    the net_output will be a distribute, do not pass through softmax calculation

"""

import numpy as np


def netoutput_2_labelmat(net_output, h, w, class_num):
    """reshape network's output, and transform to a 2d label matrix
    Note:
        in this cast, the net_output is a 2D matrix, with shape(BS*h*w, num_classes) and the value is not 1/0 .

        while, if you using  tf.argmax(logits),
        then the output will be a 2d matrix with shape(BS*h*w, 1) and the value is class label id
        then this func can be removed.


    :param
        (h, w): inter, the size of image that need to be visualized
        net_output: a tensor, the output of the network
        class_num: inter, the numbers of class that need be classified
    :return:
        a 2D array, whose value donates the label ID
    """

    if np.array(net_output).shape[0] == 0:
        print "predict error, output None!"

    label_3d = net_output.reshape((h, w, class_num))
    label_2d = np.zeros((h, w), dtype=np.uint8)

    for i in range(h):
        for j in range(w):
            label_2d[i, j] = np.where(label_3d[i, j, :] == 1)[0][0]

    return label_2d


def predict_2_labelmat_new(predict_softmax, h, w):
    """ transform 3d predict output matrix into 2d mat
    Note:
        the func accepts 3d predict matrix which after the softmax preocess,
        so if the net['output'] donot contain the softmax, pls make sure you have place the softmax precess
        in the deploy main file (deploy.py) before call this func.
    :arg
        predict_softax: a tensor, with shape(h, w, class_num), and the value is float type from 0 to 1
    :return
        return a tensor, with shape (h, w), and the value are the class ID.the type is same to input
    """

    label_2d = np.zeros((h, w), dtype=np.uint8)

    for i in range(h):
        for j in range(w):
            label_2d[i, j] = np.argmax(predict_softmax[i, j, :], axis=0)

    return label_2d


def labelmat_2_rgb(labelmat_):
    """ transform 2d label matrix to a 3D rgb image
    Note:
        the order of colormap element is [R_value,  G_value, B_value]
        so the output image is RGB image, not BGR image

        for different dataset, the only thing need to change is the  colormap list
    :param
        labelmat_: a 2D array, whose value donates the label ID

    :return:
        a rgb image with shape (h, w, 3)
    """

    VOC_COLORMAP = [[0, 0, 0], [128, 0, 0], [0, 128, 0], [128, 128, 0],
                    [0, 0, 128], [128, 0, 128], [0, 128, 128], [128, 128, 128],
                    [64, 0, 0], [192, 0, 0], [64, 128, 0], [192, 128, 0],
                    [64, 0, 128], [192, 0, 128], [64, 128, 128], [192, 128, 128],
                    [0, 64, 0], [128, 64, 0], [0, 192, 0], [128, 192, 0],
                    [0, 64, 128]]

    colormap = VOC_COLORMAP

    high, width = labelmat_.shape

    rgb = np.zeros((high, width, 3), dtype=np.uint8)

    for i in range(high):
        for j in range(width):
            rgb[i, j, :] = colormap[labelmat_[i, j]]
    return rgb
