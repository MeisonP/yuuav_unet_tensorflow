# coding: utf-8
""" module prepared for deploy.py ;  visualizing the network output,

2018/12/04
python ==2.7.15

Note:
    the net['output'] is a tensor with shape (h*w, num_classes)

    mainly, contain two step:
    1. from net['output'] --> label_2d format matrix
    2. label_2d matrix --> 3channel bgr_image using colormap

"""

import numpy as np


def netoutput_2_labelmat(net_output, h, w, class_num):
    """reshape network's output, and transform to a 2d label matrix
    :param
        (h, w): inter, the size of image that need to be visualized
        net_output: a tensor, the output of the network
        class_num: inter, the numbers of class that need be classified
    :return:
        a 2D array, whose value donates the label ID
    """

    label_3d = net_output.reshape((h, w, class_num))
    label_2d = np.zeros((h, w), dtype=np.uint8)

    for i in range(h):
        for j in range(w):
            label_2d[i, j] = np.where(label_3d[i, j, :] == 1)[0][0]

    return label_2d


def labelmat_2_rgb(labelmat_):
    """ transform 2d label matrix to a 3D rgb image
    Note:
        the order of colormap element is [R_value,  G_value, B_value]
        so the output image is RGB image, not BGR image
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
