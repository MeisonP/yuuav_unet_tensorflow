# coding: utf-8
"""module, deploy the pre-trained model, and show the predict performance.

2018/12/04
tensorflow ==1.11
python ==2.7.15

Note:
    threre mainly contain following parts:
    1. create the network form .mate file
    2. load the parameters
    3. predict
    4. visualization

    there are two way to save and load trained model, here we using frozen model,
    as the checkpoint are to complex and slow for predict.

    !!! the output of network is (BS*h*w, num_class) with the value float 0 to 1 such as 0.01
    !!! a distribution that do not pass through softmax

"""
import time
import logging

import tensorflow as tf
from visualization import *
import cv2
import numpy as np
from tensorflow.python.platform import gfile


def ckpt_single_image_predictor(img_, meta_, trained_model_path_, h_, w_, class_num_):
    """ restore the variable of model from checkpoint, and predict.
    Note:
        tf.get_collection() will return a list; to get the variable, using tf.get_collection("name")[0]
    :arg
        img_: a rgb image that wait to predict
        meta_: the .meta files from checkpoint, and the graph are stored in .meta file
        trained_model_path_: the checkpoint path
        h_, w_: the image shape of net input, (BS, h_, w_, class_num)
        class_num: length of a single distribute vector

    :return:
        return a visualized rgb image
    """

    img_ = cv2.resize(img_, (h_, w_), interpolation=cv2.INTER_LINEAR)
    img_fd = np.zeros((8, h_, w_, 3), dtype=np.uint8)
    for i in range(8):
        img_fd[i, :, :] = img_
    print img_fd.shape

    with tf.Session() as sess:
        saver = tf.train.import_meta_graph(meta_)
        saver.restore(sess, tf.train.latest_checkpoint(trained_model_path_))

        graph = tf.get_default_graph()

        # op_to_restore = graph.get_tensor_by_name("predict:0")
        op_to_restore = tf.get_collection("predict")[0]

        # print (sess.run(op_to_restore))

        # input = graph.get_tensor_by_name("image_batch:0")
        input_ = tf.get_collection("input")[0]

        # print(sess.run(graph.get_tensor_by_name('conv1_2/weights:0')))

        logging.info("predict ...")

        predict = sess.run(op_to_restore, feed_dict={input_: img_fd})

        predict = predict.reshape((8, h_, w_, class_num_))

        single_ = predict[1, :, :, :]

        single_ = single_.reshape(-1, class_num_)

        logging.info("visualization ...")
        mat_2d = netoutput_2_labelmat(single_, h_, w_, class_num_)

        rgb_image_ = labelmat_2_rgb(mat_2d)

        return rgb_image_


def frozen_predictor(pd_file_path_, single_img, h_, w_, class_num_):
    """ predictor single image, using trained frozen model file.
    Note:
        the input shape of network during triain is (batch_size, h, w, channel),
        while, the single image is (h, w, channel), so using img_mat= np.expand_dims(single_image, axis=0)
        to match the network input.

    :arg
        img_: a rgb image that wait to predict
        meta_: the .meta files from checkpoint, and the graph are stored in .meta file
        pd_file_path_: : the frozen model file model.pd path
        h_, w_: the image shape of net input, (BS, h_, w_, class_num)
        class_num: length of a single distribute vector

    :return:
        a rgb image with shape (h_, w_, 3)
    """

    single_img = cv2.resize(single_img, (h_, w_), interpolation=cv2.INTER_LINEAR)

    img_fd = np.zeros((8, h_, w_, 3), dtype=np.uint8)
    for i in range(8):
        img_fd[i, :, :] = single_img

    with tf.Session() as sess:
        with gfile.FastGFile(pd_file_path_ + 'model.pb', 'rb') as f:
            graph_def = tf.GraphDef()
            graph_def.ParseFromString(f.read())
            sess.graph.as_default()
            tf.import_graph_def(graph_def, name='')

        sess.run(tf.global_variables_initializer())

        image_tensor = sess.graph.get_tensor_by_name('source_input/image_batch/image_tensor:0')

        logging.info("{}".format(image_tensor))

        op = sess.graph.get_tensor_by_name('predict/predict:0')
        logging.info("{}".format(op))

        logging.info("predict ...")
        predict = sess.run(op, feed_dict={image_tensor: img_fd})

        #  just for tmp
        predict = predict.reshape((8, h_, w_, class_num_))
        predict = predict[1, :, :, :]
        # predict = predict.reshape(-1, class_num_)

        logging.info("{}".format(predict.shape))

        logging.info("visualization ...")
        mat_2d = predict_2_labelmat_new(predict, h_, w_)

        rgb_image_ = labelmat_2_rgb(mat_2d)

        return rgb_image_


if __name__ == "__main__":

    TM = time.strftime("%Y:%m:%d-%H:%M", time.localtime())
    LOG_FORMAT = "%(asctime)s-%(levelname)s-[line:%(lineno)d] - %(message)s"
    logging.basicConfig(level=logging.INFO, format=LOG_FORMAT)
    logging.info("**********************mason_p nn_design(%s)***********************" % TM)

    pd_file_path = "./final_model/"

    h, w = 256, 256
    class_num = 21

    img = cv2.imread('test.jpg')

    rgb_image = frozen_predictor(pd_file_path, img, h, w, class_num)

    cv2.imwrite('predict.png', rgb_image)
