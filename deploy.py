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

"""

import tensorflow as tf
from visualization import *
import cv2


def single_image_predictor(img_, meta_, trained_model_path_, h_, w_, class_num_):
    """
    Note:
        tf.get_collection() will return a list; to get the variable, using tf.get_collection("name")[0]
    :arg

    :return:
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

        print "begin to run ..."

        predict = sess.run(op_to_restore, feed_dict={input_: img_fd})

        predict = predict.reshape((8, h_, w_, class_num))

        single_ = predict[1, :, :, :]

        print "visualization"
        mat_2d = netoutput_2_labelmat(single_, h_, w_, class_num_)

        rgb_image_ = labelmat_2_rgb(mat_2d)

        return rgb_image_


if __name__ == "__main__":

    meta = "model/my_model-320.meta"
    trained_model_path = "model/"
    h = 256
    w = 256
    class_num = 21

    img = cv2.imread("test.jpg")

    rgb_image = single_image_predictor(img, meta, trained_model_path, h, w, class_num)

    cv2.imwrite('predict_single.png', rgb_image)

