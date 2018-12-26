# ===================================================================================== #
# coding:utf-8
"""main interface using for train a unet model (using tensorflow and opencv).
the purpose of this project is to do the satellite-image segmentation. yuuav building

2018/12/05
tensorflow ==1.11 (cpu)  and tensorflow-gpu (1.4.1)
python ==2.7.15
#opencv==3.4.2


project file structure:
    --train_main.py
    --config.py: all the parameters define
    --unet.py : unet network structure define
    --get_batch.py : generate the batch for network feeding
    --dateset_gen.py : generate tfrecords file using image data set

How to train (step):
    1. prepared data:
        --data
            --train
                --src
                --label
            --val
                --src
                --label
    2.




Note: the main step as follow
    --batch input
    --loss and acc
    --train_op
    --checkpoint restore if needed // save model to .pd format
    --init
    --sess.run

    ***in order to increase the efficiency, the batch queue creating should be a individual thread***
    not forget the tf.local_variables_initializer() at init step\


    the evaluate happen at the end of each epoch, eg 100 batch one epoch,
    1-99 are do the train ( pass the train data batch and do back-propagation/ cal gradient),
    and the 100th do the evaluate (pass the val data, and don't do the bp)
    optimizer:
    --GradientDescentOptimizer
    --AdagradOptimizer
    --AdagradDAOptimizer
    --MomentumOptimizer
    --AdamOptimizer

    tf_debug module is also be contained (only for dev environment debug ).
    localhost:8080 is changeable .only need to wrap the sess like:
    from tensorflow.python import debug as tf_debug
    sess = tf_debug.TensorBoardDebugWrapperSession(sess,"localhost:8080")
"""
# ===================================================================================== #


import tensorflow as tf
from config import *
from unet import unet
from get_batch import batch_input
import argparse
from tensorflow.python.framework import graph_util
from tensorflow.python import debug as tf_debug

import numpy as np
import cv2


def total_loss(net_output, label):
    """ loss calculate,
    the  shape of the inout label and logist(the network output) must be same.
    Note:
        tf.nn.softmax_cross_entropy_with_logits output a tensor/ a array,
        its shape is the same as `labels` except that
        it does not have the last dimension of `labels`.

        in some case, we can also add class_weights to particular class,
         if class distribution is not fair

         in l2_loss*0.001, the 0.001 is the weight_decay_factor on L2 regularization

    :arg
        net_output: a tensor, output after the src image pass through the network
        label: a tensor, shape is same as net_output (x, 3) 3 means RGB channel
    :return
        a scalar,

    """
    with tf.name_scope("loss"):
        with tf.name_scope("softmax_cross_entropy"):
            cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels=label, logits=net_output)
            segment_loss = tf.reduce_mean(cross_entropy)
            # loss_summary = tf.summary.scalar("{}_acc".format(phase), segment_loss)

        with tf.name_scope("l2_loss"):
            weights = [var for var in tf.trainable_variables() if var.name.endswith('weights:0')]
            l2_loss = tf.add_n([tf.nn.l2_loss(w) for w in weights])

        loss = segment_loss + l2_loss*0.001
    return loss


def accuracy(netout_softmax, label):
    """ loss calculate,
        the  shape of the inout label and logist(the network output) must be same.
        :arg
            net_output: a tensor, output after the src image pass through the network, and softmax process
            label: a tensor, shape is same as net_output (x, 3) 3 means RGB channel
            phase: string, 'train' or 'val'
        :return
            return a float scalar

        """

    labels = tf.reshape(tf.argmax(label, axis=1), [-1, 1])

    predicted_annots = tf.reshape(tf.argmax(netout_softmax, axis=1), [-1, 1])

    correct_predictions = tf.equal(predicted_annots, labels)
    # predicted_annots = tf.cast(predicted_annots, tf.float32)
    # correct_predictions = tf.nn.in_top_k(predicted_annots, labels, 1)

    seg_acc = tf.reduce_mean(tf.cast(correct_predictions, tf.float32))
    # acc_summary = tf.summary.scalar("{}_acc".format(phase), seg_acc)
    return seg_acc


def debug_main(_):
    """ main func for train
    Note:
        the checkpoint only save 20%, 40%, 60%, 80%, 100% step

        # with tf.Session(config=tf.ConfigProto(log_device_placement=True)) as sess:


        in order to convert_variables_to_constants image_tensor,
        so the out-space have to use tf.variable_scope, not tf.name_scope()

        in order to  track the cal time and memory consumption of each op during sess.run(),
        add tf.RunOptions and tf.RunMetadata to sess.run

        # the label in the tfrecords queue are with shape (h, w, class_num), so need to be
        transform to (h*w, class_num) for cal loss

    :return:
    """

    with tf.Session() as sess:

        if FLAGS.debug:
            sess = tf_debug.LocalCLIDebugWrapperSession(sess)
            # sess = tf_debug.TensorBoardDebugWrapperSession(sess, 'http://0.0.0.0:6006',
            # send_traceback_and_source_code=False)

        with tf.variable_scope("source_input"):

            name_batch, image_batch, label_batch = batch_input(tfrecord_path_train)

            with tf.variable_scope("image_batch"):
                image_tensor_batch = tf.identity(image_batch, name='image_tensor')

            with tf.name_scope("label_batch"):
                label_batch_ = tf.reshape(label_batch, (BS*image_size*image_size, num_classes))

        model_train = unet(image_tensor_batch)

        with tf.variable_scope("predict"):
            predict = tf.nn.softmax(model_train['output'], name="predict")

        with tf.name_scope("loss"):
            loss_train = total_loss(model_train['output'], label_batch_)
            loss_summary = tf.summary.scalar("train_loss", loss_train)

        with tf.name_scope("acc"):
            acc_train = accuracy(model_train['output'], label_batch_)
            acc_summary = tf.summary.scalar("train_acc", acc_train)

        with tf.name_scope('optimizer'):
            optimizer = tf.train.GradientDescentOptimizer(lr)
            # optimizer = tf.train.AdamOptimizer(lr)
            train_op = optimizer.minimize(loss_train)

        merged = tf.summary.merge([loss_summary, acc_summary])
        logging.info('variable initialization ...')

        init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())

        sess.run(init_op)

        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord)

        logging.info("saving sess.graph ...")
        writer_train = tf.summary.FileWriter(path_checker(summary_path + "train"), sess.graph)

        # using for deploy # has a indicate sess.run ?
        constant_graph = graph_util.convert_variables_to_constants(sess, sess.graph_def,
                                                                   ["predict/predict",
                                                                    "source_input/image_batch/image_tensor"])

        for i in range(100):
            names_, images_, labels_ = sess.run([name_batch, image_batch, label_batch])
            print names_[0]
            label_in = labels_[0]

            label_2d = np.zeros((256, 256), dtype=np.uint8)
            for i_ in range(256):
                for j_ in range(256):
                    label_2d[i_, j_] = np.argmax(label_in[i_, j_, :], axis=0)

            VOC_COLORMAP = [[0, 0, 0], [128, 0, 0], [0, 128, 0], [128, 128, 0],
                            [0, 0, 128], [128, 0, 128], [0, 128, 128], [128, 128, 128],
                            [64, 0, 0], [192, 0, 0], [64, 128, 0], [192, 128, 0],
                            [64, 0, 128], [192, 0, 128], [64, 128, 128], [192, 128, 128],
                            [0, 64, 0], [128, 64, 0], [0, 192, 0], [128, 192, 0],
                            [0, 64, 128]]

            rgb = np.zeros((256, 256, 3), dtype=np.uint8)
            for ii in range(256):
                for jj in range(256):
                    rgb[ii, jj, :] = VOC_COLORMAP[label_2d[ii, jj]]

            rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)

            cv2.imwrite('net_label.png', rgb)

            net_output = sess.run(model_train['output'])
            output_ = sess.run(tf.nn.softmax(net_output))

            output_1 = output_.reshape((BS, 256, 256, 21))
            print output_1.shape
            output_2 = output_1 [0, :, :, :]
            print output_2.shape
            print output_2[120, 120, :]


            label_in = output_2
            label_2d = np.zeros((256, 256), dtype=np.uint8)
            for i_ in range(256):
                for j_ in range(256):
                    label_2d[i_, j_] = np.argmax(label_in[i_, j_, :], axis=0)

            print label_2d[120, 120]

            VOC_COLORMAP = [[0, 0, 0], [128, 0, 0], [0, 128, 0], [128, 128, 0],
                            [0, 0, 128], [128, 0, 128], [0, 128, 128], [128, 128, 128],
                            [64, 0, 0], [192, 0, 0], [64, 128, 0], [192, 128, 0],
                            [64, 0, 128], [192, 0, 128], [64, 128, 128], [192, 128, 128],
                            [0, 64, 0], [128, 64, 0], [0, 192, 0], [128, 192, 0],
                            [0, 64, 128]]

            rgb = np.zeros((256, 256, 3), dtype=np.uint8)
            for ii in range(256):
                for jj in range(256):
                    rgb[ii, jj, :] = VOC_COLORMAP[label_2d[ii, jj]]

            rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)
            cv2.imwrite('net_out.png', rgb)

            print 'acc:', sess.run(acc_train)
            print 'train...'
            sess.run(train_op)
            print 'next train'


def main(_):
    """ main func for train
    Note:
        the checkpoint only save 20%, 40%, 60%, 80%, 100% step

        # with tf.Session(config=tf.ConfigProto(log_device_placement=True)) as sess:


        in order to convert_variables_to_constants image_tensor,
        so the out-space have to use tf.variable_scope, not tf.name_scope()

        in order to  track the cal time and memory consumption of each op during sess.run(),
        add tf.RunOptions and tf.RunMetadata to sess.run

        # the label in the tfrecords queue are with shape (h, w, class_num), so need to be
        transform to (h*w, class_num) for cal loss

    :return:
    """

    with tf.Session() as sess:

        if FLAGS.debug:
            sess = tf_debug.LocalCLIDebugWrapperSession(sess)
            # sess = tf_debug.TensorBoardDebugWrapperSession(sess, 'http://0.0.0.0:6006',
            # send_traceback_and_source_code=False)

        with tf.variable_scope("source_input"):

            name_batch, image_batch, label_batch = batch_input(tfrecord_path_train)

            with tf.variable_scope("image_batch"):
                image_tensor_batch = tf.identity(image_batch, name='image_tensor')

            with tf.name_scope("label_batch"):
                label_batch_ = tf.reshape(label_batch, (BS*image_size*image_size, num_classes))

        model_train = unet(image_tensor_batch)

        with tf.variable_scope("predict"):
            predict_softmax = tf.nn.softmax(model_train['output'], name="predict")

        with tf.name_scope("loss"):
            loss_train = total_loss(model_train['output'], label_batch_)
            loss_summary = tf.summary.scalar("train_loss", loss_train)

        with tf.name_scope("acc"):
            acc_train = accuracy(predict_softmax, label_batch_)
            acc_summary = tf.summary.scalar("train_acc", acc_train)

        with tf.name_scope('optimizer'):
            optimizer = tf.train.GradientDescentOptimizer(lr)
            # optimizer = tf.train.AdamOptimizer(lr)
            train_op = optimizer.minimize(loss_train)

        merged = tf.summary.merge([loss_summary, acc_summary])
        logging.info('variable initialization ...')

        init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())

        sess.run(init_op)

        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord)

        logging.info("saving sess.graph ...")
        writer_train = tf.summary.FileWriter(path_checker(summary_path + "train"), sess.graph)

        # using for deploy # has a indicate sess.run ?
        constant_graph = graph_util.convert_variables_to_constants(sess, sess.graph_def,
                                                                   ["predict/predict",
                                                                    "source_input/image_batch/image_tensor"])

        try:
            while not coord.should_stop():
                logging.info('sess run for image pass through the network, please waite...')
                pc_bar = ShowProcess(iter_each_epoch, '')

                for epoch_i in range(1, epochs+1):
                    print ('Epoch {}'.format(epoch_i) + '/{}'.format(epochs))

                    for j in range(1, iter_each_epoch + 1):

                        run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
                        run_metadata = tf.RunMetadata()

                        loss_train_, acc_train_,\
                        summary_train, _ = sess.run([loss_train, acc_train,
                                                     merged, train_op],
                                                    options=run_options, run_metadata=run_metadata)

                        step_ = (epoch_i - 1) * iter_each_epoch + j
                        writer_train.add_summary(summary_train, global_step=step_)
                        writer_train.add_run_metadata(run_metadata=run_metadata,
                                                      tag=("tag%d" % step_), global_step=step_)

                        pc_bar.show_process(j, iter_each_epoch, loss_train_, acc_train_)

                coord.request_stop()

        except tf.errors.OutOfRangeError:
            print '\ndone! string queue is empty,limit epochs achieved.'

        finally:

            logging.info('store the model to pd frozen file...')
            with tf.gfile.FastGFile(FLAGS.model_save_path + 'model.pb', mode='wb') as f:  # save the final model
                f.write(constant_graph.SerializeToString())

            logging.info("train completed!")
            writer_train.close()

            logging.info('close all threads and stop !')
            coord.request_stop()
            coord.join(threads)     # wait until coord finished , and then go to next step.


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_save_path', '-m',
                        help="final_model_path",
                        required=True, default='./final_model/')
    parser.add_argument('--debug',
                        help="debug model", default=False)
    FLAGS, _ = parser.parse_known_args()

    logging.info('****FLAGES****\n--model_save_path:{}'
                 '\n--debug{}\n****FLAGES****'.format(FLAGS.model_save_path,
                                                         FLAGS.debug))

    tf.app.run()
    # debug_main()

