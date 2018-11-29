# coding:utf-8
"""main interface using for train a unet model (using tensorflow and opencv).
the purpose of this project is to do the satellite-image segmentation. yuuav building

#2018/11/19
#python==2.7.15
#tensorflow==1.11
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
    2. command line:
        $ python dataset_gen.py -p "path to the tfrecords file"
    3. modify the config.py parameters:
       lr, path, dataset_size, image_size, batch_size, class_num, filters ....
    4. then run command line:
        $ python train_main.py -m "final model save path"




Note: the main step as follow
    --batch input
    --loss and acc
    --train_op
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


"""


import tensorflow as tf
from config import *
from unet import unet
from get_batch import batch_input
import argparse


def total_loss(net_output, label, phase):
    """ loss calculate,
    the  shape of the inout label and logist(the network output) must be same.
    Note:
        tf.nn.softmax_cross_entropy_with_logits output a tensor/ a array,
        its shape is the same as `labels` except that
        it does not have the last dimension of `labels`.

    :arg
        net_output: a tensor, output after the src image pass through the network
        label: a tensor, shape is same as net_output (x, 3) 3 means RGB channel
    :return
        a scalar,


    """
    cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels=label, logits=net_output)
    segment_loss = tf.reduce_mean(cross_entropy)
    # tf.summary.scalar("{}_loss".format(phase), segment_loss)
    return segment_loss


def accuracy(net_output, label, phase):
    """ loss calculate,
        the  shape of the inout label and logist(the network output) must be same.
        :arg
            net_output: a tensor, output after the src image pass through the network
            label: a tensor, shape is same as net_output (x, 3) 3 means RGB channel
            phase: string, 'train' or 'val'
        :return
            return a float scalar

        """
    labels = tf.reshape(tf.argmax(label, axis=1), [-1, 1])
    predicted_annots = tf.reshape(tf.argmax(net_output, axis=1), [-1, 1])

    correct_predictions = tf.equal(predicted_annots, labels)

    seg_acc = tf.reduce_mean(tf.cast(correct_predictions, tf.float32))
    # tf.summary.scalar("{}_acc".format(phase), seg_acc)
    return seg_acc


def train(sess, train_op, loss_train, acc_train):
    """ def train_op and run, then add it into the summary
    :arg
        sess: the current tf.Session() Object
        loss_train: A tensor, op to run
        acc_train: A tensor, op to run
        writer_val:  A tf.summary.FileWriter Object, which using for add_summary

    :return:
        loss_train: A tensor, op to run
        acc_train: A tensor, op to run


    """

    loss_train, acc_train, _ = sess.run([loss_train, acc_train, train_op])
    # merged = tf.summary.merge(loss_train, acc_train)
    # summary_train = sess.run(merged)
    # writer_train.add_summary(summary_train)
    return loss_train, acc_train


def main(_):
    """ main func for train

    :return:
    """
    # with tf.Session(config=tf.ConfigProto(log_device_placement=True)) as sess:
    with tf.Session() as sess:

        name_batch, image_batch, label_batch = batch_input(tfrecord_path_train)

        label_batch_ = tf.reshape(label_batch, (-1, 3))

        model_train = unet(image_batch, 'train')

        loss_train = total_loss(model_train['output'], label_batch_, 'train')
        acc_train = accuracy(model_train['output'], label_batch_, 'train')

        optimizer = tf.train.AdamOptimizer(lr)
        train_op = optimizer.minimize(loss_train)

        logging.info('variable initialization')

        sess.run(tf.local_variables_initializer())
        sess.run(tf.global_variables_initializer())

        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord)
        try:
            while not coord.should_stop():
                logging.info('sess run for image pass through the network, please waite...')
                pc_bar = ShowProcess(iter_each_epoch, '')

                for epoch_i in range(epochs):
                    print ('Epoch {}'.format(epoch_i) + '/{}'.format(epochs))
                    for j in range(1, iter_each_epoch + 1):
                        loss_train_, acc_train_ = train(sess, train_op, loss_train, acc_train)

                        pc_bar.show_process(j, iter_each_epoch, loss_train_, acc_train_)

                coord.request_stop()
        except tf.errors.OutOfRangeError:
            print '\ndone! limit epochs achieved.'
        finally:
            coord.request_stop()
            coord.join(threads)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_save_path', '-m',
                        help="final_model_path",
                        required=True, default='./final_model/')

    FLAGS, _ = parser.parse_known_args()

    tf.app.run()
    logging.info("done! ")