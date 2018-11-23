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
    3. command line:
        $ python train_main.py -m "final model save path"




Note: the main step as follow
    --batch input
    --loss and acc
    --train_op
    --init
    --sess.run

    ***in order to increase the efficiency, the batch queue creating should be a individual thread***
    not forget the tf.local_variables_initializer() at init step\


    the evaluate happen at the end of each epoch, eg 100 epoch,
    1-99 are do the train ( pass the train data batch and do back-propagation/ cal gradient),
    and the 100th do the evaluate (pass the val data, and don't do the bp)

"""


import tensorflow as tf
from config import *
from unet import unet
from get_batch import batch_input
import argparse


def total_loss(net_output, label, phase):
    cross_entropy = tf.nn.softmax_cross_entropy_with_logits_v2(labels=label, logits=net_output)
    segment_loss = cross_entropy
    tf.summary.scalar("{}_loss".format(phase), segment_loss)
    return segment_loss


def accuracy(net_output, label, phase):
    labels = tf.reshape(tf.argmax(label, axis=1), [-1, 1])
    predicted_annots = tf.reshape(tf.argmax(net_output, axis=1), [-1, 1])

    correct_predictions = tf.equal(predicted_annots, labels)

    seg_acc = tf.reduce_mean(tf.cast(correct_predictions, tf.float32))
    tf.summary.scalar("{}_acc".format(phase), seg_acc)
    return seg_acc


def evaluate(sess, loss_val, acc_val, writer_val):
    """
    :arg
        sess: the current tf.Session() Object
        loss_val: A tensor, op to run
        acc_val: A tensor, op to run
        writer_val:  A tf.summary.FileWriter Object, which using for add_summary

    :return:
        loss_val: A tensor, op to run
        acc_val: A tensor, op to run

    """

    loss_val, acc_val = sess.run([loss_val, acc_val])
    merged = tf.summary.merge(loss_val, acc_val)
    summary_val = sess.run(merged)
    writer_val.add_summary(summary_val)
    return loss_val, acc_val


def train(sess, loss_train, acc_train, writer_train):
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

    optimizer = tf.train.AdamOptimizer(lr)
    train_op = optimizer.minimize(loss_train)

    sess.run(train_op)
    loss_train, acc_train = sess.run([loss_train, acc_train])
    merged = tf.summary.merge(loss_train, acc_train)
    summary_train = sess.run(merged)
    writer_train.add_summary(summary_train)
    return loss_train, acc_train


def main(_):
    """ main func for train

    :return:
    """

    name_batch, image_batch, label_batch = batch_input(tfrecord_path_train)
    name_val, image_val, label_val = batch_input(tfrecord_path_val, batch_size=64)

    model_train = unet(input_=image_batch)
    model_val = unet(input_=image_val)

    label_batch_ = tf.reshape(label_batch, (-1, class_num))
    label_val_ = tf.reshape(label_val, (-1, class_num))

    loss_train = total_loss(model_train['output'], label_batch_, 'train')
    loss_val = total_loss(model_val['output'], label_val_, 'val')

    acc_train = accuracy(model_train['output'], label_batch_, 'train')
    acc_val = accuracy(model_val(['output'], label_val, 'val'))

    writer_train = tf.summary.FileWriter(path_checker(summary_path+"train"))
    writer_val = tf.summary.FileWriter(path_checker(summary_path + "val"))
    saver = tf.train.Saver()

    with tf.Session() as sess:

        logging.info("initialization ...")
        sess.run([tf.local_variables_initializer(), tf.global_variables_initializer()])

        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess, coord)

        logging.info("run the Session ...")

        try:
            while not coord.should_stop():
                loss_val, acc_val = None, None

                for epoch_i in range(epochs):
                    pc_bar = ShowProcess(iter_each_epoch)
                    for i in range(1, iter_each_epoch+1):   # i =1 ....iter_each_epoch

                        if i != iter_each_epoch:
                            loss_train, acc_train = train(sess, loss_train, acc_train, writer_train)
                        else:
                            loss_val, acc_val = evaluate(sess, loss_val, acc_val, writer_val)

                        pc_bar.show_process(i, epoch_i, iter_each_epoch,
                                            loss_train, acc_train,
                                            loss_val, acc_val)

                coord.request_stop()

        except tf.errors.OutOfRangeError:
            logging.info("done! user ask to stop coord-threads")

        finally:
            coord.request_stop()
            logging.info('all threads are asked to stop!')
            coord.join(threads)

        writer_train.close()
        writer_val.close()
        saver.save(sess, FLAGS.model_save_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_save_path', '-m',
                        help="final_model_path",
                        required=True, default='./final_model/')
    FLAGS, _ = parser.parse_known_args()

    tf.aap.run()
