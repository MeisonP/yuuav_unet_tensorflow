# coding:utf-8
# masom: Unet-tensorflow version
import tensorflow as tf
from config import *
import layers
from net import net
from load_data import batch_queue


def main():
    # input
    # this should op by cpu
    image_batch, label_batch = batch_queue(phase="train")

    with tf.device(device), tf.Session() as sess:
        tf.train.start_queue_runners(sess=sess)

        #



        #
        # network

        model = net(input_=image_batch)

        #
        # loss func, related between the output image and label image
        # total loss = segment_loss + l2_regularization
        label_batch_ = tf.reshape(label_batch, (-1, class_num))

        logging.info("input: {0}  {1}".format(label_batch.shape, image_batch.shape))
        logging.info("output: {0}  {1}".format(label_batch_.shape, model['output'].shape))

        loss_func = layers.segment_loss(labels_=label_batch_,  net_output=model['output'])  # + layers.l2_regular()
        # tf.summary.scalar("loss", loss_func)

        #
        # acc and summary writer

        acc = layers.acc(model['output'], label_batch_)
        tf.summary.scalar("segment_acc", acc)
        writer_train = tf.summary.FileWriter(path_checker(summary_path+"train"))

        writer_test = tf.summary.FileWriter(path_checker(summary_path + "test"))

        #
        # optimization

        optimizer = tf.train.AdamOptimizer(lr)
        train_op = optimizer.minimize(loss_func)

        #
        # init
        logging.info("initialization...")
        sess.run(tf.global_variables_initializer())

        #
        # sess.run, contain test and train
        logging.info("run the Session begin")
        for i in range(0, iter_max):
            if (i+1) % display == 0:  # display at iter
                loss, acc = sess.run([loss_func, acc])
                logging.info("loss:{}\tacc:{}".format(loss_func, acc))

                merged = tf.summary.merge_all()
                summary_test = sess.run(merged)
                writer_test.add_summary(summary_test)

            else:
                sess.run(train_op)
                #   summary
                merged = tf.summary.merge_all()
                summary_train = sess.run(merged)
                writer_train.add_summary(summary_train)

        writer_train.close()
        writer_test.close()


if __name__ == "__main__":
    main()
