# coding:utf-8
"""main interface using for train a unet model (using tensorflow and opencv).
the purpose of this project is to do the satellite-image segmentation. yuuav building

#2018/11/19
#python==2.7.15
#tensorflow==1.11
#opencv==3.4.2

Note: the main step as follow
    --batch input
    --loss and acc
    --train_op
    --init
    --sess.run

    ***in order to increase the efficiency, the batch queue creating should be a individual thread***
    not forget the tf.local_variables_initializer() at init step

"""


import tensorflow as tf
from config import *
from unet import unet
from get_batch import batch_input



def total_loss(net_output, label):
    cross_entropy = tf.nn.softmax_cross_entropy_with_logits_v2(labels=label, logits=net_output)
    segment_loss = cross_entropy
    return segment_loss


def accuracy(net_output, label):
    labels = tf.reshape(tf.argmax(label, axis=1), [-1, 1])
    predicted_annots = tf.reshape(tf.argmax(net_output, axis=1), [-1, 1])

    correct_predictions = tf.equal(predicted_annots, labels)

    seg_acc = tf.reduce_mean(tf.cast(correct_predictions, tf.float32))
    return seg_acc


def main():
    with tf.Session() as sess:
        name_batch, image_batch, label_batch = batch_input(tfrecord_path_list)

        model = unet(input_=image_batch)
        label_batch_ = tf.reshape(label_batch, (-1, class_num))

        loss_ = total_loss(model['output'], label_batch_)
        tf.summary.scalar("segment_loss", loss_)
        acc = accuracy(model['output'], label_batch_)
        tf.summary.scalar("segment_acc", acc)

        writer_train = tf.summary.FileWriter(path_checker(summary_path+"train"))
        writer_test = tf.summary.FileWriter(path_checker(summary_path + "test"))

        optimizer = tf.train.AdamOptimizer(lr)
        train_op = optimizer.minimize(loss_)

        logging.info("initialization ...")
        sess.run(tf.local_variables_initializer())

        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess, coord)

        sess.run(tf.global_variables_initializer())

        logging.info("run the Session ...")

        try:
            while not coord.should_stop():
                for i in range(0, iter_max):
                    if (i + 1) % display == 0:

                        loss, acc = sess.run([loss_, acc])
                        logging.info("loss:{}\tacc:{}".format(loss_, acc))

                        merged = tf.summary.merge_all()
                        summary_test = sess.run(merged)
                        writer_test.add_summary(summary_test)

                    else:
                        sess.run(train_op)
                        merged = tf.summary.merge_all()
                        summary_train = sess.run(merged)
                        writer_train.add_summary(summary_train)
                coord.request_stop()

        except tf.errors.OutOfRangeError:
            logging.info("done! user ask to stop coord-threads")

        finally:
            coord.request_stop()
            logging.info('all threads are asked to stop!')
            coord.join(threads)

            writer_train.close()
            writer_test.close()

if __name__ == "__main__":
    main()
