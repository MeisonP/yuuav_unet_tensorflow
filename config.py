# coding:utf-8
"""the config module of the project: training a unet model.

#2018/11/19

include:
    process bar class
    logging config
    wrap
    parameters


"""


import time
import logging
import os

import sys
TM = time.strftime("%Y:%m:%d-%H:%M", time.localtime())
#   LOG_FORMAT="%(asctime)s - %(levelname)s - [%(filename)s,line:%(lineno)d] - %(message)s"
LOG_FORMAT="%(asctime)s-%(levelname)s-[line:%(lineno)d] - %(message)s"
logging.basicConfig(level=logging.INFO, format=LOG_FORMAT)
logging.info("**********************mason_p nn_design(%s)***********************" % TM)


class ShowProcess():
    """class for process bar.
    Note:
        how to use, eg:
        process_bar = ShowProcess(1000, 'OK')
        for i in range(max_steps):
            process_bar.show_process()
            time.sleep(0.01)

        the result should be like :
        [>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>]100.00%
        "info when done..."
    """

    def __init__(self, max_steps, info_done='Done'):
        """class param init.
        :param max_steps: inter, how many step for process
            max_arrow: inter, the total num arrow
            i: inter, the step right now, init set to 0
        :param infoDone: String, def info when finishing process
        """
        self.max_steps = max_steps
        self.max_arrow = 50
        self.i = 0
        self.info_done = info_done

    def close(self):
        """
        print info, when finished
        """
        print('')
        print(self.info_done)
        self.i = 0

    def show_process(self, i, epoch_images,
                     loss_, acc_):
        """ core method, show the process bar.
        Note:
            1. calculate how many >
            2. calculate how may -
            3. calculate the percentage: "\r" must be at left ,means that begin form left
            4. sys.stdout.write(process_bar) and sys.stdout.flush() is to print to terminal
        :param
            i: inter, the step right now, init set to 0, total step== num of images for each epoch
            epoch_i: inter, show the current processed epoch
            epoch_images: inter, the num of total images for each epoch
            loss_train, acc_train: loss and acc for current calculation batch
            loss_vel, acc_vel: test at validate dataset, test at given iter, eg: each 1000 batch iter
        :return: no return

        """

        if i is not None:
            self.i = i
        else:
            self.i += 1

        num_arrow = int(self.i * self.max_arrow / self.max_steps)
        num_line = self.max_arrow - num_arrow
        percent = self.i * 100.0 / self.max_steps

        process_bar_ = '\r'\
                       + '{0}/{1}'.format(i, epoch_images)\
                       + '[' + '>' * num_arrow + ' ' * num_line + ']' \
                       + ' - loss:{:.2f}\t'.format(float(loss_)) \
                       + ' - acc:{:.2f}'.format(float(acc_))

        sys.stdout.write(process_bar_)
        sys.stdout.flush()
        if self.i >= self.max_steps:
            self.close()


def func_track(func):
    def track(*args, **kwargs):
        name = func.__name__
        logging.info("func %s..."%name)
        result = func(*args, **kwargs)
        logging.info("...func %s out"%name)
        return result
    return track


def path_checker(path):
    if os.path.exists(path):
        return path
    else:
        os.mkdir(path)
        return path


# ##########train param

summary_path = path_checker("./tensorboard/")

batch_normalization = 1
class_num = 20+1
classes = [0, 1]
keep_prob = 0.8
lr = 0.001

image_size = 256
dataset_size = 132   # 2913

BS = 8  # batch_size
epochs = 10     # the epoch mean the count, start from 0, so  epochs=10 means 0-9
iter_each_epoch = dataset_size/BS
iter_max = iter_each_epoch*epochs
queue_capacity = 10

num_queue_threads = 2

tfrecord_path_train = "./data/train.tfrecords"
tfrecord_path_val = "./data/val.tfrecords"


logging.info("\nparameters: batch_normalization={}\nclass_num={}\n"
             "keep_prob={}\nsummary_path={}\nlearning_rate={}\nbatch_size={}\nimage_size={}\ndataset_size={}\n"
             "epochs={}\niter_each_epoch{}\niter_max={}\n".
             format(bool(batch_normalization), class_num,
                    keep_prob, summary_path, lr, BS, image_size, dataset_size,
                    epochs, iter_each_epoch, iter_max))


if __name__ == '__main__':
    pc_bar_ = ShowProcess(10, '')
    ar = [1,2,3,4,5]
    for epoch_i_ in range(2):
        print ('Epoch {}'.format(epoch_i_) + '/{}'.format(epochs))
        for i in range(1,  10+1):
            pc_bar_.show_process(i, 10, ar[1], 1.0)
            time.sleep(1)
