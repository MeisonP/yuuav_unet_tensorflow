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

    def __init__(self, max_steps, infoDone = 'Done'):
        """class param init.
        :param max_steps: inter, how many step for process
            max_arrow: inter, the total num arrow
            i: inter, the step right now, init set to 0
        :param infoDone: String, def info when finishing process
        """
        self.max_steps = max_steps
        self.max_arrow = 50
        self.i = 0
        self.infoDone = infoDone

    def close(self):
        """
        print info, when finished
        """
        print('')
        print(self.infoDone)
        self.i = 0

    def show_process(self, i):
        """ core method, show the process bar.
        Note:
            1. calculate how many >
            2. calculate how may -
            3. calculate the percentage: "\r" must be at left ,means that begin form left
            4. sys.stdout.write(process_bar) and sys.stdout.flush() is to print to terminal
        :param i: inter, the step right now, init set to 0
        :return: no return

        """

        if i is not None:
            self.i = i
        else:
            self.i += 1
        num_arrow = int(self.i * self.max_arrow / self.max_steps)
        num_line = self.max_arrow - num_arrow
        percent = self.i * 100.0 / self.max_steps

        process_bar_ = '\r' + '[' + '>' * num_arrow + ' ' * num_line + ']'\
                       + '%.2f' % percent + '%'
        sys.stdout.write(process_bar_)
        sys.stdout.flush()
        if self.i >= self.max_steps:
            self.close()


def func_track(func):
    def track(*args, **kwargs):
        name=func.__name__
        logging.info("func %s..."%name)
        result=func(*args, **kwargs)
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
class_num = 2
classes = [0, 1]
keep_prob = 0.8
lr = 0.001

image_size = 256
dataset_size = 2913

BS = 8  # batch_size
epochs = 1000
iter_max = (dataset_size/BS)*epochs
display = 100
queue_capacity = 24

num_queue_threads = 4

tfrecord_path_list = ["./data/train.tfrecords"]


logging.info("\nparameters: batch_normalization={}\nclass_num={}\n"
             "keep_prob={}\nsummary_path={}\nlearning_rate={}\nbatch_size={}\nimage_size={}\n"
             "epochs={}\niter_max={}\ndisplay_iter={}\n".
             format(bool(batch_normalization), class_num,
                    keep_prob, summary_path, lr, BS, image_size,
                    epochs, iter_max, display))
