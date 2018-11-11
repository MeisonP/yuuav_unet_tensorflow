# coding:utf-8
# masom: Unet-tensorflow version
# 2018/11/5

import time
import logging
import os

TM = time.strftime("%Y:%m:%d-%H:%M",time.localtime())


#   LOG_FORMAT="%(asctime)s - %(levelname)s - [%(filename)s,line:%(lineno)d] - %(message)s"
LOG_FORMAT="%(asctime)s-%(levelname)s-[line:%(lineno)d] - %(message)s"
logging.basicConfig(level=logging.INFO, format=LOG_FORMAT)
logging.info("**********************mason_p nn_design(%s)***********************" % TM)



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


##########################
#
# parameters


device = '/cpu:0'
batch_normalization = 1

class_num = 2   # is building or not buliding
classes = [0, 1]

keep_prob = 0.8
lr = 0.001
batch_size = 64

image_size = 300
threads_data = 4

summary_path = "./tensorboard/"


dataset_size = 2913  # how many images that you have
epochs = 1000
iter_max = (dataset_size/batch_size)*epochs

display=1000

logging.info("\nparameters: device={}\nbatch_normalization={}\nclass_num={}\n"
             "keep_prob={}\nsummary_path={}\nlearning_rate={}\nbatch_size={}\nimage_size={}\nimage_size={}\n"
             "epochs={}\niter_max={}\ndisplay_iter={}\n".
             format(device, bool(batch_normalization), class_num,
                    keep_prob, summary_path, lr, batch_size, image_size, image_size,
                    epochs, iter_max, display))
