# coding:utf-8
"""mason_P 2018/11/08
the data prepared for model, which is the most complex part for me !!!!
for the tensorflow, the raw dataset format is: image files and images files
the data feed for network is : input is images, labels is images


Note:
    the data consist of  two parts
    1) dataset to tfrecord  (this module)
    2) parse tfrecord for tf graph input

"""

import tensorflow as tf
import numpy as np
import os
from PIL import Image

from config import func_track, classes, image_size

@func_track
def create_tfrecord(phase, path_):
    """TFrecord is a binary data files, which contains the tf.train.Example() protocol memory block (protocol buffer).
    Note:
        data_image --feed--> protocol_buffer ---sequential---> str ---into--> TFRecord
        the data path structure show below, and the 0,1,2, is the classes
        train/
            train ---img1.jpg
                     img2.jpg
                     img3.jpg
                     ...
            label ---img1.jpg
                     img2.jpg
                    ...
        test/
            test -- ...
        #########################################################
        #if for classify issues: where the number 0, 1 .... is the class name
        train/
            0 ---img1.jpg
                img2.jpg
                img3.jpg
                ...
            1 ---img1.jpg
                img2.jpg
                ..
        test/
            0 -- ...

        #then:  classes=[1,2,....]
        writer = tf.python_io.TFRecordWriter("train.tfrecords")
        for index, name in enumerate(classes):
            class_path = os.cwd() + name + "/"
            for img_name in os.listdir(class_path):
                img_path = class_path + img_name
                    img = Image.open(img_path)
                    img = img.resize((224, 224))
                img_raw = img.tobytes()
                example = tf.train.Example(features=tf.train.Features(feature={
                    "label": tf.train.Feature(int64_list=tf.train.Int64List(value=[index])),
                    'img_raw': tf.train.Feature(bytes_list=tf.train.BytesList(value=[img_raw]))
                }))
                writer.write(example.SerializeToString())
        writer.close()

        ###########################################################
        the method 'enumerate(classes )' return a index and its related value, the classes is a class_name list
        the features of Example is the protocol info (or prototext massages)

        # since i directly using to_bytes() which makes image to binary, so the image buffer is useless.

    :arg
        phase: train or test



    :return no return, but
        create a TFRecord file, which then pass to TFRecordReader

    """

    writer = tf.python_io.TFRecordWriter("{}.tfrecord".format(phase))

    for image_name in os.listdir(path_+"images/"):
        name = os.path.splitext(image_name)[0]
        image_path = path_ + "images/" + name + ".jpg"
        image = Image.open(image_path)
        image = image.resize((image_size, image_size))
        image_raw = image.tobytes()

        label_path = path_ + "labels/" + name + ".png"
        label_img = Image.open(label_path)
        label_ = label_img.resize((image_size, image_size))
        label_raw = label_.tobytes()

        example_ = tf.train.Example(
            features=tf.train.Features(
                feature={
                    "label": tf.train.Feature(bytes_list=tf.train.BytesList(value=[label_raw])),
                    "image_binary": tf.train.Feature(bytes_list=tf.train.BytesList(value=[image_raw]))
                }
            )
        )

        writer.write(example_.SerializeToString())

    writer.close()


def main():
    if os.path.exists(os.path.join(os.getcwd(), "train/")):
        create_tfrecord(phase="train", path_=os.path.join(os.getcwd(), "train/"))
    if os.path.exists(os.path.join(os.getcwd(), "test/")):
        create_tfrecord(phase="test", path_=os.path.join(os.getcwd(), "test/"))


if __name__ == "__main__":
    main()