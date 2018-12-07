# coding:utf-8
"""individual main module, that used to creat tfrecord file.
for the tensorflow, the raw dataset format is: images files (src is images, label is images)
#of course there are other wary to feed data into net, not only this way:generate tfrecord;
# such feet the image batch directly

2018/12/04
tensorflow ==1.11
python ==2.7.15

Note:
    the dataset structure should be
    data
        ---train_dataset
        ---val_dataset (using for train evaluate during the training,
                        and do not take part in the  back-propagation)
        ---test_dataset (same as the evaluate but after the train process as a individual module)

    as using the argparse module , the main() func must have one arg, eg def main(_):
    how to use the flag: eg,
    import argparse
    parser = argparse.ArgumentParser(description='Eval Unet on given tfrecords.')
    parser.add_argument('--image_size', help='inter, size of reshape', default=256)
    FLAGS, _ = parser.parse_known_args()    # return  namespace, upparsed args
    tf.app.run()

"""


import tensorflow as tf
import numpy as np
import os
import cv2
import sys
import argparse
import time


def rgb_label_maker(rgb_label_image, class_num):
    """only for VOC dataset segment, (need change the num_classes, COLORMAP for other dataset)
    transfer the RGB label image(h, w, 3) to input label matrix (h, w, num_classes),
    mapping the rgb value to class distribution

    Note:
        VOC rgb segment label, each color related to particular class,
        transf_value: is used for rgb value to class id which mapping to class name
        label_1d: is a (h, w) matrix , which's value is the class id. the label_1d then
        transform to label (h, w, num_)

        the label inpput network is not RGB image !!!!!!



    :arg
        rgb_label_image: a rgb image with shape (h, w, 3), dtype is uint8

    :return
        a np array (only has value 0 and 1) with shape (h, w, num_classes), dtype is int64

        but you can change to uint8 by using astype(np.uint8)
        the third dimension  is the class distribution(which index =1, means which class)
        eg:  [0 0 0 1 0 0 0 ...]  means 3 class of class and then mapping to the class name.
    """

    VOC_COLORMAP = [[0, 0, 0], [128, 0, 0], [0, 128, 0], [128, 128, 0],
                    [0, 0, 128], [128, 0, 128], [0, 128, 128], [128, 128, 128],
                    [64, 0, 0], [192, 0, 0], [64, 128, 0], [192, 128, 0],
                    [64, 0, 128], [192, 0, 128], [64, 128, 128], [192, 128, 128],
                    [0, 64, 0], [128, 64, 0], [0, 192, 0], [128, 192, 0],
                    [0, 64, 128]]

    VOC_CLASSES = ['background', 'aeroplane', 'bicycle', 'bird', 'boat',
                   'bottle', 'bus', 'car', 'cat', 'chair', 'cow',
                   'diningtable', 'dog', 'horse', 'motorbike', 'person',
                   'potted plant', 'sheep', 'sofa', 'train', 'tv/monitor']

    transf_value = np.zeros((256**3), dtype=np.uint8)

    for i, colormap in enumerate(VOC_COLORMAP):
        transf_value[(colormap[0]*256*256 + colormap[1]*256 + colormap[2])] = i

    idx = (rgb_label_image[:, :, 0]*256*256 + rgb_label_image[:, :, 1]*256 + rgb_label_image[:, :, 2])

    label_2d = transf_value[idx]

    h, w, _ = rgb_label_image.shape
    label = np.zeros(shape=(h, w, class_num), dtype=np.uint8)

    for i in range(h):
        for j in range(w):
            k = label_2d[i, j]
            label[i, j, k] = 1

    return label


def gray_label_maker(gray_image, class_num):
    """ the class name and id is changeable

    :param
        gray_image:  2d image /a matrix with shape (h, w); this equal to label_2d in rgb_label_maker func

    :return:
        a np array (only has value 0 and 1) with shape (h, w, num_classes), dtype is int64
    """

    h, w, _ = gray_image.shape
    label = np.zeros(shape=(h, w, class_num), dtype=np.uint8)

    for i in range(h):
        for j in range(w):
            k = gray_image[i, j]
            label[i, j, k] = 1

    return label


def create_tfrecord(record_path_, dataset_path_, process_bar_, image_size, class_num):
    """method, to create a TFrecord file, a byte data files,
    which contains the tf.train.Example() protocol memory block (protocol buffer).

    Note:
        ! for diff label maker,
        pls change : mask = rgb_label_maker(mask,class)  or mask = gray_label_maker(mask, class)

        *resize. make sure the shape of image are same to the tensor shape fed into the network
        dataset to tfrecord. the main question is **hwo to define the example feature**
        inorder process bar up to 100%, the counter must start from 1

        1) image filename list
            it is necessary to do some data enhancement before
            at the origin dataset (if the dataset is much small)
            the dtype = uint8 after cv2.imread,
            make sure you  have the same dtype after decode_raw, at read_tfrecord

        2) image to byte

        3) example  //set up features, be careful to write feature dict
        * check and make sure: "img.tobytes & tf.train.BytesList" pair; "value = []";

        4) write into tfrecord // TFRecordWriter

    :param
        path_: the path to data, (data/train, or data/test)


    :return no return, but
        create a TFRecord file, the binary data are stored with feature construct

    """

    writer = tf.python_io.TFRecordWriter(record_path_)

    count = 1

    for image_name in os.listdir(dataset_path_ + "src/"):
        name = os.path.splitext(image_name)[0]

        image_path = dataset_path_ + "src/" + name + ".jpg"
        image = cv2.imread(image_path)
        image = cv2.resize(image, (image_size, image_size), interpolation=cv2.INTER_LINEAR)

        i = count
        process_bar_. show_process(i)

        image_raw = image.tobytes()

        label_path = dataset_path_ + "labels/" + name + ".png"
        mask = cv2.imread(label_path)
        mask = cv2.resize(mask, (image_size, image_size), interpolation=cv2.INTER_LINEAR)
        mask = rgb_label_maker(mask, class_num)    # output (h, w, num_classes)
        mask_raw = mask.tobytes()

        feature_dict = {
                        'name': tf.train.Feature(bytes_list=tf.train.BytesList(value=[name])),
                        'image': tf.train.Feature(bytes_list=tf.train.BytesList(value=[image_raw])),
                        'mask': tf.train.Feature(bytes_list=tf.train.BytesList(value=[mask_raw]))}

        example = tf.train.Example(features=tf.train.Features(feature=feature_dict))

        writer.write(example.SerializeToString())

        count = count+1
    writer.close()


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


def main(_):
    ds = 0
    for fn in os.listdir(FLAGS.dataset_path + "src/"):
        ds = ds + 1
    dataset_size = ds
    max_steps = dataset_size

    process_bar_ = ShowProcess(max_steps, '{}: TFRecords Done!'.format(FLAGS.record_path))
    create_tfrecord(FLAGS.record_path, FLAGS.dataset_path, process_bar_, FLAGS.image_size, FLAGS.num_classes)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Eval Unet on given tfrecords.')

    parser.add_argument('--image_size', help='inter, size of reshape', default=256)
    parser.add_argument('--num_classes', '-c', help='inter, the number of classes',
                        required=True, default=21, type=np.uint8)

    parser.add_argument('--batch_size', help='inter, size of batch', default=8)

    parser.add_argument('--dataset_path', '-d', help='the dir of the data folder',
                        required=True, default="./data/train/")
    parser.add_argument('--record_path', '-r', help='path of the created tfrecords file',
                        required=True, default="./data/train.tfrecords")    # "./data/val.tfrecords"

    FLAGS, _ = parser.parse_known_args()

    tf.app.run()




