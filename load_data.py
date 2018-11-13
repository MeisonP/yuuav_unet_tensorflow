# coding:utf-8
"""mason_P 2018/11/08
the data prepared for model, which is the most complex part for me !!!!
for the tensorflow, the raw dataset format is: image files and images files
the data feed for network is : input is images, labels is images


Note:
    the data consist of  two parts
    1) dataset to tfrecord
    2) parse tfrecord for tf graph input   (this module)
        a) parse tfrecord to be example queue
        b) using example queue to create batch_queue
"""


import tensorflow as tf
from config import image_size, batch_size, epochs, threads_data, func_track


@func_track
def example_queue(phase):
    """to create a example queue from dataset, this queue then will be package as batch_input.
    Note:
    the are total 4 step:1)filenema_list; 2)filename_queue; 3)reader and decoder 4)make example_queue;
    and the image process will be happen after the step 3); the reader is used to read the data_recorded,
    and the decoder is for trans the data_recorded to tensor
    :arg
        phase: test or trian
    :return
        example_image: the input is the image and label that parse from the example TFRecord file
        example_label:
    """

    if phase == "train":
        shuffle_ = True
    else:
        shuffle_ = False

    # pattern is the filename of TFRecord file/path
    filename_list = tf.train.match_filenames_once(pattern="./{0}/{1}.tfrecord".format(phase, phase))

    filename_queue = tf.train.string_input_producer(string_tensor=filename_list,
                                                    num_epochs=epochs, shuffle=shuffle_, capacity=32)

    reader = tf.TextLineReader()
    _, serialized_example = reader.read(filename_queue)
    features = tf.parse_single_example(serialized_example,
                                       features={
                                           'label': tf.FixedLenFeature([], tf.string),
                                           'image_binary': tf.FixedLenFeature([], tf.string)
                                       })

    img = tf.decode_raw(features['image_binary'], tf.uint8)

    img_pro = tf.reshape(img, [image_size, image_size, 3])

    example_image = tf.cast(img_pro, tf.float32)    # * (1./225) - 0.5

    label_ = tf.decode_raw(features['label'], tf.uint8)
    label_pro = tf.reshape(label_, [image_size, image_size, 3])
    example_label = tf.cast(label_pro, tf.float32)

    return example_image, example_label

@func_track
def batch_queue(phase):
    """using the example to create batch queue.
    Note:
        the capacity
    :arg
        phase: train or test
    :return
        return the batch queue which then will be putinto sess.run()

    """

    image, label = example_queue(phase)

    min_after_dequeue_ = 1000
    capacity_ = min_after_dequeue_ + 3 * batch_size

    image_batch, label_batch = tf.train.shuffle_batch([image, label], batch_size, capacity=capacity_,
                                                      min_after_dequeue=min_after_dequeue_,
                                                      num_threads=threads_data)

    return image_batch, label_batch

