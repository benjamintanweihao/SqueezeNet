import multiprocessing
import os
from imutils import paths
import tensorflow as tf


def input_fn(file_names, labels, params, mode):

    def _parse_function(filename, label):
        image_string = tf.read_file(filename)
        image = tf.image.decode_jpeg(image_string, channels=3)
        image = tf.image.convert_image_dtype(image, tf.float32)
        image = tf.image.resize_images(image, list(params['input_shape'])[:-1])
        image = tf.image.per_image_standardization(image)
        # image /= 255.

        label = tf.string_to_number(label, tf.int64)

        return image, label

    def _train_preprocess(image, label):
        # image = tf.image.random_flip_left_right(image)
        # image = tf.image.random_brightness(image, max_delta=32.0 / 255.0)
        # image = tf.image.random_saturation(image, lower=0.5, upper=1.5)

        # Make sure the image is still in [0, 1]
        # image = tf.clip_by_value(image, 0.0, 1.0)

        return image, label

    dataset = tf.data.Dataset.from_tensor_slices((file_names, labels))
    if mode == tf.estimator.ModeKeys.TRAIN:
        dataset = dataset \
            .shuffle(len(file_names), seed=123456789) \
            .map(_parse_function, num_parallel_calls=multiprocessing.cpu_count()) \
            .map(_train_preprocess, num_parallel_calls=multiprocessing.cpu_count()) \
            .batch(params['batch_size'], drop_remainder=True) \
            .repeat() \
            .prefetch(1)

    if mode == tf.estimator.ModeKeys.EVAL:
        dataset = dataset \
            .map(_parse_function, num_parallel_calls=multiprocessing.cpu_count()) \
            .batch(params['batch_size'], drop_remainder=False) \
            .prefetch(1)

    return dataset


def load_filenames_and_labels():
    image_paths = paths.list_images(os.path.normpath(os.path.join(os.getcwd(), 'data', '256_ObjectCategories')))
    filenames = []
    labels = []

    for p in image_paths:
        label = p.split('/')[-2].split('.')[0]
        filenames.append(p)
        labels.append(label)

    return filenames, labels
