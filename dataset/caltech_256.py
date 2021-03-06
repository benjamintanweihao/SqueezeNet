import tensorflow as tf
import os
import multiprocessing
from sklearn.model_selection import train_test_split
from imutils import paths


class Caltech256:
    def __init__(self):
        pass


    @staticmethod
    def train_test_split(test_size=0.2):
        """

        :param test_size:
        :return: (X_train, X_val, y_train, y_val)
        """

        X, y = Caltech256.load_file_names_and_labels()
        assert len(X) == len(y)

        return train_test_split(X, y, test_size=test_size)

    @staticmethod
    def params(X_train, X_val):

        return  {'input_shape': (224, 224, 3),
                 'n_classes': 256,
                 'n_epochs': 100,
                 'batch_size': 128,
                 'lr': 0.0005,
                 'optimizer': 'adam',
                 'activation': tf.nn.elu,
                 'n_images': len(X_train),
                 'n_val_images': len(X_val) }


    @staticmethod
    def input_fn(file_names, labels, params, mode):

        def _parse_function(filename, label):
            image_string = tf.read_file(filename)
            image = tf.image.decode_jpeg(image_string, channels=3)
            image = tf.image.convert_image_dtype(image, tf.float32)
            image = tf.image.resize_images(image, list(params['input_shape'])[:-1])
            image = tf.image.per_image_standardization(image)

            label = tf.string_to_number(label, tf.int64)

            return image, label

        def _train_preprocess(image, label):
            image = tf.image.random_flip_left_right(image)
            image = tf.image.random_brightness(image, max_delta=32.0 / 255.0)
            image = tf.image.random_saturation(image, lower=0.5, upper=1.5)

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


    @staticmethod
    def load_file_names_and_labels():
        image_paths = paths.list_images(os.path.normpath(os.path.join(os.getcwd(), 'data', '256_ObjectCategories')))
        filenames = []
        labels = []

        for p in image_paths:
            label = p.split('/')[-2].split('.')[0]
            filenames.append(p)
            labels.append(label)

        return filenames, labels
