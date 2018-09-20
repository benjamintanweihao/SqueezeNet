import tensorflow as tf
import os
import multiprocessing
from sklearn.model_selection import train_test_split
from imutils import paths


class TinyImageNet:
    def __init__(self):
        pass


    @staticmethod
    def train_test_split(test_size=0.2):
        """

        :param test_size:
        :return: (X_train, X_val, y_train, y_val)
        """

        X, y = TinyImageNet.load_file_names_and_labels()
        assert len(X) == len(y)

        return train_test_split(X, y, test_size=test_size)

    @staticmethod
    def params(X_train, X_val):
        return  {'input_shape': (64, 64, 3),
                 'n_classes': 200,
                 'n_epochs': 30,
                 'batch_size': 128,
                 'lr': 0.001 * 0.5 * 0.5 * 0.5 * 0.5 * 0.5 * 0.5,
                 'optimizer': 'adam',
                 'activation': tf.nn.elu,
                 'n_images': len(X_train),
                 'n_val_images': len(X_val) }


    @staticmethod
    def input_fn(file_names, labels, params, mode):
        def _parse_function(filename, label):
            (height, width, channels) = params['input_shape']

            image_string = tf.read_file(filename)
            image = tf.image.decode_jpeg(image_string, channels=3)
            image = tf.image.convert_image_dtype(image, tf.float32)
            image = tf.image.resize_image_with_crop_or_pad(image, height, width)
            image = tf.image.per_image_standardization(image)

            label = tf.string_to_number(label, tf.int32)

            return image, label

        def _train_preprocess(image, label):
            # image = tf.image.random_flip_left_right(image)
            # image = tf.image.random_brightness(image, max_delta=32.0 / 255.0)
            # image = tf.image.random_saturation(image, lower=0.5, upper=1.5)
            #
            # # Make sure the image is still in [0, 1]
            # image = tf.clip_by_value(image, 0.0, 1.0)

            return image, label

        # NOTE: See https://stackoverflow.com/questions/46444018/meaning-of-buffer-size-in-dataset-map-dataset-prefetch
        # -and-dataset-shuffle/48096625#48096625
        # on shuffling data
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
    def build_label_dicts():
        wnids_path = os.path.join(os.getcwd(), 'data/tiny-imagenet-200/wnids.txt')
        words_path = os.path.join(os.getcwd(), 'data/tiny-imagenet-200/words.txt')

        label_dict, class_description = {}, {}

        with open(wnids_path, 'r') as f:
            for i, line in enumerate(f.readlines()):
                sysnet = line[:-1]
                label_dict[sysnet] = i

        with open(words_path, 'r') as f:
            for i, line in enumerate(f.readlines()):
                sysnet, desc = line.split('\t')
                desc = desc[:-1]
                if sysnet in label_dict:
                    class_description[label_dict[sysnet]] = desc

        return label_dict, class_description


    @staticmethod
    def load_file_names_and_labels():
        label_dict, class_description = TinyImageNet.build_label_dicts()
        filenames, labels = [], []

        # if mode == tf.estimator.ModeKeys.TRAIN:
        for filename in list(paths.list_images(os.path.join(os.getcwd(), 'data/tiny-imagenet-200/train'))):
            label = filename.split('/')[-1].split('_')[0]
            label = str(label_dict[label])
            filenames.append(filename)
            labels.append(label)

        # elif mode == tf.estimator.ModeKeys.EVAL:
        #     with open(os.path.join(os.getcwd(), 'data/tiny-imagenet-200/val/val_annotations.txt'), 'r') as f:
        #         for line in f.readlines():
        #             split_line = line.split('\t')
        #             filename = os.path.join(os.getcwd(), 'data/tiny-imagenet-200/val/images/' + split_line[0])
        #             label = str(label_dict[split_line[1]])
        #             filenames.append(filename)
        #             labels.append(label)

        assert len(filenames) == len(labels)
        assert len(filenames) > 0

        return filenames, labels

