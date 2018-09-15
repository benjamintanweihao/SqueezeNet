import multiprocessing
import os
from imutils import paths
import tensorflow as tf


def tiny_imagenet_input_fn(params, mode):

    filenames, labels = load_filenames_labels(mode)

    def _parse_function(filename, label):
        image_string = tf.read_file(filename)
        image = tf.image.decode_jpeg(image_string, channels=3)
        image = tf.image.resize_images(image, list(params['input_shape'])[:-1])
        image = tf.image.per_image_standardization(image)
        image = tf.image.convert_image_dtype(image, tf.float32)

        label = tf.string_to_number(label, tf.int32)
        one_hot = tf.one_hot(label, depth=params['n_classes'], dtype=tf.int32)

        return image, one_hot

    def _train_preprocess(image, label):
        image = tf.image.random_flip_left_right(image)
        image = tf.image.random_brightness(image, max_delta=32.0 / 255.0)
        image = tf.image.random_saturation(image, lower=0.5, upper=1.5)

        # Make sure the image is still in [0, 1]
        image = tf.clip_by_value(image, 0.0, 1.0)

        return image, label

    # NOTE: See https://stackoverflow.com/questions/46444018/meaning-of-buffer-size-in-dataset-map-dataset-prefetch
    # -and-dataset-shuffle/48096625#48096625
    # on shuffling data
    dataset = tf.data.Dataset.from_tensor_slices((filenames, labels))
    if mode == tf.estimator.ModeKeys.TRAIN:
        dataset = dataset \
            .shuffle(len(filenames), seed=123456789) \
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


def load_filenames_labels(mode):
    label_dict, class_description = build_label_dicts()
    filenames, labels = [], []

    if mode == tf.estimator.ModeKeys.TRAIN:
        for filename in list(paths.list_images(os.path.join(os.getcwd(), 'data/tiny-imagenet-200/train'))):
            label = filename.split('/')[-1].split('_')[0]
            label = str(label_dict[label])
            filenames.append(filename)
            labels.append(label)

    elif mode == tf.estimator.ModeKeys.EVAL:
        with open(os.path.join(os.getcwd(), 'data/tiny-imagenet-200/val/val_annotations.txt'), 'r') as f:
            for line in f.readlines():
                split_line = line.split('\t')
                filename = os.path.join(os.getcwd(), 'data/tiny-imagenet-200/val/images/' + split_line[0])
                label = str(label_dict[split_line[1]])
                filenames.append(filename)
                labels.append(label)

    assert len(filenames) == len(labels)
    assert len(filenames) > 0

    return filenames, labels
