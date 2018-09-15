import multiprocessing
import os
from imutils import paths
import tensorflow as tf
import random


def tiny_imagenet_input_fn(params, mode):
    imagenet_mean = tf.constant([123.68, 116.779, 103.939], dtype=tf.float32)

    def _parse_function(filename, label):
        image_string = tf.read_file(filename)
        image = tf.image.decode_jpeg(image_string, channels=3)
        image = tf.image.resize_images(image, list(params['input_shape'])[:-1])
        image = tf.subtract(image, imagenet_mean)
        image = tf.image.per_image_standardization(image)

        label = tf.string_to_number(label, tf.int32)
        one_hot = tf.one_hot(label, depth=params['n_classes'], dtype=tf.int32)

        return image, one_hot

    split_idx = int(params['n_images'] * 0.8)

    filenames, labels = load_filenames_labels() # TRAIN + EVAL
    combined = list(zip(filenames, labels))
    random.shuffle(combined)

    filenames, labels = zip(*combined)
    train_filenames = filenames[]


    dataset = tf.data.Dataset.from_tensor_slices((filenames, labels))
    if mode == tf.estimator.ModeKeys.TRAIN:
        # NOTE: Pass in a small number to .take() as a sanity check to see
        #       if the network overfits.
        dataset = dataset \
            .shuffle(10000, seed=123456789) \
            .repeat()

    if mode == tf.estimator.ModeKeys.EVAL:
        dataset = dataset \
            .repeat()

    dataset = dataset \
        .map(_parse_function, num_parallel_calls=multiprocessing.cpu_count()) \
        .batch(params['batch_size'], drop_remainder=True) \
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


def load_filenames_labels():
    label_dict, class_description = build_label_dicts()
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
