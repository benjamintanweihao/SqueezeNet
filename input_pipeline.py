import multiprocessing
import os
from imutils import paths
import tensorflow as tf


def tiny_imagenet_input_fn(params, mode):
    def _parse_function(filename, label):
        image_string = tf.read_file(filename)
        image_decoded = tf.image.decode_jpeg(image_string, channels=3)
        image_resized = tf.image.resize_images(image_decoded, list(params['input_shape'])[:-1])
        image_std = tf.image.per_image_standardization(image_resized)

        label = tf.string_to_number(label, tf.float32)
        label = tf.cast(label, tf.uint8)

        return image_std, label

    filenames, labels = load_filenames_labels(mode)

    dataset = tf.data.Dataset.from_tensor_slices((filenames, labels))
    if mode == tf.estimator.ModeKeys.TRAIN:
        dataset = dataset.shuffle(1000).repeat()

    with tf.device('/cpu:0'):
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
