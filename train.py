import multiprocessing

import tensorflow as tf
from tensorflow import keras
import os
import numpy as np

tf.logging.set_verbosity(tf.logging.INFO)

from squeezenet import SqueezeNet

train_labels_path = os.path.join(os.getcwd(), 'data/fashionmnist/train-labels-idx1-ubyte')
train_images_path = os.path.join(os.getcwd(), 'data/fashionmnist/train-images-idx3-ubyte')

train_labels = np.frombuffer(open(train_labels_path, 'rb').read(),
                             dtype=np.uint8, offset=8)
train_images = np.frombuffer(open(train_images_path, 'rb').read(),
                             dtype=np.uint8, offset=16).reshape(len(train_labels), -1)

test_labels_path = os.path.join(os.getcwd(), 'data/fashionmnist/t10k-labels-idx1-ubyte')
test_images_path = os.path.join(os.getcwd(), 'data/fashionmnist/t10k-images-idx3-ubyte')

test_labels = np.frombuffer(open(test_labels_path, 'rb').read(),
                            dtype=np.uint8, offset=8)
test_images = np.frombuffer(open(test_images_path, 'rb').read(),
                            dtype=np.uint8, offset=16).reshape(len(test_labels), -1)

# Turn single channel into 3 channels
train_images = np.dstack((train_images,) * 3)
train_images = train_images.reshape(-1, 28, 28, 3)

test_images = np.dstack((test_images,) * 3)
test_images = test_images.reshape(-1, 28, 28, 3)

CLASSES = 10
BATCH_SIZE = 128
EPOCHS = 15

decay_steps = len(train_images) * EPOCHS // BATCH_SIZE

model = SqueezeNet(input_shape=(227, 227, 3), classes=CLASSES)

# learning_rate = tf.train.polynomial_decay(learning_rate=0.04,
#                                           global_step=tf.train.get_or_create_global_step(),
#                                           decay_steps=decay_steps,
#                                           end_learning_rate=0.005,
#                                           power=1.0,
#                                           cycle=False)
#
# optimizer = tf.train.MomentumOptimizer(learning_rate=learning_rate,
#                                        momentum=0.9,
#                                        use_nesterov=True)

optimizer = tf.train.RMSPropOptimizer(learning_rate=0.001)

model.compile(optimizer=optimizer,
              loss=keras.losses.categorical_crossentropy,
              metrics=['accuracy'])


def input_fn(features, labels, batch_size, nb_classes, training=True):
    def _parse_function(feature, label):
        feature = tf.image.resize_images(feature / 255, [227, 227])

        # NOTE: Must turn into one hot so that we get (10,) for labels
        return feature, tf.one_hot(label, nb_classes)

    dataset = tf.data.Dataset.from_tensor_slices((features, labels))

    if training:
        dataset = dataset.shuffle(1000).repeat()

    dataset = dataset.batch(batch_size, drop_remainder=True) \
        .map(_parse_function, num_parallel_calls=multiprocessing.cpu_count())

    return dataset


print(model.summary())

# NOTE: steps = (no of ex / batch_size) * no_of_epochs
steps = len(train_labels) // BATCH_SIZE * EPOCHS

print('Number of steps {}'.format(steps))

estimator = keras.estimator.model_to_estimator(keras_model=model,
                                               model_dir=os.path.join(os.getcwd(), 'data', 'model'))

estimator.train(input_fn=lambda: input_fn(train_images,
                                          train_labels,
                                          batch_size=BATCH_SIZE,
                                          nb_classes=CLASSES),
                steps=steps)

eval_result = estimator.evaluate(input_fn=lambda: input_fn(test_images,
                                                           test_labels,
                                                           batch_size=BATCH_SIZE,
                                                           nb_classes=CLASSES,
                                                           training=False))

print('\nTest set accuracy: {accuracy:0.3f}\n'.format(**eval_result))
