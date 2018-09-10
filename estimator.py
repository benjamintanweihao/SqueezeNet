import multiprocessing
import os
import numpy as np
import tensorflow as tf
from tensorflow import keras

tf.enable_eager_execution()
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


def input_fn(features, labels, params, training=True):
    def _parse_function(feature, label):
        feature = tf.image.resize_images(feature / 255,
                                         list(params['input_shape'])[:-1])  # [227, 227]

        return feature, label

    dataset = tf.data.Dataset.from_tensor_slices((features, labels))

    if training:
        dataset = dataset.shuffle(1000).repeat()

    dataset = dataset.batch(params['batch_size'], drop_remainder=True) \
        .map(_parse_function, num_parallel_calls=multiprocessing.cpu_count())

    return dataset


def model_fn(features, labels, mode, params):
    net = SqueezeNet(features,
                     classes=params['n_classes'],
                     training=(mode == tf.estimator.ModeKeys.TRAIN))

    # logits.shape == (?, 10)
    logits = keras.layers.Flatten()(net)

    # Compute predictions
    predictions = {
        'classes': tf.argmax(input=logits, axis=1),
        'probabilities': tf.nn.softmax(logits, name='softmax_tensor')
    }

    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(mode, predictions=predictions)

    # Compute loss.
    onehot_labels = tf.one_hot(indices=tf.cast(labels, tf.int32),
                               depth=params['n_classes'])

    loss = tf.losses.softmax_cross_entropy(onehot_labels=onehot_labels,
                                           logits=logits)

    # Compute evaluation metrics.
    accuracy = tf.metrics.accuracy(labels=labels,
                                   predictions=predictions['classes'],
                                   name='acc_op')

    metrics = {'accuracy': accuracy}
    tf.summary.scalar('accuracy', accuracy[1])

    if mode == tf.estimator.ModeKeys.EVAL:
        return tf.estimator.EstimatorSpec(mode,
                                          loss=loss,
                                          eval_metric_ops=metrics)

    # Create training op.
    assert mode == tf.estimator.ModeKeys.TRAIN

    decay_steps = len(train_images) * params['n_epochs'] // params['batch_size']

    optimizer = None
    if params['optimizer'] == 'poly':
        # NOTE: Setting the learning rate to 0.04 gave `NanLossDuringTrainingError` as per the paper.
        learning_rate = tf.train.polynomial_decay(learning_rate=0.01,
                                                  global_step=tf.train.get_global_step(),
                                                  decay_steps=decay_steps,
                                                  end_learning_rate=0.005,
                                                  power=1.0,
                                                  cycle=False)

        optimizer = tf.train.MomentumOptimizer(learning_rate=learning_rate,
                                               momentum=0.9,
                                               use_nesterov=True)
    elif params['optimizer'] == 'rms':
        optimizer = tf.train.RMSPropOptimizer(learning_rate=0.001)
    else:
        assert 'No optimizer defined in params!'

    train_op = optimizer.minimize(loss, global_step=tf.train.get_global_step())

    return tf.estimator.EstimatorSpec(mode, loss=loss, train_op=train_op)


params = {
    'n_classes': 10,
    'n_epochs': 15,
    'batch_size': 128,
    'input_shape': (227, 227, 3),
    'optimizer': 'poly'  # rms | poly
}

estimator = tf.estimator.Estimator(
    model_fn=model_fn,
    model_dir=os.path.join(os.getcwd(), 'data', 'model'),
    params=params)

# NOTE: steps = (no of ex / batch_size) * no_of_epochs
STEPS = len(train_labels) // params['batch_size'] * params['n_epochs']

tensors_to_log = {"probabilities": "softmax_tensor"}
logging_hook = tf.train.LoggingTensorHook(tensors=tensors_to_log,
                                          every_n_iter=50)

# Train the model
estimator.train(input_fn=lambda: input_fn(train_images,
                                          train_labels,
                                          params=params),
                steps=STEPS, hooks=[logging_hook])

eval_result = estimator.evaluate(input_fn=lambda: input_fn(test_images,
                                                           test_labels,
                                                           params=params,
                                                           training=False))

print('\nTest set accuracy: {accuracy:0.3f}\n'.format(**eval_result))
