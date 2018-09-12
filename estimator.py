import os
import tensorflow as tf
from tensorflow import keras

from input_pipeline import tiny_imagenet_input_fn

tf.enable_eager_execution()
tf.logging.set_verbosity(tf.logging.INFO)

from squeezenet import SqueezeNet


def model_fn(features, labels, mode, params):
    net = SqueezeNet(features,
                     classes=params['n_classes'],
                     training=(mode == tf.estimator.ModeKeys.TRAIN))

    # logits.shape == (?, n_classes)
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

    optimizer = None
    if params['optimizer'] == 'poly':
        # NOTE: Setting the learning rate to 0.04 gave `NanLossDuringTrainingError` as per the paper.
        decay_steps = params['n_images'] * params['n_epochs'] // params['batch_size']
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
    'n_images': 100000,
    'n_classes': 200, # Tiny ImageNet has 200 classes
    'n_epochs': 15,
    'batch_size': 256,
    'input_shape': (227, 227, 3),
    'optimizer': 'rms'  # rms | poly
}

# NOTE: steps = (no of ex / batch_size) * no_of_epochs
steps = params['n_images'] // params['batch_size'] * params['n_epochs']

estimator = tf.estimator.Estimator(
    model_fn=model_fn,
    model_dir=os.path.join(os.getcwd(), 'data', 'model'),
    params=params)


tensors_to_log = {"probabilities": "softmax_tensor"}
logging_hook = tf.train.LoggingTensorHook(tensors=tensors_to_log,
                                          every_n_iter=50)

estimator.train(input_fn=lambda: tiny_imagenet_input_fn(params=params,
                                                        mode=tf.estimator.ModeKeys.TRAIN),
                steps=steps, hooks=[]) # hooks = [logging_hook]

eval_result = estimator.evaluate(input_fn=lambda: tiny_imagenet_input_fn(params=params,
                                                                         mode=tf.estimator.ModeKeys.EVAL))

print('\nTest set accuracy: {accuracy:0.3f}\n'.format(**eval_result))
