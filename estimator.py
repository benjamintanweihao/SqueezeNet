import os
import tensorflow as tf

tf.logging.set_verbosity(tf.logging.INFO)

from input_pipeline import tiny_imagenet_input_fn
from squeezenet import SqueezeNet


def model_fn(features, labels, mode, params):
    # logits.shape == (?, n_classes)
    logits = SqueezeNet(features,
                        classes=params['n_classes'],
                        training=(mode == tf.estimator.ModeKeys.TRAIN))

    # Compute predictions
    predictions = {
        'classes': tf.argmax(input=logits, axis=1),
        'probabilities': tf.nn.softmax(logits, name='softmax_tensor')
    }

    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(mode, predictions=predictions)

    # Compute loss.
    # See https://stats.stackexchange.com/questions/306862/cross-entropy-versus-mean-of-cross-entropy
    loss = tf.reduce_mean(tf.losses.softmax_cross_entropy(onehot_labels=labels,
                                                          logits=logits))

    # NOTE: Useful for debugging!
    loss = tf.Print(loss, [loss, tf.argmax(tf.nn.softmax(logits), axis=1)],
                    message='[Loss|Logits]',
                    summarize=1 + params['batch_size'] * 3)
    # Compute evaluation metrics.
    accuracy = tf.metrics.accuracy(labels=tf.argmax(labels, axis=1),
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
        learning_rate = tf.train.polynomial_decay(learning_rate=0.001,
                                                  global_step=tf.train.get_global_step(),
                                                  decay_steps=decay_steps,
                                                  end_learning_rate=0.0005,
                                                  power=1.0,
                                                  cycle=False)

        optimizer = tf.train.MomentumOptimizer(learning_rate=learning_rate,
                                               momentum=0.9,
                                               use_nesterov=False)
    elif params['optimizer'] == 'rms':
        optimizer = tf.train.RMSPropOptimizer(learning_rate=0.001,
                                              momentum=0.9)
    elif params['optimizer'] == 'sgd':
        optimizer = tf.train.MomentumOptimizer(learning_rate=0.001,
                                               momentum=0.9,
                                               use_nesterov=False)
    elif params['optimizer'] == 'adam':
        optimizer = tf.train.AdamOptimizer(learning_rate=0.001)
    else:
        assert 'No optimizer defined in params!'

    train_op = optimizer.minimize(loss, global_step=tf.train.get_global_step())

    return tf.estimator.EstimatorSpec(mode, loss=loss, train_op=train_op)


params = {
    'n_images': 100000,
    'n_val_images': 10000,
    'n_classes': 200,  # Tiny ImageNet has 200 classes
    'n_epochs': 100,
    'batch_size': 64,
    'input_shape': (64, 64, 3),
    'optimizer': 'sgd'  # rms | poly | sgd | adam
}

# NOTE: steps = (no of ex / batch_size) * no_of_epochs
steps = params['n_epochs'] * params['n_images'] // params['batch_size']

print('Steps: {}'.format(steps))

run_config = tf.estimator.RunConfig(save_checkpoints_steps=500)

estimator = tf.estimator.Estimator(
    model_fn=model_fn,
    model_dir=os.path.join(os.getcwd(), 'data', 'model'),
    params=params,
    config=run_config)

tensors_to_log = {"probabilities": "softmax_tensor"}
logging_hook = tf.train.LoggingTensorHook(tensors=tensors_to_log,
                                          every_n_iter=50)

train_spec = tf.estimator.TrainSpec(
    input_fn=lambda: tiny_imagenet_input_fn(params=params,
                                            mode=tf.estimator.ModeKeys.TRAIN),
    max_steps=steps)

# Evaluation happens after a checkpoint is created.
# See: https://www.tensorflow.org/guide/checkpoints#checkpointing_frequency
eval_spec = tf.estimator.EvalSpec(
    input_fn=lambda: tiny_imagenet_input_fn(params=params,
                                            mode=tf.estimator.ModeKeys.EVAL),
    steps=params['n_val_images'] // params['batch_size'],
    start_delay_secs=0,  # start evaluating every 60 seconds
    throttle_secs=0)

tf.estimator.train_and_evaluate(estimator, train_spec, eval_spec)
