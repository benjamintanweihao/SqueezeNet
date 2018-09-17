import os
import tensorflow as tf
import argparse
import datetime

from sklearn.model_selection import train_test_split

from input_pipeline import caltech_256
from squeezenet import SqueezeNet

tf.logging.set_verbosity(tf.logging.INFO)


def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


# ap = argparse.ArgumentParser(description='SqueezeNet')
# ap.add_argument('--n_classes', help="Total number of classes", type=int, required=True)
# ap.add_argument('--n_epochs', help="Total number of epochs", type=int, required=True)
# ap.add_argument('--batch_size', help="Batch size", type=int, required=True)
# ap.add_argument('--optimizer', help="Optimizer. One of: [rms|poly|sgd|adam]", default='sgd')
# ap.add_argument('--lr', help="Learning rate", required=True, type=float)
# ap.add_argument('--activation', help="Activation function to use [relu|elu]", default='relu')
# ap.add_argument('--exp_name', help="Experiment name", required=True)

# args = vars(ap.parse_args())
# args['input_shape'] = (64, 64, 3) # Tiny ImageNet
# args['input_shape'] = (227, 227, 3)

args = {'input_shape': (227, 227, 3),
        'n_classes': 256,
        'n_epochs': 50,
        'batch_size': 64,
        'lr': 0.0004,
        'optimizer': 'adam',
        'activation': tf.nn.elu,
        'exp_name': 'exp123'}

X, y = caltech_256.load_filenames_and_labels()
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2)

assert len(X_train) == len(y_train)
assert len(X_val) == len(y_val)


def model_fn(features, labels, mode, params):
    assert params['n_classes'] > 0

    logits = SqueezeNet(features,
                        classes=params['n_classes'],
                        training=(mode == tf.estimator.ModeKeys.TRAIN))

    onehot_labels = tf.one_hot(labels, params['n_classes'], axis=1)

    # Compute loss.
    # See https://stats.stackexchange.com/questions/306862/cross-entropy-versus-mean-of-cross-entropy
    loss = tf.reduce_mean(tf.losses.softmax_cross_entropy(onehot_labels=onehot_labels,
                                                          logits=logits))

    # Compute predictions
    predictions = tf.cast(tf.argmax(logits, 1), tf.int32)
    labels = tf.cast(labels, tf.int64)

    predictions = tf.Print(predictions, [predictions], message='predictions', summarize=10000)

    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(mode, predictions={
            "classes": tf.argmax(input=logits, axis=1),
            "probabilities": tf.nn.softmax(logits, name="softmax_tensor")
        })

    # NOTE: Useful for debugging!
    loss = tf.Print(loss, [loss, tf.argmax(tf.nn.softmax(logits), axis=1)],
                    message='[Loss|Logits]',
                    summarize=1 + params['batch_size'] * 3)
    # Compute evaluation metrics.

    metrics = {
        "accuracy": tf.metrics.accuracy(labels, predictions),
        "recall_at_5": tf.metrics.recall_at_k(labels, logits, 5),
        "recall_at_1": tf.metrics.recall_at_k(labels, logits, 1)
    }

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
        learning_rate = tf.train.polynomial_decay(learning_rate=params['lr'],
                                                  global_step=tf.train.get_global_step(),
                                                  decay_steps=decay_steps,
                                                  end_learning_rate=0.0005,
                                                  power=1.0,
                                                  cycle=False)

        optimizer = tf.train.MomentumOptimizer(learning_rate=learning_rate,
                                               momentum=0.9,
                                               use_nesterov=False)
    elif params['optimizer'] == 'rms':
        optimizer = tf.train.RMSPropOptimizer(learning_rate=params['lr'],
                                              momentum=0.9)
    elif params['optimizer'] == 'sgd':
        optimizer = tf.train.MomentumOptimizer(learning_rate=params['lr'],
                                               momentum=0.9,
                                               use_nesterov=True)
    elif params['optimizer'] == 'adam':
        optimizer = tf.train.AdamOptimizer(learning_rate=params['lr'])
    else:
        assert 'No optimizer defined in params!'

    train_op = optimizer.minimize(loss, global_step=tf.train.get_global_step())

    return tf.estimator.EstimatorSpec(mode, loss=loss, train_op=train_op)


# NOTE: steps = (no of ex / batch_size) * no_of_epochs
steps = args['n_epochs'] * len(X_train) // args['batch_size']
assert steps > 0

print('Steps: {}'.format(steps))

run_config = tf.estimator.RunConfig(save_checkpoints_steps=500)

timestamp = '-{:%Y-%m-%d-%H:%M:%S}'.format(datetime.datetime.now())
exp_name = args['exp_name'].lower().replace(' ', '_') + timestamp

args['n_images'] = len(X_train)
args['n_val_images'] = len(X_val)

estimator = tf.estimator.Estimator(
    model_fn=model_fn,
    model_dir=os.path.join(os.getcwd(), 'data', 'model', exp_name),
    params=args,
    config=run_config)

tensors_to_log = {"probabilities": "softmax_tensor"}
logging_hook = tf.train.LoggingTensorHook(tensors=tensors_to_log,
                                          every_n_iter=50)

# train_spec = tf.estimator.TrainSpec(
#     input_fn=lambda: tiny_imagenet.input_fn(params=args,
#                                             mode=tf.estimator.ModeKeys.TRAIN),
#     max_steps=steps)
#
# # Evaluation happens after a checkpoint is created.
# # See: https://www.tensorflow.org/guide/checkpoints#checkpointing_frequency
# eval_spec = tf.estimator.EvalSpec(
#     input_fn=lambda: tiny_imagenet.input_fn(params=args,
#                                             mode=tf.estimator.ModeKeys.EVAL),
#     steps=args['n_val_images'] // args['batch_size'],
#     start_delay_secs=0,  # start evaluating every 60 seconds
#     throttle_secs=0)

train_spec = tf.estimator.TrainSpec(
    input_fn=lambda: caltech_256.input_fn(X_train,
                                          y_train,
                                          params=args,
                                          mode=tf.estimator.ModeKeys.TRAIN),
    max_steps=steps)

eval_spec = tf.estimator.EvalSpec(
    input_fn=lambda: caltech_256.input_fn(X_val,
                                          y_val,
                                          params=args,
                                          mode=tf.estimator.ModeKeys.EVAL),
    steps=len(X_val) // args['batch_size'],
    start_delay_secs=0,  # start evaluating every 60 seconds
    throttle_secs=0)

tf.estimator.train_and_evaluate(estimator, train_spec, eval_spec)
