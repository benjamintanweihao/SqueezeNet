import os
import tensorflow as tf
import argparse
import datetime
from input_pipeline.tiny_imagenet import input_fn
from squeezenet import SqueezeNet

tf.logging.set_verbosity(tf.logging.INFO)


def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


ap = argparse.ArgumentParser(description='SqueezeNet')
ap.add_argument('--n_images', help="Total number of images", type=int, default=100000)
ap.add_argument('--n_val_images', help="Total number of validation images", type=int, default=10000)
ap.add_argument('--n_classes', help="Total number of classes", type=int, default=200)
ap.add_argument('--n_epochs', help="Total number of epochs", type=int, required=True)
ap.add_argument('--batch_size', help="Batch size", type=int, required=True)
ap.add_argument('--optimizer', help="Optimizer. One of: [rms|poly|sgd|adam]", default='sgd')
ap.add_argument('--lr', help="Learning rate", required=True, type=float)
ap.add_argument('--pool_size', help="Pool Size of SqueezeNet", type=int, default=3)
ap.add_argument('--include_bn', help="Include BN layer after CONV", type=str2bool, default=False)
ap.add_argument('--activation', help="Activation function to use [relu|elu]", default='relu')
ap.add_argument('--exp_name', help="Experiment name", required=True)

args = vars(ap.parse_args())
args['input_shape'] = (64, 64, 3)


def model_fn(features, labels, mode, params):
    # logits.shape == (?, n_classes)
    logits = SqueezeNet(features,
                        classes=params['n_classes'],
                        training=(mode == tf.estimator.ModeKeys.TRAIN),
                        pool_size=params['pool_size'],
                        include_bn=params['include_bn'])

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
                                               use_nesterov=False)
    elif params['optimizer'] == 'adam':
        optimizer = tf.train.AdamOptimizer(learning_rate=params['lr'])
    else:
        assert 'No optimizer defined in params!'

    train_op = optimizer.minimize(loss, global_step=tf.train.get_global_step())

    return tf.estimator.EstimatorSpec(mode, loss=loss, train_op=train_op)


# NOTE: steps = (no of ex / batch_size) * no_of_epochs
steps = args['n_epochs'] * args['n_images'] // args['batch_size']

print('Steps: {}'.format(steps))

run_config = tf.estimator.RunConfig(save_checkpoints_steps=500)

timestamp = '-{:%Y-%m-%d-%H:%M:%S}'.format(datetime.datetime.now())
exp_name = args['exp_name'].lower().replace(' ', '_') + timestamp

estimator = tf.estimator.Estimator(
    model_fn=model_fn,
    model_dir=os.path.join(os.getcwd(), 'data', 'model', exp_name),
    params=args,
    config=run_config)

tensors_to_log = {"probabilities": "softmax_tensor"}
logging_hook = tf.train.LoggingTensorHook(tensors=tensors_to_log,
                                          every_n_iter=50)

train_spec = tf.estimator.TrainSpec(
    input_fn=lambda: input_fn(params=args,
                              mode=tf.estimator.ModeKeys.TRAIN),
    max_steps=steps)

# Evaluation happens after a checkpoint is created.
# See: https://www.tensorflow.org/guide/checkpoints#checkpointing_frequency
eval_spec = tf.estimator.EvalSpec(
    input_fn=lambda: input_fn(params=args,
                              mode=tf.estimator.ModeKeys.EVAL),
    steps=args['n_val_images'] // args['batch_size'],
    start_delay_secs=0,  # start evaluating every 60 seconds
    throttle_secs=0)

tf.estimator.train_and_evaluate(estimator, train_spec, eval_spec)
