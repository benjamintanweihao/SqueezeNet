import os
import tensorflow as tf
from dataset.tiny_imagenet import TinyImageNet
from squeezenet import SqueezeNet

tf.logging.set_verbosity(tf.logging.INFO)


DATASET = TinyImageNet

X_train, X_val, y_train, y_val = DATASET.train_test_split()

print('X train = {}'.format(len(X_train)))
print('X val = {}'.format(len(X_val)))

PARAMS = DATASET.params(X_train, X_val)


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
    accuracy = tf.metrics.accuracy(labels, predictions, name='accuracy_1')

    metrics = {
        "accuracy": accuracy,
        "recall_at_5": tf.metrics.recall_at_k(labels, logits, 5),
        "recall_at_1": tf.metrics.recall_at_k(labels, logits, 1)
    }

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
                                               use_nesterov=True)
    elif params['optimizer'] == 'adam':
        optimizer = tf.train.AdamOptimizer(learning_rate=params['lr'])
    else:
        assert 'No optimizer defined in params!'

    train_op = optimizer.minimize(loss, global_step=tf.train.get_global_step())

    return tf.estimator.EstimatorSpec(mode, loss=loss, train_op=train_op)


# NOTE: steps = (no of ex / batch_size) * no_of_epochs
steps = PARAMS['n_epochs'] * len(X_train) // PARAMS['batch_size']
assert steps > 0

print('Steps: {}'.format(steps))

run_config = tf.estimator.RunConfig(save_checkpoints_steps=100)

# timestamp = '-{:%Y-%m-%d-%H:%M:%S}'.format(datetime.datetime.now())
# exp_name = args['exp_name'].lower().replace(' ', '_') + timestamp




estimator = tf.estimator.Estimator(
    model_fn=model_fn,
    model_dir=os.path.join(os.getcwd(), 'data', 'model'),
    # model_dir=os.path.join(os.getcwd(), 'data', 'model', exp_name),
    params=DATASET.params(X_train, X_val),
    config=run_config)


train_spec = tf.estimator.TrainSpec(
    input_fn=lambda: DATASET.input_fn(X_train,
                                      y_train,
                                      params=DATASET.params(X_train, X_val),
                                      mode=tf.estimator.ModeKeys.TRAIN),
    max_steps=steps)

eval_spec = tf.estimator.EvalSpec(
    input_fn=lambda: DATASET.input_fn(X_val,
                                      y_val,
                                      params=DATASET.params(X_train, X_val),
                                      mode=tf.estimator.ModeKeys.EVAL),
    steps=len(X_val) // PARAMS['batch_size'],
    start_delay_secs=30,  # start evaluating every 30 seconds
    throttle_secs=0, hooks=[])

tf.estimator.train_and_evaluate(estimator, train_spec, eval_spec)
