import tensorflow as tf
from tensorflow import keras


def SqueezeNet(features, classes=None, training=True, activation=tf.nn.relu):
    x = conv_2d(features, 96, 7, 2, activation, 'conv1')

    x = keras.layers.MaxPooling2D(pool_size=[3, 3],
                                  strides=[2, 2],
                                  name='maxpool1')(x)

    x = fire_module(x, 16, 64, name='fire2', activation=activation)
    x = fire_module(x, 16, 64, name='fire3', activation=activation)
    x = fire_module(x, 32, 128, name='fire4', activation=activation)

    x = keras.layers.MaxPooling2D(pool_size=[3, 3],
                                  strides=[2, 2],
                                  name='maxpool4')(x)

    x = fire_module(x, 32, 128, name='fire5', activation=activation)
    x = fire_module(x, 48, 192, name='fire6', activation=activation)
    x = fire_module(x, 48, 192, name='fire7', activation=activation)
    x = fire_module(x, 64, 256, name='fire8', activation=activation)

    x = keras.layers.MaxPooling2D(pool_size=[3, 3],
                                  strides=[2, 2],
                                  name='maxpool8')(x)

    x = fire_module(x, 64, 256, name='fire9', activation=activation)

    if training:
        x = keras.layers.Dropout(0.5)(x)

    x = conv_2d(x, classes, 1, 1, activation, 'conv10')

    # NOTE: Had a lot of trouble with this all because MaxPool was set to 'same'
    # NOTE: instead of 'valid'.
    # x = tf.layers.average_pooling2d(x, pool_size=[13, 13], strides=[1, 1])
    # logits = tf.layers.flatten(x)

    # NOTE: This works better for smaller datasets?
    x = keras.layers.GlobalAveragePooling2D()(x)
    logits = tf.layers.flatten(x)

    return logits


def conv_2d(inputs, filters, kernel_size, strides, activation, name):
    x = keras.layers.Conv2D(filters=filters,
                            kernel_size=[kernel_size, kernel_size],
                            strides=[strides, strides],
                            padding='same',
                            kernel_initializer=keras.initializers.he_normal(seed=4242),
                            bias_initializer=keras.initializers.zeros(),
                            kernel_regularizer=tf.contrib.layers.l2_regularizer(scale=0.0002),
                            trainable=True,
                            activation=activation,
                            name=name)(inputs)

    return keras.layers.BatchNormalization()(x)


def fire_module(x, nb_squeeze_filters, nb_expand_filters, name, activation):
    squeeze = conv_2d(x, nb_squeeze_filters, 1, 1, activation, name + '/squeeze')
    expand_1x1 = conv_2d(squeeze, nb_expand_filters, 1, 1, activation, name + '/e1x1')
    expand_3x3 = conv_2d(squeeze, nb_expand_filters, 3, 1, activation, name + '/e3x3')

    return keras.layers.Concatenate(name=name + '/concat', axis=3)([expand_1x1, expand_3x3])