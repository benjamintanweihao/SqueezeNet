from tensorflow import keras
import random


def fire_module(x, nb_squeeze_filters, nb_expand_filters, name):
    squeeze = keras.layers.Conv2D(filters=nb_squeeze_filters,
                                  kernel_size=1,
                                  kernel_initializer=keras.initializers.he_normal(random.randint(0, 1000)),

                                  padding='same',
                                  activation=keras.activations.elu,
                                  name=name + '/squeeze')(x)

    squeeze = keras.layers.BatchNormalization(name=name+'_bn')(squeeze)

    expand_1x1 = keras.layers.Conv2D(filters=nb_expand_filters,
                                     kernel_size=1,
                                     kernel_initializer=keras.initializers.he_normal(random.randint(0, 1000)),
                                     padding='same',
                                     activation=keras.activations.elu,
                                     name=name + '/e1x1')(squeeze)

    expand_3x3 = keras.layers.Conv2D(filters=nb_expand_filters,
                                     kernel_size=3,
                                     padding='same',
                                     kernel_initializer=keras.initializers.he_normal(random.randint(0, 1000)),
                                     activation=keras.activations.elu,
                                     name=name + '/e3x3')(squeeze)

    return keras.layers.Concatenate(name=name + '/concat', axis=3)([expand_1x1, expand_3x3])


def SqueezeNet(features, classes=None, training=True):
    x = keras.layers.Conv2D(filters=32,
                            kernel_size=7,
                            strides=2,
                            kernel_initializer=keras.initializers.he_normal(random.randint(0, 1000)),
                            padding='same',
                            activation=keras.activations.elu,
                            name='conv1')(features)

    x = keras.layers.MaxPooling2D(pool_size=2,
                                  strides=1,
                                  padding='valid',
                                  name='maxpool1')(x)

    x = fire_module(x, 16, 64, name='fire2')
    x = fire_module(x, 16, 64, name='fire3')
    x = fire_module(x, 32, 128, name='fire4')

    x = keras.layers.MaxPooling2D(pool_size=2,
                                  strides=1,
                                  padding='same',
                                  name='maxpool4')(x)

    x = fire_module(x, 32, 128, name='fire5')
    x = fire_module(x, 48, 192, name='fire6')
    x = fire_module(x, 48, 192, name='fire7')
    x = fire_module(x, 64, 256, name='fire8')

    x = keras.layers.MaxPooling2D(pool_size=2,
                                  strides=1,
                                  padding='same',
                                  name='maxpool8')(x)

    x = fire_module(x, 64, 256, name='fire9')

    x = keras.layers.Dropout(0.5 if training else 0.0)(x)

    x = keras.layers.Conv2D(filters=classes,
                            kernel_size=1,
                            strides=1,
                            padding='same',
                            kernel_initializer=keras.initializers.he_normal(random.randint(0, 1000)),
                            activation=keras.activations.elu,
                            name='conv10')(x)

    x = keras.layers.BatchNormalization(name='conv10_bn')(x)

    logits = keras.layers.GlobalAveragePooling2D()(x)

    return logits
