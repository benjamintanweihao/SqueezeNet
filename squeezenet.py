from tensorflow import keras


def fire_module(x, nb_filters, name):
    (squeeze_nb_filters, expand_1x1_nb_filters, expand_3x3_nb_filters) = nb_filters

    squeeze = keras.layers.Conv2D(filters=squeeze_nb_filters,
                                  kernel_size=1,
                                  padding='same',
                                  activation=keras.activations.relu,
                                  name=name + '/squeeze')(x)

    expand_1x1 = keras.layers.Conv2D(filters=expand_1x1_nb_filters,
                                     kernel_size=1,
                                     padding='same',
                                     activation=keras.activations.relu,
                                     name=name + '/expand_1_1')(squeeze)

    expand_3x3 = keras.layers.Conv2D(filters=expand_3x3_nb_filters,
                                     kernel_size=3,
                                     padding='same',
                                     activation=keras.activations.relu,
                                     name=name + '/expand_3_3')(squeeze)

    return keras.layers.concatenate([expand_1x1, expand_3x3], axis=3, name=name + '/concat')


def SqueezeNet(input_shape=None, classes=None):

    input_tensor = keras.layers.Input(shape=input_shape)

    x = keras.layers.Conv2D(filters=96,
                            kernel_size=7,
                            strides=2,
                            padding='same',
                            name='conv1')(input_tensor)

    x = keras.layers.MaxPooling2D(pool_size=3,
                                  strides=2,
                                  name='maxpool1')(x)

    x = fire_module(x, (16, 64, 64), name='fire2')
    x = fire_module(x, (16, 64, 64), name='fire3')
    x = fire_module(x, (32, 128, 128), name='fire4')

    x = keras.layers.MaxPooling2D(pool_size=3,
                                  strides=2,
                                  padding='same',
                                  name='maxpool4')(x)

    x = fire_module(x, (32, 128, 128), name='fire5')
    x = fire_module(x, (48, 192, 192), name='fire6')
    x = fire_module(x, (48, 192, 192), name='fire7')
    x = fire_module(x, (64, 256, 256), name='fire8')

    x = keras.layers.MaxPooling2D(pool_size=3,
                                  strides=2,
                                  name='maxpool8')(x)

    x = fire_module(x, (64, 256, 256), name='fire9')

    x = keras.layers.Conv2D(filters=classes,
                            kernel_size=1,
                            strides=1,
                            padding='same',
                            name='conv10')(x)

    x = keras.layers.AveragePooling2D(13, strides=1)(x)

    model = keras.Model(input_tensor, x, name='squeezenet')

    return model

