from tensorflow import keras


def fire_module(x, nb_squeeze_filters, nb_expand_filters, name):
    squeeze = keras.layers.Conv2D(filters=nb_squeeze_filters,
                                  kernel_size=1,
                                  padding='same',
                                  activation=keras.activations.relu,
                                  name=name + '/squeeze')(x)

    expand_1x1 = keras.layers.Conv2D(filters=nb_expand_filters,
                                     kernel_size=1,
                                     padding='same',
                                     activation=keras.activations.relu,
                                     name=name + '/e1x1')(squeeze)

    expand_3x3 = keras.layers.Conv2D(filters=nb_expand_filters,
                                     kernel_size=3,
                                     padding='same',
                                     activation=keras.activations.relu,
                                     name=name + '/e3x3')(squeeze)

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
                                  padding='same',
                                  name='maxpool1')(x)

    x = fire_module(x, 16, 64, name='fire2')
    x = fire_module(x, 16, 64, name='fire3')
    x = fire_module(x, 32, 128, name='fire4')

    x = keras.layers.MaxPooling2D(pool_size=3,
                                  strides=2,
                                  padding='same',
                                  name='maxpool4')(x)

    x = fire_module(x, 32, 128, name='fire5')
    x = fire_module(x, 48, 192, name='fire6')
    x = fire_module(x, 48, 192, name='fire7')
    x = fire_module(x, 64, 256, name='fire8')

    x = keras.layers.MaxPooling2D(pool_size=3,
                                  strides=2,
                                  padding='same',
                                  name='maxpool8')(x)

    x = fire_module(x, 64, 256, name='fire9')

    x = keras.layers.Dropout(0.5)(x)

    x = keras.layers.Conv2D(filters=classes,
                            kernel_size=1,
                            strides=1,
                            padding='same',
                            name='conv10')(x)

    x = keras.layers.AveragePooling2D(pool_size=13)(x)
    x = keras.layers.Flatten()(x)
    x = keras.layers.Activation('softmax')(x)

    return keras.Model(input_tensor, x, name='squeezenet')
