from tensorflow import keras


class SqueezeNet:

    def __init__(self):
        pass

    def build(self, inputs):
        conv1 = keras.layers.Conv2D(filters=96,
                                    kernel_size=7,
                                    strides=2,
                                    name='conv1')

        maxpool1 = keras.layers.MaxPooling2D(pool_size=3,
                                             strides=2,
                                             name='maxpool1')

        x = conv1(inputs)
        x = maxpool1(x)
        fire2 = self.fire(x, (16, 64, 64))
        fire3 = self.fire(fire2, (16, 64, 64))
        fire4 = self.fire(fire3, (32, 128, 128))

        maxpool4 = keras.layers.MaxPooling2D(pool_size=3,
                                             strides=2,
                                             padding='same',
                                             name='maxpool4')
        fire4 = maxpool4(fire4)
        fire5 = self.fire(fire4, (32, 128, 128))
        fire6 = self.fire(fire5, (48, 192, 192))
        fire7 = self.fire(fire6, (48, 192, 192))
        fire8 = self.fire(fire7, (64, 256, 256))

        maxpool8 = keras.layers.MaxPooling2D(pool_size=3,
                                             strides=2,
                                             name='maxpool8')
        fire8 = maxpool8(fire8)
        fire9 = self.fire(fire8, (64, 256, 256))
        conv10 = keras.layers.Conv2D(filters=1000,
                                     kernel_size=1,
                                     strides=1,
                                     padding='same',
                                     name='conv10')
        conv10 = conv10(fire9)
        avgpool10 = keras.layers.AveragePooling2D(13, strides=1)

        return avgpool10(conv10)

    def fire(self, inputs, nb_filters):
        (squeeze_nb_filters, expand_1x1_nb_filters, expand_3x3_nb_filters) = nb_filters
        x = self.squeeze(inputs, squeeze_nb_filters)
        x = self.expand(x, (expand_1x1_nb_filters, expand_3x3_nb_filters))

        return x

    @staticmethod
    def squeeze(inputs, nb_filters):
        # TODO: Supply a name
        return keras.layers.Conv2D(filters=nb_filters,
                                   kernel_size=1,
                                   padding='same',
                                   activation=keras.activations.relu)(inputs)

    @staticmethod
    def expand(inputs, nb_filters):
        assert len(nb_filters) == 2
        (nb_filters_1x1, nb_filters_3x3) = nb_filters

        expand_1x1 = keras.layers.Conv2D(filters=nb_filters_1x1,
                                         kernel_size=1,
                                         padding='same',
                                         activation=keras.activations.relu)(inputs)

        expand_3x3 = keras.layers.Conv2D(filters=nb_filters_3x3,
                                         kernel_size=3,
                                         padding='same',
                                         activation=keras.activations.relu)(inputs)

        return keras.layers.concatenate([expand_1x1, expand_3x3], axis=3)


input = keras.layers.Input(shape=(224, 224, 3))
model = SqueezeNet()
model.build(input)
