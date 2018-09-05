import tensorflow as tf
from tensorflow import keras


class SqueezeNet(keras.Model):

    def __init__(self):
        super(SqueezeNet, self).__init__(name='squeezenet')

        self.conv1 = keras.layers.Conv2D(filters=96,
                                         kernel_size=7,
                                         strides=2,
                                         name='conv1')

        self.maxpool1 = keras.layers.MaxPooling2D(pool_size=3,
                                                  strides=2,
                                                  name='maxpool1')

        self.fire2 = self.fire((16, 64, 64), name='fire2')
        self.fire3 = self.fire((16, 64, 64), name='fire3')
        self.fire4 = self.fire((32, 128, 128), name='fire4')

        self.maxpool4 = keras.layers.MaxPooling2D(pool_size=3,
                                                  strides=2,
                                                  padding='same',
                                                  name='maxpool4')

        self.fire5 = self.fire((32, 128, 128), name='fire5')
        self.fire6 = self.fire((48, 192, 192), name='fire6')
        self.fire7 = self.fire((48, 192, 192), name='fire7')
        self.fire8 = self.fire((64, 256, 256), name='fire8')

        self.maxpool8 = keras.layers.MaxPooling2D(pool_size=3,
                                                  strides=2,
                                                  name='maxpool8')

        self.fire9 = self.fire((64, 256, 256), name='fire9')

        self.conv10 = keras.layers.Conv2D(filters=1000,
                                          kernel_size=1,
                                          strides=1,
                                          padding='same',
                                          name='conv10')

        self.avgpool10 = keras.layers.AveragePooling2D(13, strides=1)

    def call(self, inputs, training=False, **kwargs):
        conv1 = self.conv1(inputs)
        conv1 = self.maxpool1(conv1)
        fire2 = self.fire2(conv1)
        fire3 = self.fire3(fire2)
        fire4 = self.fire4(fire3)
        fire4 = self.maxpool4(fire4)
        fire5 = self.fire5(fire4)
        fire6 = self.fire6(fire5)
        fire7 = self.fire7(fire6)
        fire8 = self.fire8(fire7)
        fire8 = self.maxpool8(fire8)
        fire9 = self.fire9(fire8)
        conv10 = self.conv10(fire9)

        return self.avgpool10(conv10)

    # TODO: Convert this to as a legit layer
    def fire(self, nb_filters, name):
        (squeeze_nb_filters, expand_1x1_nb_filters, expand_3x3_nb_filters) = nb_filters

        assert squeeze_nb_filters < expand_1x1_nb_filters + expand_3x3_nb_filters, \
            'Invalid number of filters. See Section 3.1'

        def _fire(inputs):
            x = self.squeeze(squeeze_nb_filters, prefix=name)(inputs)
            return self.expand((expand_1x1_nb_filters, expand_3x3_nb_filters), prefix=name)(x)

        return _fire

    @staticmethod
    def squeeze(nb_filters, prefix):
        return lambda inputs: keras.layers.Conv2D(filters=nb_filters,
                                                  kernel_size=1,
                                                  padding='same',
                                                  activation=keras.activations.relu,
                                                  name=prefix + '/squeeze')(inputs)

    @staticmethod
    def expand(nb_filters, prefix):
        assert len(nb_filters) == 2
        (nb_filters_1x1, nb_filters_3x3) = nb_filters

        def expand_1x1(inputs):
            return keras.layers.Conv2D(filters=nb_filters_1x1,
                                       kernel_size=1,
                                       padding='same',
                                       activation=keras.activations.relu,
                                       name=prefix + '/expand_1x1')(inputs)

        def expand_3x3(inputs):
            return keras.layers.Conv2D(filters=nb_filters_3x3,
                                       kernel_size=3,
                                       padding='same',
                                       activation=keras.activations.relu,
                                       name=prefix + '/expand_3x3')(inputs)

        return lambda inputs: keras.layers.concatenate([expand_1x1(inputs), expand_3x3(inputs)], axis=3)


inputs = keras.layers.Input(shape=(224, 224, 3))
model = SqueezeNet()
# NOTE: Need to call the model to initialize the weights
# NOTE: so that we can see the summary
model(inputs)
print(model.summary())




