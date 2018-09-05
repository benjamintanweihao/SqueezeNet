import tensorflow as tf
from tensorflow import keras


class SqueezeNet(keras.Model):

    def __init__(self):
        super(SqueezeNet, self).__init__(name='squeezenet')

        self.conv1 = keras.layers.Conv2D(filters=96,
                                         kernel_size=7,
                                         strides=2,
                                         padding='same',
                                         name='conv1')

        self.maxpool1 = keras.layers.MaxPooling2D(pool_size=3,
                                                  strides=2,
                                                  name='maxpool1')

        self.maxpool4 = keras.layers.MaxPooling2D(pool_size=3,
                                                  strides=2,
                                                  padding='same',
                                                  name='maxpool4')

        self.maxpool8 = keras.layers.MaxPooling2D(pool_size=3,
                                                  strides=2,
                                                  name='maxpool8')

        self.conv10 = keras.layers.Conv2D(filters=1000,
                                          kernel_size=1,
                                          strides=1,
                                          padding='same',
                                          name='conv10')

        self.avgpool10 = keras.layers.AveragePooling2D(13, strides=1)

    def call(self, inputs, training=False, **kwargs):
        conv1 = self.conv1(inputs)
        conv1 = self.maxpool1(conv1)
        fire2 = self.fire(conv1, (16, 64, 64), name='fire2')
        fire3 = self.fire(fire2, (16, 64, 64), name='fire3')
        fire4 = self.fire(fire3, (32, 128, 128), name='fire4')
        fire4 = self.maxpool4(fire4)
        fire5 = self.fire(fire4, (32, 128, 128), name='fire5')
        fire6 = self.fire(fire5, (48, 192, 192), name='fire6')
        fire7 = self.fire(fire6, (48, 192, 192), name='fire7')
        fire8 = self.fire(fire7, (64, 256, 256), name='fire8')
        fire8 = self.maxpool8(fire8)
        fire9 = self.fire(fire8, (64, 256, 256), name='fire9')
        conv10 = self.conv10(fire9)

        return self.avgpool10(conv10)

    @staticmethod
    def fire(inputs, nb_filters, name):
        (squeeze_nb_filters, expand_1x1_nb_filters, expand_3x3_nb_filters) = nb_filters

        assert squeeze_nb_filters < expand_1x1_nb_filters + expand_3x3_nb_filters, \
            'Invalid number of filters. See Section 3.1'

        squeeze = keras.layers.Conv2D(filters=squeeze_nb_filters,
                                      kernel_size=1,
                                      padding='same',
                                      activation=keras.activations.relu,
                                      name=name + '/squeeze')(inputs)

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


input_layer = keras.layers.Input(shape=(224, 224, 3))
model = SqueezeNet()
# NOTE: Need to call the model to initialize the weights
# NOTE: so that we can see the summary
model(input_layer)
print(model.summary())
