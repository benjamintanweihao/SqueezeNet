from tensorflow import keras
from fire_module import FireModule


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

        self.fire2 = FireModule((16, 64, 64), name='fire2')
        self.fire3 = FireModule((16, 64, 64), name='fire3')
        self.fire4 = FireModule((32, 128, 128), name='fire4')

        self.maxpool4 = keras.layers.MaxPooling2D(pool_size=3,
                                                  strides=2,
                                                  padding='same',
                                                  name='maxpool4')

        self.fire5 = FireModule((32, 128, 128), name='fire5')
        self.fire6 = FireModule((48, 192, 192), name='fire6')
        self.fire7 = FireModule((48, 192, 192), name='fire7')
        self.fire8 = FireModule((64, 256, 256), name='fire8')

        self.maxpool8 = keras.layers.MaxPooling2D(pool_size=3,
                                                  strides=2,
                                                  name='maxpool8')

        self.fire9 = FireModule((64, 256, 256), name='fire9')

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


input_layer = keras.layers.Input(shape=(224, 224, 3))
model = SqueezeNet()
# NOTE: Need to call the model to initialize the weights
# NOTE: so that we can see the summary
model(input_layer)
print(model.summary())
