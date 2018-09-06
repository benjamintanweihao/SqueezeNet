from tensorflow import keras


class FireModule(keras.Model):

    def __init__(self, nb_filters, name):
        super(FireModule, self).__init__(name=name)
        (squeeze_nb_filters, expand_1x1_nb_filters, expand_3x3_nb_filters) = nb_filters

        self.squeeze = keras.layers.Conv2D(filters=squeeze_nb_filters,
                                           kernel_size=1,
                                           padding='same',
                                           activation=keras.activations.relu,
                                           name=name + '/squeeze')

        self.expand_1x1 = keras.layers.Conv2D(filters=expand_1x1_nb_filters,
                                              kernel_size=1,
                                              padding='same',
                                              activation=keras.activations.relu,
                                              name=name + '/expand_1_1')

        self.expand_3x3 = keras.layers.Conv2D(filters=expand_3x3_nb_filters,
                                              kernel_size=3,
                                              padding='same',
                                              activation=keras.activations.relu,
                                              name=name + '/expand_3_3')

        self.concat = keras.layers.Concatenate(axis=3, name=name + '/concat')

    def call(self, inputs, training=False, **kwargs):
        squeeze = self.squeeze(inputs)
        expand_1x1 = self.expand_1x1(squeeze)
        expand_3x3 = self.expand_3x3(squeeze)

        return self.concat([expand_1x1, expand_3x3])

    def get_config(self):
        pass
