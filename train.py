import tensorflow as tf
from tensorflow import keras
import os
import numpy as np

from squeezenet import SqueezeNet

train_labels_path = os.path.join(os.getcwd(), 'data/fashionmnist/train-labels-idx1-ubyte')
train_images_path = os.path.join(os.getcwd(), 'data/fashionmnist/train-images-idx3-ubyte')

train_labels = np.frombuffer(open(train_labels_path, 'rb').read(),
                             dtype=np.uint8, offset=8)
train_images = np.frombuffer(open(train_images_path, 'rb').read(),
                             dtype=np.uint8, offset=16).reshape(len(train_labels), 784)

model = SqueezeNet(input_shape=(224, 224, 3), classes=10)
model.compile(optimizer=tf.train.AdamOptimizer(0.001),
              loss=keras.losses.categorical_crossentropy,
              metrics=[keras.metrics.categorical_accuracy])


def train_input_fn(features, labels, batch_size):
    # 'input_1' comes from `print(model.input_names)`
    train_dataset = tf.data.Dataset.from_tensor_slices(({'input_1': features}, labels))
    train_dataset = train_dataset.shuffle(1000).repeat().batch(batch_size)

    return train_dataset


estimator = keras.estimator.model_to_estimator(keras_model=model)
estimator.train(input_fn=lambda: train_input_fn(train_images, train_labels, batch_size=128), steps=2000)
