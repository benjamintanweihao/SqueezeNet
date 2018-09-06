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


def train_input_fn(features, labels, batch_size):
    train_dataset = tf.data.Dataset.from_tensor_slices((features, labels))
    train_dataset = train_dataset.shuffle(1000).repeat().batch(batch_size)

    return train_dataset

# TODO: Remove this later. This is just for forcing TensorFlow to use the CPU
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = ""

model = SqueezeNet()
model.compile(optimizer=tf.train.AdamOptimizer(0.001),
              loss=keras.losses.categorical_crossentropy,
              metrics=[keras.metrics.categorical_accuracy])

estimator = keras.estimator.model_to_estimator(keras_model=model)
estimator.train(input_fn=lambda: train_input_fn(train_images, train_labels, batch_size=128))
