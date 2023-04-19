import calssdcgan
import CIFARData
import tensorflow as tf
import numpy as np

BUFFER_SIZE = 60000
BATCH_SIZE = 256
EPOCHS = 50
noise_dim = 100
num_examples_to_generate = 16
input_shape=[32, 32, 3]

CIFAR = CIFARData.CIFARDATA()
data = np.array(CIFAR.GetDataSet())
(train_images, train_labels), (_, _) = tf.keras.datasets.mnist.load_data()
print(type(data))
print(data.shape)
data = data.reshape(
    data.shape[1] * data.shape[0], 32, 32, 3).astype('float32')

train_images = train_images.reshape(
    train_images.shape[0], 28, 28, 1).astype('float32')


model = calssdcgan.class_dcgan(
    data,
    BUFFER_SIZE,
    BATCH_SIZE,
    EPOCHS,
    noise_dim,
    num_examples_to_generate,
    input_shape
)

model.train()