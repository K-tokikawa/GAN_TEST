from tensorflow.keras.datasets import cifar10  # tf.kerasを使う場合（通常）
import calssdcgan
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import readmovie
import os

physical_devices = tf.config.list_physical_devices('GPU')
if len(physical_devices) > 0:
    for device in physical_devices:
        tf.config.experimental.set_memory_growth(device, True)
        print('{} memory growth: {}'.format(
            device, tf.config.experimental.get_memory_growth(device)))
else:
    print("Not enough GPU hardware devices available")

datapath = '.\\data.npy'
with tf.device('/cpu:0'):
    if (os.path.exists(datapath)):
        flames = np.load(datapath)
        print('VideoData Load')
    else :
        moviepath = '.\\1785097565.mp4'
        movie = readmovie.ReadMovie(moviepath, (256, 256))
        flames = movie.readvideo()
        print(type(flames))
        print(type(flames[0]))
        np.save(datapath, flames)
        print('VideoData Save')

BUFFER_SIZE = 60000
BATCH_SIZE = 256
EPOCHS = 100
noise_dim = 100
num_examples_to_generate = 16
input_shape = (64, 64, 3)
# from keras.datasets import cifar10  # tf.kerasではなく、Kerasを使う必要がある場合はこちらを有効にする

(train_images, train_labels), (test_images, test_labels) = cifar10.load_data()
# print(type(train_images))
# print(train_images.shape)
# print(type(train_images[0]))
# print(train_images[0].shape)
# print(type(flames[0]))
# print(flames[0].shape)
# print(type(flames))
# print(flames.shape)

model = calssdcgan.class_dcgan(
    flames,
    BUFFER_SIZE,
    BATCH_SIZE,
    EPOCHS,
    noise_dim,
    num_examples_to_generate,
    input_shape
)

# model.train()