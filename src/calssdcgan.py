from IPython import display
import time
from tensorflow.keras import layers
import tensorflow as tf
import PIL
import numpy as np
import matplotlib.pyplot as plt
import imageio
import glob
import tensorflow_docs.vis.embed as embed
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

class class_dcgan:
    def __init__(
            self,
            data,
            BUFFER_SIZE,
            BATCH_SIZE,
            EPOCHS,
            noise_dim,
            num_examples_to_generate,
            input_shape) -> None:
        self.BUFFER_SIZE = BUFFER_SIZE
        self.BATCH_SIZE = BATCH_SIZE

        self.EPOCHS = EPOCHS
        self.noise_dim = noise_dim
        self.num_examples_to_generate = num_examples_to_generate

        self.input_shape = input_shape

        # You will reuse this seed overtime (so it's easier)
        # to visualize progress in the animated GIF)

        num_attenmps = 1
        while os.path.exists('.\\image\\{:d}'.format(num_attenmps)):
            num_attenmps = num_attenmps + 1
        self.num_attenmps = num_attenmps
        os.mkdir('.\\image\\{:d}'.format(self.num_attenmps))

        train_images = (data - 127.5) / 127.5
        self.train_dataset = tf.data.Dataset.from_tensor_slices(
            train_images).shuffle(self.BUFFER_SIZE).batch(self.BATCH_SIZE)

        self.generator = self.__make_generator_model()
        noise = tf.random.normal([1, 100])
        generated_image = self.generator(noise, training=False)


        self.discriminator = self.__make_discriminator_model()
        self.decision = self.discriminator(generated_image)

        self.cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)

        self.generator_optimizer = tf.keras.optimizers.Adam(1e-4)
        self.discriminator_optimizer = tf.keras.optimizers.Adam(1e-4)

        checkpoint_dir = './training_checkpoints'
        self.checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
        self.checkpoint = tf.train.Checkpoint(generator_optimizer=self.generator_optimizer,
                                        discriminator_optimizer=self.discriminator_optimizer,
                                        generator=self.generator,
                                        discriminator=self.discriminator)
        self.seed = tf.random.normal([self.num_examples_to_generate, self.noise_dim])


    def __make_generator_model(self):
        model = tf.keras.Sequential()
        model.add(layers.Dense(8*8*256, use_bias=False, input_shape=(100,)))
        model.add(layers.BatchNormalization())
        model.add(layers.LeakyReLU())

        model.add(layers.Reshape((8, 8, 256)))
        # Note: None is the batch size
        assert model.output_shape == (None, 8, 8, 256)

        model.add(layers.Conv2DTranspose(
            128, (5, 5), strides=(1, 1), padding='same', use_bias=False,))
        assert model.output_shape == (None, 8, 8, 128)
        model.add(layers.BatchNormalization())
        model.add(layers.LeakyReLU())

        model.add(layers.Conv2DTranspose(
            64, (5, 5), strides=(2, 2), padding='same', use_bias=False))
        assert model.output_shape == (None, 16, 16, 64)
        model.add(layers.BatchNormalization())
        model.add(layers.LeakyReLU())

        model.add(layers.Conv2DTranspose(3, (5, 5), strides=(2, 2),
                padding='same', use_bias=False, activation='tanh'))
        assert model.output_shape == (None, 32, 32, 3)

        return model


    def __make_discriminator_model(self):
        model = tf.keras.Sequential()
        model.add(layers.Conv2D(64, (5, 5), strides=(2, 2), padding='same',
                                input_shape=self.input_shape))
        model.add(layers.LeakyReLU())
        model.add(layers.Dropout(0.3))

        model.add(layers.Conv2D(128, (5, 5), strides=(2, 2), padding='same'))
        model.add(layers.LeakyReLU())
        model.add(layers.Dropout(0.3))

        model.add(layers.Flatten())
        model.add(layers.Dense(1))

        return model
    def __discriminator_loss(self, real_output, fake_output):
        real_loss = self.cross_entropy(tf.ones_like(real_output), real_output)
        fake_loss = self.cross_entropy(tf.zeros_like(fake_output), fake_output)
        total_loss = real_loss + fake_loss
        return total_loss

    def __generator_loss(self, fake_output):
        return self.cross_entropy(tf.ones_like(fake_output), fake_output)


    def __display_image(self, epoch_no):
        return PIL.Image.open('image_at_epoch_{:04d}.png'.format(epoch_no))

    @tf.function
    def __train_step(self, images):
        noise = tf.random.normal([self.BATCH_SIZE, self.noise_dim])

        with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
            generated_images = self.generator(noise, training=True)

            real_output = self.discriminator(images, training=True)
            fake_output = self.discriminator(generated_images, training=True)

            gen_loss = self.__generator_loss(fake_output)
            disc_loss = self.__discriminator_loss(real_output, fake_output)

            gradients_of_generator = gen_tape.gradient(
                gen_loss, self.generator.trainable_variables)
            gradients_of_discriminator = disc_tape.gradient(
                disc_loss, self.discriminator.trainable_variables)

            self.generator_optimizer.apply_gradients(
                zip(gradients_of_generator, self.generator.trainable_variables))
            self.discriminator_optimizer.apply_gradients(
                zip(gradients_of_discriminator, self.discriminator.trainable_variables))


    def __generate_and_save_images(self, model, epoch, test_input):
        # Notice `training` is set to False.
        # This is so all layers run in inference mode (batchnorm).
        predictions = model(test_input, training=False)

        fig = plt.figure(figsize=(4, 4))

        for i in range(predictions.shape[0]):
            plt.subplot(4, 4, i+1)
            plt.imshow(predictions[i, :, :, 0] * 127.5 + 127.5)
            plt.axis('off')
        imagepath = '.\\image\\{:d}'.format(
            self.num_attenmps) + '\\image_at_epoch_{:04d}.png'.format(epoch)
        plt.savefig(imagepath)
        #   plt.show()

    def train(self):
        for epoch in range(self.EPOCHS):
            start = time.time()

            for image_batch in self.train_dataset:
                self.__train_step(image_batch)

            # Produce images for the GIF as you go
            display.clear_output(wait=True)
            self.__generate_and_save_images(self.generator,
                                    epoch + 1,
                                    self.seed)

            # Save the model every 15 epochs
            if (epoch + 1) % 15 == 0:
                self.checkpoint.save(file_prefix=self.checkpoint_prefix)

            print('Time for epoch {} is {} sec'.format(epoch + 1, time.time()-start))

        # Generate after the final epoch
        display.clear_output(wait=True)
        self.__generate_and_save_images(self.generator,
                                self.EPOCHS,
                                self.seed)
        anim_file = '.\\image\\{:d}\\dcgan.gif'.format(self.num_attenmps)


        with imageio.get_writer(anim_file, mode='I') as writer:
            print('.\\image\\{:d}'.format(self.num_attenmps)+'\\image*.png')
            filenames = glob.glob('.\\image\\{:d}'.format(self.num_attenmps)+'\\image*.png')
            filenames = sorted(filenames)
            for filename in filenames:
                print(filename)
                image = imageio.imread(filename)
                writer.append_data(image)


        embed.embed_file(anim_file)

