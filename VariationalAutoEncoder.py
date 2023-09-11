import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import tensorflow as tf
from tensorflow import keras
import numpy as np


####################################################
# References 
# - https://keras.io/examples/generative/vae/
# - https://arxiv.org/pdf/1606.05908.pdf
#
####################################################



class Sampling(keras.layers.Layer):
    """
        Given a latent vector z's probability distribution (mean and log-covariance)
        this class creates a layer to sample from it - P(X|z)
    """

    def __call__(self, inputs):
        z_mean, z_log_var = inputs
        batch = tf.shape(z_mean)[0]
        dim = tf.shape(z_mean)[1]
        epsilon = tf.keras.backend.random_normal(shape=(batch, dim))
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon


class encoder():
    """
        This builds a general 6-layered encoder NN model
        It uses the relu activation function and
        Dense layers by default.
    """

    def __init__(self, in_shape, latent_dim, hl1_num_nodes, hl2_num_nodes, hl3_num_nodes):
        self.in_shape = in_shape
        self.latent_dim = latent_dim
        self.hl1_num_nodes = hl1_num_nodes
        self.hl2_num_nodes = hl2_num_nodes
        self.hl3_num_nodes = hl3_num_nodes
        self.model = None

    def make_model(self):
        inp_layer = keras.Input(shape=self.in_shape)
        # Conv-1
        x = keras.layers.Conv2D(64,4,strides=4,activation="relu",padding= "same",input_shape=self.in_shape)(inp_layer)
        #x = keras.layers.MaxPooling2D(2)(x)
        # Conv-2
        x = keras.layers.Conv2D(128,4,strides=4,activation="relu",padding= "same",input_shape=self.in_shape)(x)
        #x = keras.layers.MaxPooling2D(2)(x)
        # Conv-3
        x = keras.layers.Conv2D(256,(2,8),strides=(2,8),activation="relu",padding= "same",input_shape=self.in_shape)(inp_layer)
        #x = keras.layers.MaxPooling2D(2)(x)
        x = keras.layers.Dense(256, activation="linear")(x)
        x = keras.layers.Flatten()(x)
        # x = keras.layers.Dense(self.hl1_num_nodes, activation="relu")(x)
        # x = keras.layers.Dense(self.hl2_num_nodes, activation="relu")(x)
        # x = keras.layers.Dense(self.hl3_num_nodes, activation="relu")(x)

        z_mean = keras.layers.Dense(self.latent_dim, name="z_mean")(x)
        z_log_var = keras.layers.Dense(self.latent_dim, name="z_log_var")(x)

        z = Sampling()([z_mean, z_log_var])
        model = keras.Model(inp_layer, [z_mean,z_log_var,z], name="encoder")
        self.model = model
        
        return model

    def my_summary(self):
        return self.model.summary()


class decoder():
    """
        This builds a general 5-layered decoder NN model
        It uses the relu activation function and
        Dense layers by default.
    """

    def __init__(self, out_shape, latent_dim, hl1_num_nodes, hl2_num_nodes, hl3_num_nodes):
        self.out_shape = out_shape
        self.latent_dim = latent_dim
        self.hl1_num_nodes = hl1_num_nodes
        self.hl2_num_nodes = hl2_num_nodes
        self.hl3_num_nodes = hl3_num_nodes
        self.model = None
    
    def make_model(self):
        latent_inputs = keras.Input(shape=(self.latent_dim,))

        # If we want to use de-convolution
        x = keras.layers.Dense(3072, activation="linear")(latent_inputs)
        x = keras.layers.Reshape((3,4,256))(x)
        x = keras.layers.Conv2DTranspose(filters=128,kernel_size=(2,2),activation="relu", strides=(2,2), padding="same")(x)
        x = keras.layers.Conv2DTranspose(filters=64,kernel_size=(2,2),activation="relu", strides=(2,2), padding="same")(x)
        x = keras.layers.Conv2DTranspose(filters=32,kernel_size=(2,2),activation="relu", strides=(2,2), padding="same")(x)
        x = keras.layers.Conv2DTranspose(filters=1,kernel_size=(2,2),activation="sigmoid", strides=(2,2), padding="same")(x)
        x = keras.layers.Flatten()(x)
        # x = keras.layers.Dense(self.out_shape[0]*self.out_shape[1], activation="sigmoid")(x)

        # # If we want to use Dense Layers
        # x = keras.layers.Dense(self.hl1_num_nodes,activation="relu")(latent_inputs)
        # x = keras.layers.Dense(self.hl2_num_nodes,activation="relu")(x)
        # x = keras.layers.Dense(self.hl2_num_nodes,activation="relu")(x)
        # x = keras.layers.Dense(self.hl3_num_nodes,activation="relu")(x)

        # x = keras.layers.Dense(self.out_shape[0]*self.out_shape[1],activation="sigmoid")(x)

        decoder_outputs = keras.layers.Reshape(self.out_shape)(x)

        model = keras.Model(latent_inputs,decoder_outputs,name="decoder")
        self.model = model

        return model
    
    def my_summary(self):
        return self.model.summary()

class VAE(keras.Model):
    def __init__(self, encoder, decoder, **kwargs):
        super(VAE, self).__init__(**kwargs)
        self.encoder = encoder
        self.decoder = decoder
        self.total_loss_tracker = keras.metrics.Mean(name="total_loss")
        self.reconstruction_loss_tracker = keras.metrics.Mean(name="reconstruction_loss")
        self.kl_loss_tracker = keras.metrics.Mean(name="kl_loss")

    @property
    def metrics(self):
        return [
            self.total_loss_tracker,
            self.reconstruction_loss_tracker,
            self.kl_loss_tracker,
        ]

    def train_step(self, data):
        with tf.GradientTape() as tape:
            z_mean, z_log_var, z = self.encoder(data)
            reconstruction = self.decoder(z)
            reconstruction_loss = tf.reduce_mean(
                tf.reduce_sum(
                    keras.losses.binary_crossentropy(data, reconstruction), axis=1
                )
            )

            kl_loss = -0.5 * (1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var))
            kl_loss = tf.reduce_mean(tf.reduce_sum(kl_loss, axis=1))
            total_loss = reconstruction_loss + kl_loss

        grads = tape.gradient(total_loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))
        self.total_loss_tracker.update_state(total_loss)
        self.reconstruction_loss_tracker.update_state(reconstruction_loss)
        self.kl_loss_tracker.update_state(kl_loss)
        return {
            "loss": self.total_loss_tracker.result(),
            "reconstruction_loss": self.reconstruction_loss_tracker.result(),
            "kl_loss": self.kl_loss_tracker.result(),
        }
