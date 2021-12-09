import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input, Dropout
from tensorflow.keras.optimizers import Adam, SGD, RMSprop
from tensorflow.keras import layers
from tensorflow.keras.callbacks import LearningRateScheduler



def step_lr(epoch, lr):
    """
    This function takes the current epoch and learning rate during training and
    divides the learning rate by 5 every 200 epochs. This function serves as
    the main argument for the LearningRateScheduler class. You don't need to
    initialize the function arguments
    """

    if (epoch != 0) & (epoch % 200 == 0): # Change every 200 epochs
        lr = lr / 5

    return lr

#########################################
############### VAE #####################
#########################################
class Sampling(layers.Layer):
    """
    Reparametrization trick. Uses (z_mean, z_log_var) to sample z, the encoding vector
    """

    def call(self, inputs):
        z_mean, z_log_var = inputs
        batch = tf.shape(z_mean)[0]
        dim = tf.shape(z_mean)[1]
        epsilon = K.random_normal(shape=(batch, dim))
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon

    # We use log std (z_log_var) because it is numerically more stable

class VAE(tf.keras.Model):
    """
    This class implements the VAE model
    """
    def __init__(self, encoder=None, decoder=None, clip_gradients=False, **kwargs):
        super(VAE,self).__init__(**kwargs)
        self.encoder = encoder
        self.decoder = decoder
        self.beta = tf.Variable(1, trainable=False, dtype=np.float32)
        self.clip_gradients = clip_gradients

    def train_step(self, data):
        """
        This function defines the calculations made in each step of training. This
        is impplicitly called by VAE.train(). It is NOT supposed to be used outside that.
        Default loss function is MSE (for scRNAseq data).
        """

        if isinstance(data, tuple):
            data = data[0]

        with tf.GradientTape() as tape:
            z_mean, z_log_var, z = self.encoder(data)
            reconstruction = self.decoder(z)

            reconstruction_loss = tf.keras.losses.MSE(data, reconstruction)
            reconstruction_loss *= data.shape[1] # Scale the loss by the dimensionality of the input space
            reconstruction_loss = K.mean(reconstruction_loss)

            # Calculate KL Divergence (KL Loss)
            kl_batch = - .5 * K.sum(1 + z_log_var - K.square(z_mean) - K.exp(z_log_var), axis=-1)
            kl_loss = K.mean(kl_batch)

            # Compute total loss
            total_loss = reconstruction_loss + self.beta.read_value() * kl_loss

        # Calculate and apply gradients
        grads = tape.gradient(total_loss, self.trainable_weights)

        # OPTIONAL: clip gradients
        if self.clip_gradients:
            grads = [tf.clip_by_norm(g, 1.0) for g in grads]

        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))
        self.compiled_metrics.update_state(data, reconstruction)

        return {
            "loss": total_loss,
            "reconstruction_loss": reconstruction_loss,
            "kl_loss": self.beta.read_value() * kl_loss,
            "KL Weight": self.beta
            }

    def predict(self, data):
        """
        Computes reconstruction after encoding the input. This method IS to be called
        by the user.
        """
        z_mean, z_log_var, z = self.encoder(data)
        return self.decoder(z)

    def call(self, inputs):
        """
        Same as predict, but is called IMPLICITLY during training of the model. Use predict
        instead.
        """
        z_mean, z_log_var, z = self.encoder(inputs)
        reconstructed = self.decoder(z)
        # Add KL divergence regularization loss.
        kl_loss = -0.5 * tf.reduce_mean(z_log_var - tf.square(z_mean) - tf.exp(z_log_var) + 1)
        self.add_loss(kl_loss)
        return reconstructed


    def get_encoder(self):
        """Returns the encoder model"""
        return self.encoder

    def get_decoder(self):
        """Returns the decoder model"""
        return self.decoder

    def set_encoder(self, n_input, hidden_structure):
        """
        This method can be used to set up the encoder model if it has not been giving
        when instantiating the scVAE class. Default activation function is PReLU (Parametric ReLU)

        n_input: dimensionality of the original data
        hidden_structure: list with the number of hidden units PER layer. Last item of
            the list is the latent space dimensionality.
            For example: [10, 5, 3, 2] cretes 4 layers from n_input to 2 (the latent space)
        """
        self.n_input = n_input
        encoder_inputs = tf.keras.Input(n_input)
        for units in hidden_structure:
            if units == hidden_structure[0]:
                x = Dense(units=units)(encoder_inputs)
                x = tf.keras.layers.PReLU()(x)

            else:
                x = Dense(units=units)(x)
                x = tf.keras.layers.PReLU()(x)

        z_mean = Dense(hidden_structure[-1], name="z_mean")(x)
        z_log_var = Dense(hidden_structure[-1], name="z_log_var")(x)
        z = Sampling()([z_mean, z_log_var])

        encoder = tf.keras.Model(encoder_inputs, [z_mean, z_log_var, z], name="encoder")
        self.encoder = encoder

    def set_decoder(self, n_output, hidden_structure, n_latent):
        """
        This method can be used to set up the decoder model if it has not been giving
        when instantiating the scVAE class. Default activation function is PReLU (Parametric ReLU)

        n_output: dimensionality of the original data
        hidden_structure: list with the number of hidden units PER layer. The first and last
            units do not correspond to n_latent or n_output in this case.
            For example: [3, 5, 10, 15] cretes 4 layers from n_latent to n_output
        """

        latent_inputs = tf.keras.Input(n_latent)
        for units in hidden_structure:
            if units == hidden_structure[0]:
                x = Dense(units=units)(latent_inputs)
                x = tf.keras.layers.PReLU()(x)

            else:
                x = Dense(units=units)(x)
                x = tf.keras.layers.PReLU()(x)

        output = Dense(units=n_output)(x)
        output = tf.keras.layers.PReLU()(output)

        decoder = tf.keras.Model(latent_inputs, output, name="decoder")
        self.decoder = decoder

    def set_linear_encoder(self, n_input, hidden_structure):
        """
        This method can be used to set up the (linear) encoder model if it has not been giving
        when instantiating the scVAE class. NO activation function is used in this setup

        n_input: dimensionality of the original data
        hidden_structure: list with the number of hidden units PER layer. Last item of
            the list is the latent space dimensionality.
            For example: [10, 5, 3, 2] cretes 4 layers from n_input to 2 (the latent space)
        """
        self.n_input = n_input
        encoder_inputs = tf.keras.Input(n_input)
        for units in hidden_structure:
            if units == hidden_structure[0]:
                x = Dense(units=units)(encoder_inputs)

            else:
                x = Dense(units=units)(x)

        z_mean = Dense(hidden_structure[-1], name="z_mean")(x)
        z_log_var = Dense(hidden_structure[-1], name="z_log_var")(x)
        z = Sampling()([z_mean, z_log_var])

        encoder = tf.keras.Model(encoder_inputs, [z_mean, z_log_var, z], name="encoder")

        self.encoder = encoder

    def set_linear_decoder(self, n_output, hidden_structure, n_latent):
        """
        This method can be used to set up the (linear) decoder model if it has not been giving
        when instantiating the scVAE class. NO activation function is used in this setup

        n_output: dimensionality of the original data
        hidden_structure: list with the number of hidden units PER layer. The first and last
            units do not correspond to n_latent or n_output in this case.
            For example: [3, 5, 10, 15] cretes 4 layers from n_latent to n_output
        """
        latent_inputs = tf.keras.Input(n_latent)
        for units in hidden_structure:
            if units == hidden_structure[0]:
                x = Dense(units=units)(latent_inputs)

            else:
                x = Dense(units=units)(x)

        # output = Dense(units=n_output,activation="sigmoid")(x)  ## SIGMOID ACT FUNC AS IN VASC
        #output = Dense(units=n_output,activation="relu")(x)
        output = Dense(units=n_output)(x)

        decoder = tf.keras.Model(latent_inputs, output, name="linear_decoder")
        self.decoder = decoder

    def set_kl_weight(self, beta):
        """Sets the weight of the KL loss in the total loss function"""
        self.beta.assign(beta)

    def save_model(self, filepath="./", format="tf", overwrite=True, optimizer=True):
        """Saves the model parameters in the specified path in tensorflow format."""
        # inputs = encoder.get_layer(index=0)
        model = tf.keras.Model(inputs=tf.keras.Input(self.n_input), outputs=self.decoder(self.encoder.get_layer(index=-1)))
        model.save(filepath, save_format=format, overwrite=overwrite, include_optimizer=optimizer)
