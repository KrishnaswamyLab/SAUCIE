# import os

# import numpy as np
# import sklearn.metrics
# import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import Input, Dense, LeakyReLU
from tensorflow.keras.initializers import GlorotUniform
from tensorflow.random import set_seed
from tensorflow import reduce_mean

# from utils import calculate_mmd, obn, tbn


class SAUCIE_BN(object):
    def __init__(self,
                 input_dim,
                 lambda_b=0,
                 lambda_c=0,
                 layer_c=0,
                 lambda_d=0,
                 layers=[512, 256, 128, 2],
                 seed=None
                 ):
        self.input_dim = input_dim
        self.lambda_b = lambda_b
        self.lambda_c = lambda_c
        self.layer_c = layer_c
        self.lambda_d = lambda_d
        self.layers = layers
        self.seed = seed

    def _build_layers(self):
        # ENCODER
        input_shape = (self.input_dim, )
        inputs = Input(shape=input_shape, name='encoder_input')
        batches = Input(shape=(1, ), name="batches_input")

        h1 = Dense(self.layers[0],
                   kernel_initializer=GlorotUniform(seed=self.seed),
                   name='encoder0', use_bias=True)(inputs)
        h1 = LeakyReLU(alpha=0.2, name='encoder0_activation')(h1)

        if not self.lambda_b and not self.lambda_c:
            h2 = Dense(self.layers[1],
                       kernel_initializer=GlorotUniform(seed=self.seed),
                       activation="sigmoid",
                       name='encoder1', use_bias=True)(h1)
        else:
            h2 = Dense(self.layers[1],
                       kernel_initializer=GlorotUniform(seed=self.seed),
                       name='encoder1', use_bias=True)(h1)
            h2 = LeakyReLU(alpha=0.2, name='encoder1_activation')(h2)

        h3 = Dense(self.layers[2],
                   kernel_initializer=GlorotUniform(seed=self.seed),
                   name='encoder2', use_bias=True)(h2)
        h3 = LeakyReLU(alpha=0.2, name='encoder2_activation')(h3)

        embedded = Dense(self.layers[3],
                         kernel_initializer=GlorotUniform(seed=self.seed),
                         name='embeddings', use_bias=True)(h3)
        encoder = Model(inputs, embedded, name='encoder')

        # DECODER
        latent_inputs = Input(shape=(self.layers[3], ),
                              name='latentspace')
        h5 = Dense(self.layers[2],
                   kernel_initializer=GlorotUniform(seed=self.seed),
                   name='decoder0', use_bias=True)(latent_inputs)
        h5 = LeakyReLU(alpha=0.2, name='decoder0_activation')(h5)

        h6 = Dense(self.layers[1],
                   kernel_initializer=GlorotUniform(seed=self.seed),
                   name='decoder1', use_bias=True)(h5)
        h6 = LeakyReLU(alpha=0.2, name='decoder0_activation')(h6)

        if self.lambda_c:
            layer_c = Dense(self.layers[0],
                            kernel_initializer=GlorotUniform(seed=self.seed),
                            activation="relu",
                            name='layer_c', use_bias=True)(h6)
        else:
            layer_c = Dense(self.layers[0],
                            kernel_initializer=GlorotUniform(seed=self.seed),
                            name='layer_c', use_bias=True)(h6)
            layer_c = LeakyReLU(alpha=0.2, name='layer_c')(layer_c)

        outputs = Dense(self.input_dim,
                        kernel_initializer=GlorotUniform(seed=self.seed),
                        name='outputs', use_bias=True)(layer_c)
        decoder = Model(latent_inputs, outputs, name='decoder')

        # get classifier layer
        classifier = Model(inputs, layer_c, name='classifier')

        # combine models
        outputs = decoder(encoder(inputs))
        SAUCIE_BN_model = Model([inputs, batches], outputs, name="SAUCIE_BN")

        # add losses
        if self.lambda_b:
            recon_loss = self._build_reconstruction_loss_mmd(inputs,
                                                             outputs,
                                                             batches)
            SAUCIE_BN_model.add_loss(recon_loss)
            mmd_loss = self._build_reg_b(embedded, batches)
            SAUCIE_BN_model.add_loss(mmd_loss)
        else:
            recon_loss = self._build_reconstruction_loss(inputs, outputs)
            SAUCIE_BN_model.add_loss(recon_loss)

        if self.lambda_c:
            id_reg_loss = self._build_reg_c(layer_c)
            SAUCIE_BN_model.add_loss(id_reg_loss)
        if self.lambda_d:
            intracluster_loss = self._build_reg_d(inputs, layer_c)
            SAUCIE_BN_model.add_loss(intracluster_loss)

        return SAUCIE_BN_model, encoder, classifier

    def _build_reconstruction_loss(self, input, reconstructed):
        """
        Build the reconstruction loss part of the network
        if batch correction isn't being performed.

        :param reconstructed: the tensor that was output by the decoder
        :param input: the tensor that was the input of the encoder
        """
        loss_recon = reduce_mean((reconstructed - input)**2)
        return loss_recon

    def _build_reconstruction_loss_mmd(self, input, reconstructed, batches):
        """
        Build the reconstruction loss part of the network
        if batch correction is being performed.

        :param reconstructed: the tensor that was output by the decoder
        :param input: the tensor that was the input of the encoder
        """
        return 0

    def _build_reg_b(self, embedded, batches):
        """
        Build the loss for the MMD regularization.

        :param embedded: the latent space embedding of the autoencoder
        """
        return 0

    def _build_reg_c(self, codes):
        """
        Build loss for the ID regularization.

        :param codes: the codes that will be binarized
                      and used to determine cluster assignment
        """
        return 0

    def _build_reg_d(self, input, codes):
        """
        Calculate the intracluster distances in the original data
        given binary-like codes.

        :param input: the tensor that was the input of the encoder
        :param codes: the codes that will be binarized
        and used to determine cluster assignment
        """
        return 0

    def get_architecture(self, lr):
        set_seed(self.seed)
        SAUCIE_BN_model, encoder, classifier = self._build_layers()
        # optimizer will take care of all loses itself
        SAUCIE_BN_model.compile(optimizer=Adam(learning_rate=lr))

        return SAUCIE_BN_model, encoder, classifier
