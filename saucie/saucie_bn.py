# import os

# import numpy as np
# import sklearn.metrics
# import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import Input, Dense, LeakyReLU
from tensorflow.keras.initializers import GlorotUniform
from tensorflow.random import set_seed

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
        latent_inputs = Input(shape=(self.layers[3]),
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
        SAUCIE_BN_model = Model(inputs, outputs, name="SAUCIE_BN")

        return SAUCIE_BN_model, encoder, classifier

    def _get_losses(self, model):
        # add loss just adds them together in fit function,
        # so that will be automated
        # model.add_loss(model_Loss)
        return model

    def get_architecture(self, lr):
        set_seed(self.seed)
        SAUCIE_BN_model, encoder, classifier = self._build_layers()
        SAUCIE_BN_model = self._get_losses(SAUCIE_BN_model)
        # optimizer will take care of all loses itself
        SAUCIE_BN_model.compile(optimizer=Adam(learning_rate=lr))

        return SAUCIE_BN_model, encoder, classifier
