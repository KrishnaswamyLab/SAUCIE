# import os

# import numpy as np
# import sklearn.metrics
# import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import Input
from tensorflow.keras.initializers import GlorotUniform
from tensorflow.random import set_seed

from utils import calculate_mmd, lrelu, obn, tbn


class SAUCIE_BN(object):
    def __init__(self,
                 input_dim,
                 lambda_b=0,
                 lambda_c=0,
                 layer_c=0,
                 lambda_d=0,
                 layers=[512, 256, 128, 2],
                 activation=lrelu,
                 seed=None
                 ):
        self.input_dim = input_dim
        self.lambda_b = lambda_b
        self.lambda_c = lambda_c
        self.layer_c = layer_c
        self.lambda_d = lambda_d
        self.layers = layers
        self.activation = activation
        self.seed = seed

    def _build_layers(self):
        input_shape = (self.input_dim, )
        inputs = Input(shape=input_shape, name='encoder_input')

        outputs = 0
        SAUCIE_BN_model = Model(inputs, outputs, name="SAUCIE_BN")
        return SAUCIE_BN_model  # layer_c, embedding

    def get_architecture(self, lr):
        set_seed(self.seed)
        SAUCIE_BN_model = self._build_layers()
        # SAUCIE_BN_model.add_loss(model_Loss)
        SAUCIE_BN_model.compile(optimizer=Adam(learning_rate=lr))
        return SAUCIE_BN_model, 0
