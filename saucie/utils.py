import math

import numpy as np
import tensorflow as tf


def asinh(x, scale=5.):
    """Asinh transform."""
    f = np.vectorize(lambda y: math.asinh(y / scale))
    return f(x)


def sinh(x, scale=5.):
    """Reverse transform for asinh."""
    return scale * np.sinh(x)


def lrelu(x, leak=0.2, name="lrelu"):
    """Leaky ReLU activation."""
    return tf.maximum(x, leak * x)


def tbn(name):
    """Get the tensor in the default graph of the given name."""
    return tf.get_default_graph().get_tensor_by_name(name)


def obn(name):
    """Get the operation node in the default graph of the given name."""
    return tf.get_default_graph().get_operation_by_name(name)


def calculate_mmd(k1, k2, k12):
    """ Calculate MMD given kernels for batch1, batch2, and between batches """
    k1_part = k1.sum()/(k1.shape[0]*k1.shape[1])
    k2_part = k2.sum()/(k2.shape[0]*k2.shape[1])
    k12_part = 2*k12.sum()/(k12.shape[0]*k12.shape[1])
    return k1_part + k2_part - k12_part
