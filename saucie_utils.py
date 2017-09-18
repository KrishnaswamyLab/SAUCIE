# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2017 Krishnan Srinivasan <krishnan1994@gmail.com>
#
# Distributed under terms of the MIT license.
# ==============================================================================

"""
Utils for SAUCIE
"""
import numpy as np
import tensorflow as tf
from typing import NamedTuple

EPS = 1e-9

class SparseLayerConfig(NamedTuple): # NamedTuple('SparseLayerConfig', ['num_layers', 'id_lam', 'l1_lam'])
    num_layers: int = 3
    id_lam: np.array = np.zeros(num_layers)
    l1_lam: np.array = np.zeros(num_layers)

    def __repr__(self):
        return 'SparseLayerConfig(id_lam={},l1_lam={})'.format(self.id_lam.tolist(), self.l1_lam.tolist())


def mean_squared_error(predicted, actual, name='mean_squared_error')
    return tf.reduce_mean(tf.square(predicted - actual), name=name)


def binary_crossentropy(predicted, actual, name='binary_crossentropy_error'):
    return -tf.reduce_mean(actual * tf.log(predicted + EPS) + (1 - actual) * tf.log(1 - predicted + EPS), name=name)


# information dimension regularization penalty
def id_penalty(act, lam, name='id_loss'):
    return tf.multiply(lam, -tf.reduce_mean(act * tf.log(act + EPS)), name=name)


def l1_act_penalty(act, lam, name='l1_loss'):
    return tf.multiply(lam, tf.reduce_mean(tf.abs(act)), name=name)

def make_dict_str(d={}, custom_keys=[], subset=[], kv_sep=': ', item_sep=', ',
                  float_format='{:6.5E}'):
    if not custom_keys:
        if subset:
            d = d.copy()
            for k in d.keys():
                if k not in subset:
                    del d[key]
        custom_keys = [(k,k) for k in d.keys()]

    item_list = []
    for c_key, key in custom_keys:
        item = d[key]
        if type(item) == float and item < 1e-4:
            item = float_format.format(item)
        elif type(item) == list:
            for i,j in enumerate(item):
                if type(j) == float and j < 1e-4:
                    item[i] = float_format.format(j)
            item = ','.join(item)
        else:
            item = str(item)
        kv_str = kv_sep.join([c_key, item])
        item_list.append(kv_str)
    dict_str = item_sep.join(item_list)
    return dict_str



# Code from Danijar Hafner gist:
# https://gist.github.com/danijar/8663d3bbfd586bffecf6a0094cd116f2

import functools

def doublewrap(function):
    """
    A decorator decorator, allowing to use the decorator to be used without
    parentheses if no arguments are provided. All arguments must be optional.
    """
    @functools.wraps(function)
    def decorator(*args, **kwargs):
        if len(args) == 1 and len(kwargs) == 0 and callable(args[0]):
            return function(args[0])
        else:
            return lambda wrapee: function(wrapee, *args, **kwargs)
    return decorator


@doublewrap
def define_scope(function, scope=None, *args, **kwargs):
    """
    A decorator for functions that define TensorFlow operations. The wrapped
    function will only be executed once. Subsequent calls to it will directly
    return the result so that operations are added to the graph only once.
    The operations added by the function live within a tf.variable_scope(). If
    this decorator is used with arguments, they will be forwarded to the
    variable scope. The scope name defaults to the name of the wrapped
    function.
    """
    attribute = '_cache_' + function.__name__
    name = scope or function.__name__
    @property
    @functools.wraps(function)
    def decorator(self):
        if not hasattr(self, attribute):
            with tf.variable_scope(name, *args, **kwargs):
                setattr(self, attribute, function(self))
        return getattr(self, attribute)
    return decorator


# Code from TensorFlow source example:
# https://github.com/tensorflow/tensorflow/blob/master/tensorflow/contrib/learn/python/learn/datasets/mnist.py

class DataSet:
    """Base data set class
    """

    def __init__(self, shuffle=True, labeled=True, **data_dict):
        assert '_data' in data_dict
        if labeled:
            assert '_labels' in data_dict
            assert data_dict['_data'].shape[0] == data_dict['_labels'].shape[0]
        self._labeled = labeled
        self._shuffle = shuffle
        self.__dict__.update(data_dict)
        self._num_samples = self._data.shape[0]
        self._index_in_epoch = 0
        self._epochs_trained = 0
        self._batch_number = 0
        if self._shuffle:
            self._shuffle_data()

    def __len__(self):
        return len(self._data) + len(self._test_data)

    @property
    def epochs_trained(self):
        return self._epochs_trained

    @epochs_trained.setter
    def epochs_trained(self, new_epochs_trained):
        self._epochs_trained = new_epochs_trained

    @property
    def batch_number(self):
        return self._batch_number

    @property
    def index_in_epoch(self):
        return self._index_in_epoch

    @property
    def num_samples(self):
        return self._num_samples

    @property
    def data(self):
        return self._data

    @property
    def labels(self):
        return self._labels

    @property
    def labeled(self):
        return self._labeled

    @property
    def test_data(self):
        return self._test_data

    @property
    def test_labels(self):
        return self._test_labels

    @classmethod
    def load(cls, filename):
        data_dict = np.load(filename)
        labeled = data_dict['_labeled']
        return cls(labeled=labeled, **data_dict)

    def save(self, filename):
        data_dict = self.__dict__
        np.savez_compressed(filename, **data_dict)

    def _shuffle_data(self):
        shuffled_idx = np.arange(self._num_samples)
        np.random.shuffle(shuffled_idx)
        self._data = self._data[shuffled_idx]
        if self._labeled:
            self._labels = self._labels[shuffled_idx]

    def next_batch(self, batch_size):
        assert batch_size <= self._num_samples
        start = self._index_in_epoch
        if start + batch_size > self._num_samples:
            self._epochs_trained += 1
            self._batch_number = 0
            data_batch = self._data[start:]
            if self._labeled:
                labels_batch = self._labels[start:]
            remaining = batch_size - (self._num_samples - start)
            if self._shuffle:
                self._shuffle_data()
            start = 0
            data_batch = np.concatenate([data_batch, self._data[:remaining]],
                                        axis=0)
            if self._labeled:
                labels_batch = np.concatenate([labels_batch,
                                               self._labels[:remaining]],
                                              axis=0)
            self._index_in_epoch = remaining
        else:
            data_batch = self._data[start:start + batch_size]
            if self._labeled:
                labels_batch = self._labels[start:start + batch_size]
            self._index_in_epoch = start + batch_size
        self._batch_number += 1
        batch = (data_batch, labels_batch) if self._labeled else data_batch
        return batch


