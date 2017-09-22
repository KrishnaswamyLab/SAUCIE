# -*- coding: utf-8 -*-
# File: saucie_utils.py
# Author: Krishnan Srinivasan <krishnan1994 at gmail>
# Date: 21.09.2017
# Last Modified Date: 21.09.2017

"""
Utils for SAUCIE
"""
import numpy as np
import pandas as pd
import tensorflow as tf

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from typing import NamedTuple

EPS = np.float32(1e-8)
RAND_SEED = 42

class SparseLayerConfig(NamedTuple): # NamedTuple('SparseLayerConfig', ['num_layers', 'id_lam', 'l1_lam'])
    num_layers: int = 3
    id_lam: np.array = np.zeros(num_layers)
    l1_lam: np.array = np.zeros(num_layers)

    def __repr__(self):
        return 'SparseLayerConfig(id_lam={},l1_lam={})'.format(self.id_lam.tolist(), self.l1_lam.tolist())

    def __eq__(self, other):
        if isinstance(other, self.__class__):
            return (other.num_layers == self.num_layers and
                    (other.id_lam == self.id_lam).all() and 
                    (other.l1_lam == self.l1_lam).all())
        return False

    def __ne__(self, other):
        return not self.__eq__(other)


def mean_squared_error(predicted, actual, name='mean_squared_error'):
    return tf.reduce_mean(tf.square(predicted - actual), name=name)


def binary_crossentropy(predicted, actual, name='binary_crossentropy_error'):
    return -tf.reduce_mean(actual * tf.log(predicted + EPS)
                           + (1 - actual) * tf.log(1 - predicted + EPS), name=name)


def binarize(acts, thresh=.5):
    binarized = np.greater(acts, thresh).astype(int)
    unique_rows = np.vstack({tuple(row) for row in binarized})
    num_clusters = unique_rows.shape[0]
    new_labels = np.zeros(acts.shape[0])

    if num_clusters > 1:
        print('Unique binary clusters: {}'.format(num_clusters))
        for i, row in enumerate(unique_rows[1:]):
            subs = np.where(np.all(binarized == row, axis=1))[0]
            new_labels[subs] = i
    return new_labels


# information dimension regularization penalty
def id_penalty(act, lam, name='id_loss'):
    return tf.multiply(lam, -tf.reduce_sum(act * tf.log(act + EPS)), name=name)


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


# Code from TensorFlow Github issue 4079
# https://github.com/tensorflow/tensorflow/issues/4079

def lrelu(x, leak=0.2, name="lrelu"):
     with tf.variable_scope(name):
         f1 = 0.5 * (1 + leak)
         f2 = 0.5 * (1 - leak)
         return f1 * x + f2 * abs(x)


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
def define_var_scope(function, scope=None, *args, **kwargs):
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


@doublewrap
def define_name_scope(function, scope=None, *args, **kwargs):
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
            with tf.name_scope(name, *args, **kwargs):
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


def load_dataset_from_csv(csv_file, header=None, index_col=None, labels=None,
                          train_ratio=0.9, colnames=None, markers=None, keep_cols=True):
    if type(csv_file) == list:
        data = []
        for csv_fn in csv_file:
            data.append(pd.read_csv(csv_fn, header=header, index_col=index_col).values)
        data = np.concatenate(data)
    else:
        data = pd.read_csv(csv_file, header=header, index_col=index_col).values

    data_dict = {}

    if colnames:
        if type(colnames) == str:
            colnames = [x.strip() for x in open(colnames).readlines()]
            data_dict['_colnames'] = colnames
        if markers:
            data_dict['_markers'] = None
            if type(markers) == str:
                markers = [x.strip() for x in open(markers).readlines()]
            if not keep_cols:
                data = pd.DataFrame(data, columns=colnames)
                data = data[markers].values
                data_dict['_colnames'] = markers
            else:
                data_dict['_markers'] = markers

    if labels:
        if type(labels) == list:
            for labels_fn in labels_file:
                labels.append(pd.read_csv(labels_fn, header=None, index_col=None).values)
            labels = np.concatenate(labels)
        elif type(labels) == str:
            labels = pd.read_csv(labels_file, header=None, index_col=None).values
        elif type(labels) == int:
            labels = data[:,labels_col]
            data = np.delete(data, labels_col, 1)

        train_data, test_data, train_labels, test_labels = train_test_split(
            data, labels, train_size=train_ratio, random_state=RAND_SEED)
        data_dict['_data'], data_dict['_test_data'] = train_data, test_data
        data_dict['_labels'], data_dict['_test_labels'] = train_labels, test_labels
    else:
        train_data, test_data = train_test_split(data, train_size=train_ratio,
                                                 random_state=RAND_SEED)
        data_dict['_data'], data_dict['_test_data'] = train_data, test_data
        data_dict['labeled'] = False

    dataset = DataSet(**data_dict)
    return dataset

