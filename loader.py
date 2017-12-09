import os, sys, glob, math, io, contextlib, random
import numpy as np
import matplotlib.pyplot as plt
import fcsparser
import scipy.io
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.model_selection import train_test_split
np.random.seed(1)

class Loader(object):
    def __init__(self, data, labels=None, spikein_mask=None):
        self.start = 0
        self.epoch = 0
        self.data = [x for x in [data, labels, spikein_mask] if x is not None]
        self.input_dim = data.shape[1]

        self.r = list(range(data.shape[0]))
        np.random.shuffle(self.r)
        self.data = [x[self.r] for x in self.data]

    def next_batch(self, sess, batch_size=100):
        num_rows = self.data[0].shape[0]

        if self.start + batch_size < num_rows:
            batch = [x[self.start:self.start+batch_size] for x in self.data]
            self.start += batch_size
        else:
            self.epoch += 1
            batch_part1 = [x[self.start:] for x in self.data]
            batch_part2 = [x[:batch_size - (x.shape[0]-self.start)] for x in self.data]
            batch = [np.concatenate([x1, x2], axis=0) for x1,x2 in zip(batch_part1, batch_part2)]

            self.start = batch_size - (num_rows-self.start)

        return batch

    def iter_batches(self, batch_size=100):
        num_rows = self.data[0].shape[0]

        for i in range(num_rows//batch_size):
            start = i*batch_size
            end = (i+1)*batch_size
            
            yield [x[start:end] for x in self.data]

        if end != num_rows:
            yield [x[end:] for x in self.data]

    def restore_order(self, data):
        data_out = np.zeros_like(data)
        for i,j in enumerate(self.r):
            data_out[j] = data[i]
        return data_out

































