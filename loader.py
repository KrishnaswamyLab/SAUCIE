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


class LoaderTFRecord(object):

    def __init__(self, base_directory='/data/amodio/zika/tfrecords', batch_size=100):
        self.test_order = None
        self.data = None
        self.iteration = 0
        self.batch_size = batch_size
        self.base_directory = base_directory
        self.files = glob.glob('{}/*.tfrecords'.format(base_directory))
        #self.files = [f for f in self.files if ('Sample_LIVE_161898.2DENV' in f) or ('Sample_LIVE_161898.2Mock' in f)]
        #self.files = [f for f in self.files if ('Spike-in_LIVE_151548ZIKV' in f) or ('Spike-in_LIVE_151550ZIKV' in f)]
        #self.files = [f for f in self.files if ('Sample_LIVE_161927Mock' in f)]

        print(self.files)

        self.input_dim = self.get_input_dim()

        self.batch_label_dict = {}
        for i,f in enumerate(self.files):
            self.batch_label_dict[f] = i

        self.get_tfrecordreader()

    def get_input_dim(self):
        for record in tf.python_io.tf_record_iterator(self.files[0]):
            example = tf.train.Example()
            example.ParseFromString(record)
            row = np.array(example.features.feature['X'].float_list.value).reshape((1,-1))
            return row.shape[1]

    def get_tfrecordreader(self):
        filename_queue = tf.train.string_input_producer(self.files, num_epochs=100, shuffle=False)
        self.reader = tf.TFRecordReader()
        key, serialized_example = self.reader.read(filename_queue)

        feature = { 'X': tf.FixedLenFeature([self.input_dim], tf.float32), 'Y': tf.FixedLenFeature([1], tf.float32) }
        features = tf.parse_single_example(serialized_example, features=feature)
        self.batch_op, self.bath_labels_op = tf.train.shuffle_batch([features['X'], features['Y']], batch_size=self.batch_size, capacity=50000, num_threads=5, min_after_dequeue=40000)

    def next_batch(self, sess):
        [batch, batch_labels] = sess.run([self.batch_op, self.bath_labels_op])
        self.iteration += 1
        return batch, batch_labels.reshape((-1))

    def iter_batches(self, N=100000, in_memory=True):
        if in_memory:
            if self.data is None:
                data = []
                labels = []
                for f in range(len(self.files)):
                    for i,serialized_example in enumerate(tf.python_io.tf_record_iterator(self.files[f])):
                        example = tf.train.Example()
                        example.ParseFromString(serialized_example)
                        row = np.array(example.features.feature['X'].float_list.value).reshape((1,-1))
                        l = np.array(example.features.feature['Y'].float_list.value)
                        data.append(row)
                        labels.append(l)
                        if i>N:
                            break

                self.data = np.concatenate(data, axis=0)
                self.labels = np.concatenate(labels, axis=0).reshape((-1))


            start = 0
            end = self.batch_size
            while start+self.batch_size < self.data.shape[0]:
                yield self.data[start:end,:], self.labels[start:end]

                start += self.batch_size
                end += self.batch_size

        else:
            data = []
            labels = []
            for f in range(len(self.files)):
                for i,serialized_example in enumerate(tf.python_io.tf_record_iterator(self.files[f])):
                    if i%10000==0: print(i)
                    example = tf.train.Example()
                    example.ParseFromString(serialized_example)
                    row = np.array(example.features.feature['X'].float_list.value).reshape((1,-1))
                    l = np.array(example.features.feature['Y'].float_list.value)
                    data.append(row)
                    labels.append(f)
                    if len(labels)>=self.batch_size:
                        data = np.concatenate(data, axis=0)
                        labels = np.concatenate(labels, axis=0).reshape((-1))
                        yield data, labels
                        data = []
                        labels = []

    def get_colnames(self):
        cols = [c.strip() for c in open(os.path.join(self.base_directory, 'colnames.txt'))]
        return cols

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

































