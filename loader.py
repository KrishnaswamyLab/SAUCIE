import os, sys, glob
import numpy as np
from sklearn.decomposition import PCA
from tensorflow.examples.tutorials.mnist import input_data

class Loader(object):
	def __init__(self, args):
		self.batch_size = args.batch_size
		self.mnist = input_data.read_data_sets("MNIST_data/", one_hot=False)

	def iter_batches(self, train_or_test):
		if train_or_test=='train':
			x = self.mnist.train.images
			y = self.mnist.train.labels
		elif train_or_test=='test':
			x = self.mnist.test.images
			y = self.mnist.test.labels

		#x = x / 255.   tf already scales it
		r = range(x.shape[0])
		np.random.shuffle(r)
		x = x[r,:]
		y = y[r]



		for i in xrange(x.shape[0]/self.batch_size):

			start = i*self.batch_size
			end = (i+1)*self.batch_size

			x_batch = x[start:end,:]
			y_batch = y[start:end]
			

			yield x_batch,y_batch

