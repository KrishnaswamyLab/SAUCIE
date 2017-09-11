import os, sys, glob, math, io, contextlib
import numpy as np
from sklearn.decomposition import PCA
from tensorflow.examples.tutorials.mnist import input_data


class SilentFile(object):
    def write(self, x): pass

@contextlib.contextmanager
def silence():
    save_stdout = sys.stdout
    sys.stdout = SilentFile()
    yield
    sys.stdout = save_stdout

class Loader(object):
	def __init__(self, args):
		self.batch_size = args.batch_size
		with silence():
			self.mnist = input_data.read_data_sets("MNIST_data/", one_hot=False)

	def iter_batches(self, train_or_test):
		if train_or_test=='train':
			x = self.mnist.train.images
			y = self.mnist.train.labels
		elif train_or_test=='test':
			x = self.mnist.test.images
			y = self.mnist.test.labels

		#x = x / 255.   tf already scales it
		if train_or_test=='train':
			r = list(range(x.shape[0]))
			np.random.shuffle(r)
			x = x[r,:]
			y = y[r]

		for i in range(math.floor(x.shape[0]/self.batch_size)):
			start = i*self.batch_size
			end = (i+1)*self.batch_size

			x_batch = x[start:end,:]
			y_batch = y[start:end]
			
			yield x_batch,y_batch

class Loader_cytof_emt(object):
	def __init__(self, args):
		self.x = np.loadtxt(os.path.join(args.data_folder, 'emt_data.csv'), delimiter=',')
		with open(os.path.join(args.data_folder, 'channels.csv')) as f:
			self.columns = f.read().strip().split(',')
		self.batch_size = args.batch_size

		# self.x = (self.x + self.x.min())
		# self.x = self.x / self.x.max()

	def iter_batches(self, train_or_test):
		x = self.x
		y = np.zeros(self.x.shape[0])
		if train_or_test=='train':
			r = list(range(self.x.shape[0]))
			np.random.shuffle(r)
			x = self.x[r,:]
			y = y[r]

		for i in range(math.floor(x.shape[0]/self.batch_size)):
			start = i*self.batch_size
			end = (i+1)*self.batch_size

			x_batch = x[start:end,:]
			y_batch = y[start:end]
			
			yield x_batch,y_batch
