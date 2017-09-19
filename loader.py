import os, sys, glob, math, io, contextlib
import numpy as np
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
		self.args = args
		if args.data == 'MNIST':
			with silence():
				self.mnist = input_data.read_data_sets("MNIST_data/", one_hot=False)
		if args.data == 'ZIKA':
			colnames = []
			with open(os.path.join(args.data_folder, 'colnames.csv')) as f:	
				for line in f:
					line = line.strip()
					colnames.append(line)

			cols_to_use = []
			with open(os.path.join(args.data_folder, 'markers.csv')) as f:
				for i,line in enumerate(f):
					line = line.strip()
					cols_to_use.append(colnames.index(line))

			self.cols_to_use = cols_to_use
		if args.data == 'FLU':
			self.data = np.genfromtxt(self.args.data_folder, skip_header=1, delimiter=',')
			asinh_transform = np.vectorize(lambda x: math.asinh(x/5))
			self.data = asinh_transform(self.data)
			self.data = self.data / self.data.max()

	def iter_batches(self, train_or_test):
		if self.args.data == 'MNIST':
			return self.iter_batches_mnist(train_or_test)
		elif self.args.data == 'ZIKA':
			return self.iter_batches_zika(train_or_test)
		elif self.args.data == 'FLU':
			return self.iter_batches_flu(train_or_test)

	def iter_batches_zika(self, train_or_test):
		files = glob.glob(os.path.join(self.args.data_folder, '*'))
		for f in files:
			if any([f.find(name)>0 for name in ['markers','colnames','npz']]): continue
			if train_or_test == 'test':
				nrows = 2*self.batch_size
			else:
				nrows = 5*self.batch_size#sys.maxsize
			x = np.genfromtxt(f, delimiter=',', skip_header=1, usecols=self.cols_to_use, max_rows=nrows)
			x = (x - x.mean() ) / x.std()
			for i in range(math.floor(x.shape[0]/self.batch_size)):
				start = i*self.batch_size
				end = (i+1)*self.batch_size
				yield x[start:end,:], np.zeros(self.args.batch_size)
				if train_or_test=='test':
					break
					
	def iter_batches_mnist(self, train_or_test):
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

	def iter_batches_flu(self, train_or_test):
		x = self.data
		for i in range(math.floor(x.shape[0]/self.batch_size)):
			start = i*self.batch_size
			end = (i+1)*self.batch_size
			yield x[start:end,:], np.zeros(self.args.batch_size)

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

		if train_or_test=='train':
			r = list(range(self.x.shape[0]))
			np.random.shuffle(r)
			x = self.x[r,:]

		for i in range(math.floor(x.shape[0]/self.batch_size)):
			start = i*self.batch_size
			end = (i+1)*self.batch_size

			x_batch = x[start:end,:]
			
			yield x_batch, np.zeros(x_batch.shape[0])
