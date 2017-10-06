import os, sys, glob, math, io, contextlib, random
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data
np.random.seed(0)



class SilentFile(object):
    def write(self, x): pass

@contextlib.contextmanager
def silence():
    save_stdout = sys.stdout
    sys.stdout = SilentFile()
    yield
    sys.stdout = save_stdout

class Loader(object):
    def __init__(self, args, load_full=True):
        self.batch_size = args.batch_size
        self.args = args
        self.asinh_transform = np.vectorize(lambda x: math.asinh(x/5))
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

            files = glob.glob(os.path.join(self.args.data_folder, '161913*'))
            self.data = []
            for f in files:
                if any([f.find(name)>0 for name in ['markers','colnames','npz']]): continue
                nrows = sys.maxsize
                x = np.genfromtxt(f, delimiter=',', skip_header=1, usecols=self.cols_to_use)
                x = self.asinh_transform(x)
                self.data.append(x)
                print(x.shape)
           
            self.data = np.concatenate(self.data, axis=0)
            print(self.data.shape)
            self.data = self.data / self.data.max()  
            self.data = self.data[ZIKA_R]
            self.data_test = self.data[-10000:,:]
            self.data = self.data[:-10000,:]
            
            # self.data = DATA
            # self.data_test = DATA_TEST

        if args.data == 'FLU':
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
            files = glob.glob(os.path.join(self.args.data_folder, 'CD8', '*'))
            self.data = []
            for f in files:
                x = np.genfromtxt(f, delimiter=',', usecols=self.cols_to_use, skip_header=1)
                x = self.asinh_transform(x)
                self.data.append(x)
            self.data = np.concatenate(self.data, axis=0)
            self.data = self.data / self.data.max()
            r = list(range(self.data.shape[0]))
            np.random.shuffle(r)
            self.data = self.data[r]

        if args.data == 'TOY':
            N = 5000
            D_big = 100
            D_small = 20
            N_clusters = 10
            theta = 1

            all_v = []
            self.labels = []
            for c in range(N_clusters):
                v = np.zeros((D_big,N))
                for i in range(D_small):
                  v[c+i,:] = np.random.normal(c,1,[N])
                all_v.append(v)
                self.labels.append(c*np.ones(v.shape[1]))
            all_v = np.concatenate(all_v, axis=1)

            s = math.sin(theta)
            c = math.cos(theta)
            rotation_Ms = []
            for rotation in range(D_big-1):
              m = np.eye(D_big)
              m[rotation,rotation] = c
              m[rotation+1,rotation+1] = c
              m[rotation,rotation+1] = -s
              m[rotation+1,rotation] = s
              rotation_Ms.append(m)

            for m in rotation_Ms:
              all_v = m.dot(all_v)

            self.data = all_v.T
            self.data = self.data - self.data.min()
            self.data = self.data / self.data.max()
            self.labels = np.concatenate(self.labels, axis=0)

            self.test_order = list(range(self.data.shape[0]))
            random.shuffle(self.test_order)
            # self.data = DATA_TOY
            # self.labels = LABELS_TOY
            # self.test_order = TEST_ORDER_TOY

    def iter_batches(self, train_or_test):
        if self.args.data == 'MNIST':
            return self.iter_batches_mnist(train_or_test)
        elif self.args.data == 'ZIKA':
            return self.iter_batches_zika(train_or_test)
        elif self.args.data == 'FLU':
            return self.iter_batches_flu(train_or_test)
        elif self.args.data == 'TOY':
            return self.iter_batches_toy(train_or_test)

    def iter_batches_zika(self, train_or_test):
        if train_or_test=='train':
            x = self.data
        else:
            x = self.data_test

        for i in range(math.floor(x.shape[0]/self.batch_size)):
            start = i*self.batch_size
            end = (i+1)*self.batch_size
            yield x[start:end,:], np.zeros(self.args.batch_size)
                    
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
            if train_or_test=='test' and end>10000:
                break

    def iter_batches_toy(self, train_or_test):
        if train_or_test == 'test':
            x = self.data[self.test_order,:]
            y = self.labels[self.test_order]
        else:
            r = list(range(self.data.shape[0]))
            random.shuffle(r)
            x = self.data[r,:]
            y = self.labels[r]

        for i in range(math.floor(x.shape[0]/self.batch_size)):
            start = i*self.batch_size
            end = (i+1)*self.batch_size
            yield x[start:end,:], y[start:end]
            if train_or_test=='test' and end>10000:
                break

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
