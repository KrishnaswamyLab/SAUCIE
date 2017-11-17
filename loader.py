import os, sys, glob, math, io, contextlib, random
import numpy as np
import matplotlib.pyplot as plt
import fcsparser
import scipy.io
from tensorflow.examples.tutorials.mnist import input_data
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.model_selection import train_test_split
np.random.seed(1)

def asinh(x, scale=5.):
    f = np.vectorize(lambda y: math.asinh(y/scale))

    return f(x) 

def sinh(x, scale=5.):

    return scale*np.sinh(x)

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
        self.args = args
        self.start = 0
        self.epoch = 0
        if not isinstance(self.data, np.ndarray):
            self.load_data()

    def next_batch(self, batch_size):
        x = self.data
        if 'labels' in dir(self): y = self.labels

        if self.start + batch_size < x.shape[0]:
            batch_x = x[self.start:self.start+batch_size]

            if 'labels' in dir(self): batch_y = y[self.start:self.start+batch_size]

            self.start += batch_size
        else:
            self.epoch += 1
            batch_x1 = x[self.start:,:]
            batch_x2 = x[:batch_size - (x.shape[0]-self.start),:]
            batch_x = np.concatenate([batch_x1, batch_x2], axis=0)

            if 'labels' in dir(self):
                batch_y1 = y[self.start:]
                batch_y2 = y[:batch_size - (y.shape[0]-self.start)]
                batch_y = np.concatenate([batch_y1, batch_y2], axis=0)

            self.start = batch_size - (x.shape[0]-self.start)

        if 'labels' in dir(self):
            return batch_x, batch_y
        else:
            return batch_x

    def iter_batches(self, epochs=1, train_or_test='train', max_iters=None):
        if train_or_test=='train':
            x = self.data
            if 'labels' in dir(self):
                y = self.labels
        else:
            x = self.data_test
            if 'labels_test' in dir(self):
                y = self.labels_test

        for epoch in range(epochs):
            for i in range(x.shape[0]//self.args.batch_size):
                start = i*self.args.batch_size
                end = (i+1)*self.args.batch_size
                
                if 'labels' in dir(self):
                    yield x[start:end,:], y[start:end]
                else:
                    yield x[start:end,:]

                if max_iters and i>max_iters: return



class LoaderMNIST(Loader):
    data = None
    labels = None
    data_test = None
    labels_test = None
    input_dim = 28*28
    data_folder = "MNIST_data/"

    def __init__(self, args, shuffle=False):
        self.args = args
        self.start = 0
        self.epoch = 0

        if not isinstance(self.data, np.ndarray):
            self.load_data(shuffle)

    def load_data(self, shuffle):
        with silence():
            mnist = input_data.read_data_sets(self.data_folder, one_hot=False)

            data = mnist.train.images
            labels = mnist.train.labels

            r = list(range(len(labels)))
            np.random.shuffle(r)
            data = data[r,:]
            labels = labels[r]

            LoaderMNIST.data = data
            LoaderMNIST.labels = labels

            LoaderMNIST.data_test = mnist.test.images
            LoaderMNIST.labels_test = mnist.test.labels

class LoaderZika(Loader):
    data = None
    labels = None
    input_dim = 34
    data_folder = '/home/krishnan/data/zika_data/gated'

    def __init__(self, args, shuffle=False):
        self.args = args
        self.start = 0
        self.epoch = 0

        if not isinstance(self.data, np.ndarray):
            self.load_data(shuffle)

    def load_data(self, shuffle):
        colnames = [line.strip() for line in open(os.path.join(self.data_folder, 'colnames.csv'))] 

        cols_to_use = [colnames.index(line.strip()) for line in open(os.path.join(self.data_folder, 'markers.csv'))] 

        # files = glob.glob(os.path.join(self.data_folder, '161913*'))
        # data = []
        # labels = []
        # for i,f in enumerate(files):
        #     if any([f.find(name)>0 for name in ['markers','colnames','npz']]): continue
        #     print(f)
        #     x = np.genfromtxt(f, delimiter=',', skip_header=1, usecols=cols_to_use, max_rows=75000)
        #     x = asinh(x)

        #     data.append(x)
        #     labels.append(i*np.ones(x.shape[0]))
        #     print(x.shape)


        files = ['/data/kevin/Zika_Cytof/GatedData/Spike-in_LIVE_151548ZIKV_04May2017_01_splorm_0_normalized.fcs', 
                 #'/data/kevin/Zika_Cytof/GatedData/Spike-in_LIVE_151550ZIKV_10May2017_01_splorm_0_normalized.fcs',
                 #'/data/kevin/Zika_Cytof/GatedData/Spike-in_LIVE_151553ZIKV_27April2017_01_splorm_0_normalized.fcs',
                  #'/data/kevin/Zika_Cytof/GatedData/Spike-in_LIVE_161927ZIKV_06April2017_01_splorm_0_normalized.fcs',
                   '/data/kevin/Zika_Cytof/GatedData/Spike-in_LIVE_161924ZIKV_22Mar2017_01_splorm_0_normalized.fcs']#,
                 #'/data/kevin/Zika_Cytof/GatedData/Sample_LIVE_151550ZIKV_10May2017_01_splorm_0_normalized.fcs']
        data = []
        labels = []
        for i,f in enumerate(files):
            meta, x = fcsparser.parse(f, reformat_meta=True)
            x = x.as_matrix()[:,cols_to_use]
            x = asinh(x)
            print(x.shape)
            data.append(x)
            labels.append(i*np.ones(x.shape[0]))
       


        data = np.concatenate(data, axis=0)
        labels = np.concatenate(labels, axis=0)

        r = list(range(len(labels)))
        np.random.shuffle(r)
        data = data[r,:]
        labels = labels[r]

        LoaderZika.data = data
        LoaderZika.labels = labels

class LoaderZika2(Loader):
    data = None
    labels = None
    input_dim = 34
    data_folder = '/home/krishnan/data/zika_data/gated'

    def __init__(self, args, shuffle=False):
        self.args = args
        self.start = 0
        self.epoch = 0

        if not isinstance(self.data, np.ndarray):
            self.load_data(shuffle)

    def load_data(self, shuffle):
        colnames = [line.strip() for line in open(os.path.join(self.data_folder, 'colnames.csv'))] 

        cols_to_use = [colnames.index(line.strip()) for line in open(os.path.join(self.data_folder, 'markers.csv'))] 

        # files = glob.glob(os.path.join(self.data_folder, '161913*'))
        # data = []
        # labels = []
        # for i,f in enumerate(files):
        #     if any([f.find(name)>0 for name in ['markers','colnames','npz']]): continue
        #     print(f)
        #     x = np.genfromtxt(f, delimiter=',', skip_header=1, usecols=cols_to_use, max_rows=75000)
        #     x = asinh(x)

        #     data.append(x)
        #     labels.append(i*np.ones(x.shape[0]))
        #     print(x.shape)


        files = ['/data/kevin/Zika_Cytof/GatedData/Spike-in_LIVE_151548ZIKV_04May2017_01_splorm_0_normalized.fcs', 
                 '/data/kevin/Zika_Cytof/GatedData/Spike-in_LIVE_151550ZIKV_10May2017_01_splorm_0_normalized.fcs']
        data = []
        labels = []
        for i,f in enumerate(files):
            meta, x = fcsparser.parse(f, reformat_meta=True)
            x = x.as_matrix()[:,cols_to_use]
            x = asinh(x)
            print(x.shape)
            data.append(x)
            labels.append(i*np.ones(x.shape[0]))
       


        data = np.concatenate(data, axis=0)
        labels = np.concatenate(labels, axis=0)

        r = list(range(len(labels)))
        np.random.shuffle(r)
        data = data[r,:]
        labels = labels[r]

        LoaderZika2.data = data
        LoaderZika2.labels = labels



class LoaderGMM(Loader):
    data = None
    labels = None
    data_test = None
    labels_test = None
    input_dim = 200

    def __init__(self, args):
        self.args = args
        self.start = 0
        self.epoch = 0
        if not isinstance(self.data, np.ndarray):
            self.load_data()

    def load_data(self, n_pts=5000, n_clusters=5, theta=1, D_big=200):
        print("Loading data...")
        all_v = []
        labels = []

        partition_dim = D_big // (n_clusters*2)
        for c in range(n_clusters*2):

            v = np.zeros((D_big,n_pts))
            mean = np.random.uniform(3,3,[partition_dim])
            cov = np.diag(.5*np.ones((partition_dim))) #np.random.uniform(1,1,[25,25])
            v[partition_dim*c:partition_dim*(c+1),:] = np.random.multivariate_normal(mean,cov,(n_pts)).T
            all_v.append(v)
            labels.append((c%n_clusters)*np.ones(v.shape[1]))
            #labels.append(c*np.ones(v.shape[1]))
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

        data = all_v.T
        labels = np.concatenate(labels, axis=0)
        # data = data - data.min()
        #data = data / data.max()
       
        r = list(range(len(labels)))
        np.random.shuffle(r)
        data = data[r,:]
        labels = labels[r]
        LoaderGMM.data = data
        LoaderGMM.labels = labels

class LoaderMouse(Loader):
    data = None
    labels = None
    input_dim = 15

    def load_data(self):
        print("Loading data...")
        fns = ['/data/krishnan/mouse_data2/p3/matrix.mtx', '/data/krishnan/mouse_data2/rm3/matrix.mtx']
        data = []
        labels = []
        for i,fn in enumerate(fns):
            data.append(scipy.io.mmread(fn))
            labels.append(i*np.ones((data[-1].shape[1])))
        data = np.concatenate([d.todense() for d in data], axis=1)
        data = data.T
        labels = np.concatenate(labels, axis=0)

        pca = PCA(self.input_dim)
        data = pca.fit_transform(data)
        self.pca = pca
        LoaderMouse.data = data
        LoaderMouse.labels = labels

    

class LoaderEMT(Loader):
    data = None
    labels = None
    input_dim = 15

    def load_data(self):
        print("Loading data...")
        fn = '/data/krishnan/emt_data/data_raw.mat'
        data = scipy.io.loadmat(fn)['data']
        labels = np.ones(data.shape[0])
        LoaderEMT.data_original = data

        pca = PCA(self.input_dim)
        data = pca.fit_transform(data)
        #data = np.clip(data, np.percentile(data, 1, axis=0), np.percentile(data, 99, axis=0))

        LoaderEMT.pca = pca
        LoaderEMT.data = data
        LoaderEMT.labels = labels

class LoaderMerck(Loader):
    data = None
    labels = None
    input_dim = 48
    cols = None

    def load_data(self):
        print("Loading data...")
        files = ['/data/amodio/merck_pembro_sbrt/RD3437.060117.Run1/RD3437.060117.Run1_01.FCS']
        data = []
        labels = []
        for i,f in enumerate(files):
            meta, x = fcsparser.parse(f, reformat_meta=True)
            cols_to_use = [meta['_channel_names_'].index(c) for c in meta['_channel_names_'] if c not in ['Time', 'Event_length', 'Center', 'Offset', 'Width', 'Residual']]
            x = x.as_matrix()[:,cols_to_use]
            x = asinh(x)
            print(x.shape)
            data.append(x)
            labels.append(i*np.ones(x.shape[0]))
       

        data = np.concatenate(data, axis=0)
        labels = np.concatenate(labels, axis=0)

        r = list(range(len(labels)))
        np.random.shuffle(r)
        r = r[:100000]
        data = data[r,:]
        labels = labels[r]

        LoaderMerck.data = data
        LoaderMerck.labels = labels

        LoaderMerck.cols = [c for c in meta['_channel_names_'] if c not in ['Time', 'Event_length', 'Center', 'Offset', 'Width', 'Residual']]









