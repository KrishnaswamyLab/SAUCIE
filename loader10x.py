import time, os, sys, random
from scipy.io import mmread
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from tensorflow.contrib.layers import xavier_initializer
import seaborn as sns
sns.set_style('dark')


def tbn(name):

    return tf.get_default_graph().get_tensor_by_name(name)

def save_vars(sess):
    variables = ['w1','b1','w2','b2','w3','b3','d1','bd1','d2','bd2','d3','bd3']
    variables_dict = {}
    for v in variables:
        [v_] = sess.run([tbn(v+':0')])
        variables_dict[v] = v_

    scipy.io.savemat('/data/amodio/mouse10x/vars', variables_dict)


class Loader10x(object):
    data = None
    fn = '/data/amodio/mouse10x.npz'
    fn_small = '/data/amodio/mouse10x_small.npz'

    def __init__(self, batch_size=100, small=False):
        self.small = small
        self.batch_size = 100

        if not isinstance(self.data, np.ndarray):
            self.load_data()

    def load_data(self):
        fn = self.fn if not self.small else self.fn_small
        Loader10x.data = np.load(fn)['data']

        r = list(range(Loader10x.data.shape[0]))
        random.shuffle(r)
        Loader10x.data = Loader10x.data[r,:]

        pca = PCA(50)
        Loader10x.data = pca.fit_transform(Loader10x.data)

    def iterbatches(self, epochs=None):
        start = 0
        end = self.batch_size
        epoch = 0

        x = self.data[:,:]
        for epoch in range(epochs):
            for i in range(self.data.shape[0]//self.batch_size):

                yield self.data[start:end,:]

                start +=self.batch_size
                end +=self.batch_size

                if end>self.data.shape[0]:
                    start = 0
                    end = self.batch_size


EPOCHS = 100
layers = [1024,512,2]
# layers = [100,50,25,2]
act = tf.nn.relu


loader = Loader10x(small=False)
print(loader.data.shape)
D = loader.data.shape[1]


x = tf.placeholder(tf.float32, [None,D])

w1 = tf.get_variable(name='e1', shape=[D,layers[0]], initializer=xavier_initializer())
b1 = tf.get_variable(name='be1', shape=[layers[0]], initializer=xavier_initializer())
w2 = tf.get_variable(name='e2', shape=[layers[0],layers[1]], initializer=xavier_initializer())
b2 = tf.get_variable(name='be2', shape=[layers[1]], initializer=xavier_initializer())
w3 = tf.get_variable(name='e3', shape=[layers[1],layers[2]], initializer=xavier_initializer())
b3 = tf.get_variable(name='be3', shape=[layers[2]], initializer=xavier_initializer())



d1 = tf.get_variable(name='d1', shape=[layers[2],layers[1]], initializer=xavier_initializer())
bd1 = tf.get_variable(name='bd1', shape=[layers[1]], initializer=xavier_initializer())
d2 = tf.get_variable(name='d2', shape=[layers[1],layers[0]], initializer=xavier_initializer())
bd2 = tf.get_variable(name='bd2', shape=[layers[0]], initializer=xavier_initializer())
d3 = tf.get_variable(name='d3', shape=[layers[0],D], initializer=xavier_initializer())
bd3 = tf.get_variable(name='bd3', shape=[D], initializer=xavier_initializer())



h1 = act(tf.matmul(x,w1)+b1)
h2 = act(tf.matmul(h1,w2)+b2)
h3 = tf.matmul(h2,w3)+b3


h4 = act(tf.matmul(h3,d1)+bd1)
h5 = act(tf.matmul(h4,d2)+bd2)
h6 = tf.matmul(h5,d3)+bd3

loss = tf.reduce_mean((x - h6)**2)

opt = tf.train.AdamOptimizer(.001)
train_op = opt.minimize(loss, name='train_op')

gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.4)

sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
with sess:
    sess.run(tf.global_variables_initializer())
    saver = tf.train.Saver(tf.global_variables(), max_to_keep=5)

    for iteration,batch in enumerate(loader.iterbatches(epochs=EPOCHS)):
        feed = {x:batch}
        [l,_] = sess.run([loss,train_op], feed_dict=feed)

        if (iteration+1)%100==0:
            print("{}: {:.3f}".format(iteration+1, l))

        
    savefile = os.path.join('saved_10x', 'AEpca')
    saver.save(sess, savefile, global_step=iteration, write_meta_graph=True)
    print("Model saved to {}".format(savefile))

    
    save_vars(sess)





