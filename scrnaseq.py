from scipy.io import loadmat
import numpy as np
import tensorflow as tf
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from tensorflow.contrib.layers import xavier_initializer
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns
sns.set_style('dark')

ITERATIONS = 20000
D = 15
batch_size = 100
layers = [1000,500,2]
act = tf.nn.tanh
fn = '/data/krishnan/emt_data/data_raw.mat'
fncols = '/data/krishnan/emt_data/genes.csv'

# TODO: seaborn pair plots

def pca3d(projected_recon):
    fig = plt.figure()
    pca3d = PCA(3)
    projected_recon_pc3 = pca3d.fit_transform(projected_recon)
    ax = Axes3D(fig)
    ax.scatter(projected_recon_pc3[:,0], projected_recon_pc3[:,1], projected_recon_pc3[:,2], alpha=.1, s=2)

    high = 0
    for ii in range(0,360,90):
        ax.view_init(elev=high*50, azim=ii+30)
        fig.savefig('reconstructed_pca3d{}'.format(ii))
        if high==0: high=1
        else: high=0

def edge_plots(recon, cols):
    ys = ['SNAI1', 'ZEB1', 'MYC','SNAI2']
    xs = ['VIM', 'SNAI1']
    fig, axes = plt.subplots(len(ys), len(xs))
    for xcoli,xcol in enumerate(xs):
        for ycoli,ycol in enumerate(ys):
            ax = axes[ycoli,xcoli]
            # ax.set_xticks([])
            # ax.set_yticks([])
            if xcoli==0:
                ax.set_ylabel(ycol)
            if ycoli==len(ys)-1:
                ax.set_xlabel(xcol)
            reconx = recon[:,cols.index(xcol)]
            recony = recon[:,cols.index(ycol)]
            ax.scatter(reconx, recony, s=4, alpha=.4)
    fig.savefig('edge_plots')

# data = loadmat(fn)['data']
# # # data = data[:1000,:]
# # # data_ = data / (data.sum(axis=0)+1e-9)
# tmp = np.percentile(data,90,axis=0)
# tmp = np.where(tmp==0, 1, tmp)
# data_ = data / tmp
# pca = PCA(D, whiten=False)
# datapca = pca.fit_transform(data)
# # edge_plots(data_, cols)
# np.savez('SAUCIE/datapca{}d_raw'.format(D), datapca=datapca, pca=pca)


# library size normalization
# x = gene_bc_matrix.matrix.asfptype().transpose() # transpose so it's cells x genes
# lib_size = x.sum(axis=0)
# cols_to_keep = np.where(lib_size != 0.)
# lib_size = lib_size[cols_to_keep]
# x = x[:, cols_to_keep[1]]
# x = x / lib_size * np.median(lib_size.tolist())
# centered_x = np.sqrt(x)
# centered_x = centered_x - centered_x.mean(axis=0)
# u, s, vt = scipy.sparse.linalg.svds(centered_x, 20)
# pc = u.dot(np.diag(s))


cols = [x.strip() for x in open(fncols).readlines()]
# npzfile = np.load('datapca{}d.npz'.format(D))
npzfile = np.load('datapca{}d_raw.npz'.format(D))
datapca = npzfile['datapca']
pca = npzfile['pca'].reshape((1))[0]

np.clip(datapca, np.percentile(datapca, 1, axis=0), np.percentile(datapca, 99, axis=0))

x = tf.placeholder(tf.float32, [None,D])

w1 = tf.get_variable(name='w1', shape=[D,layers[0]], initializer=xavier_initializer())
b1 = tf.get_variable(name='b1', shape=[layers[0]], initializer=xavier_initializer())
w2 = tf.get_variable(name='w2', shape=[layers[0],layers[1]], initializer=xavier_initializer())
b2 = tf.get_variable(name='b2', shape=[layers[1]], initializer=xavier_initializer())
w3 = tf.get_variable(name='w3', shape=[layers[1],layers[2]], initializer=xavier_initializer())
b3 = tf.get_variable(name='b3', shape=[layers[2]], initializer=xavier_initializer())



d1 = tf.get_variable(name='d1', shape=[layers[2],layers[1]], initializer=xavier_initializer())
bd1 = tf.get_variable(name='bd1', shape=[layers[1]], initializer=xavier_initializer())
d2 = tf.get_variable(name='d2', shape=[layers[1],layers[0]], initializer=xavier_initializer())
bd2 = tf.get_variable(name='bd2', shape=[layers[0]], initializer=xavier_initializer())
d3 = tf.get_variable(name='d3', shape=[layers[0],D], initializer=xavier_initializer())
bd3 = tf.get_variable(name='bd3', shape=[D], initializer=xavier_initializer())

h1 = act(tf.matmul(x,w1)+b1)
h2 = act(tf.matmul(h1,w2)+b2)
# h3 = tf.matmul(h2,w3)+b3
h3 = tf.matmul(h2,w3) + b3**2
# w3_ = tf.get_variable(name='w3_', shape=[1,2], initializer=xavier_initializer())
# b3_ = tf.get_variable(name='b3_', shape=[1,2], initializer=xavier_initializer())
# print(h3)
# h3 = w3_*h3**2 + w3_*h3 + b3_

h4 = act(tf.matmul(h3,d1)+bd1)
h5 = act(tf.matmul(h4,d2)+bd2)
h6 = tf.matmul(h5,d3)+bd3

loss = tf.reduce_mean((x - h6)**2) #tf.reduce_mean(x*tf.log(h6+1e-9) + (1-x)*tf.log(1-h6+1e-9)) 

opt = tf.train.AdamOptimizer(.001)
train_op = opt.minimize(loss, name='train_op')

gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.5)

sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
with sess:
    init_op = tf.global_variables_initializer()
    sess.run(init_op)
    start = 0
    end = batch_size

    for i in range(ITERATIONS):
        x_ = datapca[start:end,:]
        feed = {x:x_}
        [l,_] = sess.run([loss, train_op], feed_dict=feed)

        if (i+1)%1000==0:
            # [embedding] = sess.run([h3], feed_dict={x:datapca})
            # fig, ax = plt.subplots(1,1)
            # ax.scatter(embedding[:,0], embedding[:,1], alpha=.5, s=5)
            # fig.savefig('plot_scrnaseq')

            [recon] = sess.run([h6], feed_dict={x:datapca})
            projected_recon = pca.inverse_transform(recon)
            # pca3d(projected_recon)
            

            edge_plots(projected_recon, cols)

            plt.close('all')
            print("{}: {:.3f}".format(i+1, l))


        start+=batch_size
        end+=batch_size
        if end>datapca.shape[0]:
            start=0
            end=batch_size






