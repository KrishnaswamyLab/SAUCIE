import sys, random, math, time, os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as colors

def tbn(name):

    return tf.get_default_graph().get_tensor_by_name(name)


batch_size = 200


asinh_transform = np.vectorize(lambda x: math.asinh(x/5))

fn1 = '/home/krishnan/data/zika_data/gated/161873.2ZIKV_05April2017_01.csv'
fn2 = '/home/krishnan/data/zika_data/gated/161898ZIKV_22Mar2017_01.csv'
# fn1 = '/home/krishnan/data/zika_data/gated/161864ZIKV_04May2017_01.csv'
# fn2 = '/home/krishnan/data/zika_data/gated/151759ZIKV_10May2017_01.csv'

zika_batch1 = np.genfromtxt(fn1, delimiter=',', skip_header=1)#, max_rows=200)
zika_batch2 = np.genfromtxt(fn2, delimiter=',', skip_header=1)#, max_rows=200)

zika_batch1 = asinh_transform(zika_batch1)
zika_batch2 = asinh_transform(zika_batch2)

zika_batch1_mmd = np.concatenate([zika_batch1,np.zeros((zika_batch1.shape[0],1))], axis=1)
zika_batch2_mmd = np.concatenate([zika_batch2,np.ones((zika_batch2.shape[0],1))], axis=1)


print(zika_batch1.shape)
print(zika_batch2.shape)

gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.5)
sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
saver = tf.train.import_meta_graph('saved_mmd/AE1-999.meta')
saver.restore(sess, 'saved_mmd/AE1-999')

with sess:

    recon_1 = []
    recon_2 = []
    for r in range(zika_batch1.shape[0]//batch_size):
        zika_batch1_ = zika_batch1_mmd[r*batch_size:(r+1)*batch_size,:]
        zika_batch2_ = zika_batch2_mmd[r*batch_size:(r+1)*batch_size,:]
        feed = {tbn('Placeholder:0'):zika_batch1_,tbn('Placeholder_1:0'):zika_batch2_}
        [r1,r2] = sess.run([tbn("MatMul_5:0"), "MatMul_11:0"], feed_dict=feed)
        recon_1.append(r1)
        recon_2.append(r2)
    recon_mmd = np.concatenate([np.concatenate(recon_1, axis=0), np.concatenate(recon_2, axis=0)], axis=0)




    b1_all_mmd = []
    b2_all_mmd = []
    for r in range(zika_batch1.shape[0]//batch_size):
        zika_batch1_ = zika_batch1_mmd[r*batch_size:(r+1)*batch_size,:]
        zika_batch2_ = zika_batch2_mmd[r*batch_size:(r+1)*batch_size,:]
        feed = {tbn('Placeholder:0'):zika_batch1_,tbn('Placeholder_1:0'):zika_batch2_}
        [b1,b2] = sess.run([tbn("MatMul_2:0"), "MatMul_8:0"], feed_dict=feed)
        b1_all_mmd.append(b1)
        b2_all_mmd.append(b2)
    b1_all_mmd = np.concatenate(b1_all_mmd, axis=0)
    b2_all_mmd = np.concatenate(b2_all_mmd, axis=0)


tf.reset_default_graph()
sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
saver = tf.train.import_meta_graph('saved_mmd/AE-999.meta')
saver.restore(sess, 'saved_mmd/AE-999')

with sess:
    b1_all = []
    b2_all = []
    for r in range(zika_batch1.shape[0]//batch_size):
        zika_batch1_ = zika_batch1[r*batch_size:(r+1)*batch_size,:]
        zika_batch2_ = zika_batch2[r*batch_size:(r+1)*batch_size,:]
        feed = {tbn('Placeholder:0'):zika_batch1_,tbn('Placeholder_1:0'):zika_batch2_}
        [b1,b2] = sess.run([tbn("MatMul_2:0"), "MatMul_8:0"], feed_dict=feed)
        b1_all.append(b1)
        b2_all.append(b2)
    b1_all = np.concatenate(b1_all, axis=0)
    b2_all = np.concatenate(b2_all, axis=0)




# fig, axes = plt.subplots(1,2, figsize=(10,5))
# ax1 = axes.flatten()[0]
# ax2 = axes.flatten()[1]
# fig.subplots_adjust(left=.01,right=.99,top=.93,bottom=.01,hspace=.01,wspace=.01)
# ax1.set_xticks([])
# ax1.set_yticks([])
# ax2.set_xticks([])
# ax2.set_yticks([])
# ax1.set_title('Before MMD')
# ax2.set_title('After MMD')

# b1_all = np.concatenate([b1_all,np.zeros((b1_all.shape[0],1))], axis=1)
# b2_all = np.concatenate([b2_all,np.ones((b2_all.shape[0],1))], axis=1)
# b12_all = np.concatenate([b1_all, b2_all], axis=0)
# r = list(range(b12_all.shape[0]))
# random.shuffle(r)
# b12_all = b12_all[r,:]
# ax1.scatter(b12_all[:,0], b12_all[:,1], c=['r' if _ else 'b' for _ in b12_all[:,2]], marker='o', s=2, alpha=.05)


b1_all_mmd = np.concatenate([b1_all_mmd,np.zeros((b1_all_mmd.shape[0],1))], axis=1)
b2_all_mmd = np.concatenate([b2_all_mmd,np.ones((b2_all_mmd.shape[0],1))], axis=1)
b12_all_mmd = np.concatenate([b1_all_mmd, b2_all_mmd], axis=0)
r = list(range(b12_all_mmd.shape[0]))
random.shuffle(r)
b12_all_mmd = b12_all_mmd[r,:]
# ax2.scatter(b12_all_mmd[:,0], b12_all_mmd[:,1], c=['r' if _ else 'b' for _ in b12_all_mmd[:,2]], marker='o', s=2, alpha=.05)

# fig.savefig('mmd_comp_before&after')

recon_mmd = recon_mmd[r,:]
print(b12_all_mmd.shape)
print(recon_mmd.shape)

fig, ax = plt.subplots(1,1)
fig.subplots_adjust(left=.01,right=.99,top=.93,bottom=.01,hspace=.01,wspace=.01)


col=24
min_ = recon_mmd[:,col].min()
max_ = recon_mmd[:,col].max()
print(min_, max_)
normalizer = colors.Normalize(min_, max_)
ax.set_xticks([])
ax.set_yticks([])
ax.scatter(b12_all_mmd[:,0], b12_all_mmd[:,1], c=cm.hot(normalizer(recon_mmd[:,col])), marker='o', s=2, alpha=.05)
fig.savefig('mmd_cd3_after')

