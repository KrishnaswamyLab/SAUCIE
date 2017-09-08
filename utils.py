import sys, os, time, math, argparse, cPickle
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from skimage.io import imsave
from sklearn.manifold import TSNE

import seaborn as sns
sns.set_style('darkgrid')
sns.set_palette('muted')
# sns.set_context("notebook", font_scale=1.5,
#                 rc={"lines.linewidth": 2.5})



def normalized(a, axis=-1, order=2):
	l2 = np.atleast_1d(np.linalg.norm(a, order, axis))
	l2[l2==0] = 1
	return a / np.expand_dims(l2, axis)

def tbn(name):

	return tf.get_default_graph().get_tensor_by_name(name)

def obn(name):

	return tf.get_default_graph().get_operation_by_name(name)

def get_layer(sess, loader, name, test_or_train='test'):
	tensor = tbn(name)
	layer = []
	labels = []
	for batch, batch_labels in loader.iter_batches(test_or_train):
		
		feed = {tbn('x:0'):batch}
		[act] = sess.run([tensor], feed_dict=feed)

		layer.append(act)
		labels.append(batch_labels)

	layer = np.concatenate(layer, axis=0)
	labels = np.concatenate(labels, axis=0)
	return layer, labels

def plot(args, data, labels, title, fn):
	fig, ax = plt.subplots(1,1)
	ax.set_title(title)

	colors = [plt.cm.jet(float(i)/len(np.unique(labels))) for i in xrange(len(np.unique(labels)))]
	for index,lab in enumerate(np.unique(labels)):
		inds = [True if l==lab else False for l in labels]
		tmp_data = data[inds,:]

		ax.scatter(tmp_data[:,0], tmp_data[:,1], c=colors[int(index)], alpha=.5, s=12, label=int(lab))

	lgnd = plt.legend(scatterpoints=1, prop={'size':6})
	for lh in lgnd.legendHandles:
		lh._sizes = [30]
		lh.set_alpha(1)

	fig.savefig( os.path.join(args.save_folder,fn) )

	plt.close('all')
	print "Plot saved to {}".format(fn)

def activations_heatmap(args, sess, loader, layer):
	all_acts, all_labels = get_layer(sess, loader, 'layer_encoder_{}_activation:0'.format(layer))

	
	nonzero = all_acts.reshape((-1))[all_acts.reshape((-1)) > 0]

	normalized, labels = get_layer(sess, loader, 'normalized_activations_layer_{}:0'.format(layer))

	# binarized = normalized(all_acts, axis=1)
	# binarized = np.where(normalized>.5, 1, 0)
	normalized = normalized.reshape((-1))[normalized.reshape((-1)) > 0]

	fig, (ax1,ax2) = plt.subplots(1,2, figsize=(20,10))
	ax1.set_title('Nonzero first layer activations (entropy reg)')
	ax1.set_xlabel('Activation')
	ax1.set_ylabel('Count')
	ax1.hist(nonzero, bins=100)
	ax2.set_title('Nonzero first layer activations (entropy reg) normalized')
	ax2.hist(normalized, bins=100)
	fig.savefig(os.path.join(args.save_folder, 'layer_{}_activations_histogram'.format(layer)))

	

	fig, axes = plt.subplots(10,10, figsize=(20,20), dpi=150)
	fig.subplots_adjust(hspace=.01, wspace=.02, left=.02, right=1, top=1, bottom=0)
	for i in xrange(10):
		# pick out this digit
		all_this_digit = all_acts[all_labels==i,:]
		axes[i,0].set_ylabel("{}".format(i))
		for j in xrange(10):
			squaredims = int(math.floor(np.sqrt( args.layers[layer] )))
			this_digit = all_this_digit[j,:squaredims**2].reshape((squaredims,squaredims))
			ax = axes[i,j]
			ax.imshow(this_digit, cmap='gray')
			ax.set_xticklabels([])
			ax.set_yticklabels([])
			ax.grid('off')
			ax.set_aspect('equal')
	fig.savefig(os.path.join(args.save_folder, 'layer_{}_activations_heatmap'.format(layer)))

	plt.close('all')
	print "Activations heatmap saved."

def show_result(batch_res, fname, grid_size=(8, 8), grid_pad=5):
    img_height = 28
    img_width = 28
    img_size = img_height * img_width
    #batch_res = 0.5 * batch_res.reshape((batch_res.shape[0], img_height, img_width)) + 0.5
    batch_res = batch_res.reshape((batch_res.shape[0], img_height, img_width))
    img_h, img_w = batch_res.shape[1], batch_res.shape[2]
    grid_h = img_h * grid_size[0] + grid_pad * (grid_size[0] - 1)
    grid_w = img_w * grid_size[1] + grid_pad * (grid_size[1] - 1)
    img_grid = np.zeros((grid_h, grid_w), dtype=np.uint8)
    for i, res in enumerate(batch_res):
        if i >= grid_size[0] * grid_size[1]:
            break
        img = (res) * 255
        img = img.astype(np.uint8)
        row = (i // grid_size[0]) * (img_h + grid_pad)
        col = (i % grid_size[1]) * (img_w + grid_pad)
        img_grid[row:row + img_h, col:col + img_w] = img
    imsave(fname, img_grid)

def calculate_loss(sess, loader, train_or_test='test'):
	loss_tensor = tbn('loss:0')
	x_tensor = tbn('x:0')
	y_tensor = tbn('y:0')
	losses = []
	for batch, batch_labels in loader.iter_batches(train_or_test):
		feed = {x_tensor:batch,
				y_tensor:batch}
		[l] = sess.run([loss_tensor], feed_dict=feed)
		losses.append(l)

	avg_loss = sum(losses) / float(len(losses))
	return avg_loss