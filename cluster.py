import sys, os, time, math, argparse, cPickle
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from loader import Loader
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score
from sklearn.manifold import TSNE
from utils import *

folder = sys.argv[1]

def add_to_args(args):
	args.binarize_entropy_layer = True
	
	return args

def run_inits(args, sess, mlp):
	init_vars = tf.global_variables()
	init_op = tf.variables_initializer(init_vars)
	sess.run(init_op)
	saver = tf.train.Saver(init_vars)
	return saver

def plot_pca(args, data, labels, layer_name):
	pca = PCA(2)
	fitted = pca.fit_transform(data)
	plot(args, fitted, labels, "pca on layer {}".format(layer_name), 'layer_{}_pca'.format(layer_name))

def plot_tsne(args, data, labels, layer_name, n=0):
	if n:
		data = data[:n,:]
		labels = labels[:n]
	t = time.time()
	tsne = TSNE(n_components=2, verbose=2, init='pca')
	fitted = tsne.fit_transform(data)
	print "tsne took {:.1f} s".format(time.time() - t)
	plot(args, fitted, labels, "tsne on layer {}".format(layer_name), 'layer_{}_tsne'.format(layer_name))

def noised_image_reconstruction(args, sess, loader):
	for batch, batch_labels in loader.iter_batches('train'):
		batch_noised = batch
		if args.add_noise:
			batch_noised = batch + np.random.normal(0,.1, batch.shape)
			batch_noised = np.maximum(np.minimum(batch_noised, 1.), 0.)
		if args.dropout_input:
			batch_noised *= np.random.binomial(1,.5, batch_noised.shape)

		x_tensor = tbn('x:0')
		reconstructed_tensor = tbn('layer_output_activation:0')

		[reconstructed] = sess.run([reconstructed_tensor], feed_dict={x_tensor:batch_noised})

		show_result(batch, os.path.join(args.save_folder, 'original_images.jpg'))
		show_result(batch_noised, os.path.join(args.save_folder, 'noisy_images.jpg'))
		show_result(reconstructed, os.path.join(args.save_folder, 'reconstructed_images.jpg'))
		break

def binarize(args, sess, loader, layer):
	data, labels = get_layer(sess, loader, 'normalized_activations_layer_{}:0'.format(layer))
	# print data.shape
	# print labels.shape

	binarized = np.where(data>.5, 1, 0)

	# print data.max(axis=1)
	# print data[:5,:10]
	# print binarized[:5,:10]
	
	# unique_rows = np.vstack({tuple(row) for row in binarized})
	# print unique_rows.shape

	# km = KMeans(10)
	# km.fit(binarized)
	# clusts = km.predict(binarized)
	# print adjusted_rand_score(labels, clusts)

	# new_labels = np.zeros(labels.shape)

	# tot = 0
	# for i,row in enumerate(unique_rows):
	# 	# if i>5: break
	# 	rows_equal_to_this_code = np.where(np.all(binarized==row, axis=1))[0]
	# 	new_labels[rows_equal_to_this_code] = i
		# labels_code = labels[rows_equal_to_this_code]
		# unique, counts = np.unique(labels_code, return_counts=True)
		# d = np.asarray((unique, counts)).T

	# indexer_for_this_digit = np.where(labels==4,True,False)
	# data_all_this_digit = data[indexer_for_this_digit]
	# codes_all_this_digit = new_labels[indexer_for_this_digit]


	fig, axes = plt.subplots(10,10, figsize=(20,20), dpi=150)
	fig.subplots_adjust(hspace=.01, wspace=.02, left=.02, right=1, top=1, bottom=0)
	for i in xrange(10):
		# pick out this digit
		all_this_digit = binarized[labels==i,:]
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
	fig.savefig(os.path.join(args.save_folder, 'layer_{}_activations_heatmap_binarized'.format(layer)))


	




def cluster():
	with open('{}/args.pkl'.format(folder), 'rb') as f:
		args = cPickle.load(f)
	args.dropout_p = 1.

	args = add_to_args(args)

	loader = Loader(args)

	# initialize
	with tf.Session() as sess:
		ckpt = tf.train.get_checkpoint_state(args.save_folder)
		saver = tf.train.import_meta_graph('{}.meta'.format(ckpt.model_checkpoint_path))
		saver.restore(sess, ckpt.model_checkpoint_path)


		activations_heatmap(args, sess, loader, 0)
		#binarize(args, sess, loader, 0)
		

		if args.add_noise:
			noised_image_reconstruction(args, sess, loader)

		train_loss = calculate_loss(sess, loader, 'train')
		test_loss = calculate_loss(sess, loader, 'test')

		print "Train loss: {:.3f} Test loss: {:.3f}".format(train_loss, test_loss)
		
		reconstruction, labels = get_layer(sess, loader, 'layer_output_activation:0')

		plot_pca(args, reconstruction, labels, layer_name='reconstruction')

		plot_tsne(args, reconstruction, labels, layer_name='reconstruction', n=0)




if __name__=='__main__':
	cluster()