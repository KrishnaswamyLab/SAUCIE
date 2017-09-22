import sys, os, time, math, argparse, pickle
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import pandas as pd
from loader import Loader
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score
from sklearn.manifold import TSNE
from utils import *


def restore_model(model_folder):
	tf.reset_default_graph()

	with open('{}/args.pkl'.format(model_folder), 'rb') as f:
		args = pickle.load(f)
	args.dropout_p = 1.
	loader = get_loader(args)

	sess = tf.Session()
	ckpt = tf.train.get_checkpoint_state(args.save_folder)
	saver = tf.train.import_meta_graph('{}.meta'.format(ckpt.model_checkpoint_path))
	saver.restore(sess, ckpt.model_checkpoint_path)

	return args, sess, loader

def analyze():
	model_folders = sys.argv[1:]

	for model_folder in model_folders:
		args, sess, loader = restore_model(model_folder)
		print(args.save_folder)
		with sess:

			fig, ax = plt.subplots(1,1)
			w = [v for v in tf.global_variables() if v.name=='b_layer_encoder_0:0']

			[w] = sess.run(w)
			print(w)
			ax.imshow(w.reshape((32,32)))
			fig.savefig(args.save_folder+'/w_heatmap.png')
			sys.exit()
			# embeddings, labels = get_layer(sess, loader, 'layer_embedding_activation:0')
			# input_layer, labels = get_layer(sess, loader, 'x:0')
			# reconstruction, labels = get_layer(sess, loader, 'layer_output_activation:0')

			# plot_mnist(args, input_layer, labels, embeddings, 'orig')
			# plot_mnist(args, reconstruction, labels, embeddings, 'recon')
			activations_heatmap(args, sess, loader, 0, thresh=1e-4)

			count, new_labels = count_clusters(args, sess, loader, 0, thresh=1e-4, return_clusters=True)
			embedded, labels = get_layer(sess, loader, 'layer_embedding_activation:0', 'test')
			print(len(np.unique(new_labels)))
			plot(args, embedded, new_labels, 'tsne', 'tsne')
			sys.exit()
			

			df = pd.DataFrame(embedded)
			df['cluster'] = new_labels
			grouped = df.groupby('cluster')
			applied = grouped.apply(lambda x: x.mean())

			e = tbn('ph_embedding:0')
			o = tbn('ph_embedding_decoded:0')

			fig, ax = plt.subplots(1,1)
			all_clusts = []
			del applied['cluster']
			for i,clust in applied.iterrows():
				clust = clust.values.reshape((1,-1))
				[decoded] = sess.run([o], feed_dict={e:clust, tbn('is_training:0'):False})
				all_clusts.append(decoded.reshape((1,-1)))

			all_clusts = np.concatenate(all_clusts, axis=0)
			all_clusts = all_clusts[:100,:]
			g = math.ceil(math.sqrt(all_clusts.shape[0]))

			show_result(args, all_clusts, 'cluster_means_decoded.png', grid_size=(g,g))


			# train_loss = calculate_loss(sess, loader, 'train')
			# test_loss = calculate_loss(sess, loader, 'test')
			# print("Train loss: {:.3f} Test loss: {:.3f}".format(train_loss, test_loss))
			





if __name__=='__main__':
	analyze()