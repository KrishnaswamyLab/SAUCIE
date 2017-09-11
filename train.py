import sys, os, time, math, argparse, pickle
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from model import MLP, tbn, obn
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score
from utils import *

def parse_args():
	parser = argparse.ArgumentParser()


	# NAME
	parser.add_argument('--save_folder', type=str, default='saved_cytof')
	parser.add_argument('--data', type=str, default='cytof_emt')
	parser.add_argument('--data_folder', type=str, default='/home/krishnan/data/emt_cytof_data')

	# TRAINING PARAMETERS
	parser.add_argument('--batch_size', type=int, default=100)
	parser.add_argument('--learning_rate', type=float, default=.001)
	parser.add_argument('--num_epochs', type=int, default=1000)
	parser.add_argument('--save_every', type=int, default=1000)
	parser.add_argument('--print_every', type=int, default=100)
	parser.add_argument('--max_iterations', type=int, default=10000)
	
	# MODEL ARCHITECTURE
	# parser.add_argument('--layers', type=str, default='1024,512,256,2')
	# parser.add_argument('--layers', type=str, default='128,64,32,2') # make it faster while testing
	parser.add_argument('--layers', type=str, default='64,32,16,2') # make it faster while testing
	parser.add_argument('--activation', type=str, default='relu')
	parser.add_argument('--loss', type=str, default='mse')
	parser.add_argument('--dropout_p', type=float, default=1.)
	parser.add_argument('--batch_norm', type=bool, default=True)
	
	# REGULARIZATIONS
	parser.add_argument('--lambda_l1', type=float, default=0)
	parser.add_argument('--lambda_l2', type=float, default=0)
	parser.add_argument('--layers_sparsity', type=str, default='')
	parser.add_argument('--lambda_sparsity', type=float, default=0)
	parser.add_argument('--layers_entropy', type=str, default='0')
	parser.add_argument('--lambda_entropy', type=float, default=.001)
	parser.add_argument('--normalization_method', type=str, default='none')
	parser.add_argument('--thresh', type=float, default=.1)

	# NOISE
	parser.add_argument('--add_noise', type=bool, default=False)
	parser.add_argument('--dropout_input', type=bool, default=False)
	parser.add_argument('--add_noise_to_embedding', type=float, default=0)


	args = parser.parse_args()

	args = process_args(args)
	
	return args

def process_args(args):
	# parse layers
	args.layers = [int(l) for l in args.layers.split(',')] if len(args.layers)>0 else []
	args.layers_sparsity = [int(l) for l in args.layers_sparsity.split(',')] if len(args.layers_sparsity)>0 else []
	args.layers_entropy = [int(l) for l in args.layers_entropy.split(',')] if len(args.layers_entropy)>0 else []

	# parse activation
	if args.activation=='relu': args.activation = tf.nn.relu
	elif args.activation=='tanh': args.activation = tf.nn.tanh
	elif args.activation=='sigmoid': args.activation = tf.nn.sigmoid
	elif args.activation=='lrelu': args.activation = lrelu
	else: print("Could not parse activation: {}".format(args.activation))

	if args.data == 'MNIST': args.input_dim = 28*28
	elif args.data == 'cytof_emt': args.input_dim = 40
	else: print("Could not parse data type for input_dim")

	# make save folder
	if not os.path.exists(args.save_folder):
		os.makedirs(args.save_folder)

	# pickle args
	with open(args.save_folder + '/args.pkl', 'wb') as f:
		pickle.dump(args, f)

	# print args in plaintext
	with open(os.path.join(args.save_folder, 'args.txt'), 'w+') as f:
		for arg in sorted(vars(args)):
			f.write("{}: {}\n".format(arg, vars(args)[arg]))

	return args

def run_inits(args, sess, mlp):
	init_vars = tf.global_variables()
	init_op = tf.variables_initializer(init_vars)
	sess.run(init_op)
	saver = tf.train.Saver(init_vars, max_to_keep=1)
	return saver

def save(args, sess, mlp, saver, loader, iteration):
	savefile = os.path.join(args.save_folder, 'AE')
	saver.save(sess, savefile , global_step=iteration, write_meta_graph=True)
	print("Model saved to {}".format(savefile))

def calc_randind(args, sess, mlp, loader):
	# get randints
	randints = []
	names =['layer_encoder_{}_activation:0'.format(i) for i in range(3)] + ['layer_embedding_activation:0']
	for name in names:
		act_tensor = tbn(name)
		labels = []
		acts = []
		for batch, batch_labels in loader.iter_batches('test'):
			feed = {mlp.x:batch}
			[act] = sess.run([act_tensor], feed_dict=feed)

			labels.append(batch_labels)
			acts.append(act)
		acts = np.concatenate(acts, axis=0)
		labels = np.concatenate(labels, axis=0)

		km = KMeans(10)
		km.fit(acts)
		clusts = km.predict(acts)

		randint = float(adjusted_rand_score(labels, clusts))
		randints.append(randint)

	print("Randinds: {}".format(str(['{:.3f}'.format(r) for r in randints])))

def train(args):

	loader = get_loader(args)

	mlp = MLP(args)


	# initialize
	with tf.Session() as sess:
		saver = run_inits(args, sess, mlp)
		#mlp.write_graph(sess)

		iteration = 1
		losses = [tbn('loss:0'), tbn('loss_recon:0'), tbn('loss_reg:0'), tbn('loss_sparse:0'), tbn('loss_entropy:0')]
		for epoch in range(args.num_epochs):
			t = time.time()
			for batch, batch_labels in loader.iter_batches('train'):
				iteration+=1

				x = batch
				if args.add_noise:
					x = batch + np.random.normal(0,.1, batch.shape)
					x = np.maximum(np.minimum(x, 1.), 0.)
				if args.dropout_input:
					x *= np.random.binomial(1,.5, x.shape)

				feed = {mlp.x:x,
						mlp.y:batch,
						mlp.learning_rate:args.learning_rate}


				[l,lrec,lreg,lspa,le, _] = sess.run(losses + [obn('train_op')], feed_dict=feed)


				# print score on test set				
				if iteration%args.print_every==0:
					print("epoch/iter: {}/{} loss: {:.3f} ({:.3f} {:.3f} {:.3f} {:.3f}) time: {:.1f}".format(epoch,
						iteration, l, lrec, lreg, lspa, le, time.time() - t))
					t = time.time()

				# save
				if args.save_every and iteration%args.save_every==0:
					save(args, sess, mlp, saver, loader, iteration)

					embeddings, labels = get_layer(sess, loader, 'layer_embedding_activation:0')

					plot(args, embeddings, labels, 'Embedding layer', 'embedding')

					count = count_clusters(args, sess, loader, 0, thresh=args.thresh)
					print("entropy/sparsity lambda: {}/{} Number of clusters: {}".format(args.lambda_entropy, args.lambda_sparsity, count))

					activations_heatmap(args, sess, loader, 0)

			# after each epoch potentiall break out
			if args.max_iterations and iteration > args.max_iterations: break

		# save final model
		if args.save_every:
			save(args, sess, mlp, saver, loader, iteration)

		count = count_clusters(args, sess, loader, 0, thresh=args.thresh)
		print("Final number of clusters (e/s: {}/{}): {}".format(args.lambda_entropy, args.lambda_sparsity, count))
			


if __name__=='__main__':
	args = parse_args()
	train(args)