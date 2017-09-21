import tensorflow as tf
import numpy as np
from tensorflow.contrib.layers import batch_norm as bn
import sys, os, time, math, argparse
import matplotlib.pyplot as plt
from utils import *


class Layer(object):
	def __init__(self, name, dim_in, dim_out, activation=tf.nn.relu, dropout_p=1., batch_norm=True):
		self.W = tf.get_variable(shape=[dim_in,dim_out], name='W_{}'.format(name))
		self.b = tf.get_variable(shape=[dim_out], name='b_{}'.format(name))

		self.activation = activation
		self.dropout_p = dropout_p
		self.name = name
		self.batch_norm = batch_norm

	def __call__(self, x):
		if self.batch_norm:
			x = bn(x)
		h = self.activation(tf.matmul(x, self.W) + self.b)
		h = tf.nn.dropout(h,self.dropout_p)
		h = tf.identity(h, name='{}_activation'.format(self.name))
		tf.add_to_collection('activations', h)
		return h

class MLP(object):
	def __init__(self, args):
		self.args = args
		self.x = tf.placeholder(tf.float32, shape=[args.batch_size, args.input_dim], name='x')
		self.y = tf.placeholder(tf.float32, shape=[args.batch_size, args.input_dim], name='y')
		self.learning_rate = tf.placeholder(tf.float32, shape=[], name='learning_rate')

		#########################################################################
		# ENCODER
		self.layers_encoder = []
		input_plus_layers = [args.input_dim] + args.layers

		act_f = lambda x: tf.nn.softmax(tf.nn.relu(x)/.1)
		for i,layer in enumerate(input_plus_layers[:-2]):
			if i in args.layers_entropy:
				print("Adding entropy to {}".format(i))
				l = Layer('layer_encoder_{}'.format(i), input_plus_layers[i], input_plus_layers[i+1], act_f, args.dropout_p, batch_norm=args.batch_norm)
			else:
				l = Layer('layer_encoder_{}'.format(i), input_plus_layers[i], input_plus_layers[i+1], args.activation, args.dropout_p, batch_norm=args.batch_norm)
			self.layers_encoder.append(l)
		# last layer is linear, and fully-connected
		self.layers_encoder.append(Layer('layer_embedding', input_plus_layers[-2], input_plus_layers[-1], tf.identity, 1., batch_norm=args.batch_norm))

		self.embedded = self.feedforward_encoder(self.x)
		if args.add_noise_to_embedding:
			noise = tf.random_normal(self.embedded.get_shape(), mean=0, stddev=args.add_noise_to_embedding, name='embedding_noise')
			self.embedded_noisy = self.embedded + tf.identity(noise, name='noisy_embedding')
		#########################################################################


		#########################################################################
		# DECODER
		self.layers_decoder = []
		layers_decoder = input_plus_layers[::-1]

		# first layer is linear, and fully-connected
		for i,layer in enumerate(layers_decoder[:-2]):
			if i==0:
				l = Layer('layer_decoder_{}'.format(i), layers_decoder[i], layers_decoder[i+1], args.activation, 1., batch_norm=args.batch_norm)
			else:
				l = Layer('layer_decoder_{}'.format(i), layers_decoder[i], layers_decoder[i+1], args.activation, args.dropout_p, batch_norm=args.batch_norm)
			self.layers_decoder.append(l)
		# last decoder layer is linear and fully-connected
		if args.loss=='mse':
			output_act = lrelu #tf.nn.relu
		elif args.loss=='bce':
			output_act = tf.nn.sigmoid
		self.layers_decoder.append(Layer('layer_output', layers_decoder[-2], layers_decoder[-1], output_act, 1., batch_norm=args.batch_norm))

		if args.add_noise_to_embedding:
			self.reconstructed = self.feedforward_decoder(self.embedded_noisy)
		else:
			self.reconstructed = self.feedforward_decoder(self.embedded)
		#########################################################################


		#########################################################################
		# LOSSES
		# reconstruction loss
		if args.loss=='mse':
			self.loss_recon = (self.reconstructed - self.y)**2
		elif args.loss=='bce':
			self.loss_recon = -(self.y*tf.log(self.reconstructed+1e-9) + (1-self.y)*tf.log(1-self.reconstructed+1e-9))
		self.loss_recon = tf.reduce_mean(self.loss_recon, name='loss_recon')

		# sparsity regularization
		self.loss_sparse = tf.constant(0.)
		for act in tf.get_collection('activations'):
			for add_sparse_to in args.layers_sparsity:
				if 'encoder_{}'.format(add_sparse_to) in act.name:
					self.loss_sparse += args.lambda_sparsity*(tf.reduce_sum(tf.abs(act)))
		self.loss_sparse = tf.identity(self.loss_sparse, name='loss_sparse')

		# fuzzy counting regularization
		self.loss_entropy = tf.constant(0.)
		for act in tf.get_collection('activations'):
			for add_entropy_to, lambda_entropy in zip(args.layers_entropy, args.lambdas_entropy):
				if 'encoder_{}'.format(add_entropy_to) in act.name:
					if args.normalization_method=='l2norm':
						norm = tf.norm(act+1e-9, axis=1, keep_dims=True)
						normalized = act / (norm+1e-9)
					elif args.normalization_method=='softmax':
						normalized = tf.nn.softmax(act, dim=1)
					elif args.normalization_method=='none':
						normalized = act
					normalized = tf.identity(normalized, 'normalized_activations_layer_{}'.format(add_entropy_to))
					self.loss_entropy += lambda_entropy*tf.reduce_sum(-normalized*tf.log(normalized+1e-9) - (1-normalized)*tf.log(1-normalized+1e-9))
		self.loss_entropy = tf.identity(self.loss_entropy, name='loss_entropy')

		# l2 regularization
		# self.loss_reg = tf.identity(args.lambda_l2*sum([tf.nn.l2_loss(tv) for tv in tf.global_variables() if 'W_' in tv.name]), name='loss_reg')
		self.loss_reg = tf.identity(args.lambda_l1*tf.reduce_mean([tf.reduce_mean(tf.abs(tv)) for tv in tf.global_variables()]), name='loss_l1')
		self.loss_reg += tf.identity(args.lambda_l2*tf.reduce_mean([tf.reduce_mean(tf.nn.l2_loss(tv)) for tv in tf.global_variables()]), name='loss_reg')

		# total loss
		self.loss = self.loss_recon + self.loss_sparse + self.loss_reg + self.loss_entropy
		self.loss = tf.identity(self.loss, name='loss')
		#########################################################################


		#########################################################################
		# OPTIMIZATION
		opt = tf.train.AdamOptimizer(self.learning_rate)
		self.train_op = opt.minimize(self.loss, name='train_op')
		#########################################################################

	def feedforward_encoder(self, x):
		for i,l in enumerate(self.layers_encoder):
			print(x)
			x = l(x)
		return x

	def feedforward_decoder(self, x):
		for i,l in enumerate(self.layers_decoder):
			print(x)
			x = l(x)
		print(x)
		return x

	def write_graph(self, sess):
		writer = tf.summary.FileWriter(logdir=self.args.save_folder, graph=sess.graph)
		writer.flush()
