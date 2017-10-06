import tensorflow as tf
import numpy as np
from tensorflow.contrib.layers import batch_norm as bn
import sys, os, time, math, argparse
import matplotlib.pyplot as plt
from tensorflow.contrib.layers import xavier_initializer
from utils import *

def nameop(op, name):

    op = tf.identity(op, name=name)

class Layer(object):
    def __init__(self, name, dim_in, dim_out, activation=tf.nn.relu, dropout_p=1., batch_norm=True):
        self.W = tf.get_variable(shape=[dim_in,dim_out], initializer=xavier_initializer() , name='W_{}'.format(name))
        # self.b = tf.get_variable(shape=[dim_out], initializer=tf.zeros_initializer(), name='b_{}'.format(name))
        self.b = tf.get_variable(shape=[dim_out], initializer=tf.constant_initializer(-1.), name='b_{}'.format(name))

        self.activation = activation
        self.dropout_p = dropout_p
        self.name = name
        self.batch_norm = batch_norm

    def __call__(self, x, is_training=True):
        if self.batch_norm and 'output' not in self.name:
            reuse = not is_training
            x = bn(x, is_training=is_training, scope=self.name, reuse=reuse)
            nameop(x, '{}_bninput'.format(self.name))

        h = self.activation(tf.matmul(x, self.W)+ self.b)
        if is_training:
            h = tf.nn.dropout(h,self.dropout_p)
        h = tf.identity(h, name='{}_activation'.format(self.name))
        tf.add_to_collection('activations', h)
        return h

class MLP(object):
    def __init__(self, args):
        self.args = args
        self.x = tf.placeholder(tf.float32, shape=[args.batch_size, args.input_dim], name='x')
        self.y = tf.placeholder(tf.float32, shape=[args.batch_size, args.input_dim], name='y')
        is_training = tf.placeholder(tf.bool, shape=[], name='is_training')
        self.learning_rate = tf.placeholder(tf.float32, shape=[], name='learning_rate')

        #########################################################################
        # ENCODER
        self.layers_encoder = []
        input_plus_layers = [args.input_dim] + args.layers

        for i,layer in enumerate(input_plus_layers[:-2]):
            name  = 'layer_encoder_{}'.format(i)
            if i in args.layers_entropy:
                print("Adding entropy to {}".format(name))
                f = args.activation_idreg
                # f = lambda x: (args.activation_idreg(x) + 1) / 2
                l = Layer(name, input_plus_layers[i], input_plus_layers[i+1], f, args.dropout_p, batch_norm=args.batch_norm)
            else:
                l = Layer(name, input_plus_layers[i], input_plus_layers[i+1], args.activation, args.dropout_p, batch_norm=args.batch_norm)
            self.layers_encoder.append(l)
        # last layer is linear, and fully-connected
        # self.layers_encoder.append(Layer('layer_embedding', input_plus_layers[-2], input_plus_layers[-1], tf.nn.tanh, 1., batch_norm=args.batch_norm))
        self.layers_encoder.append(Layer('layer_embedding', input_plus_layers[-2], input_plus_layers[-1], tf.identity, 1., batch_norm=args.batch_norm))

        self.embedded = self.feedforward_encoder(self.x, is_training)
        #########################################################################


        #########################################################################
        # DECODER
        self.layers_decoder = []
        layers_decoder = input_plus_layers[::-1]

        # first layer is linear, and fully-connected
        for i,layer in enumerate(layers_decoder[:-2]):
            if i==0:
                l = Layer('layer_decoder_{}'.format(i), layers_decoder[i], layers_decoder[i+1], tf.identity, 1., batch_norm=args.batch_norm)
            else:
                l = Layer('layer_decoder_{}'.format(i), layers_decoder[i], layers_decoder[i+1], args.activation, args.dropout_p, batch_norm=args.batch_norm)
            self.layers_decoder.append(l)
        # last decoder layer is linear and fully-connected
        if args.loss=='mse':
            output_act = tf.nn.relu
        elif args.loss=='bce':
            output_act = tf.nn.sigmoid

        self.layers_decoder.append(Layer('layer_output', layers_decoder[-2], layers_decoder[-1], output_act, 1., batch_norm=args.batch_norm))

        self.reconstructed = self.feedforward_decoder(self.embedded)
        print(self.reconstructed)
        #########################################################################


        #########################################################################
        # LOSSES
        # reconstruction loss
        if args.loss=='mse':
            self.loss_recon = (self.reconstructed - self.y)**2
        elif args.loss=='bce':
            self.loss_recon = -(self.y*tf.log(self.reconstructed+1e-9) + (1-self.y)*tf.log(1-self.reconstructed+1e-9))
        self.loss_recon = tf.reduce_mean(self.loss_recon, name='loss_recon')

        
        # cluster use penalty
        self.loss_clustuse = tf.constant(0.)
        for act in tf.get_collection('activations'):    
            for layer_index, lambda_clustuse in zip(args.layers_clustuse, args.lambdas_clustuse):
                if 'encoder_{}'.format(layer_index) in act.name:
                    act = (act + 1) / 2
                    clustuse = tf.reduce_sum(act, axis=0)
                    clustuse = tf.nn.sigmoid(clustuse)
                    clustuse = tf.reduce_mean(clustuse)

                    # clustuse = tf.nn.sigmoid(act)
                    # clustuse = tf.reduce_sum(clustuse)
                    self.loss_clustuse += lambda_clustuse*clustuse
        nameop(self.loss_clustuse, 'loss_clustuse')




        # sparsity regularization
        self.loss_sparse = tf.constant(0.)
        for act in tf.get_collection('activations'):
            for layer_index, lambda_sparsity in zip(args.layers_sparsity, args.lambdas_sparsity):
                if 'encoder_{}'.format(layer_index) in act.name:
                    self.loss_sparse += lambda_sparsity*tf.reduce_mean(tf.abs(act))
        self.loss_sparse = tf.identity(self.loss_sparse, name='loss_sparse')

        # fuzzy counting regularization
        self.loss_entropy = tf.constant(0.)
        for act in tf.get_collection('activations'):
            for add_entropy_to, lambda_entropy in zip(args.layers_entropy, args.lambdas_entropy):
                if 'encoder_{}'.format(add_entropy_to) in act.name or \
                   add_entropy_to=='embedding' and 'embedding' in act.name:
                    if args.normalization_method=='l2norm':
                        norm = tf.norm(act+1e-9, axis=1, keep_dims=True)
                        normalized = act / (norm+1e-9)
                    elif args.normalization_method=='softmax':
                        normalized = tf.nn.softmax(act, dim=1)
                    elif args.normalization_method=='fixed_denom':
                        normalized = fixed_denom(act)
                    elif args.normalization_method=='tanh':
                        normalized = (act+1) / 2   
                    elif args.normalization_method=='none':
                        normalized = act
                    normalized = tf.identity(normalized, 'normalized_activations_layer_{}'.format(add_entropy_to))
                    self.loss_entropy += lambda_entropy*tf.reduce_mean(-normalized*tf.log(normalized+1e-9))

        self.loss_entropy = tf.identity(self.loss_entropy, name='loss_entropy')

        # l2 regularization
        # self.loss_reg = tf.identity(args.lambda_l2*sum([tf.nn.l2_loss(tv) for tv in tf.global_variables() if 'W_' in tv.name]), name='loss_reg')
        loss_l1 = tf.identity(args.lambda_l1*tf.reduce_mean([tf.reduce_mean(tf.abs(tv)) for tv in tf.global_variables()]), name='loss_l1')
        loss_l2 = tf.identity(args.lambda_l2*tf.reduce_mean([tf.reduce_mean(tf.nn.l2_loss(tv)) for tv in tf.global_variables()]), name='loss_l2')
        loss_reg = loss_l1 + loss_l2

        # total loss
        self.loss = self.loss_recon + self.loss_sparse + loss_reg + self.loss_entropy + self.loss_clustuse
        nameop(self.loss, 'loss')
        #########################################################################


        #########################################################################
        # OPTIMIZATION
        opt = tf.train.AdamOptimizer(self.learning_rate)
        self.train_op = opt.minimize(self.loss, name='train_op')
        #########################################################################


        self.decode_embedding()

    def feedforward_encoder(self, x, is_training):
        for i,l in enumerate(self.layers_encoder):
            print(x)
            x = l(x)
            # if i in self.args.layers_entropy:
            #     noise = tf.cond(is_training,
            #         lambda: tf.random_normal(x.get_shape(), mean=0,  stddev=self.args.noise_stddev),
            #         lambda: tf.zeros(x.get_shape()))
            #     x = tf.cast(x>10, x.dtype) * x
        return x

    def feedforward_decoder(self, x, is_training=True):
        for i,l in enumerate(self.layers_decoder):
            print(x)
            x = l(x, is_training=is_training)
        print(x)
        return x

    def decode_embedding(self):
        e = tf.placeholder(tf.float32, [1, self.args.layers[-1]], name='ph_embedding')
        out = self.feedforward_decoder(e, is_training=False)
        out = tf.identity(out, 'ph_embedding_decoded')

        return out






