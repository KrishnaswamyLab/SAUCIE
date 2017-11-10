import tensorflow as tf
import numpy as np
from tensorflow.contrib.layers import batch_norm as bn
import sys, os, time, math, argparse
import matplotlib.pyplot as plt
from tensorflow.contrib.layers import xavier_initializer
from utils import *

def nameop(op, name):

    op = tf.identity(op, name=name)
    return op

class Layer(object):
    def __init__(self, name, dim_in, dim_out, activation=tf.nn.relu, dropout_p=1., batch_norm=True):
        self.W = tf.get_variable(shape=[dim_in,dim_out], initializer=xavier_initializer() , name='W_{}'.format(name))
        self.b = tf.get_variable(shape=[dim_out], initializer=tf.zeros_initializer(), name='b_{}'.format(name))

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

class SAUCIE(object):
    def __init__(self, args):
        self.args = args
        self.x = tf.placeholder(tf.float32, shape=[None, args.input_dim], name='x')
        self.y = tf.placeholder(tf.float32, shape=[None, args.input_dim], name='y')
        self.batches = tf.placeholder(tf.int32, shape=[None], name='batches')
        self.is_training = tf.placeholder(tf.bool, shape=[], name='is_training')
        self.learning_rate = tf.placeholder(tf.float32, shape=[], name='learning_rate')

        self._build()
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.45)
        self.sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) 
        self.graph_init(self.sess)

        self.iteration = 0
        
    def _build(self):
        self._build_encoder()

        self._build_decoder()

        self._build_losses()

        self._build_optimization()

    def _feedforward_encoder(self, x, is_training):
        for i,l in enumerate(self.layers_encoder):
            print(x)
            x = l(x)
        return x

    def _feedforward_decoder(self, x, is_training=True):
        for i,l in enumerate(self.layers_decoder):
            print(x)
            x = l(x, is_training=is_training)
        return x

    def _build_encoder(self):
        self.layers_encoder = []
        input_plus_layers = [self.args.input_dim] + self.args.layers

        for i,layer in enumerate(input_plus_layers[:-2]):
            name  = 'layer_encoder_{}'.format(i)
            if i in self.args.layers_entropy:
                print("Adding entropy to {}".format(name))
                f = lambda x: self.args.activation_idreg(x)
                l = Layer(name, input_plus_layers[i], input_plus_layers[i+1], f, self.args.dropout_p, batch_norm=self.args.batch_norm)
            else:
                l = Layer(name, input_plus_layers[i], input_plus_layers[i+1], self.args.activation, self.args.dropout_p, batch_norm=self.args.batch_norm)
            self.layers_encoder.append(l)
        # last layer is linear, and fully-connected
        self.layers_encoder.append(Layer('layer_embedding', input_plus_layers[-2], input_plus_layers[-1], tf.identity, 1., batch_norm=self.args.batch_norm))

        self.embedded = self._feedforward_encoder(self.x, self.is_training)

    def _build_decoder(self):
        input_plus_layers = [self.args.input_dim] + self.args.layers
        layers_decoder = input_plus_layers[::-1]
        self.layers_decoder = []

        # first layer is linear, and fully-connected
        for i,layer in enumerate(layers_decoder[:-2]):
            if i==0:
                l = Layer('layer_decoder_{}'.format(i), layers_decoder[i], layers_decoder[i+1], self.args.activation, 1., batch_norm=self.args.batch_norm)
            else:
                l = Layer('layer_decoder_{}'.format(i), layers_decoder[i], layers_decoder[i+1], self.args.activation, self.args.dropout_p, batch_norm=self.args.batch_norm)
            self.layers_decoder.append(l)
        # last decoder layer is linear and fully-connected
        if self.args.loss=='mse':
            output_act = tf.identity
        elif self.args.loss=='bce':
            output_act = tf.nn.sigmoid

        self.layers_decoder.append(Layer('layer_output', layers_decoder[-2], layers_decoder[-1], output_act, 1., batch_norm=self.args.batch_norm))

        self.reconstructed = self._feedforward_decoder(self.embedded)
        print(self.reconstructed)

    def _build_losses(self):
        if self.args.lambda_batchcorrection:
            self._build_reconstruction_loss_mmd(self.reconstructed, self.x)
        else:
            self._build_reconstruction_loss(self.reconstructed, self.x)
        self._build_reg_clustuse()
        self._build_reg_sparsity()
        self._build_reg_entropy()
        if self.args.batchcorrection=='mmd':
            self._build_reg_mmd()
        elif self.args.batchcorrection=='adversary':
            self._build_reg_adversary()
        self._build_reg_l1weights()
        self._build_reg_l2weights()
        self._build_total_loss()

    def _build_optimization(self):
        opt = tf.train.AdamOptimizer(self.learning_rate)
        tvs = [v for v in tf.global_variables() if 'adversary' not in v.name]
        self.train_op = opt.minimize(self.loss, name='train_op', var_list=tvs)

    def _build_reconstruction_loss(self, reconstructed, y):
        if self.args.loss=='mse':
            self.loss_recon = (reconstructed - y)**2

        elif self.args.loss=='bce':
            self.loss_recon = -(y*tf.log(reconstructed+1e-9) + (1-y)*tf.log(1-reconstructed+1e-9))

        self.loss_recon = tf.reduce_mean(self.loss_recon)

        self.loss_recon = nameop(self.loss_recon, 'loss_recon')
        tf.add_to_collection('losses', self.loss_recon)

    def _build_reconstruction_loss_mmd(self, reconstructed, x):
        x_dists = self.pairwise_dists(x, x)
        x_dists = tf.sqrt(x_dists+1e-3)
        #x_dists = self.rbf_kernel(x_dists)
        #x_dists = x_dists * tf.transpose(x_dists)
        recon_dists = self.pairwise_dists(reconstructed, reconstructed)
        recon_dists = tf.sqrt(recon_dists+1e-3)
        #recon_dists = self.rbf_kernel(recon_dists)
        #recon_dists = recon_dists * tf.transpose(recon_dists)

        self.loss_recon = 0.

        for i in range(self.args.num_batches):
            if i==0:
                recon_ = tf.boolean_mask(reconstructed, tf.equal(self.batches, i))
                x_ = tf.boolean_mask(x, tf.equal(self.batches, i))
                l = tf.abs(x_-recon_)
                self.loss_recon+= tf.reduce_mean(l)

            batch_x_rows = tf.boolean_mask(x_dists, tf.equal(self.batches, i))
            batch_x_rowscols = tf.boolean_mask(tf.transpose(batch_x_rows), tf.equal(self.batches, i))

            batch_recon_rows = tf.boolean_mask(recon_dists, tf.equal(self.batches, i))
            batch_recon_rowscols = tf.boolean_mask(tf.transpose(batch_recon_rows), tf.equal(self.batches, i))

            self.loss_recon += tf.reduce_mean(tf.abs(batch_x_rowscols - batch_recon_rowscols))

        self.loss_recon = nameop(self.loss_recon, 'loss_recon')
        tf.add_to_collection('losses', self.loss_recon)

    def _build_reg_clustuse(self):
        # cluster use penalty
        self.loss_clustuse = tf.constant(0.)
        for act in tf.get_collection('activations'):    
            for layer_index, lambda_clustuse in zip(self.args.layers_clustuse, self.args.lambdas_clustuse):
                if 'encoder_{}'.format(layer_index) in act.name:
                    act = (act + 1) / 2
                    clustuse = tf.reduce_sum(act, axis=0)
                    clustuse = tf.nn.sigmoid(clustuse)
                    clustuse = tf.reduce_mean(clustuse)

                    self.loss_clustuse += lambda_clustuse*clustuse
        
        self.loss_clustuse = nameop(self.loss_clustuse, 'loss_clustuse')
        tf.add_to_collection('losses', self.loss_clustuse)

    def _build_reg_sparsity(self):
        # sparsity regularization
        self.loss_sparse = tf.constant(0.)
        for act in tf.get_collection('activations'):
            for layer_index, lambda_sparsity in zip(self.args.layers_sparsity, self.args.lambdas_sparsity):
                if 'encoder_{}'.format(layer_index) in act.name:
                    self.loss_sparse += lambda_sparsity*tf.reduce_mean(tf.abs(act))

        self.loss_sparse = nameop(self.loss_sparse, 'loss_sparse')
        tf.add_to_collection('losses', self.loss_sparse)

    def _build_reg_entropy(self):
        # fuzzy counting regularization
        self.loss_entropy = tf.constant(0.)
        for act in tf.get_collection('activations'):
            for add_entropy_to, lambda_entropy in zip(self.args.layers_entropy, self.args.lambdas_entropy):
                if 'encoder_{}'.format(add_entropy_to) in act.name or \
                   (add_entropy_to=='embedding' and 'embedding' in act.name):
                    if self.args.normalization_method=='neuronuse':
                        # normalize to (0,1)
                        act = (act+1)/2
                        # sum down neurons
                        p = tf.reduce_sum(act, axis=0, keep_dims=True)
                        # normalize neuron sums
                        normalized = p / tf.reduce_sum(p)
                    elif self.args.normalization_method=='softmax':
                        normalized = tf.nn.softmax(act, dim=1)
                    elif self.args.normalization_method=='tanh':
                        normalized = (act+1) / 2   
                    elif self.args.normalization_method=='none':
                        normalized = act
                    normalized = nameop(normalized, 'normalized_activations_layer_{}'.format(add_entropy_to))
                    self.loss_entropy += lambda_entropy*tf.reduce_sum(-normalized*tf.log(normalized+1e-9))

        self.loss_entropy = nameop(self.loss_entropy, 'loss_entropy')
        #tf.add_to_collection('losses', self.loss_entropy)
        trainop = tf.train.AdamOptimizer().minimize(self.loss_entropy, name='entropy_op', var_list=[v for v in tf.global_variables() if 'encoder_2' in v.name])
        #nameop(trainop, 'entropy_op')

    def _build_reg_mmd(self):
        var_within = {}
        var_between = {}
        batch_sizes = {}
        self.loss_mmd = tf.constant(0.)
        if not self.args.lambda_batchcorrection:
            return
        


        K = self.pairwise_dists(self.embedded, self.embedded)
        K = self.rbf_kernel(K)
        K = K * tf.transpose(K)
        for i in range(self.args.num_batches):
            if i not in var_within:
                batch1_rows = tf.boolean_mask(K, tf.equal(self.batches, i))
                batch1_rowscols = tf.boolean_mask(tf.transpose(batch1_rows), tf.equal(self.batches, i))

                K_b1 = tf.matrix_band_part(batch1_rowscols, 0, -1) - tf.matrix_band_part(batch1_rowscols, 0, 0) # just upper triangular part
                n_rows_b1 = tf.reduce_sum(0*tf.ones_like(K_b1)[:,0]+1)
                K_b1 = tf.reduce_sum(K_b1) / (n_rows_b1**2 + 1e-9)

                var_within[i] = K_b1
                batch_sizes[i] = n_rows_b1


            for j in range(i+1, self.args.num_batches):
                if not j in var_within:
                    batch2_rows = tf.boolean_mask(K, tf.equal(self.batches, j))
                    batch2_rowscols = tf.boolean_mask(tf.transpose(batch2_rows), tf.equal(self.batches, j))

                    K_b2 = tf.matrix_band_part(batch2_rowscols, 0, -1) - tf.matrix_band_part(batch2_rowscols, 0, 0) # just upper triangular part
                    n_rows_b2 = tf.reduce_sum(0*tf.ones_like(K_b2)[:,0]+1)
                    K_b2 = tf.reduce_sum(K_b2) / (n_rows_b2**2 + 1e-9)

                    var_within[j] = K_b2
                    batch_sizes[j] = n_rows_b2



                K_12 = tf.boolean_mask(K, tf.equal(self.batches, i))
                K_12 = tf.boolean_mask(tf.transpose(K_12), tf.equal(self.batches, j))
                K_12 = tf.reduce_mean(tf.transpose(K_12))
                self.loss_mmd+= var_within[i] + var_within[j] - 2 * K_12 / (batch_sizes[i]*batch_sizes[j])

        self.loss_mmd /= self.args.num_batches

        self.loss_mmd = self.args.lambda_batchcorrection*tf.abs(self.loss_mmd)
        self.loss_mmd = nameop(self.loss_mmd, 'loss_mmd')
        tf.add_to_collection('losses', self.loss_mmd)

    def _build_reg_adversary(self):
        if not self.args.lambda_batchcorrection:
            return
        
        d = self.args.layers[-1]
        layers = [d, 2000, 1000, 1000, 1]
        layers_adversary = []
        for i,l in enumerate(layers[:-1]):
            name = 'adversary_layer_{}'.format(i)
            if i<len(layers[:-1])-1:
                layers_adversary.append(Layer(name, layers[i], layers[i+1], activation=tf.nn.relu, batch_norm=True))
            else:
                layers_adversary.append(Layer(name, layers[i], layers[i+1], activation=tf.nn.sigmoid, batch_norm=True))

        def feedforward_adversary(x):
            for l_adv in layers_adversary:
                print(x)
                x = l_adv(x)
            return x


        with tf.variable_scope('adversary') as scope:
            adversary_loss = 0.
            saucie_loss = 0.
            adversary_probs = feedforward_adversary(self.embedded)

            for i in range(self.args.num_batches):
                this_batch = tf.boolean_mask(adversary_probs, tf.equal(self.batches, i))
                
                if i==0:
                    ref = this_batch
                else:
                    adversary_loss +=  -tf.reduce_mean(tf.log(ref + 1e-9)) - tf.reduce_mean(tf.log(1 - this_batch + 1e-9))
                    saucie_loss += -tf.reduce_mean(tf.log(this_batch + 1e-9))


        # saucie loss
        saucie_loss *= lambda_batchcorrection
        self.loss_batchcorrection = nameop(saucie_loss, 'loss_mmd')
        tf.add_to_collection('losses', self.loss_mmd)

        # adversary loss
        adversary_loss = nameop(adversary_loss, 'adversary_loss')
        nameop(adversary_probs, 'adversary_probs')
        opt = tf.train.AdamOptimizer(.001)
        adversary_tvs = [v for v in tf.global_variables() if 'adversary' in v.name]
        train_op_adversary = opt.minimize(adversary_loss, var_list=adversary_tvs, name='train_op_adversary')

    def _build_reg_l1weights(self):
        self.loss_l1weights = self.args.lambda_l1*tf.reduce_mean([tf.reduce_mean(tf.abs(tv)) for tv in tf.global_variables()])

        self.loss_l1weights = nameop(self.loss_l1weights, 'loss_l1')
        tf.add_to_collection('losses', self.loss_l1weights)

    def _build_reg_l2weights(self):
        self.loss_l2weights = self.args.lambda_l2*tf.reduce_mean([tf.reduce_mean(tf.nn.l2_loss(tv)) for tv in tf.global_variables() if 'W' in tv.name])
        
        self.loss_l2weights = nameop(self.loss_l2weights, 'loss_l2')
        tf.add_to_collection('losses', self.loss_l2weights)

    def _build_total_loss(self):
        self.loss = tf.constant(0.)
        for l in tf.get_collection('losses'):
            self.loss += l
        self.loss = nameop(self.loss, 'loss')

    def pairwise_dists(self, x1, x2):
        r1 = tf.reduce_sum(x1*x1, 1, keep_dims=True)
        r2 = tf.reduce_sum(x2*x2, 1, keep_dims=True)

        D = r1 - 2*tf.matmul(x1, tf.transpose(x2)) + tf.transpose(r2)
        return D

    def rbf_kernel(self, x, sigma=1):
        sigma = tf.reduce_mean(x, axis=1, keep_dims=True)
        x = np.e**(- ((x**2) / (sigma)))
        return x

    def graph_init(self, sess=None):
        if not sess: sess = self.sess

        init_vars = tf.global_variables()
        init_op = tf.variables_initializer(init_vars)
        sess.run(init_op)
        self.saver = tf.train.Saver(init_vars, max_to_keep=1)
        return self.saver

    def save(self, iteration=None, saver=None, sess=None, folder=None):
        if not iteration: iteration=self.iteration
        if not saver: saver=self.saver
        if not sess: sess=self.sess
        if not folder: folder=self.args.save_folder

        savefile = os.path.join(folder, 'AE')
        saver.save(sess, savefile , global_step=iteration, write_meta_graph=True)
        print("Model saved to {}".format(savefile))

    def get_loss_names(self):
        losses = [tns.name[:-2].replace('loss_','') for tns in tf.get_collection('losses')]
        return "Losses: {}".format(' '.join(losses))

    def train(self, load, steps):
        t = time.time()
        start = self.iteration
        while (self.iteration - start) < steps:
            self.iteration+=1

            batch = load.next_batch(self.args.batch_size)
            if isinstance(batch, tuple):
                batch, batch_labels = batch
                    
            feed = {tbn('x:0'): batch,
                    tbn('y:0'): batch,
                    tbn('batches:0'): batch_labels,
                    tbn('is_training:0'): True,
                    tbn('learning_rate:0'): self.args.learning_rate}
            
            if self.args.batchcorrection=='adversary':
                _ = self.sess.run([obn('train_op_adversary')], feed_dict=feed)

            if self.iteration>1000 and self.iteration%5==0:
                _ = self.sess.run([obn('entropy_op')], feed_dict=feed)
            _ = self.sess.run([obn('train_op')], feed_dict=feed)

    def get_loss(self, load):
        losses = None

        for i,batch in enumerate(load.iter_batches()):
            if isinstance(batch, tuple):
                batch, batch_labels = batch
                
            feed = {tbn('x:0'): batch,
                    tbn('y:0'): batch,
                    tbn('batches:0'): batch_labels,
                    tbn('is_training:0'): True,
                    tbn('learning_rate:0'): self.args.learning_rate}


            batch_losses = self.sess.run(tf.get_collection('losses'), feed_dict=feed)
            if not losses:
                losses = batch_losses
            else:
                losses = [loss+batch_loss for loss,batch_loss in zip(losses,batch_losses)]
            
        losses = [loss/float(i+1) for loss in losses]
        lstring = ' '.join(['{:.3f}'.format(loss) for loss in losses])
        
        return lstring


















