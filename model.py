import tensorflow as tf
import numpy as np
from tensorflow.contrib.layers import batch_norm as bn
import sys, os, time, math, argparse, pickle
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
    def __init__(self, input_dim,
        layer_dimensions=[1024, 512, 256, 2],
        lambda_batchcorrection=0,
        num_batches=2,
        lambdas_entropy=[],
        layers_entropy=[],
        lambda_within_cluster_dists=0,
        lambda_l1=0,
        lambda_l2=0,
        batch_norm=True,
        dropout_p=1,
        activation=lrelu,
        activation_idreg=tf.nn.tanh,
        loss='mse',
        learning_rate=.001,
        restore_folder='',
        save_folder=''):

        if restore_folder:
            self._restore(restore_folder)
            return 

        self.input_dim = input_dim
        self.layer_dimensions = layer_dimensions
        self.batch_norm = batch_norm
        self.dropout_p = dropout_p
        self.activation = activation
        self.activation_idreg = activation_idreg
        self.lambda_l1 = lambda_l1
        self.lambda_l2 = lambda_l2
        self.lambda_batchcorrection = lambda_batchcorrection
        self.num_batches = num_batches
        self.lambdas_entropy = lambdas_entropy
        self.layers_entropy = layers_entropy
        self.lambda_within_cluster_dists = lambda_within_cluster_dists
        self.loss = loss
        self.save_folder = save_folder
        self.learning_rate = learning_rate

        self.x = tf.placeholder(tf.float32, shape=[None, input_dim], name='x')
        self.y = tf.placeholder(tf.float32, shape=[None, input_dim], name='y')
        self.batches = tf.placeholder(tf.int32, shape=[None], name='batches')
        self.spikein_mask = tf.placeholder(tf.int32, shape=[None], name='spikein_mask')
        self.is_training = tf.placeholder(tf.bool, shape=[], name='is_training')
        self.learning_rate_tensor = tf.placeholder(tf.float32, shape=[], name='learning_rate_tensor')

        self._build()
        self.init_session()
        
        self.graph_init(self.sess)

        self.iteration = 0
       
    def init_session(self, limit_gpu_fraction=.3, no_gpu=False):
        if limit_gpu_fraction:
            gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=limit_gpu_fraction)
            config = tf.ConfigProto(gpu_options=gpu_options)
            self.sess = tf.Session(config=config)
        elif no_gpu:
            config = tf.ConfigProto(device_count = {'GPU': 0})
            self.sess = tf.Session(config=config)
        else:
            self.sess = tf.Session()

    def _restore(self, restore_folder):
        tf.reset_default_graph()
        self.init_session()
        ckpt = tf.train.get_checkpoint_state(restore_folder)
        self.saver = tf.train.import_meta_graph('{}.meta'.format(ckpt.model_checkpoint_path))
        self.saver.restore(self.sess, ckpt.model_checkpoint_path)
        print("Model restored from {}".format(restore_folder))

    def _build(self):
        self._build_encoder()

        self._build_decoder()

        self._build_losses()

        self._build_optimization()

    def _feedforward_encoder(self, x, is_training):
        for i,l in enumerate(self.layers_encoder):
            #print(x)
            x = l(x)
        return x

    def _feedforward_decoder(self, x, is_training=True):
        for i,l in enumerate(self.layers_decoder):
            #print(x)
            x = l(x, is_training=is_training)
        return x

    def _build_encoder(self):
        self.layers_encoder = []
        input_plus_layers = [self.input_dim] + self.layer_dimensions

        for i,layer in enumerate(input_plus_layers[:-2]):
            name  = 'layer_encoder_{}'.format(i)
            if i in self.layers_entropy:
                #print("Adding entropy to {}".format(name))
                f = lambda x: self.activation_idreg(x)
                l = Layer(name, input_plus_layers[i], input_plus_layers[i+1], f, self.dropout_p, batch_norm=self.batch_norm)
            else:
                l = Layer(name, input_plus_layers[i], input_plus_layers[i+1], self.activation, self.dropout_p, batch_norm=self.batch_norm)
            self.layers_encoder.append(l)
        # last layer is linear, and fully-connected
        self.layers_encoder.append(Layer('layer_embedding', input_plus_layers[-2], input_plus_layers[-1], tf.identity, 1., batch_norm=self.batch_norm))

        self.embedded = self._feedforward_encoder(self.x, self.is_training)

    def _build_decoder(self):
        input_plus_layers = [self.input_dim] + self.layer_dimensions
        layers_decoder = input_plus_layers[::-1]
        self.layers_decoder = []

        # first layer is linear, and fully-connected
        for i,layer in enumerate(layers_decoder[:-2]):
            if i==0:
                l = Layer('layer_decoder_{}'.format(i), layers_decoder[i], layers_decoder[i+1], self.activation, 1., batch_norm=self.batch_norm)
            else:
                l = Layer('layer_decoder_{}'.format(i), layers_decoder[i], layers_decoder[i+1], self.activation, self.dropout_p, batch_norm=self.batch_norm)
            self.layers_decoder.append(l)
        # last decoder layer is linear and fully-connected
        if self.loss=='mse':
            output_act = tf.identity
        elif self.loss=='bce':
            output_act = tf.nn.sigmoid

        self.layers_decoder.append(Layer('layer_output', layers_decoder[-2], layers_decoder[-1], output_act, 1., batch_norm=self.batch_norm))

        self.reconstructed = self._feedforward_decoder(self.embedded)
        #print(self.reconstructed)

    def _build_losses(self):
        self.loss_recon = 0.

        if self.lambda_batchcorrection:
            with tf.variable_scope('reconstruction_mmd'):
                self._build_reconstruction_loss_mmd(self.reconstructed, self.x)
        else:
            with tf.variable_scope('reconstruction'):
                self._build_reconstruction_loss(self.reconstructed, self.x)

        with tf.variable_scope('entropy'):
            self._build_reg_entropy()

        with tf.variable_scope('batchcorrection'):
            self._build_reg_mmd()

        with tf.variable_scope('l1/l2'):
            self._build_reg_l1weights()
            self._build_reg_l2weights()

        self._build_total_loss()

    def _build_optimization(self):
        opt = tf.train.AdamOptimizer(self.learning_rate)
        tvs = tf.global_variables()

        tv_vals = opt.compute_gradients(self.loss, var_list=tvs)
        self.train_op = opt.apply_gradients(tv_vals, name='train_op')

    def _build_reconstruction_loss(self, reconstructed, y):
        if self.loss=='mse':
            self.loss_recon = (self.reconstructed - y)**2

        elif self.loss=='bce':
            self.loss_recon = -(y*tf.log(reconstructed+1e-9) + (1-y)*tf.log(1-reconstructed+1e-9))

        self.loss_recon = tf.reduce_mean(self.loss_recon)

        self.loss_recon = nameop(self.loss_recon, 'loss_recon')
        tf.add_to_collection('losses', self.loss_recon)

    def _build_reconstruction_loss_mmd(self, reconstructed, y):
        x_dists = self.pairwise_dists(y, y)
        x_dists = tf.sqrt(x_dists+1e-3)

        recon_dists = self.pairwise_dists(reconstructed, reconstructed)
        recon_dists = tf.sqrt(recon_dists+1e-3)


        for i in range(self.num_batches):
            recon_ = tf.boolean_mask(reconstructed, tf.equal(self.batches, i))
            y_ = tf.boolean_mask(y, tf.equal(self.batches, i))
            
            if i==0:
                l = (y_-recon_)**2
                self.loss_recon += tf.reduce_mean(l)
            # else:
            #     averge_direction = tf.reduce_mean(y_ - recon_, axis=0, keep_dims=True)
            #     deviation_from_averge_direction = ((y_ - recon_) - averge_direction)**2
            #     self.loss_recon += tf.reduce_mean(deviation_from_averge_direction)

            batch_x_rows = tf.boolean_mask(x_dists, tf.equal(self.batches, i))
            batch_x_rowscols = tf.boolean_mask(tf.transpose(batch_x_rows), tf.equal(self.batches, i))

            batch_recon_rows = tf.boolean_mask(recon_dists, tf.equal(self.batches, i))
            batch_recon_rowscols = tf.boolean_mask(tf.transpose(batch_recon_rows), tf.equal(self.batches, i))

            self.loss_recon += tf.reduce_mean(tf.abs(batch_x_rowscols - batch_recon_rowscols))

        self.loss_recon = nameop(self.loss_recon, 'loss_recon')
        tf.add_to_collection('losses', self.loss_recon)

    def _within_cluster_distances(self, act):
        dists = self.pairwise_dists(self.x, self.x)
        dists = tf.sqrt(dists+1e-3)

        binarized = tf.where(act>0, tf.ones_like(act), tf.zeros_like(act))
        out = self.pairwise_dists(binarized, binarized)
        same_cluster = tf.where(tf.equal(out,tf.zeros_like(out)), tf.ones_like(out), tf.zeros_like(out))

        within_cluster_distances = dists*same_cluster
        within_cluster_distances = tf.reduce_mean(within_cluster_distances)
        return within_cluster_distances

    def _build_reg_entropy(self):
        # fuzzy counting regularization
        self.loss_entropy = tf.constant(0.)
        for act in tf.get_collection('activations'):
            for add_entropy_to, lambda_entropy in zip(self.layers_entropy, self.lambdas_entropy):
                if 'encoder_{}'.format(add_entropy_to) in act.name:
                    # normalize to (0,1)
                    act = (act+1)/2
                    # sum down neurons
                    p = tf.reduce_sum(act, axis=0, keep_dims=True)
                    # normalize neuron sums
                    normalized = p / tf.reduce_sum(p)

                    within_cluster_distances = self._within_cluster_distances(act)
                    self.loss_withinclust = self.lambda_within_cluster_dists*within_cluster_distances
                    self.loss_withinclust = nameop(self.loss_withinclust, 'loss_withinclust')
                    tf.add_to_collection('losses', self.loss_withinclust)

                    normalized = nameop(normalized, 'normalized_activations_layer_{}'.format(add_entropy_to))
                    self.loss_entropy += lambda_entropy*tf.reduce_sum(-normalized*tf.log(normalized+1e-9))


        self.loss_entropy = nameop(self.loss_entropy, 'loss_entropy')
        tf.add_to_collection('losses', self.loss_entropy)

    def _build_reg_mmd(self):
        var_within = {}
        var_between = {}
        batch_sizes = {}
        self.loss_mmd = tf.constant(0.)
        if not self.lambda_batchcorrection:
            return

        K = self.pairwise_dists(self.embedded, self.embedded)
        K = tf.sqrt(K+1e-3)

        i = 0
        batch1_spikein_mask = tf.boolean_mask(self.spikein_mask, tf.equal(self.batches, i))

        batch1_rows = tf.boolean_mask(K, tf.equal(self.batches, i))
        batch1_rows = tf.boolean_mask(batch1_rows, tf.equal(batch1_spikein_mask, 1))
        batch1_rowscols = tf.boolean_mask(tf.transpose(batch1_rows), tf.equal(self.batches, i))
        batch1_rowscols = tf.boolean_mask(batch1_rowscols, tf.equal(batch1_spikein_mask, 1))


        K_b1 = tf.matrix_band_part(batch1_rowscols, 0, -1) # just upper triangular part
        #n_rows_b1 = tf.reduce_sum(0*tf.ones_like(K_b1)[:,0]+1)
        n_rows_b1 = tf.cast(tf.reduce_sum(batch1_spikein_mask), tf.float32)
        nameop(n_rows_b1, 'b1')
        K_b1 = tf.reduce_sum(K_b1) / (n_rows_b1**2 + 1)

        var_within[i] = K_b1
        batch_sizes[i] = n_rows_b1


        for j in range(1, self.num_batches):
            batch2_spikein_mask = tf.boolean_mask(self.spikein_mask, tf.equal(self.batches, j))

            batch2_rows = tf.boolean_mask(K, tf.equal(self.batches, j))
            batch2_rows = tf.boolean_mask(batch2_rows, tf.equal(batch2_spikein_mask, 1))
            batch2_rowscols = tf.boolean_mask(tf.transpose(batch2_rows), tf.equal(self.batches, j))
            batch2_rowscols = tf.boolean_mask(batch2_rowscols, tf.equal(batch2_spikein_mask, 1))


            K_b2 = tf.matrix_band_part(batch2_rowscols, 0, -1) # just upper triangular part
            #n_rows_b2 = tf.reduce_sum(0*tf.ones_like(K_b2)[:,0]+1)
            n_rows_b2 = tf.cast(tf.reduce_sum(batch2_spikein_mask), tf.float32)
            K_b2 = tf.reduce_sum(K_b2) / (n_rows_b2**2 + 1)

            var_within[j] = K_b2
            batch_sizes[j] = n_rows_b2

            K_12 = tf.boolean_mask(K, tf.equal(self.batches, i))
            K_12 = tf.boolean_mask(K_12, tf.equal(batch1_spikein_mask, 1))
            K_12 = tf.boolean_mask(tf.transpose(K_12), tf.equal(self.batches, j))
            K_12 = tf.boolean_mask(K_12, tf.equal(batch2_spikein_mask, 1))
            K_12_ = tf.reduce_sum(tf.transpose(K_12))

            mmd_pair = var_within[i] + var_within[j] - 2 * K_12_ / (batch_sizes[i]*batch_sizes[j]+1)
            self.loss_mmd += tf.abs(mmd_pair)

        self.loss_mmd = self.lambda_batchcorrection*(self.loss_mmd)
        self.loss_mmd = nameop(self.loss_mmd, 'loss_mmd')
        tf.add_to_collection('losses', self.loss_mmd)

    def _build_reg_l1weights(self):
        self.loss_l1weights = self.lambda_l1*tf.reduce_mean([tf.reduce_mean(tf.abs(tv)) for tv in tf.global_variables()])

        self.loss_l1weights = nameop(self.loss_l1weights, 'loss_l1')
        tf.add_to_collection('losses', self.loss_l1weights)

    def _build_reg_l2weights(self):
        self.loss_l2weights = self.lambda_l2*tf.reduce_mean([tf.reduce_mean(tf.nn.l2_loss(tv)) for tv in tf.global_variables() if 'W' in tv.name])
        
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

    def rbf_kernel(self, x, sigma=None):
        if not sigma:
            sigma = tf.reduce_mean(x, axis=1, keep_dims=True)
        x = np.e**(- (x / (sigma)))
        return x

    def graph_init(self, sess=None):
        if not sess: sess = self.sess

        self.saver = tf.train.Saver(tf.global_variables(), max_to_keep=1)

        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)


        return self.saver

    def save(self, iteration=None, saver=None, sess=None, folder=None):
        if not iteration: iteration=self.iteration
        if not saver: saver=self.saver
        if not sess: sess=self.sess
        if not folder: folder=self.save_folder

        savefile = os.path.join(folder, 'AE')
        saver.save(sess, savefile , global_step=iteration, write_meta_graph=True)
        print("Model saved to {}".format(savefile))

    def get_loss_names(self):
        losses = [tns.name[:-2].replace('loss_','').split('/')[-1] for tns in tf.get_collection('losses')]
        return "Losses: {}".format(' '.join(losses))

    def train(self, load, steps):
        start = self.iteration
        while (self.iteration - start) < steps:
            self.iteration+=1

            batch = load.next_batch(self.sess, batch_size=256)
            if len(batch)>1:
                batch, batch_labels, batch_spikein_mask = batch
                
                feed = {tbn('x:0'): batch,
                        tbn('y:0'): batch,
                        tbn('batches:0'): batch_labels,
                        tbn('spikein_mask:0'): batch_spikein_mask}
            else:
                batch = batch[0]
                feed = {tbn('x:0'): batch,
                        tbn('y:0'): batch}


            feed[tbn('is_training:0')] = True
            feed[tbn('learning_rate_tensor:0')] = self.learning_rate

            ops = [obn('train_op')]

            _ = self.sess.run(ops, feed_dict=feed)

    def get_loss(self, load):
        losses = None

        for i,batch in enumerate(load.iter_batches()):
            if len(batch)>1:
                batch, batch_labels, batch_spikein_mask = batch
                feed = {tbn('x:0'): batch,
                        tbn('y:0'): batch,
                        tbn('batches:0'): batch_labels,
                        tbn('spikein_mask:0'): batch_spikein_mask}
            else:
                batch = batch[0]
                feed = {tbn('x:0'): batch,
                        tbn('y:0'): batch}

            batch_losses = self.sess.run(tf.get_collection('losses'), feed_dict=feed)

            if not losses:
                losses = batch_losses
            else:
                losses = [loss+batch_loss for loss,batch_loss in zip(losses,batch_losses)]
            
        losses = [loss/float(i+1) for loss in losses]
        lstring = ' '.join(['{:.3f}'.format(loss) for loss in losses])
        
        return lstring

    def save(self, save_folder=None, sess=None):
        if not save_folder: save_folder = self.save_folder
        if not sess: sess = self.sess

        if not self.save_folder:
            print("Cannot save model, no save folder given!")
            return

        self.saver.save(sess, os.path.join(save_folder, 'SAUCIE'), write_meta_graph=True)

    def get_layer(self, load, name):
        tensor_name = "{}:0".format(name)
        tensor = tbn(tensor_name)
        layer = []
        labels = []
        spikein_mask = []
        for batch in load.iter_batches():
            if len(batch)>1:
                batch, batch_labels, batch_spikein_mask = batch
                labels.append(batch_labels)
                spikein_mask.append(batch_spikein_mask)

                feed = {tbn('x:0'): batch,
                        tbn('y:0'): batch,
                        tbn('batches:0'): batch_labels,
                        tbn('spikein_mask:0'): batch_spikein_mask}
            else:
                batch = batch[0]
                feed = {tbn('x:0'): batch,
                        tbn('y:0'): batch}
            
            [act] = self.sess.run([tensor], feed_dict=feed)
            layer.append(act)

        layer = np.concatenate(layer, axis=0)

        if labels:
            labels = np.concatenate(labels, axis=0)
            spikein_mask = np.concatenate(spikein_mask, axis=0)
            return layer, labels, spikein_mask
        else:
            return layer

    def get_clusters(self, load, thresh=0, BIN_MIN=10, verbose=True):
        acts = self.get_layer(load, 'layer_encoder_{}_activation'.format(self.layers_entropy[-1]))
        if isinstance(acts, list):
            acts, _, _ = acts

        binarized = np.where(acts>thresh, 1, 0)

        unique_rows, counts = np.unique(binarized, axis=0, return_counts=True)
        unique_rows = unique_rows[counts>BIN_MIN]

        num_clusters = unique_rows.shape[0]
        if num_clusters>1000:
            print("Too many clusters to go through...")
            return None, None
        
        num_clusters = 0
        rows_clustered = 0
        clusters = -1*np.ones(acts.shape[0])
        for i,row in enumerate(unique_rows):
            rows_equal_to_this_code = np.where(np.all(binarized==row, axis=1))[0]

            clusters[rows_equal_to_this_code] = num_clusters
            num_clusters += 1
            rows_clustered += rows_equal_to_this_code.shape[0]

        if verbose:
            print("---- Num clusters: {} ---- Percent clustered: {:.3f} ----".format(num_clusters, 1.*rows_clustered/clusters.shape[0]))

        return num_clusters, clusters






















































