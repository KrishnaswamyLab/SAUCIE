# -*- coding: utf-8 -*-
# File: saucie.py
# Author: Krishnan Srinivasan <krishnan1994 at gmail>
# Date: 21.09.2017
# Last Modified Date: 21.09.2017

"""
Classes and functions used to instantiate SAUCIE.
"""

import tensorflow as tf
import saucie_utils as utils
import pickle
import os
import numpy as np
import glob

from typing import NamedTuple
from collections import OrderedDict
from datetime import datetime
from saucie_utils import define_var_scope, define_name_scope

ACT_FNS = {'lrelu': utils.lrelu,
           'relu': tf.nn.relu,
           'softmax': tf.nn.softmax,
           'softplus': tf.nn.softplus,
           'sigmoid': tf.nn.sigmoid,
           'tanh': tf.nn.tanh}

SAVE_PATH = './saucie_models'

DATETIME_FMT = '%y-%m-%d-runs'

def load_model_from_config(dataset='mnist', config_path=SAVE_PATH+'/best.config'):
    if os.path.exists(config_path):
        with open(config_path, 'rb') as f:
            config = pickle.load(f)
        model = Saucie(**config)
    else:
        config = default_config(dataset)
        model = Saucie(**config)
    return model, config


def default_config(dataset='mnist'):
    sparse_config = utils.SparseLayerConfig(num_layers=3, id_lam=np.array([1e-3,0.,0.], dtype=utils.FLOAT_DTYPE))
    if dataset == 'mnist':
        input_dim = 784
    elif dataset == 'zika':
        input_dim = 35
    config = OrderedDict(input_dim=input_dim, encoder_layers=[1024,512,256],
                         emb_dim=2, act_fn='tanh', d_act_fn='tanh', use_bias=True,
                         loss_fn='bce', opt_method='adam', lr=1e-3, batch_norm=True,
                         sparse_config=sparse_config,
                         save_path='{}/{}'.format(SAVE_PATH,dataset))
    return config


def make_config(args):
    sparse_config = utils.SparseLayerConfig(num_layers=len(args.id_lam),
                                            id_lam=args.id_lam, l1_lam=args.l1_lam,
                                            l1_w_lam=args.l1_w_lam,
                                            l2_w_lam=args.l2_w_lam)
    if args.dataset == 'mnist':
        input_dim = 784
    elif args.dataset == 'zika':
        input_dim = 35
    config = OrderedDict(input_dim=input_dim, encoder_layers=args.encoder_layers,
                         emb_dim=args.emb_dim, act_fn=args.act_fn,
                         d_act_fn=args.d_act_fn, use_bias=args.use_bias, loss_fn=args.loss_fn,
                         opt_method=args.opt_method, lr=args.lr, batch_norm=args.batch_norm,
                         sparse_config=sparse_config,
                         save_path='{}/{}'.format(args.model_dir, args.dataset))
    return config


class Layer():
    # defined as class attribute so can be changed for all layers at once
    w_init = tf.contrib.layers.xavier_initializer
    b_init = tf.zeros_initializer

    def __init__(self, name, in_dim, out_dim, act_fn='relu', batch_norm=False, is_training=False, use_bias=True):
        self.w_, self.b_ = None, None
        self.act_ = None
        self.name = name
        self._in_dim = in_dim
        self._out_dim = out_dim
        self._batch_norm = batch_norm # whether to batch normalize layer output
        self._act_fn = act_fn # name of activation function
        self.is_training_ = is_training
        self._use_bias = use_bias

    def __call__(self, x):
        x_dtype = x.dtype
        if self.act_ is None:
            with tf.variable_scope(self.name):   
                self.w_ = tf.get_variable('w', [self._in_dim, self._out_dim], dtype=x_dtype, initializer=self.w_init(dtype=x_dtype))
                if self._use_bias:
                    self.b_ = tf.get_variable('b', [self._out_dim], dtype=x_dtype, initializer=self.b_init(dtype=x_dtype))
                    self.act_ = tf.add(tf.matmul(x, self.w_), self.b_, name='act_aff')
                else:
                    self.act_ = tf.matmul(x, self.w_, name='act_aff')
                if self._act_fn:
                    if type(self._act_fn) == list:
                        for act_fn in self._act_fn:
                            self.act_ = ACT_FNS[act_fn](self.act_, name='act_{}'.format(act_fn))
                    else:
                        self.act_ = ACT_FNS[self._act_fn](self.act_, name='act_{}'.format(self._act_fn)) 
                if self._batch_norm:
                    self.act_ = tf.layers.batch_normalization(self.act_, center=True,
                                        training=self.is_training_, name='bn')
            return self.act_
        else:
            return self.act_


class Saucie():
    def __init__(self, input_dim, encoder_layers, emb_dim, act_fn, d_act_fn,
                 use_bias, loss_fn, opt_method, lr, batch_norm, sparse_config,
                 save_path=SAVE_PATH, overwrite=False):
        """
        Args:
            input_dim: shape of input vector
            encoder_layers: shape of hidden layers of encoder
            emb_dim: shape of bottleneck layer
            act_fn: list of activation functions for each encoder layer or a single activation
                function to use everywhere
            d_act_fn: same as above for decoder
            use_bias: whether or not to use bias variables in affine transformation
            loss_fn: mse or bce for computing reconstruction loss
            opt_method: name of optimizer
            lr: learning rate for optimizer
            batch_norm: whether or not to use batch normalization between encoding layers
            sparse_config: config of type SparseLayerConfig which contains sparsity coefficients
                for l1 and id regularization on encoder layers.
            save_path: base of model save path
        """
        self._input_dim = input_dim
        self._encoder_layers = encoder_layers
        self._emb_dim = emb_dim
        self._act_fn = act_fn
        self._d_act_fn = d_act_fn
        self._use_bias = use_bias
        self._loss_fn = loss_fn
        self._opt_method = opt_method
        self._lr = float(lr)
        self._batch_norm = batch_norm
        self._sparse_config = sparse_config
        self._model_config = OrderedDict(
            input_dim=input_dim, encoder_layers=encoder_layers,
            emb_dim=emb_dim, act_fn=act_fn, d_act_fn=d_act_fn, use_bias=use_bias,
            loss_fn=loss_fn, opt_method=opt_method, lr=lr, batch_norm=batch_norm,
            sparse_config=sparse_config)
        current_date = datetime.now().strftime(DATETIME_FMT)
        run = 0
        while os.path.exists(os.path.join(save_path, current_date, str(run))):
            config_path = os.path.join(save_path, current_date, str(run), 'model.config')
            if os.path.exists(config_path):
                with open(config_path, 'rb') as f:
                    config = pickle.load(f)
                if config == self._model_config and overwrite:
                    break
            run += 1
        self.save_path = os.path.join(save_path, current_date, str(run))
        if not os.path.exists(self.save_path):
            os.makedirs(self.save_path)
        self.save_config()
        self.epochs_trained = 0

    def __dict__(self):
        return self._model_config

    def save_config(self, config_name='model.config'):
        config_path = os.path.join(self.save_path, config_name)
        with open(config_path, 'wb') as f:
            pickle.dump(self._model_config, f)

    def restore_model(self, sess, ckpt_dir, ckpt_name='best.model'):
        ckpt = tf.train.get_checkpoint_state(ckpt_dir, ckpt_name)
        if ckpt is None:
            return False
        else:
            self.saver.restore(sess, ckpt.model_checkpoint_path)
            return True

    def save_model(self, sess, filename, step=None):
        if filename != 'best.model':
            if len(self.saver.last_checkpoints) == 5:
                for ckpt in self.saver.last_checkpoints:
                    if ckpt != self.save_path + '/best.model':
                        break
                for ckpt_file in glob.glob(ckpt + '*'):
                    os.remove(ckpt_file)
                self.saver.last_checkpoints.remove(ckpt)
        self.saver.save(sess, self.save_path + '/{}'.format(filename), global_step=step)

    def build(self, sess):
        self.sess = sess
        self.x_ = tf.placeholder(utils.TF_FLOAT_DTYPE, shape=[None, self._input_dim], name='x')
        self.is_training_ = tf.placeholder(tf.bool, name='is_training')
        self.hidden_layers = []

        self.encoder
        tf.logging.debug('Built SAUCIE encoder')
        self.decoder
        tf.logging.debug('Built SAUCIE decoder')
        self.loss
        self.optimize
        tf.logging.debug('Built SAUCIE loss ops and optimizer')

        sess.run(tf.global_variables_initializer())

        self.saver = tf.train.Saver(max_to_keep=5)
        if os.path.exists(self.save_path + '/best.model'):
            restored = self.restore_model(sess, self.save_path, 'best.model')
            if restored:
                tf.logging.info('Restored model: {}'.format(self.save_path + '/best.model'))
                tf.logging.info('Already trained for {} steps, {} epochs'.format(*sess.run([self.global_step_, self.current_epoch_])))

        tf.logging.debug('Finished building SAUCIE')
        return

    @define_var_scope
    def encoder(self):
        x = self.x_
        in_dim = self._input_dim
        id_lam = self._sparse_config.id_lam
        l1_lam = self._sparse_config.l1_lam
        for i, out_dim in enumerate(self._encoder_layers):
            layer_name = 'layer-{}'.format(i)
            act_fn = self._act_fn
            if type(self._act_fn) == list: # specify act_fn for each layer
                act_fn = self._act_fn[i]
            if id_lam[i] != 0.:
                act_fn = 'relu'
            x = self.add_layer(x, in_dim, out_dim, self._batch_norm,
                               layer_name, act_fn, self._use_bias)
            if l1_lam[i] != 0.:
                tf.add_to_collection('l1_regularization', x)
            if id_lam[i] != 0.:
                tf.add_to_collection('id_regularization', x) 
                """
                noise = tf.cond(self.is_training_,
                        lambda: tf.random_normal(tf.shape(x), mean=0, stddev=1),
                        lambda: tf.zeros_like(x),
                        name='noise')
                x += noise
                """
            in_dim = out_dim
        layer_name = 'embedding'
        out_dim = self._emb_dim
        encoded_ = self.add_layer(x, in_dim, out_dim, self._batch_norm, layer_name, None,
                                  self._use_bias)
        return encoded_

    @define_var_scope
    def decoder(self):
        in_dim = self._emb_dim
        x = self.encoder
        for i, out_dim in enumerate(self._encoder_layers[::-1]):
            layer_name = 'layer-{}'.format(i)
            act_fn = self._d_act_fn if i != 0 else None
            if type(self._d_act_fn) == list:
                act_fn = self._d_act_fn[i]
            x = self.add_layer(x, in_dim, out_dim, self._batch_norm, layer_name,
                               act_fn, self._use_bias)
            in_dim = out_dim
        layer_name = 'reconstructed'
        out_dim = self._input_dim
        if self._loss_fn == 'bce':
            act_fn = 'sigmoid'
        else:
            act_fn = self._d_act_fn
        reconstructed_ = self.add_layer(x, in_dim, out_dim, None, layer_name, act_fn,
                                        self._use_bias)
        return reconstructed_

    @define_name_scope
    def recons_loss(self):
        if self._loss_fn == 'mse':
            loss_ = utils.mean_squared_error(self.decoder, self.x_,
                                             'mean_squared_error')
        elif self._loss_fn == 'bce':
            loss_ = utils.binary_crossentropy(self.decoder, self.x_,
                                              'binary_crossentropy_error')
        return loss_

    @define_name_scope
    def loss(self):
        loss_ = self.recons_loss
        id_lam = self._sparse_config.id_lam
        l1_lam = self._sparse_config.l1_lam
        l1_w_lam = self._sparse_config.l1_w_lam
        l2_w_lam = self._sparse_config.l2_w_lam
        l1_b_lam = self._sparse_config.l1_b_lam
        l2_b_lam = self._sparse_config.l2_b_lam

        sparse_acts = tf.get_collection('id_regularization')
        if sparse_acts:
            with tf.name_scope('id_reg'):
                for act_idx, layer_idx in enumerate(id_lam.nonzero()[0]):
                    lam = id_lam[layer_idx]
                    # act_ = sparse_acts[act_idx]
                    # act_ = tf.multiply(lam, (act_ + 1.) / tf.cast(2 * tf.shape(act_)[0], utils.TF_FLOAT_DTYPE), name='normalized_act-{}'.format(layer_idx))
                    act_ = ACT_FNS['softmax'](sparse_acts[act_idx], name='normalized_act-{}'.format(layer_idx))
                    id_name = 'id_loss_layer_{}'.format(layer_idx)
                    tf.add_to_collection('id_penalties', utils.id_penalty(act_, lam, id_name))
                    tf.add_to_collection('id_normalized_activations', act_)
                id_losses_ = tf.get_collection('id_penalties')
                id_loss_ = tf.reduce_sum(id_losses_, name='id_loss')
                loss_ += id_loss_

        sparse_acts = tf.get_collection('l1_regularization')
        if sparse_acts:
            with tf.name_scope('l1_reg'):
                for act_idx, layer_idx in enumerate(l1_lam.nonzero()[0]):
                    lam = l1_lam[layer_idx]
                    act_ = sparse_acts[act_idx]
                    l1_name = 'l1_loss_layer_{}'.format(layer_idx)
                    tf.add_to_collection('l1_penalties', utils.l1_penalty(act_, lam, l1_name))
                l1_losses_ = tf.get_collection('l1_penalties')
                l1_loss_ = tf.reduce_sum(l1_losses_, name='l1_loss')
                loss_ += l1_loss_

        if len(l1_w_lam.nonzero()[0]) != 0:
            with tf.name_scope('l1_w_reg'):
                for layer_idx in l1_w_lam.nonzero()[0]:
                    lam = l1_w_lam[layer_idx]
                    layer = self.hidden_layers[layer_idx]
                    l1_w_name = 'l1_w_loss_layer_{}'.format(layer_idx)
                    tf.add_to_collection('l1_w_penalties', utils.l1_penalty(layer.w_, lam, l1_w_name))
                l1_w_losses_ = tf.get_collection('l1_w_penalties')
                l1_w_loss_ = tf.reduce_sum(l1_w_losses_, name='l1_w_loss')
                loss_ += l1_w_loss

        if len(l2_w_lam.nonzero()[0]) != 0:
            with tf.name_scope('l2_w_reg'):
                for layer_idx in l2_w_lam.nonzero()[0]:
                    lam = l2_w_lam[layer_idx]
                    layer = self.hidden_layers[layer_idx]
                    l1_w_name = 'l2_w_loss_layer_{}'.format(layer_idx)
                    tf.add_to_collection('l2_w_penalties', utils.l2_penalty(layer.w_, lam, l1_w_name))
                l1_w_losses_ = tf.get_collection('l1_w_penalties')
                l1_w_loss_ = tf.reduce_sum(l1_w_losses_, name='l1_w_loss')
                loss_ += l1_w_loss
            
        if len(l1_b_lam.nonzero()[0]) != 0:
            with tf.name_scope('l1_b_reg'):
                for layer_idx in l1_b_lam.nonzero()[0]:
                    lam = l1_b_lam[layer_idx]
                    layer = self.hidden_layers[layer_idx]
                    l1_b_name = 'l1_b_loss_layer_{}'.format(layer_idx)
                    tf.add_to_collection('l1_b_penalties', utils.l1_penalty(layer.b_, lam, l1_b_name))
                l1_b_losses_ = tf.get_collection('l1_b_penalties')
                l1_b_loss_ = tf.reduce_sum(l1_b_losses_, name='l1_b_loss')
                loss_ += l1_b_loss

        if len(l2_b_lam.nonzero()[0]) != 0:
            with tf.name_scope('l2_b_reg'):
                for layer_idx in l2_b_lam.nonzero()[0]:
                    lam = l2_b_lam[layer_idx]
                    layer = self.hidden_layers[layer_idx]
                    l1_b_name = 'l2_b_loss_layer_{}'.format(layer_idx)
                    tf.add_to_collection('l2_b_penalties', utils.l2_penalty(layer.b_, lam, l1_b_name))
                l1_b_losses_ = tf.get_collection('l1_b_penalties')
                l1_b_loss_ = tf.reduce_sum(l1_b_losses_, name='l1_b_loss')
                loss_ += l1_b_loss

        loss_ = tf.identity(loss_, name='combined_loss')
        return loss_

    @define_name_scope
    def optimize(self):
        # builds optimization ops
        if self._opt_method == 'adam':
            optimizer = tf.train.AdamOptimizer(self._lr)
        elif self._opt_method == 'sgd':
            optimizer = tf.train.GradientDescentOptimizer(self._lr)
        self.global_step_ = tf.Variable(0, name='global_step', trainable=False)
        self.current_epoch_ = tf.Variable(0, name='current_epoch', trainable=False)
        # used for batch norm calculations
        if self._batch_norm:
            update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
            with tf.control_dependencies(update_ops):
                return optimizer.minimize(self.loss, self.global_step_)
        return optimizer.minimize(self.loss, self.global_step_)

    def loss_tensors_dict(self, graph):
        used_id = len(self._sparse_config.id_lam.nonzero()[0]) != 0.
        used_l1 = len(self._sparse_config.l1_lam.nonzero()[0]) != 0.
        loss_tensors = OrderedDict()
        loss_tensors['loss'] = self.loss
        if used_id or used_l1:
            loss_tensors['recons_loss'] = self.recons_loss
        if used_id:
            id_losses_ = tf.get_collection('id_penalties')
            id_loss_ = graph.get_tensor_by_name('loss/id_reg/id_loss:0')
            if len(id_losses_) > 1:
                loss_tensors['id_losses'] = id_losses_
                loss_tensors['id_loss'] = id_loss_
            else:
                loss_tensors['id_loss'] = id_loss_
        if used_l1:
            l1_losses_ = tf.get_collection('l1_penalties')
            l1_loss_ = graph.get_tensor_by_name('loss/l1_reg/l1_loss:0')
            if len(l1_losses_) > 1:
                loss_tensors['l1_losses'] = l1_losses_
                loss_tensors['l1_loss'] = l1_loss_
            else:
                loss_tensors['l1_loss'] = l1_loss_
        return loss_tensors

    def train(self, sess, data, batch_size, num_steps, patience=None, update_freq=100, ckpt_freq=100):
        """
        Args:
            sess: tf.Session object to run all ops with
            data: utils.DataSet object to load batches and test data from
            batch_size: size of batches to train with
            num_steps: number of optimizer iteration steps
            update_freq: number of steps before printing training loss
            saving_freq: number of steps before checkpointing model
        """
        self.epochs_trained = data.epochs_trained = self.current_epoch_.eval(sess)
        graph = sess.graph
        loss_tensors = self.loss_tensors_dict(graph)
        train_ops = [loss_tensors, self.optimize]
        test_ops = loss_tensors
        test_feed_dict = {self.x_: data.test_data, self.is_training_: False}
        best_test_losses = None
        epochs_since_improved = 0
        current_step = self.global_step_.eval(sess)

        for step in range(current_step + 1, num_steps + 1):
            batch = data.next_batch(batch_size)
            if data.labeled:
                batch, labels = batch
            feed_dict = {self.x_: batch, self.is_training_: True}
            train_losses, _ = sess.run(train_ops, feed_dict=feed_dict)
            log_str = (' epoch {}: step {}: '.format(self.epochs_trained, step)
                       + utils.make_dict_str(train_losses))
            tf.logging.log_every_n(tf.logging.INFO, log_str, update_freq)
            if (step % ckpt_freq) == 0:
                self.save_model(sess, 'model', step=step)
            if self.epochs_trained != data.epochs_trained:
                self.epochs_trained = sess.run(tf.assign(self.current_epoch_, data.epochs_trained))
                test_losses = sess.run(test_ops, feed_dict=test_feed_dict)
                log_str = (' test loss -- epoch {}: '.format(self.epochs_trained)
                           + utils.make_dict_str(test_losses))
                tf.logging.info(log_str)
                if best_test_losses is None or best_test_losses['loss'] > test_losses['loss']:
                    self.saver.save(sess, self.save_path + '/best.model')
                    tf.logging.info('Best model saved after {} epochs'.format(self.epochs_trained))
                    best_test_losses = test_losses
                    epochs_since_improved = 0
                else:
                    epochs_since_improved += 1
                if patience and epochs_since_improved == patience:
                    tf.logging.info('Early stopping, test loss did not improve for {} epochs'.format(epochs_since_improved))
                    break

        tf.logging.info('Trained for {} epochs'.format(self.epochs_trained))
        return test_losses

    def add_layer(self, x, in_dim, out_dim, batch_norm, name, act_fn, use_bias):
        l = Layer(name, in_dim, out_dim, act_fn, batch_norm, self.is_training_,
                  use_bias)
        self.hidden_layers.append(l)
        return l(x)

