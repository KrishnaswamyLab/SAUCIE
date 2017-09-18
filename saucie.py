# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2017 Krishnan Srinivasan <krishnan1994@gmail.com>
#
# Distributed under terms of the MIT license.
# ==============================================================================

"""
Classes and functions used to instantiate SAUCIE.
"""

import tensorflow as tf
import saucie_utils as utils
import pickle
import os

from typing import NamedTuple
from collections import OrderedDict
from saucie_utils import define_scope

ACT_FNS = {'lrelu': tf_util.lrelu,
           'relu': tf.nn.relu,
           'softmax': tf.nn.softmax,
           'softplus': tf.nn.softplus,
           'sigmoid': tf.nn.sigmoid}

SAVE_PATH = '/data/krishnan/saucie_models'


def load_model_from_config(dataset='mnist', config_path=SAVE_PATH+'/best.config'):
    if os.path.exists(config_path):
        with open(config_path, 'rb') as f:
            config = pickle.load(f)
        model = Saucie(**config)
    else:
        config = default_config(dataset)
        model = Saucie(**config)


def default_config(dataset='mnist'):
    sparse_config = utils.SparseLayerConfig(num_layers=3, id_lam=np.array([1e-5,0.,0.]))
    config = dict(encoder_layers=[1024,512,256], emb_dim=2, act_fn='relu',
                  d_act_fn='relu', use_bias=True, loss_fn='bce', lr=1e-3,
                  batch_norm=False, sparse_config=sparse_config)
    if dataset == 'mnist':
        config['input_dim'] = 784
    elif dataset == 'cytof_emt':
        config['input_dim'] = 30 # or something..
    return config

class Layer():
    # defined as class attribute so can be changed for all layers at once
    w_init = tf.contrib.layers.xavier_initializer()
    b_init = tf.zeros_initializer()

    def __init__(self, name, in_dim, out_dim, act_fn='relu', p_dropout=1., batch_norm=False, is_training=False, use_bias=True):
        self.w_, self.b_ = None, None
        self.hidden_rep_ = None
        self.name = name
        self._in_dim = in_dim
        self._out_dim = out_dim
        self._p_dropout = p_dropout # how much to dropout layer input, equiv to l2 reg of weights
        self._batch_norm = batch_norm # whether to batch normalize layer output
        self._act_fn = act_fn # name of activation function
        self.is_training_ = is_training_
        self._use_bias = use_bias

    def __call__(self, x):
        if self._p_dropout != 1.:
            x = tf.nn.dropout(x, self._p_dropout)
        if self.hidden_rep_ is None:
            with tf.variable_scope(self.name):
                self.w_ = tf.get_variable('w', [self._in_dim, self._out_dim], initializer=w_init)
                if self._use_bias:
                    self.b_ = tf.get_variable('b', [self._out_dim], initializer=b_init)
                    self.hidden_rep_ = tf.add(tf.matmul(x, self.w_), self.b_, name='hidden_rep_aff')
                else:
                    self.hidden_rep_ = tf.matmul(x, self.w_, name='hidden_rep_aff')
                if self._act_fn:
                    if type(self._act_fn) == list:
                        for act_fn in self._act_fn:
                            self.hidden_rep_ = ACT_FNS[act_fn](self.hidden_rep_, name='hidden_rep_{}'.format(act_fn))
                    else:
                        self._hidden_rep_ = ACT_FNS[self._act_fn](self.hidden_rep_, name='hidden_rep')

        else:
            if self._batch_norm:
                return self.hidden_rep_norm_
            else:
                return self.hidden_rep_
        if self._batch_norm:
            self.hidden_rep_norm_ = tf.layers.batch_normalization(
                self.hidden_rep_, training=self.is_training_, name='hidden_rep_norm')
            return self.hidden_rep_norm_
        else:
            return self.hidden_rep_


class Saucie():
    def __init__(self, input_dim, encoder_layers, emb_dim, act_fn, d_act_fn,
                 use_bias, loss_fn, opt_method, lr, batch_norm, sparse_config,
                 save_path=SAVE_PATH):
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
        self._model_config = dict(encoder_layers=','.join([str(x) for x in encoder_layers]),
                                  act_fn=act_fn, d_act_fn=d_act_fn, use_bias=use_bias,
                                  loss_fn=loss_fn, lr=lr, batch_norm=batch_norm,
                                  sparse_config=sparse_config)
        self._model_str = utils.make_dict_str(model_dict, kv_sep='=', item_sep='-')
        self._save_path = os.path.join(save_path, self._model_str)
        self.epochs_trained = 0

    def __dict__(self):
        return self._model_config

    def save_config(self, config_name='model.config'):
        with open(self._save_path + '/{}'.format(config_name), 'wb') as f:
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
            if len(self.saver.last_checkpoints()) == 5:
                for ckpt in self.saver.last_checkpoints():
                    if ckpt != self._save_path + '/best.model':
                        break
                os.remove(ckpt)
                if ckpt in self.saver.last_checkpoints():
                    self.saver.last_checkpoints().remove(ckpt)
        self.saver.save(sess, self._save_path + '/{}'.format(filename), global_step=step)

    def build(sess):
        self.sess = sess
        self.x_ = tf.placeholder(tf.float64, shape=[None, self._input_dim], 'x')
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
        if os.path.exists(self._save_path + '/best.model'):
            restored = self.restore_model(sess, self._save_path, 'best.model')
            if restored:
                tf.logging.debug('Restored model: {}'.format(self._save_path + '/best.model'))

        tf.logging.debug('Finished building SAUCIE')
        return

    @define_scope
    def encoder(self):
        x = self.x_
        in_dim = self._input_dim
        id_lam = self._sparse_config.id_lam
        l1_lam = self._sparse_config.l1_lam
        for i, out_dim in enumerate(self._encoder_layers):
            layer_name = 'layer-{}'.format(i)
            if type(self._act_fn) == list:
                act_fn = self._act_fn[i]
            if id_lam[i] != 0.:
                act_fn = [self._act_fn, 'softmax']
            else:
                act_fn = self._act_fn
            x = self.add_layer(x, in_dim, out_dim, self._p_dropout, self._batch_norm,
                               layer_name, act_fn, self._use_bias)
            if id_lam[i] != 0.:
                tf.add_to_collection(x, 'id_regularization')
            if l1_lam[i] != 0.:
                tf.add_to_collection(x, 'l1_regularization')
            in_dim = out_dim
        layer_name = 'embedding'
        out_dim = self._emb_dim
        encoded_ = self.add_layer(x, in_dim, out_dim, 1., False, layer_name, None,
                                  self._use_bias)
        return encoded_

    @define_scope
    def decoder(self):
        in_dim = self._emb_dim
        x = self.encoder
        for i, out_dim in enumerate(self._encoder_layers[::-1]):
            layer_name = 'layer-{}'.format(i)
            x = self.add_layer(x, in_dim, out_dim, self._p_dropout, False, layer_name,
                               self._d_act_fn, self._use_bias)
            in_dim = out_dim
        layer_name = 'reconstruted'
        out_dim = self._input_dim
        act_fn = 'sigmoid' if self._loss_fn == 'bce' else 'relu'
        reconstructed_ = self.add_layer(x, in_dim, out_dim, 1., False, layer_name, act_fn,
                                        self._use_bias)
        return reconstructed_

    @define_scope
    def recons_loss(self):
        if self._loss_fn == 'mse':
            loss_ = utils.mean_squared_error(self.decoder, self.x_,
                                             'mean_squared_error')
        elif self._loss_fn == 'bce':
            loss_ = utils.binary_crossentropy(self.decoder, self.x_,
                                              'binary_crossentropy_error')
        return loss_

    @define_scope
    def loss(self):
        loss_ = self.recons_loss

        sparse_acts = tf.get_collection('id_regularization')
        if sparse_acts:
            with tf.name_scope('id_reg'):
                id_losses_ = []
                for act_idx, layer_idx in enumerate(id_lam.nonzero()[0]):
                    lam = id_lam[layer_idx]
                    act = sparse_acts[act_idx]
                    id_name = 'id_loss_layer_{}'.format(layer_idx)
                    id_losses_.append(utils.id_penalty(act, lam, id_name))
                id_loss_ = tf.reduce_sum(id_losses_, name='id_loss')
                loss_ += id_loss_

        sparse_acts = tf.get_collection('l1_regularization')
        if sparse_acts:
            with tf.name_scope('l1_reg'):
                l1_losses_ = []
                for act_idx, layer_idx in enumerate(l1_lam.nonzero()[0]):
                    lam = l1_lam[layer_idx]
                    act = sparse_acts[act_idx]
                    l1_name = 'l1_loss_layer_{}'.format(layer_idx)
                    l1_losses_.append(utils.l1_act_penalty(act, lam, l1_name))
                l1_loss_ = tf.reduce_sum(l1_losses_, name='l1_loss')
                loss_ += l1_loss_

        loss_ = tf.identity(loss_, name='combined_loss')
        return loss_

    @define_scope
    def optimize(self):
        # builds optimization ops
        if self._opt_method == 'adam':
            optimizer = tf.train.AdamOptimizer(self._lr)
        elif self._opt_method == 'sgd':
            optimizer = tf.train.GradientDescentOptimizer(self._lr)
        global_step_ = tf.Variable(0, name='global_step' trainable=False)
        return optimizer.minimize(self.loss, global_step_)

    def loss_tensors_dict(self, graph):
        used_id = len(self._sparse_tensors.id_lam.nonzero()[0]) != 0.
        used_l1 = len(self._sparse_tensors.l1_lam.nonzero()[0]) != 0.
        loss_tensors = OrderedDict()
        loss_tensors['loss'] = self.loss
        if used_id or used_l1:
            loss_tensors['recons_loss'] = self.recons_loss
        if used_id:
            id_losses_ = []
            for i, idx in enumerate(self._sparse_tensors.id_lam.nonzero()[0]):
                t_name= = 'id_loss_layer_{}:0'.format(idx)
                id_losses_.append(graph.get_tensor_by_name('loss/id_reg/' + t_name))
            if len(id_losses_) > 1:
                loss_tensors['id_losses'] = id_losses_
                id_loss_ = graph.get_tensor_by_name('loss/id_reg/id_loss:0')
                loss_tensors['id_loss'] = id_loss_
            else:
                loss_tensors['id_loss'] = id_losses_[0]
        if used_l1:
            l1_losses = []
            for i, idx in enumerate(self._sparse_tensors.l1_lam.nonzero()[0]):
                t_name= = 'l1_loss_layer_{}:0'.format(idx)
                l1_losses_.append(graph.get_tensor_by_name('loss/l1_reg/' + t_name))
            if len(l1_losses_) > 1:
                loss_tensors['l1_losses'] = l1_losses_
                l1_loss_ = graph.get_tensor_by_name('loss/l1_reg/l1_loss:0')
                loss_tensors['l1_loss'] = l1_loss_
            else:
                loss_tensors['l1_loss'] = l1_losses_[0]
        return loss_tensors

    def train(self, sess, data, batch_size, num_steps, update_freq=100, ckpt_freq=100):
        """
        Args:
            sess: tf.Session object to run all ops with
            data: utils.DataSet object to load batches and test data from
            batch_size: size of batches to train with
            num_steps: number of optimizer iteration steps
            update_freq: number of steps before printing training loss
            saving_freq: number of steps before checkpointing model
        """
        self.epochs_trained = data.epochs_trained = 0
        graph = sess.graph
        loss_tensors = self.loss_tensors_dict(graph)
        train_ops = [loss_tensors, self.optimize]
        test_ops = loss_tensors
        test_feed_dict = {self.x_: data.test_data, self.is_training_: False}
        best_test_losses = None

        for step in range(1, num_steps + 1):
            batch = data.next_batch(batch_size)
            if data.labeled:
                batch, labels = batch
            feed_dict = {self.x_: batch, self.is_training_: True}
            train_losses, _ = sess.run(train_ops, feed_dict=feed_dict)
            tf.logging.log_every_n(tf.logging.INFO,
                'epoch {}: '.format(self.epochs_trained) + utils.make_dict_str(train_losses),
                update_freq)
            if (step % ckpt_freq) == 0:
                self.save_model(sess, 'model', step=step)
            if self.epochs_trained != data.epochs_trained:
                self.epochs_trained = data.epochs_trained
                test_losses = sess.run(test_ops, feed_dict=test_feed_dict)
                tf.logging.info(
                    'test loss -- epoch {}: '.format(self.epochs_trained) +
                    utils.make_dict_str(test_losses)
                )
                if best_test_loss is None or best_test_losses['loss'] > test_losses['loss']:
                    self.saver.save(sess, self._save_path + '/best.model')
                    tf.logging.info('Best model saved after {} epochs'.format(self.epochs_trained))
                    best_test_losses = test_losses

        tf.logging.info('Trained for {} epochs'.format(self.epochs_trained))
        return test_losses


    def add_layer(self, x, in_dim, out_dim, p_dropout, batch_norm, name, act_fn,
                  use_bias):
        l = Layer(name, in_dim, out_dim, act_fn, p_dropout, batch_norm, self.is_training_
                  use_bias)
        self.hidden_layers.append(l)
        return l(x)

