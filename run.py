# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2017 Krishnan Srinivasan <krishnan1994@gmail.com>
#
# Distributed under terms of the MIT license.
# ==============================================================================

"""
Run script for SAUCIE
"""

import tensorflow as tf
import saucie
import saucie_utils as utils

from collections import namedtuple
from saucie import Saucie

tf.flags.DEFINE_string('dataset', 'mnist', 'name of dataset')
tf.flags.DEFINE_string('data_path', '/data/krishnan/mnist/mnist_data.npz',
                       'path to DataSet object containing training and testing data, and feeding functionality')
tf.flags.DEFINE_string('model_config', None, 'name of model config file, if file does not exist will build a default model')
tf.flags.DEFINE_string('model_dir', './output', 'name of directory to save model variables and logs in')
tf.flags.DEFINE_string('encoder_layers', '1024,512,256', 'comma-separated list of layer shapes for encoder')
tf.flags.DEFINE_integer('emb_dim', 2, 'shape of bottle-neck layer')
tf.flags.DEFINE_string('act_fn', 'relu', 'name of activation function used in encoder')
tf.flags.DEFINE_string('d_act_fn', 'relu', 'name of activation function used in decoder')
tf.flags.DEFINE_string('id_lam', '1e-4,0,0', 'comma-separated list of id regularization scaling coefficients for each encoder layer')
tf.flags.DEFINE_string('l1_lam', '1e-6,0,0', 'comma-separated list of l1 activity regularization scaling coefficients for each encoder layer')
tf.flags.DEFINE_boolean('use_bias', True, 'boolean for whether or not to use bias')
tf.flags.DEFINE_string('loss_fn', 'bce', 'type of reconstruction loss to use. Options are: mse, bce')
tf.flags.DEFINE_string('opt_method', 'adam', 'name of optimizer to use during training')
tf.flags.DEFINE_float('lr', 1e-3, 'optimizer learning rate')
tf.flags.DEFINE_boolean('batch_norm', False, 'bool to decide whether to use batch normalization between encoder layers')
tf.flags.DEFINE_integer('batch_size', 100, 'size of batch during training')
tf.flags.DEFINE_integer('num_epochs', 50, 'number of epochs to train')
tf.flags.DEFINE_integer('patience', 20, 'number of epochs to train without improvement, early stopping')

FLAGS = tf.flags.FLAGS
RAND_SEED = 20

def main(_):
    FLAGS.encoder_layers = [int(x) for x in FLAGS.encoder_layers.split(',')]
    FLAGS.id_lam = np.array([float(x) for x in FLAGS.id_lam.split(',')])
    FLAGS.l1_lam = np.array([float(x) for x in FLAGS.l1_lam.split(',')])
    assert len(FLAGS.encoder_layers) == len(FLAGS.id_lam) == len(FLAGS.l1_lam), "id_lam, l1_lam, and encoder_layers must be comma-separated lists of the same length"
    data = utils.DataSet.load(FLAGS.data_path)
    sess = tf.get_default_session()
    if FLAGS.model_config:
        model = saucie.load_model_from_config(FLAGS.dataset, FLAGS.model_config)
    else:
        sparse_config = utils.SparseLayerConfig(num_layers=3, id_lam=FLAGS.id_lam, l1_lam=FLAGS.l1_lam)
        config = dict(encoder_layers=FLAGS.encoder_layers, emb_dim=FLAGS.emb_dim, act_fn=FLAGS.act_fn,
                      d_act_fn=FLAGS.d_act_fn, use_bias=FLAGS.use_bias, loss_fn=FLAGS.loss_fn, opt_method=FLAGS.opt_method,
                      lr=FLAGS.lr, batch_norm=FLAGS.batch_norm, sparse_config=sparse_config)
        if FLAGS.dataset == 'mnist':
            config['input_dim'] = 784
        elif FLAGS.dataset == 'emt_cytof':
            config['input_dim'] = 30
        model = Saucie(**config)
    steps_per_epoch = data.num_samples // FLAGS.batch_size
    num_steps = steps_per_epoch * FLAGS.num_epochs
    model.train(sess, data, batch_size, num_steps)


if __name__ == '__main__':
    tf.app.run()
