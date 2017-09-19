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

import numpy as np
import tensorflow as tf
import saucie
import saucie_utils as utils

from collections import namedtuple
from saucie import Saucie
from tensorflow.python import debug as tf_debug

# DATA FLAGS
tf.flags.DEFINE_string('dataset', 'mnist', 'name of dataset')
tf.flags.DEFINE_string('data_path', '/data/krishnan/mnist/mnist_data.npz',
                       'path to DataSet object containing training and testing data, and feeding functionality')

# MODEL FLAGS
tf.flags.DEFINE_string('model_config', None, 'name of model config file, if file does not exist will build a default model')
tf.flags.DEFINE_string('model_dir', '/data/krishnan/saucie_models', 'name of directory to save model variables and logs in')
tf.flags.DEFINE_string('encoder_layers', '1024,512,256', 'comma-separated list of layer shapes for encoder')
tf.flags.DEFINE_integer('emb_dim', 2, 'shape of bottle-neck layer')
tf.flags.DEFINE_string('act_fn', 'relu', 'name of activation function used in encoder')
tf.flags.DEFINE_string('d_act_fn', 'relu', 'name of activation function used in decoder')
tf.flags.DEFINE_string('id_lam', '1e-4,0,0', 'comma-separated list of id regularization scaling coefficients for each encoder layer')
tf.flags.DEFINE_string('l1_lam', '0,0,0', 'comma-separated list of l1 activity regularization scaling coefficients for each encoder layer')
tf.flags.DEFINE_boolean('use_bias', True, 'boolean for whether or not to use bias')
tf.flags.DEFINE_string('loss_fn', 'bce', 'type of reconstruction loss to use. Options are: mse, bce')
tf.flags.DEFINE_string('opt_method', 'adam', 'name of optimizer to use during training')
tf.flags.DEFINE_float('lr', 1e-3, 'optimizer learning rate')
tf.flags.DEFINE_boolean('batch_norm', False, 'bool to decide whether to use batch normalization between encoder layers')

# TRAINING FLAGS
tf.flags.DEFINE_integer('batch_size', 100, 'size of batch during training')
tf.flags.DEFINE_integer('num_epochs', 50, 'number of epochs to train')
tf.flags.DEFINE_integer('patience', 20, 'number of epochs to train without improvement, early stopping')
tf.flags.DEFINE_integer('log_every', 100, 'training loss logging frequency') 
tf.flags.DEFINE_integer('save_every', 100, 'checkpointing frequency') 
tf.flags.DEFINE_boolean('debug', False, 'enable debugging')

FLAGS = tf.flags.FLAGS
RAND_SEED = 20

def main(_):
    FLAGS.encoder_layers = [int(x) for x in FLAGS.encoder_layers.split(',')]
    FLAGS.id_lam = np.array([float(x) for x in FLAGS.id_lam.split(',')])
    FLAGS.l1_lam = np.array([float(x) for x in FLAGS.l1_lam.split(',')])
    print('id_lam: {}, l1_lam: {}'.format(FLAGS.id_lam, FLAGS.l1_lam))
    assert len(FLAGS.encoder_layers) == len(FLAGS.id_lam) == len(FLAGS.l1_lam), "id_lam, l1_lam, and encoder_layers must be comma-separated lists of the same length"
    data = utils.DataSet.load(FLAGS.data_path)
    sess = tf.Session()
    if FLAGS.debug:
        sess = tf_debug.LocalCLIDebugWrapperSession(sess)
        sess.add_tensor_filter('has_inf_or_nan', tf_debug.has_inf_or_nan)
        tf.logging.set_verbosity(tf.logging.DEBUG)
    else:
        tf.logging.set_verbosity(tf.logging.INFO)
    if FLAGS.model_config:
        model = saucie.load_model_from_config(FLAGS.dataset, FLAGS.model_config)
    else:
        config = saucie.make_config(FLAGS)
        model = Saucie(**config)
    model.build(sess)
    steps_per_epoch = data.num_samples // FLAGS.batch_size
    num_steps = steps_per_epoch * FLAGS.num_epochs
    train(model, sess, data, FLAGS.batch_size, num_steps, FLAGS.patience,
          FLAGS.log_every, FLAGS.save_every)


def train(model, sess, data, batch_size, num_steps, patience=None, update_freq=100, ckpt_freq=100):
    """
    Args:
        model: Saucie instance to train
        sess: tf.Session object to run all ops with
        data: utils.DataSet object to load batches and test data from
        batch_size: size of batches to train with
        num_steps: number of optimizer iteration steps
        update_freq: number of steps before printing training loss
        saving_freq: number of steps before checkpointing model
    """
    model.epochs_trained = data.epochs_trained = model.current_epoch_.eval(sess)
    graph = sess.graph
    loss_tensors = model.loss_tensors_dict(graph)
    train_ops = [loss_tensors, model.optimize]
    test_ops = loss_tensors
    test_feed_dict = {model.x_: data.test_data, model.is_training_: False}
    best_test_losses = None
    epochs_since_improved = 0
    current_step = model.global_step_.eval(sess)

    for step in range(current_step + 1, num_steps + 1):
        batch = data.next_batch(batch_size)
        if data.labeled:
            batch, labels = batch
        feed_dict = {model.x_: batch, model.is_training_: True}
        train_losses, _ = sess.run(train_ops, feed_dict=feed_dict)
        log_str = (' epoch {}: step {}: '.format(model.epochs_trained, step)
                   + utils.make_dict_str(train_losses))
        tf.logging.log_every_n(tf.logging.INFO, log_str, update_freq)
        if (step % ckpt_freq) == 0:
            model.save_model(sess, 'model', step=step)
        if model.epochs_trained != data.epochs_trained:
            model.epochs_trained = sess.run(tf.assign(model.current_epoch_, data.epochs_trained))
            test_losses = sess.run(test_ops, feed_dict=test_feed_dict)
            log_str = (' test loss -- epoch {}: '.format(model.epochs_trained)
                       + utils.make_dict_str(test_losses))
            tf.logging.info(log_str)
            if best_test_losses is None or best_test_losses['loss'] > test_losses['loss']:
                model.saver.save(sess, model.save_path + '/best.model')
                tf.logging.info('Best model saved after {} epochs'.format(model.epochs_trained))
                best_test_losses = test_losses
                epochs_since_improved = 0
            else:
                epochs_since_improved += 1
            if patience and epochs_since_improved == patience:
                tf.logging.info('Early stopping, test loss did not improve for {} epochs'.format(epochs_since_improved))
                break

    tf.logging.info('Trained for {} epochs'.format(model.epochs_trained))
    return test_losses

if __name__ == '__main__':
    tf.app.run()
