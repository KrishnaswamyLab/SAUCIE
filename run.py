# -*- coding: utf-8 -*-
# File: run.py
# Author: Krishnan Srinivasan <krishnan1994 at gmail>
# Date: 21.09.2017
# Last Modified Date: 28.09.2017

"""
Run script for SAUCIE
"""

import glob
import numpy as np
import os
import plotting
import tensorflow as tf
import saucie
import saucie_utils as utils

from collections import OrderedDict
from saucie import Saucie
from tensorflow.python import debug as tf_debug

# DATA FLAGS
tf.flags.DEFINE_string('dataset', 'emt_rnaseq', 'name of dataset')
tf.flags.DEFINE_string('data_path', '/data/krishnan/emt_data/emt_data.npz',
                       'path to npz (utils.DataSet object), mat, or csv file(s)')
tf.flags.DEFINE_string('labels', None, 'path to labels file if exists')
tf.flags.DEFINE_string('colnames', '/data/krishnan/emt_data/genes.csv', 'path to list of colnames for data, useful for plotting')
tf.flags.DEFINE_string('markers', None, 'path to list of subset of columns of processed data')
tf.flags.DEFINE_boolean('keep_cols', False, 'whether to keep colnames in data matrix or subset to markers list, markers must be specified')

# MODEL FLAGS
tf.flags.DEFINE_string('model_config', None, 'name of model config file, if file does not exist will build a default model')
tf.flags.DEFINE_string('model_dir', '/data/krishnan/saucie_models', 'name of directory to save model variables and logs in')
tf.flags.DEFINE_string('encoder_layers', '1024,512,256', 'comma-separated list of layer shapes for encoder')
tf.flags.DEFINE_integer('emb_dim', 10, 'shape of bottle-neck layer')
tf.flags.DEFINE_integer('vis_dim', 3, 'shape of visualization layer')
tf.flags.DEFINE_string('act_fn', 'tanh', 'name of activation function used in encoder')
tf.flags.DEFINE_string('d_act_fn', 'tanh', 'name of activation function used in decoder')
tf.flags.DEFINE_string('id_lam', None, 'comma-separated list of id regularization scaling coefficients for each encoder layer')
tf.flags.DEFINE_string('l1_lam', None, 'comma-separated list of l1 activity regularization scaling coefficients for each encoder layer')
tf.flags.DEFINE_string('l1_w_lam', None, 'comma-separated list of l1 weight regularization scaling coefficients for each encoder layer')
tf.flags.DEFINE_string('l2_w_lam', None, 'comma-separated list of l2 weight regularization scaling coefficients for each encoder layer')
tf.flags.DEFINE_string('l1_b_lam', None, 'comma-separated list of l1 bias regularization scaling coefficients for each encoder layer')
tf.flags.DEFINE_string('l2_b_lam', None, 'comma-separated list of l2 bias regularization scaling coefficients for each encoder layer')
tf.flags.DEFINE_boolean('use_bias', True, 'boolean for whether or not to use bias')
tf.flags.DEFINE_string('loss_fn', 'bce', 'type of reconstruction loss to use. Options are: mse, bce')
tf.flags.DEFINE_string('opt_method', 'adam', 'name of optimizer to use during training')
tf.flags.DEFINE_float('lr', 1e-3, 'optimizer learning rate')
tf.flags.DEFINE_boolean('batch_norm', True, 'bool to decide whether to use batch normalization between encoder layers')
tf.flags.DEFINE_string('id_norm', 'tanh', 'method for normalizing id regularization layer. Valid options are: tanh, softmax')

# TRAINING FLAGS
tf.flags.DEFINE_integer('batch_size', 100, 'size of batch during training')
tf.flags.DEFINE_integer('num_epochs', 30, 'number of epochs to train')
tf.flags.DEFINE_integer('patience', 8, 'number of epochs to train without improvement, early stopping')
tf.flags.DEFINE_integer('log_every', 100, 'training loss logging frequency') 
tf.flags.DEFINE_integer('save_every', 500, 'checkpointing frequency') 
tf.flags.DEFINE_boolean('tb_graph', True, 'whether to log graph to TensorBoard') 
tf.flags.DEFINE_boolean('tb_summs', True, 'whether to log summaries to TensorBoard') 
tf.flags.DEFINE_boolean('debug', False, 'enable debugging')
tf.flags.DEFINE_boolean('verbose', True, 'will log in debug mode if True')
tf.flags.DEFINE_float('gpu_mem', 0.45, 'percent of gpu mem to allocate')
tf.flags.DEFINE_string('mutual_info', None, 'if None, do not calculate, if kmeans, calculate for kmeans for k n_clusts, '
                       'if dist, calculate for pairwise distances')
tf.flags.DEFINE_string('n_clusts', None, 'comma separated list of k to use for mutual_info kmeans, if None defaults to utils.N_CLUSTS')

# PLOTTING FLAGS
tf.flags.DEFINE_float('thresh', .5, 'threshold to binarize id regularized layers')
tf.flags.DEFINE_boolean('save_plots', True, 'boolean determining whether or not to save plots')


FLAGS = tf.flags.FLAGS

# removes stochasticity of results
np.random.seed(utils.RAND_SEED)
tf.set_random_seed(utils.RAND_SEED)

def main(_):
    process_flags()

    data = utils.load_dataset(FLAGS.dataset, FLAGS.data_path, FLAGS.labels, FLAGS.colnames,
                              FLAGS.markers, FLAGS.keep_cols)
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=FLAGS.gpu_mem)
    sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))

    if FLAGS.verbose:
        tf.logging.set_verbosity(tf.logging.DEBUG)
    else:
        tf.logging.set_verbosity(tf.logging.INFO)
    if FLAGS.debug:
        sess = tf_debug.LocalCLIDebugWrapperSession(sess)
        sess.add_tensor_filter('has_inf_or_nan', tf_debug.has_inf_or_nan)

    tf.logging.debug('id_lam: {}, l1_lam: {}'.format(FLAGS.id_lam.tolist(), FLAGS.l1_lam.tolist()))
    tf.logging.debug('l1_w_lam: {}, l2_w_lam: {}'.format(FLAGS.l1_w_lam.tolist(), FLAGS.l2_w_lam.tolist()))
    tf.logging.debug('l1_b_lam: {}, l2_b_lam: {}'.format(FLAGS.l1_b_lam.tolist(), FLAGS.l2_b_lam.tolist()))

    if FLAGS.model_config:
        model, config = saucie.load_model_from_config(FLAGS.dataset, FLAGS.model_config)
    else:
        config = saucie.make_config(FLAGS)
        model = Saucie(**config)

    model.build(sess)
    steps_per_epoch = data.num_samples // FLAGS.batch_size
    num_steps = steps_per_epoch * FLAGS.num_epochs
    train(model, sess, data, FLAGS.batch_size, num_steps, FLAGS.thresh, FLAGS.patience,
          FLAGS.log_every, FLAGS.save_every, FLAGS.save_plots)
    return


def process_flags():
    FLAGS.encoder_layers = [int(x) for x in FLAGS.encoder_layers.split(',')]
    num_layers = len(FLAGS.encoder_layers)

    if FLAGS.id_lam is not None:
        FLAGS.id_lam = np.array([float(x) for x in FLAGS.id_lam.split(',')], dtype=utils.FLOAT_DTYPE)
    else:
        FLAGS.id_lam = np.zeros(num_layers, dtype=utils.FLOAT_DTYPE)
    if FLAGS.l1_lam is not None:
        FLAGS.l1_lam = np.array([float(x) for x in FLAGS.l1_lam.split(',')], dtype=utils.FLOAT_DTYPE)
    else:
        FLAGS.l1_lam = np.zeros(num_layers, dtype=utils.FLOAT_DTYPE)
    if FLAGS.l1_w_lam is not None:
        FLAGS.l1_w_lam = np.array([float(x) for x in FLAGS.l1_w_lam.split(',')], dtype=utils.FLOAT_DTYPE)
    else:
        FLAGS.l1_w_lam = np.zeros(num_layers, dtype=utils.FLOAT_DTYPE)
    if FLAGS.l2_w_lam is not None:
        FLAGS.l2_w_lam = np.array([float(x) for x in FLAGS.l2_w_lam.split(',')], dtype=utils.FLOAT_DTYPE)
    else:
        FLAGS.l2_w_lam = np.zeros(num_layers, dtype=utils.FLOAT_DTYPE)
    if FLAGS.l1_b_lam is not None:
        FLAGS.l1_b_lam = np.array([float(x) for x in FLAGS.l1_b_lam.split(',')], dtype=utils.FLOAT_DTYPE)
    else:
        FLAGS.l1_b_lam = np.zeros(num_layers, dtype=utils.FLOAT_DTYPE)
    if FLAGS.l2_b_lam is not None:
        FLAGS.l2_b_lam = np.array([float(x) for x in FLAGS.l2_b_lam.split(',')], dtype=utils.FLOAT_DTYPE)
    else:
        FLAGS.l2_b_lam = np.zeros(num_layers, dtype=utils.FLOAT_DTYPE)
    
    if FLAGS.n_clusts is not None:
        FLAGS.n_clusts = [int(x) for x in FLAGS.n_clusts.split(',')]

    return 


def train(model, sess, data, batch_size, num_steps, thresh=0.5, patience=None,
          log_freq=100, ckpt_freq=100, save_plots=True, train_id=True):
    """
    Args:
        model: Saucie instance to train
        sess: tf.Session object to run all ops with
        data: utils.DataSet object to load batches and test data from
        batch_size: size of batches to train with
        num_steps: number of optimizer iteration steps
        thresh: threshold for binarization
        patience: number of epochs of training allowed without improvement
        log_freq: number of steps before printing training loss
        ckpt_freq: number of steps before checkpointing model
        save_plots: boolean determining whether or not to save plots
    """
    epochs_trained = data.epochs_trained = model.current_epoch_.eval(sess)
    graph = sess.graph
    loss_tensors = model.loss_tensors_dict(graph)
    train_ops = OrderedDict(losses=loss_tensors, opt=model.optimize)
    test_ops = OrderedDict(losses=loss_tensors)
    n_clusts = utils.N_CLUSTS if FLAGS.n_clusts is None else FLAGS.n_clusts

    train_feed_dict = {model.x_: data.data[:5000,:], model.is_training_:True}
    test_feed_dict = {model.x_: data.test_data, model.is_training_: False}
    train_labels = None if not data.labeled else data.labels[:5000]
    test_labels = None if not data.labeled else data.test_labels
    # flatten if labels are one-hot encoded
    if train_labels is not None and len(train_labels.shape) != 1: 
        train_labels = np.argmax(train_labels, axis=1)
    if test_labels is not None and len(test_labels.shape) != 1:
        test_labels = np.argmax(test_labels, axis=1)

    best_test_losses = None
    epochs_since_improved = 0
    current_step = model.global_step_.eval(sess)

    print('Saving all run data to: {}'.format(model.save_path))

    if FLAGS.tb_graph or FLAGS.tb_summs: 
        train_writer = tf.summary.FileWriter(model.save_path + '/logs/train', graph=graph)
        test_writer = tf.summary.FileWriter(model.save_path + '/logs/test', graph=graph)
        tf.logging.debug('Saving graph to TensorBoard in {}/logs'.format(model.save_path))

    if FLAGS.tb_summs:
        loss_summs = [tf.summary.scalar(name, loss) for name, loss in loss_tensors.items() if type(loss) != list]
        loss_summs = tf.summary.merge(loss_summs)
        train_ops['loss_summs'] = loss_summs
        test_ops['loss_summs'] = loss_summs
        if FLAGS.mutual_info:
            train_ops['encoder_acts'] = [hl.act_ for hl in model.hidden_layers[:len(model._encoder_layers)]]
            test_ops['encoder_acts'] = [hl.act_ for hl in model.hidden_layers[:len(model._encoder_layers)]]
            mi_ = []
            for d in model._encoder_layers:
                if FLAGS.mutual_info == 'kmeans':
                    for k in n_clusts:
                        mi_.append(tf.placeholder(tf.float32, name='mutual_info-d_{}-k_{}'.format(d,k)))
                else:
                    mi_.appennd(t.placeholder(tf.float32, name='mutual_info-d_{}'.format(d)))
            mi_summ_ = tf.summary.merge([tf.summary.scalar(x.name.split(':')[0], x) for x in mi_])
        tf.logging.debug('Saving loss summaries to TensorBoard in {}/logs'.format(model.save_path))

    if save_plots:
        plot_folder = model.save_path + '/plots'
        if not os.path.exists(plot_folder):
            os.makedirs(plot_folder)
        plot_ops = OrderedDict(emb=model.encoder)
        plot_ops['cluster_acts'] = tf.get_collection('id_normalized_activations')
        if FLAGS.dataset in RNASEQ_DATASETS:
            plot_ops['recons'] = model.decoder

    tf.logging.info('Total steps: {}'.format(num_steps))
    for step in range(current_step + 1, num_steps + 1):
        batch = data.next_batch(batch_size)
        if data.labeled:
            batch, labels = batch
        feed_dict = {model.x_: batch, model.is_training_: True,
                     model.train_id_: train_id}
        train_dict = sess.run(train_ops, feed_dict=feed_dict)

        train_losses = train_dict['losses']
        if 'loss_summs' in train_dict:
            summ = train_dict['loss_summs']
            train_writer.add_summary(summ, step)
        if 'encoder_acts' in train_dict and step % log_freq == 0:
            mi = []
            for act in train_dict['encoder_acts']:
                if FLAGS.mutual_info == 'kmeans':
                    for k in n_clusts:
                        mi.append(utils.activation_mutual_info(batch, act, k=k))
                else:
                    mi.append(utils.activation_mutual_info(batch, act))
            mi_summ = sess.run(mi_summ_, feed_dict=dict(zip(mi_, mi)))
            train_writer.add_summary(mi_summ, step)
            tf.logging.info('Mutual info scores: {}'.format(mi))
        log_str = ('epoch {}, step {}/{}: '.format(epochs_trained, step-1, num_steps)
                   + utils.make_dict_str(train_losses))
        tf.logging.log_first_n(tf.logging.INFO, log_str, log_freq - 1)
        tf.logging.log_every_n(tf.logging.INFO, log_str, log_freq)

        if ckpt_freq and (step % ckpt_freq) == 0:
            tf.logging.info('Saving model, after step {}'.format(step))
            model.save_model(sess, 'model', step=step)
            if save_plots:
                tf.logging.debug('Plotting middle layer embedding')
                plot_dict = sess.run(plot_ops, feed_dict=train_feed_dict)
                make_plots(model._sparse_config, plot_folder, plot_dict, data, train_labels)

        if epochs_trained != data.epochs_trained:
            epochs_trained = sess.run(tf.assign(model.current_epoch_, data.epochs_trained))
            test_dict = sess.run(test_ops, feed_dict=test_feed_dict)
            test_losses = test_dict['losses']
            if 'loss_summs' in test_dict:
                summ = test_dict['loss_summs']
                test_writer.add_summary(summ, step)
            if 'encoder_acts' in test_dict:
                mi = []
                if FLAGS.mutual_info == 'kmeans':
                    for act in test_dict['encoder_acts']:
                        for k in n_clusts:
                            mi.append(utils.activation_mutual_info(data.test_data, act, k=k))
                else:
                    mi.append(utils.activation_mutual_info(data.test_data, act))
                mi_summ = sess.run(mi_summ_, feed_dict=dict(zip(mi_, mi)))
                test_writer.add_summary(mi_summ, step)
                tf.logging.info('Mutual info scores: {}'.format(mi))
            log_str = ('TESTING -- epoch: {}, '.format(epochs_trained)
                       + utils.make_dict_str(test_losses))
            tf.logging.info(log_str)

            if best_test_losses is None or best_test_losses['loss'] > test_losses['loss']:
                model.saver.save(sess, model.save_path + '/best.model')
                tf.logging.info('Best model saved after {} epochs'.format(epochs_trained))
                best_test_losses = test_losses
                epochs_since_improved = 0
                if save_plots:
                    tf.logging.debug('Plotting best middle layer embedding')
                    plot_dict = sess.run(plot_ops, feed_dict=test_feed_dict)
                    make_plots(model._sparse_config, plot_folder, plot_dict, data, test_labels, True, True)
            else:
                epochs_since_improved += 1
            if patience and epochs_since_improved == patience:
                tf.logging.info('Early stopping, test loss did not improve for {} epochs.'.format(epochs_since_improved))
                tf.logging.info('Best test loss: epoch {}: '.format(epochs_trained - epochs_since_improved)
                                + utils.make_dict_str(best_test_losses))
                break

    tf.logging.info('Trained for {} epochs, {} steps'.format(epochs_trained, step))

    print('Saved all run data to: {}'.format(model.save_path))
    return test_losses


def make_plots(sparse_config, plot_folder, plot_dict, data, suffix='', labels=None, heatmap_only=True):
    id_lam = sparse_config.id_lam
    l1_lam = sparse_config.l1_lam
    cluster_layers = id_lam.nonzero()[0].tolist()
    plot_fn = plotting.plot_embedding2D if plot_dict['emb'].shape[1] != 3 else plotting.plot_embedding3D
    if testing:
        plot_data = data.test_data
    else:
        plot_data = data.data[:5000]
    for i, acts in enumerate(plot_dict['cluster_acts']):
        hl_idx = cluster_layers[i]
        clusts = utils.binarize(acts, FLAGS.thresh)
        tf.logging.debug('Top 5 neurons max acts: {}'.format(acts.max(axis=1)[:5]))
        tf.logging.debug('Mean max activation: {}'.format(acts.max(axis=1).mean()))
        tf.logging.debug('Count of activated nodes: {}'.format(np.sum(acts.max(axis=0) > FLAGS.thresh)))
        if not heatmap_only:
            save_file = plot_folder + '/emb-layer={}{}.png'.format(hl_idx, suffix)
            title = 'Embedding, clust layer-{}, id_lam/l1_lam={:3.2E}/{:3.2E}'.format(hl_idx, id_lam[hl_idx], l1_lam[hl_idx]) 
            plot_fn(plot_dict['emb'], clusts, save_file, title) 
        if '_colnames' in data.__dict__ and FLAGS.dataset in utils.CYTOF_DATASETS and len(np.unique(clusts)) > 1:
            save_file = plot_folder + '/heatmap-layer={}{}.png'.format(hl_idx, suffix)
            # plotting.plot_cluster_heatmap(plot_data, clusts, data._colnames, data._markers, save_file=save_file)
            plotting.plot_cluster_linkage_heatmap(plot_data, clusts, data._colnames, data._markers, save_file)
    if not heatmap_only:
        if labels is not None:
            save_file = plot_folder + '/emb-labels{}.png'.format(suffix)
            title = 'Embedding, with true labels'
            plot_fn(plot_dict['emb'], labels, save_file, title)
        if cluster_layers == []:
            plot_fn(plot_dict['emb'], None, plot_folder + '/emb.png', 'Embedding, no clusters')
    if FLAGS.dataset in utils.RNASEQ_DATASETS and recons in plot_dict:
        if FLAGS.dataset == 'zika':
            marker_sub = ['VIM', 'CDH1', 'CDH2', 'EZH2', 'SNAI1', 'ZEB1']
            x_vars = ['VIM', 'SNAI1', 'CDH1']
            y_vars = ['CDH1', 'EZH2', 'ZEB1', 'CDH2'],
            recons_rnaseq = saucie_utils.inverse_transform(data, x=plot_dict['recons'])
            save_file = plot_folder + '/edgeplots{}.png'.format(suffix)
            plotting.plot_edge_plots(recons_rnaseq, data._colnames, marker_sub, x_vars, y_vars, title='Reconstructed edge plots', save_file=save_file)
    return


if __name__ == '__main__':
    tf.app.run()
