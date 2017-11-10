import sys, os, time, math, argparse, pickle
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score
from scipy.io import savemat
from utils import *
from model import *

def parse_args():
    parser = argparse.ArgumentParser()


    # NAME
    parser.add_argument('--save_folder', type=str, default='saved_mmd')
    parser.add_argument('--loader', type=str, default='LoaderGMM2D')

    # TRAINING PARAMETERS
    parser.add_argument('--batch_size', type=int, default=100)
    parser.add_argument('--learning_rate', type=float, default=.001)
    parser.add_argument('--num_epochs', type=int, default=100)
    parser.add_argument('--save_every', type=int, default=1000)
    parser.add_argument('--print_every', type=int, default=100)
    parser.add_argument('--max_iterations', type=int, default=10000)
    
    # MODEL ARCHITECTURE
    parser.add_argument('--layers', type=str, default='1024,512,256,2')
    # parser.add_argument('--layers', type=str, default='256,128,64,2') # make it faster while testing
    parser.add_argument('--activation', type=str, default='relu')
    parser.add_argument('--activation_idreg', type=str, default='tanh')
    parser.add_argument('--loss', type=str, default='mse')
    parser.add_argument('--dropout_p', type=float, default=1.)
    parser.add_argument('--batch_norm', type=bool, default=True)
    
    # REGULARIZATIONS
    parser.add_argument('--lambda_l1', type=float, default=0)
    parser.add_argument('--lambda_l2', type=float, default=0)
    parser.add_argument('--layers_sparsity', type=str, default='')
    parser.add_argument('--lambdas_sparsity', type=str, default='')
    parser.add_argument('--layers_clustuse', type=str, default='')
    parser.add_argument('--lambdas_clustuse', type=str, default='')
    parser.add_argument('--layers_entropy', type=str, default='')
    parser.add_argument('--lambdas_entropy', type=str, default='')
    parser.add_argument('--normalization_method', type=str, default='')
    parser.add_argument('--thresh', type=float, default=.5)
    parser.add_argument('--sigma', type=float, default=.5)
    parser.add_argument('--lambda_mmd', type=float, default=0)
    parser.add_argument('--num_batches', type=int, default=2)

    # NOISE
    parser.add_argument('--add_noise', type=float, default=0)
    parser.add_argument('--dropout_input', type=float, default=0)
    parser.add_argument('--noise_stddev', type=float, default=0)


    sys.argv = [sys.argv[0]]
    args = parser.parse_args()

    args = process_args(args)
    
    return args

def process_args(args):
    # parse layers
    args.layers = [int(l) for l in args.layers.split(',')] if len(args.layers)>0 else []
    args.layers_sparsity = [int(l) for l in args.layers_sparsity.split(',')] if len(args.layers_sparsity)>0 else []
    args.lambdas_sparsity = [float(l) for l in args.lambdas_sparsity.split(',')] if len(args.lambdas_sparsity)>0 else []
    args.layers_entropy = [l if l=='embedding' else int(l) for l in args.layers_entropy.split(',')] if len(args.layers_entropy)>0 else []
    args.lambdas_entropy= [float(l) for l in args.lambdas_entropy.split(',')] if len(args.lambdas_entropy)>0 else []
    args.layers_clustuse = [int(l) for l in args.layers_clustuse.split(',')] if len(args.layers_clustuse)>0 else []
    args.lambdas_clustuse= [float(l) for l in args.lambdas_clustuse.split(',')] if len(args.lambdas_clustuse)>0 else []

    # parse activation
    if args.activation=='relu': args.activation = tf.nn.relu
    elif args.activation=='tanh': args.activation = tf.nn.tanh
    elif args.activation=='sigmoid': args.activation = tf.nn.sigmoid
    elif args.activation=='lrelu': args.activation = lrelu
    else: print("Could not parse activation: {}".format(args.activation))

    if args.activation_idreg=='relu': args.activation_idreg = tf.nn.relu
    elif args.activation_idreg=='tanh': args.activation_idreg = tf.nn.tanh
    elif args.activation_idreg=='sigmoid': args.activation_idreg = tf.nn.sigmoid
    elif args.activation_idreg=='lrelu': args.activation_idreg = lrelu
    else: print("Could not parse activation_idreg: {}".format(args.activation_idreg))

    if args.loader == 'LoaderMNIST': args.input_dim = 28*28
    elif args.loader == 'LoaderZika': args.input_dim = 35
    elif args.loader == 'LoaderGMM2D': args.input_dim = 2
    elif args.loader == 'LoaderGMM': args.input_dim = 100
    else: raise Exception("Could not parse name of data for input_dim: {}".format(args.loader))

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

def save(args, sess, mlp, saver, iteration):
    savefile = os.path.join(args.save_folder, 'AE')
    saver.save(sess, savefile , global_step=iteration, write_meta_graph=True)
    print("Model saved to {}".format(savefile))

def train(args):
    load = get_loader(args)
    mlp = MLP(args)

    # initialize
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.30)
    sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) 
    with sess:
        saver = run_inits(args, sess, mlp)

        iteration = 1
        losses = [tbn('loss:0'), tbn('loss_recon:0'), tbn('loss_clustuse:0'), tbn('loss_entropy:0')]
        for epoch in range(args.num_epochs):
            t = time.time()
            for batch in load.iter_batches():
                if isinstance(batch, tuple):
                    batch, batch_labels = batch

                iteration+=1

                x = batch.copy()
                if args.add_noise:
                    x = batch + np.random.normal(0,args.add_noise, batch.shape)
                    x = np.maximum(np.minimum(x, 1.), 0.)
                if args.dropout_input:
                    x *= np.random.binomial(1,args.dropout_input, x.shape)

                feed = {mlp.x:x,
                        mlp.y:batch,
                        tbn('is_training:0'):True,
                        mlp.learning_rate:args.learning_rate}

                [ltotal,l1,l2,l3, _] = sess.run(losses + [obn('train_op')], feed_dict=feed)

                # print score on train set              
                if iteration%args.print_every==0:
                    print("epoch/iter: {}/{} loss: {:.3f} ({:.3f} {:.3f} {:.7f}) time: {:.1f}".format(epoch,
                        iteration, ltotal, l1, l2, l3, time.time() - t))
                    t = time.time()
                    
                # save
                # if args.save_every and iteration%args.save_every==0:
                #     save(args, sess, mlp, saver, load, iteration)

                    # embeddings, labels = get_layer(sess, load, 'layer_embedding_activation:0')
                    # embeddings, labels = get_layer(sess, load, 'layer_decoder_0_bninput:0')
                    # input_layer, labels = get_layer(sess, load, 'x:0')
                    # reconstruction, labels = get_layer(sess, load, 'layer_output_activation:0')

                    # plot(args, embeddings, labels, 'Embedding layer by label', 'embedding_by_label')
                    # if args.layers[-1]==2:
                    #     if args.data=='MNIST':
                    #         plot_mnist(args, input_layer, labels, embeddings, 'orig')
                    #         plot_mnist(args, reconstruction, labels, embeddings, 'recon')
                    # if args.data=='MNIST':
                    #     show_result(args, input_layer, 'original_images.png')
                    #     show_result(args, reconstruction, 'reconstructed_images.png')
                    # if args.dropout_input or args.add_noise:
                    #     input_layer_noisy = input_layer
                    #     if args.add_noise:
                    #         input_layer_noisy = input_layer + np.random.normal(0,args.add_noise, input_layer.shape)
                    #         input_layer_noisy = np.maximum(np.minimum(input_layer_noisy, 1.), 0.)
                    #     if args.dropout_input:
                    #         input_layer_noisy *= np.random.binomial(1,args.dropout_input, input_layer_noisy.shape)
                    #     if args.data=='MNIST':
                    #         show_result(args, input_layer_noisy, 'original_images_noisy.png')
                    
                    # for l in args.layers_entropy:
                    #     count, clusters = count_clusters(args, sess, load, l, thresh=args.thresh, return_clusters=True)
                    #     embeddings = embeddings[clusters!=-1,:]
                    #     clusters = clusters[clusters!=-1]
                    #     plot(args, embeddings, clusters, 'Embedding layer by cluster', 'embedding_by_cluster_{}'.format(l))
                    #     channel_by_cluster(args, sess, load, l)


               

            # after each epoch potentially break out
            if args.max_iterations and iteration > args.max_iterations: break

        # save final model
        if args.save_every:
            save(args, sess, mlp, saver, iteration)


args = parse_args()
load = get_loader(args)
tf.reset_default_graph()
if 'sess' in globals():
    sess.close()
mlp = MLP(args)

# initialize session
gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.45)
sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) 
saver = run_inits(args, sess, mlp)

all_axes = []
iteration = 1
#losses = [tbn('loss:0'), tbn('loss_recon:0'), tbn('loss_clustuse:0'), tbn('loss_entropy:0')]
print("Losses: {}".format(' '.join([tns.name[:-2] for tns in tf.get_collection('losses')])))
for epoch in range(args.num_epochs):
    t = time.time()
    for batch in load.iter_batches():
        if isinstance(batch, tuple):
            batch, batch_labels = batch
            
        iteration+=1
        #if iteration==3000: asdf

        feed = {mlp.x:batch,
                mlp.y:batch,
                mlp.batches:batch_labels,
                tbn('is_training:0'):True,
                mlp.learning_rate:args.learning_rate}
        
        #if iteration%1==0:
            #_ = sess.run([obn('train_op_adversary')], feed_dict=feed)
        if iteration%1==0:
            _ = sess.run([obn('train_op')], feed_dict=feed)

        # print score on train set              
        if iteration%args.print_every==0:
            ls = sess.run(tf.get_collection('losses'), feed_dict=feed)
            lstring = "epoch/iter: {}/{} ({:.1f}) ".format(epoch,iteration,time.time() - t)
            lstring+= ' '.join(['{:.3f}'.format(ls_) for ls_ in ls])
            print(lstring)
            t = time.time()




