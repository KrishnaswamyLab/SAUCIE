import sys, os, time, math, argparse, pickle
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from model import MLP, tbn, obn
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score
from scipy.io import savemat
from utils import *

def parse_args():
    parser = argparse.ArgumentParser()


    # NAME
    parser.add_argument('--save_folder', type=str, default='saved_toy_noid_')
    parser.add_argument('--data', type=str, default='ZIKA')
    # parser.add_argument('--data_folder', type=str, default='/data/krishnan/flu_data/')
    parser.add_argument('--data_folder', type=str, default='/home/krishnan/data/zika_data/gated')

    # TRAINING PARAMETERS
    parser.add_argument('--batch_size', type=int, default=100)
    parser.add_argument('--learning_rate', type=float, default=.001)
    parser.add_argument('--num_epochs', type=int, default=100)
    parser.add_argument('--save_every', type=int, default=1000)
    parser.add_argument('--print_every', type=int, default=100)
    parser.add_argument('--max_iterations', type=int, default=10000)
    
    # MODEL ARCHITECTURE
    parser.add_argument('--layers', type=str, default='1024,512,256,2')
    # parser.add_argument('--layers', type=str, default='256,128,5,2') # make it faster while testing
    parser.add_argument('--activation', type=str, default='tanh')
    parser.add_argument('--activation_idreg', type=str, default='tanh')
    parser.add_argument('--loss', type=str, default='bce')
    parser.add_argument('--dropout_p', type=float, default=1.)
    parser.add_argument('--batch_norm', type=bool, default=True)
    
    # REGULARIZATIONS
    parser.add_argument('--lambda_l1', type=float, default=0)
    parser.add_argument('--lambda_l2', type=float, default=0)
    parser.add_argument('--layers_sparsity', type=str, default='')
    parser.add_argument('--lambdas_sparsity', type=str, default='')
    parser.add_argument('--layers_clustuse', type=str, default='')
    parser.add_argument('--lambdas_clustuse', type=str, default='')
    parser.add_argument('--layers_entropy', type=str, default='0')
    parser.add_argument('--lambdas_entropy', type=str, default='')
    parser.add_argument('--normalization_method', type=str, default='tanh')
    parser.add_argument('--thresh', type=float, default=.5)
    parser.add_argument('--sigma', type=float, default=.5)

    # NOISE
    parser.add_argument('--add_noise', type=float, default=0)
    parser.add_argument('--dropout_input', type=float, default=0)
    parser.add_argument('--noise_stddev', type=float, default=0)


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

    if args.data == 'MNIST': args.input_dim = 28*28
    elif args.data == 'cytof_emt': args.input_dim = 40
    elif args.data == 'ZIKA': args.input_dim = 35
    elif args.data == 'FLU': args.input_dim = 22
    elif args.data == 'TOY': args.input_dim = 100
    else: raise Exception("Could not parse name of data for input_dim: {}".format(args.data))

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

def save(args, sess, mlp, saver, loader, iteration):
    savefile = os.path.join(args.save_folder, 'AE')
    saver.save(sess, savefile , global_step=iteration, write_meta_graph=True)
    print("Model saved to {}".format(savefile))

def train(args):
    loader = get_loader(args)
    mlp = MLP(args)

    # relevant data to save for visualizations later
    DATA = {'reconstruction_losses': []}

    # initialize
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.10)
    sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) 
    with sess:
        saver = run_inits(args, sess, mlp)

        iteration = 1
        losses = [tbn('loss:0'), tbn('loss_recon:0'), tbn('loss_clustuse:0'), tbn('loss_entropy:0')]
        for epoch in range(args.num_epochs):
            t = time.time()
            for batch, batch_labels in loader.iter_batches('train'):
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

                DATA['reconstruction_losses'].append(l1)


                # print score on train set              
                if iteration%args.print_every==0:
                    print("epoch/iter: {}/{} loss: {:.3f} ({:.3f} {:.3f} {:.3f}) time: {:.1f}".format(epoch,
                        iteration, ltotal, l1, l2, l3, time.time() - t))
                    t = time.time()
                    
                # save
                if args.save_every and iteration%args.save_every==0:
                    fig, ax = plt.subplots(1,1)
                    ax.plot(range(len(DATA['reconstruction_losses'])), DATA['reconstruction_losses'])
                    ax.set_xlabel('Iteration')
                    ax.set_ylabel('Reconstruction loss')
                    title = 'ID reg: {}'.format(str(args.lambdas_entropy))
                    ax.set_title(title)
                    fig.savefig(os.path.join(args.save_folder, 'reconstruction_loss'))
                    save(args, sess, mlp, saver, loader, iteration)

                    # embeddings, labels = get_layer(sess, loader, 'layer_embedding_activation:0')
                    embeddings, labels = get_layer(sess, loader, 'layer_decoder_0_bninput:0')
                    input_layer, labels = get_layer(sess, loader, 'x:0')
                    reconstruction, labels = get_layer(sess, loader, 'layer_output_activation:0')

                    plot(args, embeddings, labels, 'Embedding layer by label', 'embedding_by_label')
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
                    

                    # if not args.layers_entropy:
                    #     plot(args, embeddings, np.zeros(embeddings.shape[0]), 'Embedding layer', 'embedding_without_idreg')

                    for l in args.layers_entropy:
                        count, clusters = count_clusters(args, sess, loader, l, thresh=args.thresh, return_clusters=True)
                        plot(args, embeddings, clusters, 'Embedding layer by cluster', 'embedding_by_cluster_{}'.format(l))
                        # activations_heatmap(args, sess, loader, l)
                    #     print("entropy lambdas: {} Number of clusters: {}".format(args.lambdas_entropy, count))
                    #     if args.data=='MNIST':
                    #         decode_cluster_means(args, sess, loader, l, clusters)
                    #         confusion_matrix(args, sess, loader, l, clusters)
                        
                        

                    plt.close('all')

            # after each epoch potentiall break out
            if args.max_iterations and iteration > args.max_iterations: break

        # save final model
        if args.save_every:
            save(args, sess, mlp, saver, loader, iteration)

        for l in args.layers_entropy:
            count = count_clusters(args, sess, loader, l, thresh=args.thresh)
            print("Final entropy lambdas: {} Number of clusters: {}".format(args.lambdas_entropy, count))

        with open(os.path.join(args.save_folder, 'data.pkl'), 'wb+') as f:
            pickle.dump(DATA, f)
            


if __name__=='__main__':
    args = parse_args()
    train(args)