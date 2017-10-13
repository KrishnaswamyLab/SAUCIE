import sys, os, time, math, argparse, pickle
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import matplotlib.cm as cm
import pandas as pd
from loader import Loader
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score
from sklearn.manifold import TSNE
from sklearn.metrics.pairwise import pairwise_distances
from utils import *
from sklearn.feature_selection import mutual_info_classif

def restore_model(model_folder):
    tf.reset_default_graph()

    with open('{}/args.pkl'.format(model_folder), 'rb') as f:
        args = pickle.load(f)
    args.dropout_p = 1.
    loader = get_loader(args, load_full=False)

    sess = tf.Session()
    ckpt = tf.train.get_checkpoint_state(args.save_folder)
    saver = tf.train.import_meta_graph('{}.meta'.format(ckpt.model_checkpoint_path))
    saver.restore(sess, ckpt.model_checkpoint_path)

    return args, sess, loader

def color_by_marker(args, sess, loader, recon=False):
    if not recon:
        x, labels = get_layer(sess, loader, 'x:0')
        fn = 'embedding_by_markers_orig'
    else:
        x, labels = get_layer(sess, loader, 'layer_output_activation:0')
        fn = 'embedding_by_markers_recon'

    with open('/home/krishnan/data/zika_data/gated/markers.csv') as f:
        cols = f.read().strip()
        cols = [c.strip().split('_')[1] for c in cols.split('\n')]


    embedding, labels = get_layer(sess, loader, 'layer_embedding_activation:0')

    normalizer = colors.Normalize(0, 1)

    g = math.ceil(math.sqrt(x.shape[1]))
    fig, axes = plt.subplots(g,g, figsize=(20,20))
    plt.subplots_adjust(left=0.0, bottom=0.0, right=1.0, top=1.0, wspace=0.0, hspace=0.0)
    for i in range(len(axes.flatten())):
        ax = axes.flatten()[i]
        ax.set_xticks([])
        ax.set_yticks([])
    for i in range(x.shape[1]):
        ax = axes.flatten()[i]
        #normalizer = colors.Normalize(x[:,i].min(), x[:,i].max())
        ax.scatter(embedding[:,0], embedding[:,1], c=cm.jet(normalizer(x[:,i])), s=8, alpha=.7)
        ax.annotate("{}".format(cols[i]), xy=(.1,.9), xycoords='axes fraction', size=24)

    fig.savefig(os.path.join(args.save_folder,fn))

def channel_by_cluster(args, sess, loader, layer):
    x, labels = get_layer(sess, loader, 'x:0')
    count, clusters = count_clusters(args, sess, loader, layer, thresh=.5, return_clusters=True)

    df = pd.DataFrame(x)
    df['cluster'] = clusters
    grouped = df.groupby('cluster')
    means = grouped.apply(lambda x: x.mean(axis=0))
    del means['cluster']

    with open('/home/krishnan/data/zika_data/gated/markers.csv') as f:
        cols = f.read().strip()
        cols = [c.strip().split('_')[1] for c in cols.split('\n')]

    fig, ax = plt.subplots(1,1, figsize=(30,30))
    plt.subplots_adjust(left=.1, bottom=.1, right=1, top=1)
    ax.imshow(means.transpose(), cmap='jet')
    ax.set_yticks(range(len(cols)))
    ax.set_yticklabels(cols, fontsize=24)
    ax.set_xticks([])
    ax.set_ylabel('Marker', fontsize=28)
    ax.set_xlabel('Cluster', fontsize=28)
    fig.savefig(args.save_folder+'/means')

def cluster_on_embeddings(folder_embeddings, folder_clusters):
    # get embeddings
    args, sess, loader = restore_model(folder_embeddings)
    with sess:
        embeddings, labels = get_layer(sess, loader, 'layer_embedding_activation:0')

    # get clusters
    args, sess, loader = restore_model(folder_clusters)
    with sess:
        layer = args.layers_entropy[0]
        count, clusters = count_clusters(args, sess, loader, layer, thresh=.5, return_clusters=True)

    plot(args, embeddings, clusters, 'Embedding layer by cluster', 'embedding_by_cluster')  

def diffusion(input_layer, clusters, n=1000):
    # input_layer = input_layer[:n,:]
    # clusters = clusters[:n]
    # df = DiffusionMap(sigma=5, embedding_dim=10, verbose=False, k=100)
    # V,lam = df.fit_transform(input_layer, density_normalize=False)

    # mi = mutual_info_classif(input_layer, clusters)
    # print(mi)

    print(input_layer.shape)

    for i in range(input_layer.shape[1]):
        binned = np.histogram(input_layer[:,i])
        if i%100==0: print(i)
    print(binned.shape)
    # diffusion_dists = pairwise_distances(V)
    # print(diffusion_dists)

    # embeddings, labels = get_layer(sess, loader, 'layer_embedding_activation:0')
    # embeddings = embeddings[:n,:]
    # embedding_dists = pairwise_distances(embeddings)

    # print(embedding_dists)
    


def analyze():
    model_folders = sys.argv[1:]

    
    
    for model_folder in model_folders:
        # if model_folder != 'saved_zika_2d':
        #     cluster_on_embeddings('saved_zika_2d', model_folder)
        # cluster_on_embeddings('saved_toy_noid_', model_folder)
        

        args, sess, loader = restore_model(model_folder)
        print(args.save_folder)
        with sess:
            # color_by_marker(args, sess, loader, recon=False)
            # color_by_marker(args, sess, loader, recon=True)
            # activations_heatmap(args, sess, loader, 2)

            # embeddings, labels = get_layer(sess, loader, 'layer_embedding_activation:0')
            embeddings, labels = get_layer(sess, loader, 'layer_decoder_0_bninput:0')
            input_layer, labels = get_layer(sess, loader, 'x:0')
            reconstruction, labels = get_layer(sess, loader, 'layer_output_activation:0')

            plot(args, embeddings, np.zeros(embeddings.shape[0]), 'Embedding', 'embedding_without_label')
            sys.exit()
            plot(args, embeddings, labels, 'Embedding layer by label', 'embedding_by_label')
            if not args.layers_entropy:
                plot(args, embeddings, np.zeros(embeddings.shape[0]), 'Embedding layer', 'embedding_without_idreg')
            else:
                for l in args.layers_entropy:
                    count, clusters = count_clusters(args, sess, loader, l, thresh=args.thresh, return_clusters=True)
            
                    plot(args, embeddings, clusters, 'Embedding layer by cluster', 'embedding_by_cluster_{}'.format(l))
                    print("layer: {} entropy lambdas: {} Number of clusters: {}".format(l, args.lambdas_entropy, count))
                    if args.data=='MNIST':
                        decode_cluster_means(args, sess, loader, clusters)
                        confusion_matrix(args, sess, loader, l, clusters)
                    activations_heatmap(args, sess, loader, l)
                    channel_by_cluster(args, sess, loader, l)




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


                
            





if __name__=='__main__':
    analyze()