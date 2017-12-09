import sys, os, time, math, argparse, pickle, fcsparser
import numpy as np
import tensorflow as tf
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import matplotlib.cm as cm
from utils import *
from model import *
from loader import *
plt.ion()
PLOT = True


def get_data(n_batches=2, n_pts_per_cluster=5000):
    data = []
    labels = []
    spikein_mask = []
    for i in range(n_batches):
        data.append(np.random.normal(i,.1, (n_pts_per_cluster, 2)))
        labels.append(i*np.ones(n_pts_per_cluster))
        spikein_mask.append(np.ones(n_pts_per_cluster))

        col = 0
        if i%2==1:
            col = 1

        data[-1][n_pts_per_cluster//2:,col] += 1
        spikein_mask[-1][:n_pts_per_cluster//2] -= 1

    data = np.concatenate(data, axis=0)
    labels = np.concatenate(labels, axis=0)
    spikein_mask = np.concatenate(spikein_mask, axis=0)

    return data, labels, spikein_mask

def run_with_batchcorrection():
    tf.reset_default_graph()
    load = Loader(data, labels, spikein_mask)
    saucie = SAUCIE(input_dim = data.shape[1],
                    lambda_batchcorrection=.1)

    print(saucie.get_loss_names())
    t = time.time()

    if PLOT:
        fig_original = plt.figure()
        x, labels_, spikein_mask_ = saucie.get_layer(load, 'x')
        ax = fig_original.add_subplot(1,1,1)
        plot(x[spikein_mask_==1], labels_[spikein_mask_==1], ax=ax, marker='^', s=20, alpha=.2)
        plot(x[spikein_mask_==0], labels_[spikein_mask_==0], ax=ax, marker='.', s=20, alpha=.2)
        fig_original.canvas.draw()

        fig_reconstructed = plt.figure()
    for iteration in range(1,10): 
        nsteps = 500

        saucie.train(load, steps=nsteps)
        
        lstring = saucie.get_loss(load)
        print("{} ({:.1f} s): {}".format(iteration*nsteps, time.time()-t, lstring))
        t = time.time()

        if PLOT:
            reconstruction, labels_, spikein_mask_ = saucie.get_layer(load, 'layer_output_activation')
            fig_reconstructed.clf()
            ax = fig_reconstructed.add_subplot(1,1,1)
            plot(reconstruction[spikein_mask_==1], labels_[spikein_mask_==1], ax=ax, marker='^', s=20, alpha=.2)
            plot(reconstruction[spikein_mask_==0], labels_[spikein_mask_==0], ax=ax, marker='.', s=20, alpha=.2)
            fig_reconstructed.canvas.draw()

    
    reconstruction, labels_, spikein_mask_ = saucie.get_layer(load, 'layer_output_activation')

    return reconstruction

def run_with_clustering(reconstructed_data):
    tf.reset_default_graph()
    load = Loader(reconstructed_data)
    saucie = SAUCIE(input_dim = reconstructed_data.shape[1],
                    layers_entropy = [2],
                    lambdas_entropy = [1])

    print(saucie.get_loss_names())
    t = time.time()

    if PLOT:
        fig = plt.figure()
    for iteration in range(1,20): 
        nsteps = 500

        saucie.train(load, steps=nsteps)
        
        lstring = saucie.get_loss(load)
        print("{} ({:.1f} s): {}".format(iteration*nsteps, time.time()-t, lstring))
        t = time.time()

        x = saucie.get_layer(load, 'x')
        
        
        if PLOT:
            x = saucie.get_layer(load, 'x')
            count, clusters = saucie.get_clusters(load)

            fig.clf()
            ax = fig.add_subplot(1,1,1)
            plot(x[clusters==-1,:], clusters[clusters==-1], ax=ax, alpha=.1, cmap=cm.Greys)
            plot(x[clusters!=-1,:], clusters[clusters!=-1], ax=ax)
            fig.canvas.draw()

            if count==3: break



data, labels, spikein_mask = get_data()

reconstruction = run_with_batchcorrection()

run_with_clustering(reconstruction)


input("Batch correction and clustering done. Press enter to end.")
