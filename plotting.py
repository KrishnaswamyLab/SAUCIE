# -*- coding: utf-8 -*-
# File: plotting.py
# Author: Krishnan Srinivasan <krishnan1994 at gmail>
# Date: 21.09.2017
# Last Modified Date: 21.09.2017

"""
Plotting utils for SAUCIE
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.preprocessing import scale


def plot_embedding2D(data, clusts, save_file='./plots/emb.png',
                     title="Embedding with clusters", dim_red='TSNE'):
    fig, ax = plt.subplots(1,1, figsize=(8,6))
    ax.set_title(title)
    
    if data.shape[1] > 2:
        if dim_red == 'TSNE':
            tsne = TSNE(verbose=2)
            data = tsne.fit_transform(data)
        elif dim_red == 'PCA':
            pca = PCA(2)
            data = pca.fit_transform(data)

    unique_clusts = np.unique(clusts)
    colors = [plt.cm.jet(float(i) / len(unique_clusts))
              for i in range(len(unique_clusts))]
    for i, clust in enumerate(unique_clusts):
        sub_idx = clusts == clust
        ax.scatter(data[sub_idx,0], data[sub_idx,1], c=colors[i], alpha=.3,
                   s=12, label=int(clust))

    lgnd = ax.legend()
    """
    for lh in lgnd.legendHandles:
        lh._sizes = [30]
        lh.set_alpha(1)
    """

    fig.savefig(save_file, dpi=300)
    plt.close('all')


def plot_cluster_heatmap(data, clusts, colnames, markers=None, save_file='./plots/heatmap.png',
                         title="Marker vs. Cluster heatmap"):
    data = pd.DataFrame(data, columns=colnames)
    if markers:
        data = data[markers]
        colnames = markers
    agg = pd.concat([data, pd.DataFrame(dict(cluster=clusts))], axis=1)
    by_cluster = agg.groupby(['cluster']) 
    z_scaled_cluster = scale(by_cluster.mean().values, axis=0)
    unique, counts = np.unique(clusts, return_counts=True)
    counts = counts / len(clusts) * 100 # computes percentage
    clust_dict = dict(zip(unique, counts))
    f = plt.figure(figsize=(18, 10))
    ax = sns.heatmap(z_scaled_cluster, xticklabels=colnames,
                     yticklabels=['{}: ({}%)'.format(i+1, round(clust_dict[i], 2))
                                  for i in range(len(unique))])
    ax.set_title('Marker vs. Cluster Heatmap', {'fontsize': 20})
    ax.set_ylabel('Cluster Number', {'fontsize': 15})
    ax.set_xlabel('Marker', {'fontsize': 15})
    f.savefig(save_file, dpi=300)
    plt.close('all')
