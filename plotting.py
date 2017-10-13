# -*- coding: utf-8 -*-
# File: plotting.py
# Author: Krishnan Srinivasan <krishnan1994 at gmail>
# Date: 21.09.2017
# Last Modified Date: 06.10.2017

"""
Plotting utils for SAUCIE
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.preprocessing import scale
from mpl_toolkits.mplot3d import Axes3D


def plot_embedding2D(data, clusts, save_file='./plots/emb2d.png',
                     title="Embedding with clusters", dim_red='PCA'):
    fig, ax = plt.subplots(1,1, figsize=(6,6))
    ax.set_title(title)
    
    if data.shape[1] > 2:
        if dim_red == 'TSNE':
            tsne = TSNE(verbose=2)
            data = tsne.fit_transform(data)
        elif dim_red == 'PCA':
            pca = PCA(2)
            data = pca.fit_transform(data)

    if clusts:
        unique_clusts = np.unique(clusts)
        num_clusters = len(unique_clusters)
        colors = [plt.cm.jet(float(i) / len(unique_clusts))
                  for i in range(len(unique_clusts))]
        relabel = np.zeros(len(clusts))
        for i, label in enumerate(unique_clusts):
            sub_idx = clusts == label
            relabel[sub_idx] = i
            ax.scatter(data[sub_idx,0], data[sub_idx,1], c=colors[i], alpha=.3,
                       s=5, label=int(i), linewidth=0.)
        lgnd = ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05),  shadow=True, ncol=4)
	for lh in lgnd.legendHandles:
	    lh._sizes = [50]
	    lh.set_alpha(1)
    else:
        ax.scatter(data[:,0], data[:,1], alpha=.3, s=5, linewidth=0.)

    if save_file:
        save_dir = os.path.split(save_file)[0]
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        if clusts:
            fig.savefig(save_file, dpi=300, bbox_extra_artists=(lgnd,), bbox_inches='tight')
        else:
            fig.savefig(save_file, dpi=300)
    plt.close('all')
    return relabeled


def plot_embedding3D(data, clusts=None, save_file='./plots/emb3d.png', title='3D embedding', dim_red='PCA'):
    if data.shape[1] != 3:
        pca = PCA(3)
        data = pca.fit_transform(data)

    f, ax = plt.subplots(ncols=1, nrows=1, subplot_kw={'projection': '3d'}) 

    if clusts:
        unique_clusts = np.unique(clusts)
        colors = [plt.cm.jet(float(i)) for i in range(len(unique_clusts))]
        for i, clust in enumerate(unique_clusts):
            sub_idx = clusts == clust
            ax.scatter(data[sub_idx,0], data[sub_idx,1], data[sub_idx,2], c=colors[i], alpha=.3,
                      s=12, label=int(clust))
        lgnd = ax.legend()
    else:
        ax.scatter(recons_pc[:,0], recons_pc[:,1], recons_pc[:,2])
    ax.set_title(title)

    if save_file:
        save_dir = os.path.split(save_file)[0]
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        f.savefig(save_file, dpi=300)
    plt.close('all')


def plot_cluster_heatmap(data, clusts, colnames, markers=None, zscore=True,
                         save_file='./plots/heatmap.png',
                         title="Marker vs. Cluster Heatmap"):
    data = pd.DataFrame(data, columns=colnames)
    if markers:
        data = data[markers]
        colnames = markers
    data['cluster'] = clusts
    by_cluster = data.groupby(['cluster']) 
    scaled = by_cluster
    if zscore:
        scaled = scale(scaled.mean().values, axis=0)
    unique, counts = np.unique(clusts, return_counts=True)
    counts = counts / len(clusts) * 100 # computes percentage
    clust_dict = dict(zip(unique, counts))
    cluster_labels = ['{}: ({}%)'.format(i+1, round(clust_dict[i], 2))
                      for i in range(len(unique))]
    f = plt.figure(figsize=(18, 10))
    ax = sns.heatmap(scaled, xticklabels=colnames,
                     yticklabels=cluster_labels)
    ax.set_title(title, {'fontsize': 20})
    ax.set_ylabel('Cluster Number', {'fontsize': 15})
    ax.set_xlabel('Marker', {'fontsize': 15})
    if save_file:
        save_dir = os.path.split(save_file)[0]
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        f.savefig(save_file, dpi=300)
    plt.close('all')


def plot_cluster_linkage_heatmap(data, clusts, colnames, markers=None, save_file='./plots/linkage.png',
                                 title="Marker vs Cluster linkage heatmap"):
    data = pd.DataFrame(data, columns=colnames)
    if markers:
        data = data[markers]
        colnames = markers
    data['cluster'] = clusts
    by_cluster = data.groupby(['cluster'])
    unique, counts = np.unique(clusts, return_counts=True)
    counts = counts / len(clusts) * 100 # computes percentage
    clust_dict = dict(zip(unique, counts))
    cluster_labels = ['{}: ({}%)'.format(i, round(clust_dict[i], 2))
                      for i in unique]
    cg = sns.clustermap(by_cluster.mean().values.T, z_score=0, yticklabels=colnames, xticklabels=cluster_labels)

    if save_file:
        save_dir = os.path.split(save_file)[0]
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        cg.savefig(save_file, dpi=300)
    plt.close('all')


def plot_edge_plots(data, colnames, marker_sub=['VIM', 'CDH1', 'CDH2', 'EZH2', 'SNAI1', 'ZEB1'],
                    x_vars=['VIM', 'SNAI1', 'CDH1'], y_vars=['CDH1', 'EZH2', 'ZEB1', 'CDH2'],
                    title='Edge plots', save_file='./plots/edgeplots.png'):

    marker_cols = np.any([np.array(colnames) == x for x in marker_sub], axis=0)
    df = pd.DataFrame(data=data[:, marker_cols], columns=marker_sub)
    p = sns.pairplot(df, plot_kws=dict(s=10, linewidth=0), x_vars=x_vars, y_vars=y_vars)

    for row in p.axes:
	for col in row:
	    col.xaxis.reset_ticks()
	    col.yaxis.reset_ticks() # adds back ticks to subplots

    p.fig.suptitle(title)
    p.fig.tight_layout(rect=[0, 0.03, 1, 0.95])

    if save_file:
        save_dir = os.path.split(save_file)[0]
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        p.fig.savefig(save_file)
    plt.close('all')

