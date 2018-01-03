import sys, os, time, math, random
import tensorflow as tf
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as colors

def lrelu(x, leak=0.2, name="lrelu"):
    """ Leaky ReLU activation """
    return tf.maximum(x, leak*x)

def tbn(name):
    """ Get the tensor in the default graph of the given name """
    return tf.get_default_graph().get_tensor_by_name(name)

def obn(name):
    """ Get the operation node in the default graph of the given name """
    return tf.get_default_graph().get_operation_by_name(name)

def plot(data, labels=None, title='', fn='', alpha=.3, s=2, fig=None, ax=None, marker='o', cmap=plt.cm.jet):
    """ One-liner helper function for plotting two-dimensional data with the given label assignments """
    if not ax:
        fig, ax = plt.subplots(1,1)
    ax.set_title(title)
    ax.set_xticks([])
    ax.set_yticks([])

    if data.shape[1]>2:
        print("Can't plot input with >2 dimensions...")
        return 

    if labels is not None:
        r = list(range(data.shape[0]))
        np.random.shuffle(r)

        labels = labels[r]
        data = data[r,:]

        # reorder labels so that there are no missing labels between (0, l.max())
        for l in range(len(np.unique(labels))):
            if l != np.unique(labels)[l]:
                while l < np.unique(labels)[l]:
                    labels = np.where(labels>l, labels-1, labels)

        colors = [cmap(float(i)/(max(1,len(np.unique(labels)-1)))) for i in range(len(np.unique(labels)))]
        colors_ = [colors[int(l)] if l!=-1 else cm.Greys(.5) for l in labels]

        if len(np.unique(labels)) == 1: colors_ = cm.Greys(.5)

        ax.scatter(data[:,0], data[:,1], c=colors_, alpha=alpha, s=s, marker=marker)
    
    else:
        ax.scatter(data[:,0], data[:,1], alpha=alpha, s=s, marker=marker, c=cmap(0.))

    if fn:
        fig.savefig(os.path.join(args.save_folder,fn))
        plt.close('all')
        print("Plot saved to {}".format(fn))

def calculate_mmd(k1, k2, k12):
    """ Calculates MMD given kernels for batch1, batch2, and between batches """
    return k1.sum()/(k1.shape[0]*k1.shape[1]) + k2.sum()/(k2.shape[0]*k2.shape[1]) - 2*k12.sum()/(k12.shape[0]*k12.shape[1])


