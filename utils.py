import sys, os, time, math, argparse, contextlib
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as colors
import pandas as pd
from matplotlib import offsetbox
import loader
from skimage.io import imsave
from sklearn.manifold import TSNE
from sklearn.metrics import adjusted_rand_score, silhouette_score, pairwise_distances
from sklearn.decomposition import PCA
from scipy.spatial.distance import pdist, squareform
from skimage.io import imsave
import seaborn as sns


def plot_mnist(args, X, y, X_embedded, name, min_dist=15.):
    fig, ax = plt.subplots(1,1, figsize=(10,10), frameon=False)
    ax.set_xticks([])
    ax.set_yticks([])
    plt.subplots_adjust(left=0.0, bottom=0.0, right=1.0, top=1.0, wspace=0.0, hspace=0.0)
    
    colors = [plt.cm.jet(float(i)/len(np.unique(y))) for i in range(len(np.unique(y)))]
    colors_y = [colors[int(y_)] for y_ in y]
    ax.scatter(X_embedded[:, 0], X_embedded[:, 1], c=colors_y)

    shown_images = np.array([[15., 15.]])
    for i in range(1000):
        for d in range(10):
            if i >= X_embedded[y==d].shape[0]: continue

            dist = np.sum((X_embedded[y==d][i] - shown_images) ** 2, 1)
            if np.min(dist) < min_dist:
                continue
            img = X_embedded[y==d][i]
            shown_images = np.concatenate([shown_images,img.reshape((1,-1))])

            imagebox = offsetbox.AnnotationBbox( 
                offsetbox.OffsetImage(X[y==d][i].reshape(28, 28), cmap=cm.gray_r), X_embedded[y==d][i]
                )
            ax.add_artist(imagebox)
    fig.savefig(args.save_folder+'/embed_w_images'+name)

def asinh(x):
    f = np.vectorize(lambda y: math.asinh(y/5))

    return f(x) 

class SilentFile(object):
    def write(self, x): pass
    def flush(self): pass

@contextlib.contextmanager
def silence():
    save_stdout = sys.stdout
    sys.stdout = SilentFile()
    yield
    sys.stdout = save_stdout

def softmax(x):

    return np.exp(x) / np.sum(np.exp(x), axis=1).reshape((-1,1))

def show_result(args, data, fname, grid_size=(8, 8), grid_pad=5):
    img_height = 28
    img_width = 28
    data = data.reshape((data.shape[0], img_height, img_width))
    img_h, img_w = data.shape[1], data.shape[2]
    grid_h = img_h * grid_size[0] + grid_pad * (grid_size[0] - 1)
    grid_w = img_w * grid_size[1] + grid_pad * (grid_size[1] - 1)
    img_grid = np.zeros((grid_h, grid_w), dtype=np.uint8)
    for i, res in enumerate(data):
        if i >= grid_size[0] * grid_size[1]:
            break
        img = (res) * 255
        img = img.astype(np.uint8)
        row = (i // grid_size[0]) * (img_h + grid_pad)
        col = (i % grid_size[1]) * (img_w + grid_pad)
        img_grid[row:row + img_h, col:col + img_w] = img
    imsave(os.path.join(args.save_folder, fname), img_grid)

def lrelu(x, leak=0.2, name="lrelu"):

  return tf.maximum(x, leak*x)

def get_loader(args):
    try:
        l =  getattr(loader, args.loader)

    except:
        raise Exception("Couldn't parse loader to use: {}".format(str(args.loader)))

    return l(args)

def tbn(name):

    return tf.get_default_graph().get_tensor_by_name(name)

def obn(name):

    return tf.get_default_graph().get_operation_by_name(name)

def to_one_hot(y, n):
    h = np.zeros((y.shape[0], n))
    h[np.arange(y.shape[0]), y] = 1
    return h

def get_layer(sess, l, name, scope):
    tensor = tbn("{}/{}:0".format(scope,name))
    layer = []
    labels = []
    for batch in l.iter_batches():
        if isinstance(batch, tuple):
            batch, batch_labels = batch
            labels.append(batch_labels)
        
        feed = {tbn('{}/x:0'.format(scope)):batch}#, tbn('is_training:0'):False}
        #feed = {tbn('x:0'):batch, tbn('batches:0'):batch_labels}
        [act] = sess.run([tensor], feed_dict=feed)

        layer.append(act)
        

    layer = np.concatenate(layer, axis=0)

    if labels:
        labels = np.concatenate(labels, axis=0)
        return layer, labels
    else:
        return layer

def plot(args, data, labels=None, title='', fn='', alpha=.3, s=2, fig=None, ax=None, marker='o', cmap=plt.cm.jet):
    if not fig:
        fig, ax = plt.subplots(1,1)
    ax.set_title(title)
    # ax.set_xticks([])
    # ax.set_yticks([])

    if data.shape[1]>2:
        print("Can't plot input with >2 dimensions...")
        return 

    if labels is not None and len(np.unique(labels))>1:
        r = list(range(data.shape[0]))
        np.random.shuffle(r)

        labels = labels[r]
        data = data[r,:]

        colors = [cmap(float(i)/(len(np.unique(labels-1)))) for i in range(len(np.unique(labels)))]
        colors_ = [colors[int(l)] if l!=-1 else cm.Greys(.5) for l in labels]

        if len(np.unique(labels)) == 1: colors_ = cm.Greys(.5)

        ax.scatter(data[:,0], data[:,1], c=colors_, alpha=alpha, s=s, marker=marker)
    
    else:
        ax.scatter(data[:,0], data[:,1], alpha=alpha, s=s, marker=marker, c=cmap(1.))

    if fn:
        fig.savefig(os.path.join(args.save_folder,fn))
        plt.close('all')
        print("Plot saved to {}".format(fn))

def neuronuse_histogram(args, sess, loader, layer=None, neuronuse=None, ax=None, scope=''):
    if not isinstance(neuronuse, np.ndarray):
        all_acts, all_labels = get_layer(sess, loader, 'layer_encoder_{}_activation'.format(layer), scope)
        all_acts = (all_acts+1) / 2.
        neuronuse = all_acts.sum(axis=0)

    if not ax:
        fig, ax = plt.subplots(1,1, figsize=(5,5))
    
    ax.bar(range(len(neuronuse)), neuronuse, np.ones_like(neuronuse))

    return neuronuse

def activations_heatmap(args, sess, loader, layer, thresh=.5, ax=None, save=None):
    all_acts, all_labels = get_layer(sess, loader, 'layer_encoder_{}_activation:0'.format(layer))

    
    #nonzero = all_acts.reshape((-1))[all_acts.reshape((-1)) > 0]
    
    if not ax:
        fig, ax = plt.subplots(1,1, figsize=(5,5))
    #ax.set_title('ID Regularization', fontsize=18)
    #ax.set_xlabel('Activation', fontsize=18)
    #ax.set_ylabel('Count', fontsize=18, labelpad=-10)
    #ax.grid(linewidth=1, color='k', linestyle='--', alpha=.5)
    #ax.set_axisbelow(False)
    #ax.set_xticks([0,.5,1])
    

    n, bins = np.histogram(all_acts, bins=20)

    bin_starts = bins[0:bins.size-1]
    bin_widths = bins[1:bins.size] - bins[0:bins.size-1]
    bin_starts = bin_starts + bin_widths/2. 
    ax.bar(bin_starts,n,bin_widths)

    if save:
        fig.savefig(os.path.join(args.save_folder, 'layer_{}_activations_histogram.png'.format(layer)), format='png', dpi=600)

    return


    if layer in args.layers_entropy:
        acts_normalized, labels = get_layer(sess, loader, 'normalized_activations_layer_{}:0'.format(layer))
        normalized_nonzero = acts_normalized.reshape((-1))#[acts_normalized.reshape((-1)) > 0]
        fig, ax = plt.subplots(1,1, figsize=(5,5))
        ax.set_title('ID Regularization', fontsize=18)
        ax.set_xlabel('Activation', fontsize=18)
        ax.set_ylabel('Count', fontsize=18, labelpad=-10)
        ax.grid(linewidth=1, color='k', linestyle='--', alpha=.5)
        ax.set_axisbelow(False)
        ax.set_xticks([0,.5,1])
        
        

        # ax.hist(np.log(normalized_nonzero), bins=1000)
        # ax.hist(normalized_nonzero, bins=1000)
        n, bins, patches = plt.hist(normalized_nonzero, bins=1000, log=True)
        bin_starts = bins[0:bins.size-1]
        bin_widths = bins[1:bins.size] - bins[0:bins.size-1]
        ax.bar(bin_starts,np.log10(n),bin_widths)
        ax.set_ylim([0,10**6])
        ax.set_xlim([0,1])
        ax.set_yticks([10**2,10**4,10**6])
        ax.set_yticklabels(['100', '10000', '1000000'])

        [tick.label.set_fontsize(6) for tick in ax.xaxis.get_major_ticks()]
        [tick.label.set_fontsize(6) for tick in ax.yaxis.get_major_ticks()]
        fig.subplots_adjust(right=.98)
        fig.set_figheight(5)
        fig.set_figwidth(5.3)
        fig.savefig(os.path.join(args.save_folder, 'layer_{}_activations_histogram.png'.format(layer)), format='png', dpi=600)
    else:
        acts_normalized = all_acts
        fig, ax = plt.subplots(1,1, figsize=(10,10))
        fn = 'No regularization' if not (args.layers_entropy or args.layers_sparsity) else 'L1 Regularization'
        ax.set_title(fn, fontsize=18)
        ax.set_xlabel('Activation', fontsize=18)
        ax.set_ylabel('Count', fontsize=18, labelpad=-10)
        ax.grid(linewidth=1, color='k', linestyle='--', alpha=.5)
        ax.set_axisbelow(False)
        ax.set_xticks([0,.5,1])
        [tick.label.set_fontsize(6) for tick in ax.xaxis.get_major_ticks()]
        ax.set_xlim([0,1])
        
        # ax.hist(np.log(nonzero), bins=1000)
        # ax.hist(nonzero, bins=1000)

        n, bins, patches = plt.hist(nonzero, bins=1000, log=True)
        bin_starts = bins[0:bins.size-1]
        bin_widths = bins[1:bins.size] - bins[0:bins.size-1]
        ax.bar(bin_starts,np.log10(n),bin_widths)
        #[tick.label.set_fontsize(6) for tick in ax.yaxis.get_major_ticks()]
        ax.set_ylim([0,10**6])
        ax.set_yticks([10**2,10**4,10**6])
        ax.set_yticklabels(['100', '10000', '1000000'])

        fig.subplots_adjust(right=.98)
        fig.set_figheight(5)
        fig.set_figwidth(5.3)
        fig.savefig(os.path.join(args.save_folder, 'layer_{}_activations_histogram.png'.format(layer)), format='png', dpi=600)

        
    return

    binarized = acts_normalized
    # binarized = np.where(acts_normalized>thresh, 1, 0)
    fig, ax = plt.subplots(1,1, figsize=(10,10), dpi=150)
    fig.subplots_adjust(hspace=0, wspace=0, left=0, right=1, top=1, bottom=0)
    all_argmaxes = np.zeros((10,10))
    for i in range(1):
        
        squaredims = int(math.floor(np.sqrt( args.layers[layer] )))
        this_digit = binarized[i,:]
        this_digit = this_digit[:squaredims**2].reshape((squaredims,squaredims))
        # ax = axes.flatten()[i]
        # ax.set_xlabel("{}".format(i), fontsize=14)
        ax.imshow(this_digit, cmap='gray', vmin=0, vmax=1)
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        # ax.grid('off')
        ax.set_aspect('equal')
    fig.savefig(os.path.join(args.save_folder, 'layer_{}_activations_heatmap'.format(layer)))
    plt.close('all')
    print("Activations heatmap saved.")
    return





    binarized = acts_normalized
    # binarized = np.where(acts_normalized>thresh, 1, 0)
    fig, axes = plt.subplots(10,10, figsize=(20,20), dpi=150)
    fig.subplots_adjust(hspace=.01, wspace=.02, left=.02, right=1, top=1, bottom=0)
    all_argmaxes = np.zeros((10,10))
    for i in range(10):
        if args.data == 'MNIST':
            # pick out this digit
            all_this_digit = binarized[all_labels==i,:]
        else:
            all_this_digit = binarized[[ii*10+i for ii in range(10)],:]
        axes[i,0].set_ylabel("{}".format(i))
        for j in range(10):
            squaredims = int(math.floor(np.sqrt( args.layers[layer] )))
            this_digit = all_this_digit[j,:squaredims**2].reshape((squaredims,squaredims))
            ax = axes[i,j]
            all_argmaxes[i,j] = this_digit.argmax() 
            ax.imshow(this_digit, cmap='gray')
            ax.set_xticklabels([])
            ax.set_yticklabels([])
            ax.grid('off')
            ax.set_aspect('equal')
    fig.savefig(os.path.join(args.save_folder, 'layer_{}_activations_heatmap'.format(layer)))
    plt.close('all')
    print("Activations heatmap saved.")

def calculate_loss(sess, loader, train_or_test='test'):
    loss_tensor = tbn('loss:0')
    x_tensor = tbn('x:0')
    y_tensor = tbn('y:0')
    losses = []
    for batch, batch_labels in loader.iter_batches(train_or_test):
        feed = {x_tensor:batch,
                y_tensor:batch,
                tbn('is_training:0'):False}
        [l] = sess.run([loss_tensor], feed_dict=feed)
        losses.append(l)

    avg_loss = sum(losses) / float(len(losses))
    return avg_loss

def count_clusters(args, sess, loader, layer, thresh=0, return_clusters=False, BIN_MIN=50, scope=''):
    '''Counts the number of clusters after binarizing the activations of the given layer.'''
    #acts, labels = get_layer(sess, loader, 'normalized_activations_layer_{}:0'.format(layer))
    acts, labels = get_layer(sess, loader, 'layer_encoder_{}_activation'.format(layer), scope=scope)

    unique_argmaxes, unique_argmaxes_counts = np.unique(acts.argmax(axis=1), return_counts=True)
    unique_argmaxes_counts = list(reversed(sorted(unique_argmaxes_counts.tolist())))
    # for i in range(len(unique_argmaxes)):
    #     if i>10: break
    #     print(unique_argmaxes[i], unique_argmaxes_counts[i])
    # print("Max neuron values: ", acts.max(axis=1)[:5], "...")
    # print("Number of unique max neurons: ", len(np.unique(acts.argmax(axis=1))))


    binarized = np.where(acts>thresh, 1, 0)

    # k = 10
    # binarized = np.zeros(acts.shape)
    # topk = np.argpartition(acts, -k, axis=1)[:,-k:]
    # for i,row in enumerate(topk):
    #     for j in row:
    #         binarized[i,j] = 1
    unique_rows, counts = np.unique(binarized, axis=0, return_counts=True)
    unique_rows = unique_rows[counts>BIN_MIN]

    #unique_rows = np.vstack({tuple(row) for row in binarized})
    num_clusters = unique_rows.shape[0]
    print(num_clusters)
    if num_clusters>5000:
        print("Too many clusters to go through...")
        return None, None
    
    num_clusters = 0
    rows_clustered = 0
    new_labels = -1*np.ones(labels.shape)
    for i,row in enumerate(unique_rows):
        if i and i%100==0:
            print(i)
        rows_equal_to_this_code = np.where(np.all(binarized==row, axis=1))[0]

        new_labels[rows_equal_to_this_code] = num_clusters
        num_clusters += 1
        rows_clustered += rows_equal_to_this_code.shape[0]

    print("---- Num clusters: {} ---- Pct clustered: {:.3f} ----".format(num_clusters, 1.*rows_clustered/new_labels.shape[0]))

    if return_clusters:
        return num_clusters, new_labels 

    new_labels_vals, new_labels_counts = np.unique(new_labels, return_counts=True)
    fig, ax = plt.subplots(1,1)
    ax.bar(range(len(new_labels_counts)), new_labels_counts.tolist())#list(reversed(sorted(new_labels_counts.tolist()))))
    ax.set_xlabel('Cluster')
    ax.set_ylabel('Count')
    ax.set_title('Test Set Cluster Counts')
    fig.savefig(os.path.join(args.save_folder, 'cluster_counts'))


    if return_clusters:
        return num_clusters, new_labels 
    return num_clusters     

def calculate_randinds(labels1, labels2):

    return adjusted_rand_score(labels1, labels2)

def calculate_confusion_matrix(true_labels, clusters):
    true_labels = true_labels[clusters!=-1]
    clusters = clusters[clusters!=-1]

    table = pd.crosstab(true_labels, clusters)

    return table

def decode_cluster_means(args, sess, loader, cluster_labels):
    embedded, labels = get_layer(sess, loader, 'layer_embedding_activation:0', 'test')

    df = pd.DataFrame(embedded)
    df['cluster'] = cluster_labels
    grouped = df.groupby('cluster')
    applied = grouped.apply(lambda x: x.mean())

    e = tbn('ph_embedding:0')
    o = tbn('ph_embedding_decoded:0')

    fig, ax = plt.subplots(1,1)
    all_clusts = []
    del applied['cluster']
    for i,clust in applied.iterrows():
        clust = clust.values.reshape((1,-1))
        [decoded] = sess.run([o], feed_dict={e:clust, tbn('is_training:0'):False})
        all_clusts.append(decoded.reshape((1,-1)))

    all_clusts = np.concatenate(all_clusts, axis=0)
    g = math.ceil(math.sqrt(all_clusts.shape[0]))
    # all_clusts = all_clusts[:,:g**2]
    print(all_clusts.shape)

    show_result(args, all_clusts, 'cluster_means_decoded.png', grid_size=(g,g))

def confusion_matrix(args, sess, loader, layer, clusters):
    input_layer, labels = get_layer(sess, loader, 'x:0')


    table = calculate_confusion_matrix(labels, clusters)
    table = table / table.sum(axis=0)
    # table = table.transpose()
    # table['amax'] = table.idxmax(axis=1)
    # table = table.sort_values('amax')
    # del table['amax']
    # table = table.transpose()
    fig, ax = plt.subplots(1,1)
    ax.imshow(table.as_matrix(), cmap='jet')
    fig.savefig(args.save_folder + '/confusion_matrix_{}.png'.format(layer))

def channel_by_cluster(args, sess, loader, layer, cols, ax=None, savefile=None, zscore=False):
    x, labels = get_layer(sess, loader, 'x:0')
    count, clusters = count_clusters(args, sess, loader, layer, thresh=0, return_clusters=True)

    x = x[clusters!=-1,:]
    clusters = clusters[clusters!=-1]

    df = pd.DataFrame(x)
    print(df.shape)
    df['cluster'] = clusters

    grouped = df.groupby('cluster')
    means = grouped.apply(lambda x: x.mean(axis=0))
    del means['cluster']

    if zscore:
        means = (means - means.mean(axis=0)) / (means.std(axis=0)**2)

    normalizer = colors.Normalize(means.min().min(), means.max().max())
    means = normalizer(means)
    means = means.data

    if not ax:
        fig, ax = plt.subplots(1,1, figsize=(30,30))
        fig.subplots_adjust(left=.1, bottom=.1, right=1, top=1)
    ax.imshow(means.transpose(), cmap='bwr')
    ax.set_yticks(range(len(cols)))
    ax.set_yticklabels(cols)
    ax.set_xticks([])
    ax.set_ylabel('Marker')
    ax.set_xlabel('Cluster')
    if savefile:
        fig.savefig(args.save_folder+savefile)

def calculate_mmd(k1, k2, k12):
    k1 = np.triu(k1)
    k2 = np.triu(k2)
    k12 = np.triu(k12)

    return k1.sum()/(k1.shape[0]*k1.shape[1]) + k2.sum()/(k2.shape[0]*k2.shape[1]) - 2*k12.sum()/(k12.shape[0]*k12.shape[1])























