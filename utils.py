import sys, os, time, math, argparse, contextlib
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from skimage.io import imsave
from sklearn.manifold import TSNE
import seaborn as sns
import phenograph
from loader import Loader, Loader_cytof_emt
from sklearn.metrics import adjusted_rand_score, silhouette_score, normalized_mutual_info_score
from scipy.spatial.distance import pdist, squareform
import matplotlib.cm as cm
from skimage.io import imsave
from matplotlib import offsetbox
# sns.set_style('darkgrid')
# sns.set_palette('muted')
# sns.set_context("notebook", font_scale=1.5,
#                 rc={"lines.linewidth": 2.5})

def plot_mnist(args, X, y, X_embedded, name, min_dist=15.):
        fig, ax = plt.subplots(1,1, figsize=(10,10), frameon=False)
        ax.set_xticks([])
        ax.set_yticks([])
        plt.subplots_adjust(left=0.0, bottom=0.0, right=1.0, top=0.9, wspace=0.0, hspace=0.0)
        ax.scatter(X_embedded[:, 0], X_embedded[:, 1], c=y)

        if min_dist is not None:
                for i in range(10):
                        for d in range(10):
                                shown_images = np.array([[15., 15.]])
                                # indices = np.arange(X_embedded.shape[0])
                                # np.random.shuffle(indices)
                                # for i in indices[:5000]:
                                dist = np.sum((X_embedded[y==d][i] - shown_images) ** 2, 1)
                                if np.min(dist) < min_dist:
                                        continue
                                shown_images = np.r_[shown_images, [X_embedded[y==d][i]]]
                                imagebox = offsetbox.AnnotationBbox( 
                                        offsetbox.OffsetImage(X[y==d][i].reshape(28, 28), cmap=cm.gray_r), X_embedded[y==d][i]
                                        )
                                ax.add_artist(imagebox)
        fig.savefig(args.save_folder+'/embed_w_images'+name)



class SilentFile(object):
    def write(self, x): pass
    def flush(self): pass

@contextlib.contextmanager
def silence():
    save_stdout = sys.stdout
    sys.stdout = SilentFile()
    yield
    sys.stdout = save_stdout


def show_result(batch_res, fname, grid_size=(8, 8), grid_pad=5):
        img_height = 28
        img_width = 28
        batch_res = batch_res.reshape((batch_res.shape[0], img_height, img_width))
        img_h, img_w = batch_res.shape[1], batch_res.shape[2]
        grid_h = img_h * grid_size[0] + grid_pad * (grid_size[0] - 1)
        grid_w = img_w * grid_size[1] + grid_pad * (grid_size[1] - 1)
        img_grid = np.zeros((grid_h, grid_w), dtype=np.uint8)
        for i, res in enumerate(batch_res):
                if i >= grid_size[0] * grid_size[1]:
                        break
                img = (res) * 255
                img = img.astype(np.uint8)
                row = (i // grid_size[0]) * (img_h + grid_pad)
                col = (i % grid_size[1]) * (img_w + grid_pad)
                img_grid[row:row + img_h, col:col + img_w] = img
        imsave(fname, img_grid)

def lrelu(x, leak=0.2, name="lrelu"):

  return tf.maximum(x, leak*x)

def get_loader(args):
        if args.data == 'MNIST':
                loader = Loader(args)
        elif args.data == 'cytof_emt':
                loader = Loader_cytof_emt(args)
        return loader

def tbn(name):

        return tf.get_default_graph().get_tensor_by_name(name)

def obn(name):

        return tf.get_default_graph().get_operation_by_name(name)

def to_one_hot(y, n):
        h = np.zeros((y.shape[0], n))
        h[np.arange(y.shape[0]), y] = 1
        return h

def get_layer(sess, loader, name, test_or_train='test'):
        tensor = tbn(name)
        layer = []
        labels = []
        for batch, batch_labels in loader.iter_batches(test_or_train):
                
                feed = {tbn('x:0'):batch}
                [act] = sess.run([tensor], feed_dict=feed)

                layer.append(act)
                labels.append(batch_labels)

        layer = np.concatenate(layer, axis=0)
        labels = np.concatenate(labels, axis=0)
        return layer, labels

def plot(args, data, labels, title, fn):
        fig, ax = plt.subplots(1,1)
        ax.set_title(title)

        colors = [plt.cm.jet(float(i)/len(np.unique(labels))) for i in range(len(np.unique(labels)))]
        for index,lab in enumerate(np.unique(labels)):
                inds = [True if l==lab else False for l in labels]
                tmp_data = data[inds,:]

                ax.scatter(tmp_data[:,0], tmp_data[:,1], c=colors[int(index)], alpha=.5, s=12, marker='${}$'.format(index), label=int(lab))

        lgnd = plt.legend(scatterpoints=1, prop={'size':6})
        for lh in lgnd.legendHandles:
                lh._sizes = [30]
                lh.set_alpha(1)

        fig.savefig( os.path.join(args.save_folder,fn), dpi=300)

        plt.close('all')
        print("Plot saved to {}".format(fn))

def activations_heatmap(args, sess, loader, layer):
        all_acts, all_labels = get_layer(sess, loader, 'layer_encoder_{}_activation:0'.format(layer))

        
        nonzero = all_acts.reshape((-1))[all_acts.reshape((-1)) > 0]

        acts_normalized, labels = get_layer(sess, loader, 'normalized_activations_layer_{}:0'.format(layer))

        binarized = np.where(acts_normalized>.5, 1, 0)
        normalized_nonzero = acts_normalized.reshape((-1))[acts_normalized.reshape((-1)) > 0]

        fig, (ax1,ax2) = plt.subplots(1,2, figsize=(20,10))
        ax1.set_title('Nonzero first layer activations (entropy reg)')
        ax1.set_xlabel('Activation')
        ax1.set_ylabel('Count')
        ax1.hist(nonzero, bins=100)
        ax2.set_title('Nonzero first layer activations (entropy reg) normalized')
        ax2.hist(normalized_nonzero, bins=100)
        fig.savefig(os.path.join(args.save_folder, 'layer_{}_activations_histogram'.format(layer)))

        

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
        # print(all_argmaxes)
        plt.close('all')
        print("Activations heatmap saved.")

def calculate_loss(sess, loader, train_or_test='test'):
        loss_tensor = tbn('loss:0')
        x_tensor = tbn('x:0')
        y_tensor = tbn('y:0')
        losses = []
        for batch, batch_labels in loader.iter_batches(train_or_test):
                feed = {x_tensor:batch,
                                y_tensor:batch}
                [l] = sess.run([loss_tensor], feed_dict=feed)
                losses.append(l)

        avg_loss = sum(losses) / float(len(losses))
        return avg_loss

def count_clusters(args, sess, loader, layer, thresh=.5, return_clusters=False):
        '''Counts the number of clusters after binarizing the activations of the given layer.'''
        acts, labels = get_layer(sess, loader, 'normalized_activations_layer_{}:0'.format(layer))
        # print(acts.argmax(axis=1)[:5])
        # print(acts.max(axis=1)[:5])
        binarized = np.where(acts>thresh, 1, 0)
        unique_rows = np.vstack({tuple(row) for row in binarized})
        num_clusters = unique_rows.shape[0]

        new_labels = np.zeros(labels.shape)

        for i,row in enumerate(unique_rows):
                # if i>5: break
                rows_equal_to_this_code = np.where(np.all(binarized==row, axis=1))[0]
                new_labels[rows_equal_to_this_code] = i
                labels_code = labels[rows_equal_to_this_code]
                unique, counts = np.unique(labels_code, return_counts=True)
                # print(np.array([unique,counts]).T)

        acts, _ = get_layer(sess, loader, 'layer_embedding_activation:0')
        plot(args, acts, new_labels, 'embedding by cluster', 'embedding_by_cluster_{}'.format(layer))

        if return_clusters:
                return num_clusters, new_labels 
        return num_clusters             

def calculate_randinds(labels1, labels2):

        return adjusted_rand_score(labels1, labels2)

def calculate_mutual_info(labels1, labels2):

        return normalized_mutual_info_score(labels1, labels2)

def calculate_modularity(x, labels, sigma):
        labels = to_one_hot(labels, labels.max()+1)
        pairwise_sq_dists = squareform(pdist(x, 'sqeuclidean'))
        A = np.exp(-pairwise_sq_dists / sigma**2)
        A = A - np.eye(x.shape[0])
        k = A.sum(axis=0)
        M = A.sum()
        B = A - k.reshape((-1,1)).dot(k.reshape((1,-1))) / M
        Q = np.trace(labels.T.dot(B).dot(labels)) / M
        return Q

def calculate_silhouette(x, labels):

        return silhouette_score(x, labels)
