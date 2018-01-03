from utils import *
from tensorflow.contrib.layers import batch_norm as bn
from tensorflow.contrib.layers import xavier_initializer

def nameop(op, name):
    """ Gives a name to a tensorflow op
    :param op: a tensorflow op
    :param name: a string name for the op
    """

    op = tf.identity(op, name=name)
    return op

class Layer(object):
    def __init__(self, name, dim_in, dim_out, activation=tf.nn.relu, dropout_p=1., batch_norm=True):
        """
        A helper class representing one layer of the neural network.
        :param name: string name for the layer
        :param dim_in: the dimensionality of the data from the previous layer
        :param dim_out: the dimensionality of this layer
        :param activation: the activation function (usually a nonlinearity) to use after the weights/biases are applied
        :param dropout_p: the probability of keeping a variable if dropout is being used for training
        :param batch_norm: whether this layer should apply batch_norm to its input
        """
        self.W = tf.get_variable(shape=[dim_in,dim_out], initializer=xavier_initializer() , name='W_{}'.format(name))
        self.b = tf.get_variable(shape=[dim_out], initializer=tf.zeros_initializer(), name='b_{}'.format(name))

        self.activation = activation
        self.dropout_p = dropout_p
        self.name = name
        self.batch_norm = batch_norm

    def __call__(self, x, is_training):
        """
        Returns the result of applying this layer to x
        :param x: the tensorflow node input
        :param is_training: whether this is a call during training (when batch_norm mean/variance are updated) or not
        """
        if self.batch_norm:
            x = bn(x, is_training=is_training)

        h = self.activation(tf.matmul(x, self.W)+ self.b)
        h = tf.nn.dropout(h,self.dropout_p)

        h = tf.identity(h, name='{}_activation'.format(self.name))

        tf.add_to_collection('activations', h)
        return h

class SAUCIE(object):
    def __init__(self, input_dim,
        layer_dimensions=[1024, 512, 100, 2],
        lambda_b=0,
        num_batches=2,
        lambda_c=0,
        layer_c=0,
        lambda_d=0,
        batch_norm=False,
        dropout_p=1,
        activation=lrelu,
        loss='mse',
        learning_rate=.001,
        restore_folder='',
        save_folder=''):
        """
        The SAUCIE model.
        :param input_dim: the dimensionality of the data
        :param layer_dimensions: a list of ints for the sizes of all layers up to and encluding the embedding layer. a symmetric
                                 decoder will be created as well
        :param lambda_b: the coefficient for the MMD regularization
        :param num_batches: the number of batches in the data requiring batch correction
        :param lambda_c: the coefficient for the ID regularization
        :param layer_c: the index of layer_dimensions that ID regularization should be applied to (usually len(layer_dimensions)-2)
        :param lambda_d: the coefficient for the intracluster distance regularization
        :param batch_norm: whether to apply batch_norm to the input before layers
        :param dropout_p: whether to apply dropout to the layers while training
        :param activation: the nonlinearity to use in the hidden layers
        :param loss: the loss function to use, one of 'mse' or 'bce'
        :param learning_rate: the learning_rate to use while training
        :param restore_folder: string of the directory where a previous model is saved, if present will return a new Python object
                               with the old SAUCIE tensorflow graph
        :param save_folder: string of the directory to save SAUCIE to by default when save() is called
        """
        if restore_folder:
            self._restore(restore_folder)
            return 

        self.input_dim = input_dim
        self.layer_dimensions = layer_dimensions
        self.lambda_b = lambda_b
        self.num_batches = num_batches
        self.lambda_c = lambda_c
        self.layer_c = layer_c
        self.lambda_d = lambda_d
        self.batch_norm = batch_norm
        self.dropout_p = dropout_p
        self.activation = activation
        self.loss = loss
        self.learning_rate = learning_rate
        self.save_folder = save_folder
        self.iteration = 0

        self.x = tf.placeholder(tf.float32, shape=[None, input_dim], name='x')
        self.y = tf.placeholder(tf.float32, shape=[None, input_dim], name='y')
        self.batches = tf.placeholder(tf.int32, shape=[None], name='batches')
        self.is_training = tf.placeholder(tf.bool, shape=[], name='is_training')
        self.learning_rate_tensor = tf.placeholder(tf.float32, shape=[], name='learning_rate_tensor')

        self._build()
        self.init_session()
        
        self.graph_init(self.sess) 
       
    def init_session(self, limit_gpu_fraction=.3, no_gpu=False):
        """
        Initialize a tensorflow session for SAUCIE.
        :param limit_gpu_fraction: float percentage of the avaiable gpu to use
        :param no_gpu: bool for whether or not to use the gpu if available
        """
        if no_gpu:
            os.environ['CUDA_VISIBLE_DEVICES'] = ''
            config = tf.ConfigProto(device_count = {'GPU': 0})
            self.sess = tf.Session(config=config)
        elif limit_gpu_fraction:
            gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=limit_gpu_fraction)
            config = tf.ConfigProto(gpu_options=gpu_options)
            self.sess = tf.Session(config=config)
        else:
            self.sess = tf.Session()

    def _restore(self, restore_folder):
        """
        Restores the tensorflow graph stored in restore_folder.
        :param restore_folder: the location of the directory where the saved SAUCIE model resides.
        """
        tf.reset_default_graph()
        self.init_session()
        ckpt = tf.train.get_checkpoint_state(restore_folder)
        self.saver = tf.train.import_meta_graph('{}.meta'.format(ckpt.model_checkpoint_path))
        self.saver.restore(self.sess, ckpt.model_checkpoint_path)
        print("Model restored from {}".format(restore_folder))

    def _build(self):
        """
        Coordinates the building of each part of SAUCIE
        """
        self._build_encoder()

        self._build_decoder()

        self._build_losses()

        self._build_optimization()

    def _feedforward_encoder(self, x, is_training):
        """
        Returns the result of inputing x into the encoder.
        :param x: the tensorflow node input
        :param is_training: whether this call is during training or testing
        """
        for i,l in enumerate(self.layers_encoder):
            x = l(x, is_training=is_training)
        return x

    def _feedforward_decoder(self, x, is_training):
        """
        Returns the result of inputing x into the decoder.
        :param x: the tensorflow node input
        :param is_training: whether this call is during training or testing
        """
        for i,l in enumerate(self.layers_decoder):
            x = l(x, is_training=is_training)
        return x

    def _build_encoder(self):
        """
        Build the part of the tensorflow graph for the encoder.
        """
        self.layers_encoder = []
        input_plus_layers = [self.input_dim] + self.layer_dimensions

        for i,layer in enumerate(input_plus_layers[:-2]):
            name  = 'layer_encoder_{}'.format(i)
            layer_act = self.activation
            layer_bn = self.batch_norm
            if i==0:
                layer_bn = False
            if i==self.layer_c:
                layer_act = tf.nn.tanh

            l = Layer(name, input_plus_layers[i], input_plus_layers[i+1], layer_act, self.dropout_p, batch_norm=layer_bn)

            self.layers_encoder.append(l)

        # last layer is linear, and fully-connected
        self.layers_encoder.append(Layer('layer_embedding', input_plus_layers[-2], input_plus_layers[-1], tf.identity, 1., batch_norm=self.batch_norm))

        self.embedded = self._feedforward_encoder(self.x, self.is_training)

    def _build_decoder(self):
        """
        Build the part of the tensorflow graph for the decoder.
        """
        input_plus_layers = [self.input_dim] + self.layer_dimensions
        layers_decoder = input_plus_layers[::-1]
        self.layers_decoder = []

        # first layer is linear, and fully-connected
        for i,layer in enumerate(layers_decoder[:-2]):
            l = Layer('layer_decoder_{}'.format(i), layers_decoder[i], layers_decoder[i+1], self.activation, self.dropout_p, batch_norm=self.batch_norm)

            self.layers_decoder.append(l)

        # last decoder layer is linear and fully-connected
        if self.loss=='mse':
            output_act = tf.identity
        elif self.loss=='bce':
            output_act = tf.nn.sigmoid

        self.layers_decoder.append(Layer('layer_output', layers_decoder[-2], layers_decoder[-1], output_act, 1., batch_norm=False))

        self.reconstructed = self._feedforward_decoder(self.embedded, self.is_training)

    def _build_losses(self):
        """
        Build all the loss ops for the network.
        """
        self.loss_recon = 0.

        if self.lambda_b:
            with tf.variable_scope('reconstruction_mmd'):
                self._build_reconstruction_loss_mmd(self.reconstructed, self.x)
        else:
            with tf.variable_scope('reconstruction'):
                self._build_reconstruction_loss(self.reconstructed, self.x)

        with tf.variable_scope('clustering'):
            self._build_reg_c()

        with tf.variable_scope('batchcorrection'):
            self._build_reg_b()

        self._build_total_loss()

    def _build_optimization(self, norm_clip=5.):
        """
        Build all the optimization ops for the network.
        """
        opt = tf.train.AdamOptimizer(self.learning_rate)
        self.train_op = opt.minimize(self.loss, name='train_op')

    def _build_reconstruction_loss(self, reconstructed, y):
        """
        Build the reconstruction loss part of the network if batch correction isn't being performed.
        :param reconstructed: the tensorflow op that was output by the decoder
        :param y: the tensorflow op for the target
        """
        if self.loss=='mse':
            self.loss_recon = (self.reconstructed - y)**2

        elif self.loss=='bce':
            self.loss_recon = -(y*tf.log(reconstructed+1e-9) + (1-y)*tf.log(1-reconstructed+1e-9))

        self.loss_recon = tf.reduce_mean(self.loss_recon)

        self.loss_recon = nameop(self.loss_recon, 'loss_recon')
        tf.add_to_collection('losses', self.loss_recon)

    def _build_reconstruction_loss_mmd(self, reconstructed, y):
        """
        Build the reconstruction loss part of the network if batch correction is being performed.
        :param reconstructed: the tensorflow op that was output by the decoder
        :param y: the tensorflow op for the target
        """
        x_dists = self.pairwise_dists(y, y)
        x_dists = tf.sqrt(x_dists+1e-3)

        recon_dists = self.pairwise_dists(reconstructed, reconstructed)
        recon_dists = tf.sqrt(recon_dists+1e-3)

        for i in range(self.num_batches):
            recon_ = tf.boolean_mask(reconstructed, tf.equal(self.batches, i))
            y_ = tf.boolean_mask(y, tf.equal(self.batches, i))
            
            # reconstruct the reference batch exactly
            if i==0:
                l = (y_-recon_)**2
                self.loss_recon += tf.reduce_mean(l)

            # reconstruct non-reference batches only to preserve pairwise distances    
            batch_x_rows = tf.boolean_mask(x_dists, tf.equal(self.batches, i))
            batch_x_rowscols = tf.boolean_mask(tf.transpose(batch_x_rows), tf.equal(self.batches, i))

            batch_recon_rows = tf.boolean_mask(recon_dists, tf.equal(self.batches, i))
            batch_recon_rowscols = tf.boolean_mask(tf.transpose(batch_recon_rows), tf.equal(self.batches, i))

            self.loss_recon += tf.reduce_mean((batch_x_rowscols - batch_recon_rowscols)**2)

        self.loss_recon = nameop(self.loss_recon, 'loss_recon')
        tf.add_to_collection('losses', self.loss_recon)

    def _intracluster_distances(self, act):
        """
        Calculate the intracluster distances in the original data given binary-like codes
        :param act: the codes that will be binarized and used to determine cluster assignment
        """
        dists = self.pairwise_dists(self.x, self.x)
        dists = tf.sqrt(dists+1e-3)

        binarized = tf.where(act>0, tf.ones_like(act), tf.zeros_like(act))
        out = self.pairwise_dists(binarized, binarized)
        same_cluster = tf.where(tf.equal(out,tf.zeros_like(out)), tf.ones_like(out), tf.zeros_like(out))

        within_cluster_distances = dists*same_cluster
        within_cluster_distances = tf.reduce_mean(within_cluster_distances)
        return within_cluster_distances

    def _build_reg_c(self):
        """
        Build the tensorflow graph for the ID regularization
        """
        self.loss_c = tf.constant(0.)
        for act in tf.get_collection('activations'):
            if 'encoder_{}'.format(self.layer_c) in act.name:
                # normalize to (0,1)
                act = (act+1)/2
                # sum down neurons
                p = tf.reduce_sum(act, axis=0, keep_dims=True)
                # normalize neuron sums
                normalized = p / tf.reduce_sum(p)

                intracluster_distances = self._intracluster_distances(act)
                self.loss_d = self.lambda_d*intracluster_distances
                self.loss_d = nameop(self.loss_d, 'loss_d')
                tf.add_to_collection('losses', self.loss_d)

                normalized = nameop(normalized, 'normalized_activations_layer_{}'.format(self.layer_c))
                self.loss_c += self.lambda_c*tf.reduce_sum(-normalized*tf.log(normalized+1e-9))

        self.loss_c = nameop(self.loss_c, 'loss_c')
        tf.add_to_collection('losses', self.loss_c)

    def _build_reg_b(self):
        """
        Build the tensorflow graph for the MMD regularization
        """
        var_within = {}
        var_between = {}
        batch_sizes = {}
        self.loss_b = tf.constant(0.)
        if not self.lambda_b:
            return

        K = self.pairwise_dists(self.embedded, self.embedded)
        K = tf.sqrt(K+1e-3)

        # reference batch
        i = 0
        batch1_rows = tf.boolean_mask(K, tf.equal(self.batches, i))
        batch1_rowscols = tf.boolean_mask(tf.transpose(batch1_rows), tf.equal(self.batches, i))

        K_b1 = tf.matrix_band_part(batch1_rowscols, 0, -1) # just upper triangular part
        n_rows_b1 = tf.cast(tf.reduce_sum(tf.boolean_mask(tf.ones_like(self.batches), tf.equal(self.batches, i))), tf.float32)
        K_b1 = tf.reduce_sum(K_b1) / (n_rows_b1**2 + 1)

        var_within[i] = K_b1
        batch_sizes[i] = n_rows_b1

        # nonreference batches
        for j in range(1, self.num_batches):
            batch2_rows = tf.boolean_mask(K, tf.equal(self.batches, j))
            batch2_rowscols = tf.boolean_mask(tf.transpose(batch2_rows), tf.equal(self.batches, j))

            K_b2 = tf.matrix_band_part(batch2_rowscols, 0, -1) # just upper triangular part
            n_rows_b2 = tf.cast(tf.reduce_sum(tf.boolean_mask(tf.ones_like(self.batches), tf.equal(self.batches, j))), tf.float32)
            K_b2 = tf.reduce_sum(K_b2) / (n_rows_b2**2 + 1)

            var_within[j] = K_b2
            batch_sizes[j] = n_rows_b2

            K_12 = tf.boolean_mask(K, tf.equal(self.batches, i))
            K_12 = tf.boolean_mask(tf.transpose(K_12), tf.equal(self.batches, j))
            K_12_ = tf.reduce_sum(tf.transpose(K_12))

            mmd_pair = var_within[i] + var_within[j] - 2 * K_12_ / (batch_sizes[i]*batch_sizes[j]+1)
            self.loss_b += tf.abs(mmd_pair)

        self.loss_b = self.lambda_b*(self.loss_b)
        self.loss_b = nameop(self.loss_b, 'loss_b')
        tf.add_to_collection('losses', self.loss_b)

    def _build_total_loss(self):
        """
        Collect all of the losses together.
        """
        self.loss = tf.constant(0.)
        for l in tf.get_collection('losses'):
            self.loss += l
        self.loss = nameop(self.loss, 'loss')

    def pairwise_dists(self, x1, x2):
        """ Helper function to calculate pairwise distances between tensors x1 and x2 """
        r1 = tf.reduce_sum(x1*x1, 1, keep_dims=True)
        r2 = tf.reduce_sum(x2*x2, 1, keep_dims=True)

        D = r1 - 2*tf.matmul(x1, tf.transpose(x2)) + tf.transpose(r2)

        return D

    def rbf_kernel(self, x, sigma=None):
        """
        Calculates the RBF kernel for distances in x
        :param x: tensor of distances to have kernel applied to
        :param sigma: the constant bandwidth for the RBF kernel, if not provided, use adaptive one based on each point's mean distance
        """
        if not sigma:
            sigma = tf.reduce_mean(x, axis=1, keep_dims=True)
        x = np.e**(- (x / (sigma)))
        return x

    def graph_init(self, sess=None):
        """
        Initialize the tensorflow graph that's been created.
        :param sess: the session to use while initializing, if different from SAUCIE's sess member
        """
        if not sess: sess = self.sess

        self.saver = tf.train.Saver(tf.global_variables(), max_to_keep=1)

        sess.run(tf.global_variables_initializer())

    def save(self, iteration=None, saver=None, sess=None, folder=None):
        """
        Save the current state of SAUCIE.
        :param iteration: the number of training steps SAUCIE has taken, which distinguishes the saved states
                          throughout training
        :param saver: the saver instance to use
        :param sess: the session to save
        :param folder: the location to save SAUCIE's state to
        """
        if not iteration: iteration=self.iteration
        if not saver: saver=self.saver
        if not sess: sess=self.sess
        if not folder: folder=self.save_folder

        savefile = os.path.join(folder, 'SAUCIE')
        saver.save(sess, savefile , global_step=iteration, write_meta_graph=True)
        print("Model saved to {}".format(savefile))

    def get_loss_names(self):
        """ Returns the strings of the loss names in the order they're printed during training"""
        losses = [tns.name[:-2].replace('loss_','').split('/')[-1] for tns in tf.get_collection('losses')]
        return "Losses: {}".format(' '.join(losses))

    def train(self, load, steps, batch_size=256):
        """
        Train SAUCIE.
        :param load: the loader object to yield batches from
        :param steps: the number of steps to train for
        :param batch_size: the number of points to train on in each step
        """
        start = self.iteration
        while (self.iteration - start) < steps:
            self.iteration+=1

            batch = load.next_batch(batch_size=batch_size)

            feed = {tbn('x:0'): batch[0],
                    tbn('y:0'): batch[0],
                    tbn('is_training:0'): True,
                    tbn('learning_rate_tensor:0'): self.learning_rate}
            if len(batch)==2:
                feed[tbn('batches:0')] = batch[1]

            ops = [obn('train_op')]

            _ = self.sess.run(ops, feed_dict=feed)

    def get_loss(self, load):
        """
        Get the current losses over the dataset.
        :param load: the loader object to iterate over
        """
        losses = None

        for i,batch in enumerate(load.iter_batches()):

            feed = {tbn('x:0'): batch[0],
                    tbn('y:0'): batch[0],
                    tbn('is_training:0'): False}
            if len(batch)==2:
                feed[tbn('batches:0')] = batch[1]

            batch_losses = self.sess.run(tf.get_collection('losses'), feed_dict=feed)

            if not losses:
                losses = batch_losses
            else:
                losses = [loss+batch_loss for loss,batch_loss in zip(losses,batch_losses)]
            
        losses = [loss/float(i+1) for loss in losses]
        lstring = ' '.join(['{:.3f}'.format(loss) for loss in losses])
        
        return lstring

    def get_layer(self, load, name):
        """
        Get the actual values in array_like form from an abstract tensor
        :param load: the loader object to iterate over
        :param name: the name of the tensor to evaluate for each point
        """
        tensor_name = "{}:0".format(name)
        tensor = tbn(tensor_name)
        layer = []
        labels = []
        for batch in load.iter_batches():

            feed = {tbn('x:0'): batch[0],
                    tbn('y:0'): batch[0],
                    tbn('is_training:0'): False}
            if len(batch)==2:
                feed[tbn('batches:0')] = batch[1]
                labels.append(batch[1])
            
            [act] = self.sess.run([tensor], feed_dict=feed)
            layer.append(act)

        layer = np.concatenate(layer, axis=0)

        if labels:
            labels = np.concatenate(labels, axis=0)
            return layer, labels
        else:
            return layer

    def get_clusters(self, load, BIN_MIN=10, max_clusters=1000, verbose=True):
        """
        Get cluster assignments from the ID regularization layer
        :param load: the loader object to iterate over
        :param BIN_MIN: points in a cluster of less than this many points will be assigned the unclustered "-1" label
        :param max_clusters: going through the clusters can take a long time, so optionally abort any attempt to go
                             through more than a certain number of clusters
        :param verbose: whether or not to print the results of the clustering
        """
        acts = self.get_layer(load, 'layer_encoder_{}_activation'.format(self.layer_c))
        if isinstance(acts, list) or isinstance(acts, tuple):
            acts = acts[0]

        binarized = np.where(acts>0, 1, 0)

        unique_rows, counts = np.unique(binarized, axis=0, return_counts=True)
        unique_rows = unique_rows[counts>BIN_MIN]

        num_clusters = unique_rows.shape[0]
        if num_clusters>max_clusters:
            print("Too many clusters ({}) to go through...".format(num_clusters))
            return num_clusters, None
        
        num_clusters = 0
        rows_clustered = 0
        clusters = -1*np.ones(acts.shape[0])
        for i,row in enumerate(unique_rows):
            rows_equal_to_this_code = np.where(np.all(binarized==row, axis=1))[0]

            clusters[rows_equal_to_this_code] = num_clusters
            num_clusters += 1
            rows_clustered += rows_equal_to_this_code.shape[0]

        if verbose:
            print("---- Num clusters: {} ---- Percent clustered: {:.3f} ----".format(num_clusters, 1.*rows_clustered/clusters.shape[0]))

        return num_clusters, clusters

