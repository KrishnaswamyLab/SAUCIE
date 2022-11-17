import tensorflow as tf
from tensorflow.keras.initializers import GlorotUniform
from tensorflow.keras.layers import Dense, Input, LeakyReLU
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.random import set_seed
from tensorflow.nn import moments

# from utils import calculate_mmd


class SAUCIE_BN(object):
    def __init__(self,
                 input_dim,
                 lambda_b=0,
                 lambda_c=0,
                 layer_c=0,
                 lambda_d=0,
                 layers=[512, 256, 128, 2],
                 seed=None
                 ):
        self.input_dim = input_dim
        self.lambda_b = lambda_b
        self.lambda_c = lambda_c
        self.layer_c = layer_c
        self.lambda_d = lambda_d
        self.layers = layers
        self.seed = seed

    def _build_layers(self):
        """ Add SAUCIE layers and consecutive losses. """
        # ENCODER
        input_shape = (self.input_dim, )
        inputs = Input(shape=input_shape, name='encoder_input')
        batches = Input(shape=(1, ), name="batches_input")

        h1 = Dense(self.layers[0],
                   kernel_initializer=GlorotUniform(seed=self.seed),
                   name='encoder0', use_bias=True)(inputs)
        h1 = LeakyReLU(alpha=0.2, name='encoder0_activation')(h1)

        if not self.lambda_b and not self.lambda_c:
            h2 = Dense(self.layers[1],
                       kernel_initializer=GlorotUniform(seed=self.seed),
                       activation="sigmoid",
                       name='encoder1', use_bias=True)(h1)
        else:
            h2 = Dense(self.layers[1],
                       kernel_initializer=GlorotUniform(seed=self.seed),
                       name='encoder1', use_bias=True)(h1)
            h2 = LeakyReLU(alpha=0.2, name='encoder1_activation')(h2)

        h3 = Dense(self.layers[2],
                   kernel_initializer=GlorotUniform(seed=self.seed),
                   name='encoder2', use_bias=True)(h2)
        h3 = LeakyReLU(alpha=0.2, name='encoder2_activation')(h3)

        embedded = Dense(self.layers[3],
                         kernel_initializer=GlorotUniform(seed=self.seed),
                         name='embeddings', use_bias=True)(h3)
        encoder = Model(inputs, embedded, name='encoder')

        # DECODER
        # latent_inputs = Input(shape=(self.layers[3], ),
        #                       name='latentspace')
        h5 = Dense(self.layers[2],
                   kernel_initializer=GlorotUniform(seed=self.seed),
                   name='decoder0', use_bias=True)(embedded)
        h5 = LeakyReLU(alpha=0.2, name='decoder0_activation')(h5)

        h6 = Dense(self.layers[1],
                   kernel_initializer=GlorotUniform(seed=self.seed),
                   name='decoder1', use_bias=True)(h5)
        h6 = LeakyReLU(alpha=0.2, name='decoder1_activation')(h6)

        if self.lambda_c:
            layer_c = Dense(self.layers[0],
                            kernel_initializer=GlorotUniform(seed=self.seed),
                            activation="relu",
                            name='layer_c', use_bias=True)(h6)
        else:
            layer_c = Dense(self.layers[0],
                            kernel_initializer=GlorotUniform(seed=self.seed),
                            name='layer_c', use_bias=True)(h6)
            layer_c = LeakyReLU(alpha=0.2, name='layer_c')(layer_c)

        outputs = Dense(self.input_dim,
                        kernel_initializer=GlorotUniform(seed=self.seed),
                        name='outputs', use_bias=True)(layer_c)

        # get classifier layer
        classifier = Model(inputs, layer_c, name='classifier')

        # combine models
        SAUCIE_BN_model = Model([inputs, batches], outputs, name="SAUCIE_BN")

        # add losses
        if self.lambda_b:
            recon_loss = self._build_reconstruction_loss_mmd(inputs,
                                                             outputs,
                                                             batches)
            SAUCIE_BN_model.add_loss(recon_loss)
            mmd_loss = self._build_reg_b(embedded, batches)
            SAUCIE_BN_model.add_loss(mmd_loss)
            SAUCIE_BN_model.add_metric(mmd_loss, name='mmd_loss',
                                       aggregation='mean')
        else:
            recon_loss = self._build_reconstruction_loss(inputs, outputs)
            SAUCIE_BN_model.add_loss(recon_loss)
            SAUCIE_BN_model.add_metric(recon_loss, name='recon_loss',
                                       aggregation='mean')

        if self.lambda_c:
            id_reg_loss = self._build_reg_c(layer_c)
            SAUCIE_BN_model.add_loss(id_reg_loss)
            SAUCIE_BN_model.add_metric(id_reg_loss, name='id_reg_loss',
                                       aggregation='mean')
        if self.lambda_d:
            intracluster_loss = self._build_reg_d(inputs, layer_c)
            SAUCIE_BN_model.add_loss(intracluster_loss)
            SAUCIE_BN_model.add_metric(intracluster_loss, name='intra_loss',
                                       aggregation='mean')            

        return SAUCIE_BN_model, encoder, classifier

    def _build_reconstruction_loss(self, input, reconstructed):
        """
        Build the  classical autoencoder reconstruction loss.

        :param reconstructed: the tensor that was output by the decoder
        :param input: the tensor that was the input of the encoder
        """
        loss_recon = tf.reduce_mean((reconstructed - input)**2)
        return loss_recon

    def _normalize_dist(samples):
        """
        Normalize the given samples.

        :param samples: the tensor of samples to be normalized
        """
        u, var = moments(samples, 0)
        dist = (samples - u)/(tf.sqrt(var+1e-6)+1e-6)
        return dist

    def _build_reconstruction_loss_mmd(self, input, reconstructed, batches):
        """
        Build the reconstruction loss if batch correction is being performed.

        :param reconstructed: the tensor that was output by the decoder
        :param input: the tensor that was the input of the encoder
        :param batches: the tensor of batch labels of the data
        """
        # reference batch and normal autoencoder loss
        ref_el = tf.equal(batches, 0)
        ref_recon = tf.boolean_mask(reconstructed, ref_el)
        ref_input = tf.boolean_mask(input, ref_el)
        ref_loss = self._build_reconstruction_loss(ref_input, ref_recon)

        # non-reference batch
        nonrefel = tf.equal(batches, 1)
        nonrefrecon = tf.boolean_mask(reconstructed, nonrefel)
        nonrefin = tf.boolean_mask(input, nonrefel)
        nonrefrecon_dist = self._normalize_dist(nonrefrecon)
        nonrefin_dist = self._normalize_dist(nonrefin)
        nonref_loss = self._build_reconstruction_loss(nonrefrecon_dist,
                                                      nonrefin_dist)

        return ref_loss + 0.1*nonref_loss

    def _calculate_batch_var(self, dists, batches, batch):
        batch_rows = tf.boolean_mask(dists, tf.equal(batches, batch))
        K_b = tf.boolean_mask(tf.transpose(batch_rows),
                              tf.equal(batches, batch))

        nrows_batch = tf.reduce_sum(tf.boolean_mask(tf.ones_like(batches),
                                    tf.equal(batches, batches)))
        nrows_batch = tf.cast(nrows_batch, tf.float32)

        K_b = tf.reduce_sum(K_b) / (nrows_batch**2)

        return K_b, nrows_batch

    def _build_reg_b(self, embedded, batches):
        """
        Build the loss for the MMD regularization.

        :param embedded: the latent space embedding of the autoencoder
        """
        embedded = embedded/tf.reduce_mean(embedded)
        K = self._pairwise_dists(embedded, embedded)
        K = K/tf.reduce_max()
        K = self._gaussian_kernel_matrix(K)

        # reference batch
        var_in_ref, bsize_ref = self._calculate_batch_var(K, batches, 0)
        # nonreference batch
        var_in_nonref, bsize_nonref = self._calculate_batch_var(K, batches, 1)
        # between batches
        K_12 = tf.boolean_mask(K, tf.equal(batches, 0))
        K_12 = tf.boolean_mask(tf.transpose(K_12), tf.equal(batches, 1))
        K_12_ = tf.reduce_sum(tf.transpose(K_12))
        k_12_coeff = K_12_ / (bsize_ref * bsize_nonref)

        loss_b = var_in_ref + var_in_nonref - 2 * k_12_coeff
        loss_b = self.lambda_b * tf.abs(loss_b)
        return loss_b

    def _build_reg_c(self, codes):
        """
        Build loss for the ID regularization.

        :param codes: the codes that will be binarized
                      and used to determine cluster assignment
        """
        # sum down neurons
        neuron_sums = tf.math.reduce_sum(codes, axis=0, keepdims=True)
        # normalize neuron sums
        normalized_sums = neuron_sums/tf.math.reduce_sum(neuron_sums)

        log_norm = tf.math.log(normalized_sums + 1e-9)
        loss_c = self.lambda_c*tf.math.reduce_sum(-normalized_sums*log_norm)
        return loss_c

    def _build_reg_d(self, input, codes):
        """
        Calculate the intracluster distances in the original data
        given binary-like codes.

        :param input: the tensor that was the input of the encoder
        :param codes: the codes that will be binarized
        and used to determine cluster assignment
        """
        out = self._pairwise_dists(codes, codes)
        same_cluster = self._gaussian_kernel_matrix(out)
        same_cluster = same_cluster - tf.reduce_min(same_cluster)
        same_cluster = same_cluster / tf.reduce_max(same_cluster)

        dists = self._pairwise_dists(input, input)
        dists = tf.sqrt(dists + 1e-3)

        intracluster_distances = dists * same_cluster
        intracluster_distances = tf.reduce_mean(intracluster_distances)
        intra_loss = self.lambda_d * intracluster_distances
        return intra_loss

    def _pairwise_dists(self, x1, x2):
        """Helper function to calculate pairwise distances
        between tensors x1 and x2."""
        r1 = tf.reduce_sum(x1 * x1, 1, keepdims=True)
        r2 = tf.reduce_sum(x2 * x2, 1, keepdims=True)

        D = r1 - 2 * tf.matmul(x1, tf.transpose(x2)) + tf.transpose(r2)

        return D

    def _gaussian_kernel_matrix(self, dist):
        """Multi-scale RBF kernel."""
        sigmas = [1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1,
                  5, 10, 15, 20, 25, 30, 35, 100, 1e3,
                  1e4, 1e5, 1e6]

        beta = 1. / (2. * (tf.expand_dims(sigmas, 1)))

        s = tf.matmul(beta, tf.reshape(dist, (1, -1)))

        return tf.reshape(tf.reduce_sum(tf.exp(-s), 0),
                          tf.shape(dist)) / len(sigmas)

    def get_architecture(self, lr):
        """
        Get the SAUCIE architecture
        and compile the model with a given learning rate.

        :param lr: learning rate to compile the model with
        """
        set_seed(self.seed)
        SAUCIE_BN_model, encoder, classifier = self._build_layers()
        # optimizer will take care of all loses itself
        SAUCIE_BN_model.compile(optimizer=Adam(learning_rate=lr))

        return SAUCIE_BN_model, encoder, classifier
