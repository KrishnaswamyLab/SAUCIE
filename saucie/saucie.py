from copy import deepcopy

import numpy as np
from sklearn.base import BaseEstimator, ClusterMixin, TransformerMixin

from saucie.saucie_bn import SAUCIE_BN


class SAUCIE_batches(BaseEstimator, TransformerMixin):
    def __init__(self,
                 lambda_b=0.1,
                 lr=1e-3,
                 epochs=100,
                 layers=[512, 256, 128, 2],
                 batch_size=128,
                 verbose="auto",
                 normalize=False,
                 random_state=None):
        self.lambda_b = lambda_b
        self.lr = lr
        self.epochs = epochs
        self.layers = layers
        self.batch_size = batch_size
        self.verbose = verbose
        self.normalize = normalize
        self.random_state = random_state
        self.history = []

    def fit(self, X, y=None):
        self.layers[3] = 2
        if y is None:
            y = np.zeros(X.shape[0])
        if self.normalize:
            self._min_x = np.min(X, axis=0)
            X = X - self._min_x
            self._max_x = np.max(X, axis=0)
            X = X/self._max_x
            X = np.arcsinh(X)
        self.y_ = y[np.where(y == np.unique(y)[0])]
        self._fit_X = X[np.where(y == np.unique(y)[0])]
        if self.normalize:
            self._fit_X = np.arcsinh(self._fit_X)
        ncol = X.shape[1]
        saucie_bn = SAUCIE_BN(input_dim=ncol,
                              lambda_b=self.lambda_b,
                              layers=self.layers,
                              seed=self.random_state)
        # list and then unpack if too long line
        self.ae_, _, _ = saucie_bn.get_architecture(self.lr)
        return self

    def transform(self, X, y=None):
        ref_batch = np.zeros(self.y_.shape[0])
        if self.normalize:
            X = (X - self._min_x)/self._max_x
            X = np.arcsinh(X)
        if y is None:
            y = np.repeat(self.y_[0]+1, X.shape[0])
        for batch_name in np.unique(y):
            if batch_name == self.y_[0]:
                # leaving reference batch unchanged
                continue
            cur_x = np.append(self._fit_X, X[np.where(y == batch_name)],
                              axis=0)
            nonref_batch = np.ones(y[np.where(y == batch_name)].shape[0])
            cur_y = np.append(ref_batch, nonref_batch,
                              axis=0)
            ae_ = deepcopy(self.ae_)
            self.history += [ae_.fit(x=[cur_x, cur_y],
                                     y=None,
                                     epochs=self.epochs,
                                     batch_size=self.batch_size,
                                     shuffle=True,
                                     verbose=self.verbose
                                     )]
            X_trans = ae_.predict([X[np.where(y == batch_name)],
                                   y[np.where(y == batch_name)]])
            X[np.where(y == batch_name)] = X_trans

        return X


class SAUCIE_labels(BaseEstimator, ClusterMixin, TransformerMixin):
    def __init__(self,
                 lambda_c=0.1,
                 lambda_d=0.2,
                 lr=1e-3,
                 epochs=100,
                 layers=[512, 256, 128, 2],
                 batch_size=128,
                 shuffle="batch",
                 verbose="auto",
                 normalize=False,
                 random_state=None):
        self.lambda_c = lambda_c
        self.lambda_d = lambda_d
        self.lr = lr
        self.epochs = epochs
        self.layers = layers
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.verbose = verbose
        self.normalize = normalize
        self.random_state = random_state

    def fit(self, X, y=None):
        self.layers[3] = 2
        ncol = X.shape[1]
        if self.normalize:
            self._min_x = np.min(X, axis=0)
            X = X - self._min_x
            self._max_x = np.max(X, axis=0)
            X = X/self._max_x
            X = np.arcsinh(X)
        if y is None:
            y = np.zeros(X.shape[0])
        saucie_bn = SAUCIE_BN(input_dim=ncol,
                              lambda_c=self.lambda_c,
                              lambda_d=self.lambda_d,
                              layers=self.layers,
                              seed=self.random_state
                              )
        models = saucie_bn.get_architecture(self.lr)
        self.ae_, self.encoder_, self.classifier_ = models
        self.history = self.ae_.fit(x=[X, y],
                                    y=None,
                                    epochs=self.epochs,
                                    batch_size=self.batch_size,
                                    shuffle=self.shuffle,
                                    verbose=self.verbose
                                    )

        labels = self.classifier_.predict(X)
        self.labels_ = self._decode_labels(labels)
        return self

    def transform(self, X, y=None):
        if self.normalize:
            X = (X - self._min_x)/self._max_x
            X = np.arcsinh(X)
        encoded = self.encoder_.predict(X)
        return encoded

    def _decode_labels(self, labels):
        labels = labels/labels.max()
        binarized = np.where(labels > 1e-6, 1, 0)
        unique_codes = np.unique(binarized, axis=0)
        clusters = np.zeros(labels.shape[0])
        for i, code in enumerate(unique_codes):
            clusters[(binarized == code).all(axis=1).nonzero()] = i
        return clusters

    def predict(self, X, y=None):
        if self.normalize:
            X = (X - self._min_x)/self._max_x
            X = np.arcsinh(X)
        labels = self.classifier_.predict(X)
        labels = self._decode_labels(labels)
        return labels

    def fit_predict(self, X, y=None):
        return self.fit(X).labels_
