import numpy as np
from saucie_bn import SAUCIE_BN
from sklearn.base import BaseEstimator, ClusterMixin, TransformerMixin


class SAUCIE_batches(BaseEstimator, TransformerMixin):
    def __init__(self,
                 lambda_b=0.1,
                 lr=1e-3,
                 epochs=100,
                 batch_size=128,
                 shuffle="batch",
                 verbose="auto",
                 random_state=None):
        self.lambda_b = lambda_b
        self.lr = lr
        self.epochs = epochs
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.verbose = verbose
        self.random_state = random_state
        self.history = []

    def fit(self, X, y=None):
        if y is None:
            y = np.zeros(X.shape[0])
        self.y_ = y[np.where(y == np.unique(y)[0])]
        self._fit_X = X[np.where(y == np.unique(y)[0])]
        ncol = X.shape[1]
        saucie_bn = SAUCIE_BN(input_dim=ncol,
                              lambda_b=self.lambda_b,
                              seed=self.random_state)
        # list and then unpack if too long line
        self.ae_, _, _ = saucie_bn.get_architecture(self.lr)
        return self

    def transform(self, X, y=None):
        if y is None:
            y = np.repeat(self.y_[0]+1, X.shape[0])
        for batch_name in np.unique(y):
            if batch_name == self.y_[0]:
                # leaving reference batch unchanged
                continue
            cur_x = np.append(self._fit_X, X[np.where(y == batch_name)],
                              axis=0)
            cur_y = np.append(self.y_, y[np.where(y == batch_name)],
                              axis=0)

            self.history += self.ae_.fit(x=[cur_x, cur_y],
                                         y=None,
                                         epochs=self.epochs,
                                         batch_size=self.batch_size,
                                         shuffle=self.shuffle,
                                         verbose=self.verbose
                                         )
            X_trans = self.ae_.predict([X[np.where(y == batch_name)],
                                        y[np.where(y == batch_name)]])
            X[np.where(y == batch_name)] = X_trans

        return X


class SAUCIE_labels(BaseEstimator, ClusterMixin, TransformerMixin):
    def __init__(self,
                 lambda_c=0.1,
                 lambda_d=0.2,
                 lr=1e-3,
                 epochs=100,
                 batch_size=128,
                 shuffle="batch",
                 verbose="auto",
                 random_state=None):
        self.lambda_c = lambda_c
        self.lambda_d = lambda_d
        self.lr = lr
        self.epochs = epochs
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.verbose = verbose
        self.random_state = random_state

    def fit(self, X, y=None):
        ncol = X.shape[1]
        if y is None:
            y = np.zeros(X.shape[0])
        saucie_bn = SAUCIE_BN(input_dim=ncol,
                              lambda_c=self.lambda_c,
                              lambda_d=self.lambda_d,
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
        return self

    def transform(self, X, y=None):
        encoded = self.encoder_.predict(X)
        return encoded

    def predict(self, X, y=None):
        # TODO: change this according to decoding original code
        labels = self.classifier_.predict(X)
        return labels
