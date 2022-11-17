# import pickle

import numpy as np
# import tensorflow as tf
from sklearn.datasets import make_blobs
from sklearn.base import clone

from saucie.wrappers import SAUCIE_batches, SAUCIE_labels


def data_saucie():
    data = make_blobs(
        n_samples=10000,
        n_features=20,
        centers=2,
        random_state=42,
    )[0]
    data = data - np.min(data)
    return data


def data_batches():
    data = np.random.randint(0, 3, 10000)
    return data

# def test_SAUCIE_compresses_data():
#     assert 0 == 1


# def test_SAUCIE_batches_compresses_data():
#     assert 0 == 1


# def test_SAUCIE_yields_stable_results_without_training():
#     assert 0 == 1


# def test_SAUCIE_yields_stable_results_with_training():
#     assert 0 == 1


# def test_SAUCIE_batches_yields_stable_results_without_training():
#     assert 0 == 1


# def test_SAUCIE_batches_yields_stable_results_with_training():
#     assert 0 == 1


def test_SAUCIE_is_clonable():
    data = data_saucie()
    saucie = SAUCIE_labels(epochs=3, lr=1e-6, normalize=True, random_state=42)
    saucie.fit(data)
    labels1 = saucie.predict(data)
    encoded1 = saucie.transform(data)

    saucie1 = clone(saucie)
    saucie1.fit(data)
    labels2 = saucie1.predict(data)
    encoded2 = saucie1.transform(data)
    np.testing.assert_array_equal(labels1, labels2)
    np.testing.assert_array_equal(encoded1, encoded2)


def test_SAUCIE_batches_is_clonable():
    data = data_saucie()
    batches = data_batches()
    saucie = SAUCIE_batches(epochs=2, lr=1e-9, normalize=True, random_state=42)
    saucie.fit(data, batches)
    cleaned1 = saucie.transform(data, batches)

    saucie1 = clone(saucie)
    saucie1.fit(data, batches)
    cleaned2 = saucie1.transform(data, batches)

    np.testing.assert_array_equal(cleaned1, cleaned2)


# def test_SAUCIE_batches_yields_stable_results_batches_order():
#     assert 0 == 1


# def test_SAUCIE_labels_data():
#     assert 0 == 1


# def test_SAUCIE_batches_preserves_ref_batch():
#     assert 0 == 1


# def test_SAUCIE_is_picklable():
#     assert 0 == 1


# def test_SAUCIE_batches_is_picklable():
#     assert 0 == 1


# def test_SAUCIE_pickling_restores_tf_graph():
#     assert 0 == 1


# def test_SAUCIE_batches_pickling_restores_tf_graph():
#     assert 0 == 1
