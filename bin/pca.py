import argparse
import os
import pickle

import numpy as np
from polyaxon.tracking import Run
from sklearn.decomposition import PCA

# Polyaxon
experiment = Run()


def path(file):
    if os.path.exists(file) and os.path.isfile(file):
        return file
    raise ValueError(f"File not found or not a file: {file}")


def model(X):
    transformer = PCA(2).fit(X)
    results = transformer.transform(X)
    return transformer, results


def load_data(fname):
    if fname.lower().endswith(".npy"):
        data = np.load()
        return data
    raise ValueError(f'Unsupported file format: {fname}')


parser = argparse.ArgumentParser()
parser.add_argument('--data', type=path, default="")
parser.add_argument('--true_labels', type=path, default="")
args = parser.parse_args()

X = load_data(args.data)
y = load_data(args.true_labels)

# Polyaxon
# https://polyaxon.com/docs/experimentation/tracking/module/#log_data_ref

experiment.log_data_ref('dataset_X', content=X)
experiment.log_data_ref('dataset_y', content=y)

transformer, results = model(X=X)

outpath = os.path.join(experiment.get_outputs_path(), 'model.pkl')
with (open(outpath, 'wb')) as outfile:
    pickle.dump(transformer, outfile)

result_path = os.path.join(experiment.get_outputs_path(), 'pca.csv')
with (open(result_path, 'wb')) as outfile:
    np.savetxt(outfile, results, delimiter=",")

experiment.log_model(
    outpath,
    name='PCA model',
    framework='sklearn'
)
