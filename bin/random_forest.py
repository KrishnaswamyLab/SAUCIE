import argparse
import pickle
import os
import numpy as np
from polyaxon.tracking import Run
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score

from my_project.data import load_data

# Polyaxon
experiment = Run()

def model(X, y, n_estimators, max_features, min_samples_leaf):
    classifier = RandomForestClassifier(n_estimators=n_estimators,
                                        max_features=max_features,
                                        min_samples_leaf=min_samples_leaf)
    return cross_val_score(classifier, X, y, cv=5), classifier


parser = argparse.ArgumentParser()
parser.add_argument('--n_estimators', type=int, default=3)
parser.add_argument('--max_features', type=int, default=3)
parser.add_argument('--min_samples_leaf', type=int, default=80)
args = parser.parse_args()

(X, y) = load_data()

# Polyaxon
# https://polyaxon.com/docs/experimentation/tracking/module/#log_data_ref

experiment.log_data_ref('dataset_X', content=X)
experiment.log_data_ref('dataset_y', content=y)

accuracies, classifier = model(X=X,
                               y=y,
                               n_estimators=args.n_estimators,
                               max_features=args.max_features,
                               min_samples_leaf=args.min_samples_leaf)

accuracy_mean, accuracy_std = (np.mean(accuracies), np.std(accuracies))
values, counts = np.histogram(accuracies)

# Polyaxon

experiment.log_metrics(accuracy_mean=accuracy_mean,
                       accuracy_std=accuracy_std)
for step in range(accuracies.size):
    experiment.log_metrics(accuracy=accuracies[step], step=step)

outpath = os.path.join(experiment.get_outputs_path(), 'model.pkl')
with(open(outpath, 'wb')) as outfile:
    pickle.dump(classifier, outfile)

experiment.log_model(
    outpath,
    name='top cross validation model',
    framework='sklearn'
)
