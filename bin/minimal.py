import pickle
import os
from sklearn.ensemble import RandomForestClassifier

from my_project.data import load_data


(X, y) = load_data()

classifier = RandomForestClassifier(
    n_estimators=3, max_features=3, min_samples_leaf=80).fit(X, y)

output_root = os.environ['POLYAXON_RUN_OUTPUTS_PATH']
outpath = os.path.join(output_root, 'model.pkl')

with(open(outpath, 'wb')) as outfile:
    pickle.dump(classifier, outfile)
