# SAUCIE
An implementation of SAUCIE (Sparse Autoencoder for Clustering, Imputing, and Embedding) in Tensorflow.

## Requirements
All tests performed with:
```
tensorflow 1.4.0
numpy 1.13.3
```

## Installation

Download with

```
git clone https://github.com/KrishnaswamyLab/SAUCIE
```

install requirements with

```
pip install -r SAUCIE/requirements.txt
```

and then add SAUCIE to your Python path (e.g. by running Python in the same directory in which you ran `git clone`, or by adding that directory with `sys.path.append("/path/to/git/")`.


## Usage
SAUCIE is a python object that loads data from a numpy matrix and produces numpy matrix output for the reconstruction, visualization, and/or clusters. Standard usage is to train a model from a numpy matrix and get the embedding, reconstruction, or clusters for that data. This can be done with:
```
data = ...

import SAUCIE

saucie = SAUCIE.SAUCIE(data.shape[1])
loadtrain = SAUCIE.Loader(data, shuffle=True)
saucie.train(loadtrain, steps=1000)

loadeval = SAUCIE.Loader(data, shuffle=False)
embedding = saucie.get_embedding(loadeval)
number_of_clusters, clusters = saucie.get_clusters(loadeval)
reconstruction = saucie.get_reconstruction(loadeval)

... work with numpy results as desired ...
```

## Example
See `scripts/example.py` for an example of running SAUCIE on data. You can also see a tutorial on Google Colab [here](https://colab.research.google.com/github/KrishnaswamyLab/SingleCellWorkshop/blob/master/exercises/Deep_Learning/notebooks/02_Answers_Exploratory_analysis_of_single_cell_data_with_SAUCIE.ipynb).

## Running
SAUCIE also comes with the option of running a full cohort of samples if the data is prepared in a specific way under `scripts/SAUCIE.py`. Namely, for a folder of CSV (or FCS files if the flag --fcs is provided), an example of how to use SAUCIE for both batch correction and clustering is:
```
python SAUCIE.py --input_dir path/to/input/files
                 --output_dir path/for/output/files
                 --batch_correct
                 --cluster
                 [--lambda_b .1]
                 [--lambda_c .1]
                 [--lambda_d .2];
```
The input directory must contain the CSV (or FCS if you specify --fcs) you wish to run SAUCIE on. If you do not want to run SAUCIE on all columns in the input file, a file named cols_to_use.txt with the 0-indexed column numbers, one per line can be provided.

In the output directory, if batch correction was done, there will be a folder ```batch_corrected``` with a batch-corrected CSV (or FCS) file corresponding to each original file. If clustering was done, there will also be a folder ```clustered``` with a clustered file corresponding to each original file. In each clustered file, there is either the original or batch-corrected data with additional columns giving the cluster number and the X and Y coordinate for the visualization.
