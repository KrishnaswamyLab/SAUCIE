# SAUCIE
An implementation of SAUCIE (Sparse Autoencoder for Clustering, Imputing, and Embedding) in Tensorflow.

## Requirements
All tests performed with:
```
tensorflow 1.4.0
numpy 1.13.3
```

## Running
An example of how to use it for both batch correction and clustering, and optionally specifying the
values of the regularization parameters:
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
