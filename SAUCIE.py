import glob
import os
import argparse
import pickle
import fcsparser
import fcswrite
import numpy as np
import pandas as pd
import tensorflow as tf
from model import SAUCIE
from loader import Loader
import shutil
from utils import asinh, sinh
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

def cluster_done():
    """Return True if clustering has already been done."""
    if os.path.exists(os.path.join(args.output_dir, 'clustered')):
        numfiles = len(glob.glob(os.path.join(args.output_dir, 'clustered', '*')))
        numfiles_total = len(glob.glob(os.path.join(args.input_dir, '*.fcs')))
        if numfiles == numfiles_total:
            return True

    return False

def cluster_training_done():
    """Return True if cluster model has already been trained."""
    if os.path.exists(os.path.join(args.output_dir, 'models', 'clustered')):
        numfiles = len(glob.glob(os.path.join(args.output_dir, 'models', 'clustered', '*')))
        if numfiles > 0:
            return True

    return False

def batch_correction_training_done():
    """Return True if batch correction models have already been trained."""
    if os.path.exists(os.path.join(args.output_dir, 'models', 'batch_corrected')):
        numfiles = len(glob.glob(os.path.join(args.output_dir, 'models', 'batch_corrected', '*')))
        numfiles_total = len(glob.glob(os.path.join(args.input_dir, '*.fcs')))
        # -1 because there is 1 reference file
        print("Found {} batch-corrected models (out of {} total models)".format(numfiles, numfiles_total - 1))
        if numfiles == numfiles_total - 1:
            return True

    return False

def batch_correction_done():
    """Return True if batch correction has already been performed."""
    if os.path.exists(os.path.join(args.output_dir, 'batch_corrected')):
        numfiles = len(glob.glob(os.path.join(args.output_dir, 'batch_corrected', '*')))
        numfiles_total = len(glob.glob(os.path.join(args.input_dir, '*.fcs')))
        print("Found {} batch-corrected files (out of {} total files)".format(numfiles, numfiles_total))
        if numfiles == numfiles_total:
            return True

    return False

def get_data(fn, sample=0, return_rawfile=False):
    """Return DataFrame of an FCS file."""
    meta, x = fcsparser.parse(fn)
    if return_rawfile:
        return x

    x = x.iloc[:, args.cols]

    newvals = asinh(x)
    x = pd.DataFrame(newvals, columns=x.columns)

    if sample:
        r = list(range(x.shape[0]))
        np.random.shuffle(r)
        r = r[:sample]
        x = x.iloc[r, :]

    return x

def train_batch_correction(rawfiles):
    """Run batch correction on all files."""
    try:
        model_dir = os.path.join(args.output_dir, 'models', 'batch_corrected')
        if os.path.exists(model_dir):
            shutil.rmtree(model_dir)
        os.mkdir(model_dir)
        ref = rawfiles[0]
        refx = get_data(ref)

        print("Starting to train {} batch correction models...".format(len(rawfiles[1:])))
        for counter, nonref in enumerate(rawfiles[1:]):
            nonrefname = os.path.split(nonref)[-1]
            print("Training model {}".format(counter))

            nonrefx = get_data(nonref)
            alldata = np.concatenate([refx.as_matrix(), nonrefx.as_matrix()], axis=0)
            alllabels = np.concatenate([np.zeros(refx.shape[0]), np.ones(nonrefx.shape[0])], axis=0)

            load = Loader(data=alldata, labels=alllabels, shuffle=True)

            tf.reset_default_graph()

            saucie = SAUCIE(input_dim=refx.shape[1], lambda_b=args.lambda_b)

            for i in range(args.num_iterations):
                saucie.train(load, steps=1000, batch_size=200)

            saucie.save(folder=os.path.join(model_dir, nonrefname))

    except Exception as ex:
        # if it didn't run all the way through, clean everything up and remove it
        shutil.rmtree(model_dir)
        raise(ex)

def output_batch_correction(rawfiles):
    """Use already trained models to output batch corrected data."""
    try:
        model_dir = os.path.join(args.output_dir, 'models', 'batch_corrected')
        data_dir = os.path.join(args.output_dir, 'batch_corrected')
        if os.path.exists(data_dir):
            shutil.rmtree(data_dir)
        os.mkdir(data_dir)
        ref = rawfiles[0]
        refx = get_data(ref)
        refname = os.path.split(ref)[-1]

        print("Starting to output {} batch corrected files...".format(len(rawfiles)))
        for counter, nonref in enumerate(rawfiles[1:]):
            nonrefname = os.path.split(nonref)[-1]
            print("Outputing file {}".format(counter))

            nonrefx = get_data(nonref)
            alldata = np.concatenate([refx.as_matrix(), nonrefx.as_matrix()], axis=0)
            alllabels = np.concatenate([np.zeros(refx.shape[0]), np.ones(nonrefx.shape[0])], axis=0)

            load = Loader(data=alldata, labels=alllabels, shuffle=False)

            tf.reset_default_graph()
            restore_folder = os.path.join(model_dir, nonrefname)
            saucie = SAUCIE(None, restore_folder=restore_folder)

            recon, labels = saucie.get_layer(load, 'output')

            recon = sinh(recon)

            # write out reference file
            if counter == 0:
                reconref = recon[labels == 0]
                rawdata = get_data(ref, return_rawfile=True)
                for ind, c in enumerate(args.cols):
                    rawdata.iloc[:, c] = reconref[:, ind]

                outfileref = os.path.join(data_dir, refname)
                fcswrite.write_fcs(outfileref, rawdata.columns.tolist(), rawdata)

            # write out nonreference file
            reconnonref = recon[labels == 1]
            rawdata = get_data(nonref, return_rawfile=True)
            for ind, c in enumerate(args.cols):
                rawdata.iloc[:, c] = reconnonref[:, ind]
            outfilenonref = os.path.join(data_dir, nonrefname)
            fcswrite.write_fcs(outfilenonref, rawdata.columns.tolist(), rawdata)

    except Exception as ex:
        # if it didn't run all the way through, clean everything up and remove it
        shutil.rmtree(data_dir)
        raise(ex)

def train_cluster(inputfiles):
    """Run clustering on all files."""
    try:
        model_dir = os.path.join(args.output_dir, 'models', 'clustered')
        if os.path.exists(model_dir):
            shutil.rmtree(model_dir)
        os.mkdir(model_dir)

        tf.reset_default_graph()
        x = get_data(inputfiles[0], sample=2)
        saucie = SAUCIE(input_dim=x.shape[1], lambda_d=args.lambda_d, lambda_c=args.lambda_c)

        for i in range(args.num_iterations):
            alldata = []
            for f in inputfiles:
                x = get_data(f, sample=args.num_points_sample)
                alldata.append(x)
            alldata = np.concatenate(alldata, axis=0)

            load = Loader(data=alldata, shuffle=True)

            saucie.train(load, steps=1000, batch_size=400)

        saucie.save(folder=model_dir)

    except Exception as ex:
        # if it didn't run all the way through, clean everything up and remove it
        shutil.rmtree(model_dir)
        raise(ex)

def output_cluster(inputfiles):
    """Use already trained model to output clustered data."""
    try:
        model_dir = os.path.join(args.output_dir, 'models', 'clustered')
        data_dir = os.path.join(args.output_dir, 'clustered')
        if os.path.exists(data_dir):
            shutil.rmtree(data_dir)
        os.mkdir(data_dir)

        tf.reset_default_graph()
        saucie = SAUCIE(None, restore_folder=model_dir)

        print("Finding all binary codes")
        all_codes = {}
        for counter, f in enumerate(inputfiles):
            x = get_data(f)
            load = Loader(data=x, shuffle=False)

            acts = saucie.get_layer(load, 'layer_c')
            acts = acts / acts.max()
            binarized = np.where(acts > .000001, 1, 0)

            unique_rows, counts = np.unique(binarized, axis=0, return_counts=True)
            for unique_row in unique_rows:
                unique_row = tuple(unique_row.tolist())
                if unique_row not in all_codes:
                    all_codes[unique_row] = len(all_codes)

        print("Found {} clusters".format(len(all_codes)))

        print("Starting to output {} clustered files...".format(len(inputfiles)))
        for counter, f in enumerate(inputfiles):
            fname = os.path.split(f)[-1]
            print("Outputing file {}".format(counter))
            x = get_data(f)
            load = Loader(data=x, shuffle=False)
            acts = saucie.get_layer(load, 'layer_c')
            acts = acts / acts.max()
            binarized = np.where(acts > .000001, 1, 0)

            clusters = -1 * np.ones(x.shape[0])
            for code in all_codes:
                rows_equal_to_this_code = np.where(np.all(binarized == code, axis=1))[0]
                clusters[rows_equal_to_this_code] = all_codes[code]

            embeddings = saucie.get_layer(load, 'embeddings')

            rawdata = get_data(f, return_rawfile=True)
            outcols = rawdata.columns.tolist() + ['Cluster', 'Embedding_SAUCIE1', 'Embedding_SAUCIE2']
            rawdata = pd.concat([rawdata, pd.DataFrame(clusters), pd.DataFrame(embeddings[:, 0]), pd.DataFrame(embeddings[:, 1])], axis=1)
            outfile = os.path.join(data_dir, fname)
            fcswrite.write_fcs(outfile, outcols, rawdata)

    except Exception as ex:
        # if it didn't run all the way through, clean everything up and remove it
        shutil.rmtree(data_dir)
        raise(ex)

def parse_args():
    """Parse args."""
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_dir', type=str, help='directory of input FCS files')
    parser.add_argument('--output_dir', type=str, help='directory to create for output files')
    parser.add_argument('--batch_correct', action='store_true', default=False, help='whether or not to do batch correction on the files')
    parser.add_argument('--cluster', action='store_true', default=False, help='whether or not to do clustering on the files')
    parser.add_argument('--lambda_c', default=.1, type=float, help='if clustering, the value of lambda_c')
    parser.add_argument('--lambda_d', default=.2, type=float, help='if clustering, the value of lambda_d')
    parser.add_argument('--lambda_b', default=.1, type=float, help='if batch correcting, the value of lambda_b')
    parser.add_argument('--num_iterations', default=10, type=int, help='number of iterations to train (in thousands)')
    parser.add_argument('--num_points_sample', default=100, type=int, 
        help='''when loading data into memory, number of points to sample from each file. if all of the data from all files fits into
        memory at the same time, set to 0 for no sampling.''')

    args = parser.parse_args()

    # make sure there is a file for the columns to use
    if not os.path.exists(os.path.join(args.input_dir, 'cols_to_use.txt')):
        raise Exception("The input directory must include a file named cols_to_use.txt with the column numbers of the FCS files \\\
            that you want to analyze, one name per line (0-indexed).")

    with open(os.path.join(args.input_dir, 'cols_to_use.txt')) as f:
        args.cols = [int(line.strip()) for line in f]

    if not os.path.exists(args.output_dir):
        os.mkdir(args.output_dir)

    if not os.path.exists(os.path.join(args.output_dir, 'models')):
        os.mkdir(os.path.join(args.output_dir, 'models'))

    with open(os.path.join(args.output_dir, 'args.pkl'), 'wb+') as f:
        pickle.dump(args, f)

    return args


##################################
##################################
# PREPROCESSING

args = parse_args()

rawfiles = sorted(glob.glob(os.path.join(args.input_dir, '*.fcs')))

##################################
##################################
# BATCH CORRECTION
# check if we are supposed to do batch correction and whether it already has been done
if args.batch_correct:
    if not batch_correction_training_done():
        print("Training batch correction models.")
        train_batch_correction(rawfiles)
    else:
        print("Found batch correction models.\n")

    if not batch_correction_done():
        print("Outputing batch corrected data.")
        output_batch_correction(rawfiles)
    else:
        print("Found batch corrected data.\n")

##################################
##################################
# CLUSTERING
if args.cluster:
    if args.batch_correct:
        input_files = sorted(glob.glob(os.path.join(args.output_dir, 'batch_corrected', '*.fcs')))
    else:
        input_files = rawfiles

    if not cluster_training_done():
        print("Training cluster model.")
        train_cluster(input_files)
    else:
        print("Found cluster model.\n")

    if not cluster_done():
        print("Outputing clustered data.")
        output_cluster(input_files)
    else:
        print("Found clustered data.\n")




print("Finished training models and outputing data!")














