#!/usr/bin/env python3

import argparse
import numpy as np
from sklearn.cluster import Birch

from loader import *
from features import *
from visualisation import *
from merging import *
from analysis import *

parser = argparse.ArgumentParser(description="Parameters for algorithm")
parser.add_argument("-p", "--path", type=str, default="../data/level_1p0a/*.h5", help="Path to level 1.0a data")
parser.add_argument("-b", "--branching_factor", type=int, default=50, help="Branching factor of BIRCH algorithm")
parser.add_argument("-t", "--threshold", type=float, default=3, help="Threshold for BIRCH algorithm")
parser.add_argument("-n", "--n_clusters", type=int, default=None, help="n_clusters for BIRCH algorithm")
parser.add_argument("--no_compute_labels", action="store_false", help="Whether or not BIRCH computes labels for each fit")
parser.add_argument("--no_copy", action="store_false", help="Whether or not BIRCH makes a copy of given data, or overwrites initial data")
parser.add_argument("-o", "--order", type=int, default=4, help="Order of polynomial fit to use for smoothing soundings")
parser.add_argument("-s", "--smoothing", type=str, default="poly", help="Method for smoothing noise in spectra")
parser.add_argument("-r", "--remove_flat", action="store_true", help="Remove flat soundings manually before BIRCH")
parser.add_argument("-v", "--view_classes", type=str, default=None, help="Creates plots of soundings in spectral classes and outputs to folder name provided")
args = parser.parse_args()

## Loading data ##
wav, soundings, shift_std, shift_mean, space, time = load_soundings(
                                                args.path, smoothing=args.smoothing, remove_flat=args.remove_flat, order=args.order
                                                                    )
centred_soundings = centre(soundings)

## Creating features ##
grad, polyres, poly_coeffs, curv = create_features(wav, centred_soundings, standardize, "grad", "polyres", "poly_coeffs", "curv")
print("Feature space created")
features = np.hstack([poly_coeffs[:,0:1], polyres[:,1:2], grad[:,4:5]])

combined_features = np.hstack([standardize(centred_soundings), features])

## Fitting model ##
model = Birch(n_clusters=args.n_clusters, threshold=args.threshold, branching_factor=args.branching_factor, 
                compute_labels=args.no_compute_labels, copy=args.no_copy).fit(combined_features)
print("BIRCH algorithm fitted to data")

minorlabels = model.labels_

## Merging clusters ##
majorlabels = merge_labels(soundings, minorlabels, grad, polyres, poly_coeffs, curv)

finallabels = add_flat_labels(soundings, majorlabels)

if args.view_classes:
    view_major_groups(args.view_classes, finallabels, minorlabels, centred_soundings, wav)

## Analysis ## 
create_class_view(wav, soundings, space, time, finallabels)
create_occultation_view(time, finallabels)
print("Output written to filepath ../data/out")

### Desired functionality ###

# what waveforms are present in an occultation
# spatial and temporal distribution of given waveforms
# original average transmission
# altitudes
# transmission shift over soundings

### Label index ###
# 0 - flat above atmosphere
# 1 - flat below atmosphere
# 2 - flat in atmosphere
# 3 - high gradient
# 4 - negative curvature
# 5 - periodic 
# 6 - positive curvature
# 7 - small gradient