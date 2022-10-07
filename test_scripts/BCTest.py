#!/usr/bin/env python3

import argparse
import os
import numpy as np
from sklearn.cluster import Birch
from sklearn.metrics import pairwise_distances
from loader import *
from features import *
from visualisation import *
np.set_printoptions(suppress=False)


parser = argparse.ArgumentParser(description='Parameters for algorithm')
parser.add_argument('-t', '--test', action='store_true',
                    help='run on full dataset')
parser.add_argument('-b', '--branching_factor', type=int, default=50,
                    help='branching factor')
parser.add_argument('-th', '--threshold', type=float, default=3,
                    help='threshold')

args = parser.parse_args()


wav, soundings, shift_std, shift_mean, space, time = load_soundings("../data/level_1p0a/*[I|E].h5", smoothing="poly", test=args.test, remove_flat=False)
centred_soundings = centre(soundings)

grad, polyres, poly_coeffs = create_features(wav, centred_soundings, standardize, "grad", "polyres", "poly_coeffs")

features = np.hstack([poly_coeffs[:,0:1], polyres[:,1:2], grad[:,4:5]])

combined_features = np.hstack([standardize(centred_soundings), features])

savepath = f"../bctest/standardized_combined_features_unfiltered/{args.threshold}/{args.branching_factor}/"

model = Birch(n_clusters=None, threshold=args.threshold, branching_factor=args.branching_factor).fit(combined_features)

try: os.makedirs(savepath)
except FileExistsError: pass

labels = model.labels_

view_grouped_soundings(labels, wav, centred_soundings, savepath=savepath)