#!/usr/bin/env python3

import argparse
import os
import numpy as np
from sklearn.cluster import MeanShift
from features import *
from visualisation import *
np.set_printoptions(suppress=False)


parser = argparse.ArgumentParser(description='Parameters for algorithm')
parser.add_argument('-t', '--test', action='store_true',
                    help='run on full dataset')
parser.add_argument('-b', '--bandwidth', type=float,
                    help='bandwidth')
parser.add_argument('-ca', '--cluster_all', action="store_true",
                    help='Cluster all soundings')

args = parser.parse_args()


wav, soundings, shift_std, shift_mean, locations = load_soundings("../data/level_1p0a/*[I|E].h5", smoothing="poly", test=args.test)
centred_soundings = centre(soundings)
comp_wav, comp_centred_soundings = dropout(wav, centred_soundings, 11)

grad, polyres, poly_coeffs, curv = create_features(wav, centred_soundings, normalize, "grad", "polyres", "poly_coeffs", "curv")

features = np.hstack([poly_coeffs[:,0:1], polyres[:,1:2], grad[:,4:5]])

combined_features = np.hstack([normalize(centred_soundings), features])

savepath = f"../mstest/normalized_combined_features/{args.bandwidth}/"

model = MeanShift(bandwidth=args.bandwidth, cluster_all=args.cluster_all, n_jobs=-1).fit(combined_features)

try: os.makedirs(savepath)
except FileExistsError: pass

labels = model.labels_

view_grouped_soundings(labels, wav, centred_soundings, savepath=savepath)