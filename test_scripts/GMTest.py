#!/usr/bin/env python3

import argparse
import os
import numpy as np
from sklearn.mixture import GaussianMixture
from sklearn.metrics import mean_squared_error as mse, silhouette_score
from features import *
from visualisation import *
np.set_printoptions(suppress=False)


parser = argparse.ArgumentParser(description='Parameters for algorithm')
parser.add_argument('-t', '--test', action='store_true',
                    help='run on full dataset')
parser.add_argument('-f', '--features', type=str,
                    help='feature space to run algorithm on')
parser.add_argument('-c', '--cov', type=str,
                    help='covariance type')
parser.add_argument('-n', '--n_components', type=int,
                    help='number of integers')

args = parser.parse_args()


wav, soundings, shift_std, shift_mean, locations = load_soundings("../data/level_1p0a/*[I|E].h5", smoothing="poly", test=args.test)
centred_soundings = centre(soundings)
comp_wav, comp_centred_soundings = dropout(wav, centred_soundings, 11)

grad, polyres, poly_coeffs = create_features(wav, centred_soundings, normalize, "grad", "polyres", "poly_coeffs")

logpolyres = np.log(polyres)/np.log(polyres).std(axis=0)
cbrt_poly_coeffs = np.cbrt(poly_coeffs)
cbrt_grad = np.cbrt(grad)

eng_features = np.hstack([poly_coeffs[:,0:1], polyres[:,1:2], grad[:,4:5]])
att_eng_features = np.hstack([cbrt_poly_coeffs[:,0:1], logpolyres[:,1:2], cbrt_grad[:,4:5]])

standardised_combined_features = np.hstack([comp_centred_soundings/comp_centred_soundings.std(axis=0), eng_features])
att_standardised_combined_features = np.hstack([comp_centred_soundings/comp_centred_soundings.std(axis=0), att_eng_features])

index = {"att_standardised_combined_features": att_standardised_combined_features, "att_eng_features": att_eng_features,
         "standardised_combined_features": standardised_combined_features, "eng_features": eng_features,
         "comp_centred_soundings": comp_centred_soundings}

savepath = f"../gmtest/{args.features}/{args.cov}/{args.n_components}/"


gm = GaussianMixture(
    n_components=args.n_components, covariance_type=args.cov, init_params="random", verbose=False, tol=1e-7, max_iter=int(1e15)
                ).fit(index[args.features])

try: os.makedirs(savepath)
except FileExistsError: pass

gmlabels = gm.predict(index[args.features])

view_grouped_soundings(gmlabels, wav, centred_soundings, savepath=savepath)