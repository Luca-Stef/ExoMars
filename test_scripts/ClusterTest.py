#!/usr/bin/env python3

import numpy as np
import h5py
from matplotlib import pyplot as plt
import sklearn
from sklearn.cluster import KMeans, SpectralClustering, AgglomerativeClustering, DBSCAN, OPTICS, Birch
from sklearn.mixture import GaussianMixture
from sklearn.metrics import mean_squared_error as mse
from glob import glob
from math import floor, ceil
from itertools import combinations
from tqdm import tqdm
from functions import *
np.set_printoptions(suppress=False)

wav, soundings, scaled_soundings, shift_std = load_soundings("../data/level_1p0a/*[I|E].h5")

grad, curv, polyres, poly_coeffs = create_features(wav, soundings)

algorithms = {"kmeans": KMeans, "spectralClustering": SpectralClustering, "agglomerativeClustering": AgglomerativeClustering, 
                "gaussianMixtures": GaussianMixture}
features = {"coeffs": poly_coeffs, "coeffs_res": np.hstack([poly_coeffs, polyres]), 
            "coeffs_res_grad": np.hstack([poly_coeffs, polyres, grad]), "soundings": scaled_soundings, 
            "less_soundings": scaled_soundings[:,::10]}

"""for alg_name, alg in algorithms.items():
    for feature_name, feature in features.items():
        for n in range(3,20):
            analyse_model(wav, scaled_soundings[::10], alg, feature[::10], n_clusters=n, view="constituents", alg_name=alg_name, feature_name=feature_name)"""

for alg_name, alg in {"dbscan": DBSCAN, "optics": OPTICS, "birch": Birch}.items():
    for feature_name, feature in features.items():
        analyse_model(wav, scaled_soundings, alg, feature, n_clusters=0, view="constituents", alg_name=alg_name, feature_name=feature_name)