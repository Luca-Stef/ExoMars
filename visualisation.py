import numpy as np
from matplotlib import pyplot as plt
from math import ceil
import os
from itertools import combinations
import shutil


def analyse_model(model, features, wav, soundings, n_clusters, view):
    """
    Create a KMeans model with n clusters and visualise the soundings in every cluster as well as the virtual sounding
    created from the cluster centroid
    """
    model = model(n_clusters).fit(features)

    if view == "deviations":
        for c in range(n_clusters):
            try: cluster_mask = model.labels_ == c
            except AttributeError: cluster_mask = model.predict(features) == c
            cluster_c = soundings[cluster_mask]

            if cluster_c.shape[0] == 0:
                plt.plot(wav, np.poly1d(model.cluster_centers_[c])(wav))
                plt.ylim([-0.5, 0.5])
                plt.title(f"Cluster {c} with {cluster_c.shape[0]} soundings")
                plt.savefig(f"./results/{c}.png")
                plt.show()
                continue

            mx = cluster_c.max(axis=0)
            mn = cluster_c.min(axis=0)
            
            plt.plot(wav, np.poly1d(model.cluster_centers_[c])(wav))
            plt.fill_between(wav, mn, mx, alpha=0.3)
            plt.ylim([-0.5, 0.5])
            plt.title(f"{cluster_c.shape[0]} soundings in cluster {c}")
            plt.savefig(f"./results/{c}.png")
            plt.show()
            
    elif view == "constituents":
        for c in range(n_clusters):
            try: cluster_mask = model.labels_ == c
            except AttributeError: cluster_mask = model.predict(features) == c
            cluster_c = soundings[:200][cluster_mask]
            n_soundings = cluster_c.shape[0]
            step = ceil(n_soundings/20) if n_soundings != 0 else 1
            
            if cluster_c.shape[0] == 0:
                print(f"Cluster {c} with {cluster_c.shape[0]} soundings")
                continue
            
            plt.plot(wav, cluster_c.T[:,::step])
            plt.ylim([-0.5,0.5])
            plt.title(f"{cluster_c.shape[0]} soundings in cluster {c}")
            plt.savefig(f"./results/{c}.png")
            plt.show()
           
    return_cluster = input("Inspect cluster: ")
    
    if return_cluster == "":
        return None, None
    
    return_cluster = int(return_cluster)
    try: return_mask = model.labels_ == return_cluster
    except AttributeError: return_mask = model.predict(features) == return_cluster
    return features[return_mask], soundings[return_mask]

    return None, None

def view_soundings(wav, soundings):
    """
    Plot soundings in a grid of plots
    """
    n = soundings.shape[0]
    rows = ceil(n/4)
    fig = plt.figure(figsize=(20,4*rows))
    axes = []
    for i in range(n):
        axes.append(fig.add_subplot(rows,4,i+1))
        axes[i].plot(wav, soundings[i])
        axes[i].set_ylim([-0.5,0.5]) if soundings[i].min() < 0 else axes[i].set_ylim([0,1])
    fig.tight_layout()
    
def view_many_soundings(wav, soundings):
    """
    Plot soundings with many soundings per plot
    """
    n = soundings.shape[0]
    rows = ceil(n/4)
    fig = plt.figure(figsize=(20,4*rows))
    axes = []
    for i in range(ceil(n/50)):
        axes.append(fig.add_subplot(rows,4,i+1))
        for j in range(i*50, min((i+1)*50, soundings.shape[0])):
            axes[i].plot(wav, soundings[j])
        axes[i].set_ylim([0,1])
    fig.tight_layout()
    
def scatter_features(features):
    """
    Plots scatter plots of every 2 element combination of our features
    """
    combs = list(combinations(range(features.shape[1]), 2))
    n = len(combs)
    rows = ceil(n/4)
    fig = plt.figure(figsize=(20,4*rows))
    axes = []
    for i in range(n):
        axes.append(fig.add_subplot(rows,4,i+1))
        axes[i].scatter(features[:,combs[i][0]], features[:,combs[i][1]])
        axes[i].set_xlabel(combs[i][0])
        axes[i].set_ylabel(combs[i][1])
    fig.tight_layout()
    
def view_clusters3d(labels, feature1, feature2, feature3, selected_cluster):
    """
    Visualise a 3D feature space from given features, with selected cluster highlighted a different color
    """
    fig = plt.figure()
    ax = fig.add_subplot(projection="3d")
    p = ax.scatter(feature1, feature2, feature3, c=(labels == selected_cluster).astype(float))
    
def view_grouped_soundings(labels, wav, soundings, return_cluster=False, savepath=None):
    """
    View a sample of soundings in every cluster created by a clustering algorithm
    """
    for c in set(labels):
        cluster_c = soundings[labels==c]
        if cluster_c.shape[0] > 1:
            dispersion = np.sqrt(np.cov(cluster_c.T).trace())
        else:
            dispersion = 0

        if cluster_c.shape[0] == 0:
            print(f"Cluster {c} with {cluster_c.shape[0]} soundings")
            continue
        
        plt.plot(wav, cluster_c.T)
        plt.ylim([0, 1]) if cluster_c.max() > 0.5 else plt.ylim([-0.5,0.5])
        plt.title(f"{cluster_c.shape[0]} soundings in cluster {c}\ndispersion: {dispersion}\n")
        if not savepath:
            plt.show()
        else:
            plt.savefig(savepath + f"{c}", bbox_inches='tight')
            plt.close()
    
    if return_cluster:
        return_cluster = input("Select a cluster: ")

        try: return soundings[labels==int(return_cluster)]
        except ValueError: pass
        
def view_feature_space(features, names, labels, c=None):
    """
    View feature space with specified cluster highlighted in yellow
    """
    n_features = features.shape[1]
    
    if n_features == 2:
        fig = plt.figure()
        ax = fig.add_subplot()
        
        if c == None:
            ax.scatter(features[:,0], features[:,1], c=labels.astype(float))
        else:
            ax.scatter(features[:,0], features[:,1], c=(labels==c).astype(float))
            
        ax.set_xlabel(names[0])
        ax.set_ylabel(names[1])

    elif n_features == 3:
        fig = plt.figure()
        ax = fig.add_subplot(projection="3d")
        
        if c == None:
            ax.scatter(features[:,0], features[:,1], features[:,2], c=labels.astype(float))
        else:
            ax.scatter(features[:,0], features[:,1], features[:,2], c=(labels==c).astype(float))
            
        ax.set_xlabel(names[0])
        ax.set_ylabel(names[1])
        ax.set_zlabel(names[2])
        
    else:
        fig = plt.figure()
        ax = fig.add_subplot()
        
        if c == None:
            ax.scatter(features[:,0], features[:,1], c=labels.astype(float))
        else:
            ax.scatter(features[:,0], features[:,1], c=(labels==c).astype(float))
            
        ax.set_xlabel(names[0])
        ax.set_ylabel(names[1])
        view_feature_space(features[:,2:], names[2:], labels, c)

def view_major_groups(foldername, majorlabels, minorlabels, soundings, wav):
    """
    Plot soundings present in each major class and save to foldername
    """
    try: shutil.rmtree(f"{foldername}")
    except FileNotFoundError: pass

    # Iterate over every class
    for majorlabel in set(majorlabels):
        mask = majorlabels == majorlabel

        # Iterate over every cluster present in the current major class
        for c in set(minorlabels[mask]):
            cluster = soundings[minorlabels == c]

            try: os.makedirs(f"{foldername}/class {majorlabel}/")
            except FileExistsError: pass
            
            plt.plot(wav, cluster.T)
            plt.ylim([-0.5, 0.5])
            plt.title(f"{cluster.shape[0]} soundings in cluster {c}")
            plt.savefig(f"{foldername}/class {majorlabel}/cluster {c}", bbox_inches="tight")
            plt.close()