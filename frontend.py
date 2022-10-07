#!/usr/bin/env python3

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from PIL import Image
from glob import glob
import h5py
import os
import argparse

parser = argparse.ArgumentParser(description="Parameters for algorithm")
parser.add_argument("-m", "--mode", type=int, help="1 for viewing class, 2 for occultation")
parser.add_argument("-c", "--spectralclass", type=int, help="Which spectral class to view")
parser.add_argument("-p", "--plottype", type=int, help="1 to view geolocations, 2 to view altitude distribution, 3 to view seasonal distribution")
parser.add_argument("-d", "--destination", type=str, default="save", help="Type 'show' to show plot or 'save' to automatically save or provide a file path to save to")
parser.add_argument("-n", "--name", type=str, help="File name of occultation (without .h5)") 
parser.add_argument("-s", "--view_soundings", action="store_true", help="View soundings in a class/occultation")
args = parser.parse_args()

marsImage = Image.open("../images/mars.png")
occ_view = pd.read_csv("../data/out/occ_view.csv")
spectralclasses = [pd.read_csv(file) for file in glob("../data/out/class_view/*")]
labels = np.genfromtxt("../data/labels.txt")

index = {0: "flat above atmosphere", 1: "flat below atmosphere", 2: "flat in atmosphere", 3: "high gradient", 
        4: "negative curvature", 5: "periodic", 6: "positive curvature", 7: "small gradient"}
xedni = {v: k for k, v in index.items()}

print("Class index\n")
for key, value in index.items(): print(f"{key}: {value}")
print()


# class view
if args.mode == 1:
    nspectra = spectralclasses[args.spectralclass].shape[0]

    # geolocations
    if args.plottype == 1:
        print(f"Viewing geolocations of spectral class {args.spectralclass}: {nspectra} spectra present\n")

        fig = plt.figure(figsize=(15,9))
        ax = fig.gca()
        im = ax.scatter(spectralclasses[args.spectralclass]["lon"], spectralclasses[args.spectralclass]["lat"], 
                    c=spectralclasses[args.spectralclass]["alt"], linewidths=0.5)
        ax.grid()
        ax.set_title(f"Locations of {index[args.spectralclass]} soundings")
        ax.set_xlabel("Longitude")
        ax.set_ylabel("Latitude")
        ax.set_xticks(range(-180, 181, 30))
        ax.set_yticks(range(-90, 91, 30))
        ax.set_xlim(-180, 180)
        ax.set_ylim(-90, 90)

        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        cbar = plt.colorbar(im, cax=cax)
        cbar.set_label("Altitudes / km")
        ax.imshow(marsImage, extent=[-180, 180, 90, -90])
        plt.tight_layout()
        
        if args.destination == "show":
            plt.show()

        elif args.destination == "save":
            try: os.makedirs(f"../class plots/geolocations/")
            except FileExistsError: pass

            plt.savefig(f"../class plots/geolocations/class {args.spectralclass}.png")
            print(f"Saved to ../class plots/geolocations/class {args.spectralclass}.png\n")
            plt.close()
            
        else:
            plt.savefig(args.destination)
            print(f"Saved to {args.destination}\n")
            plt.close()
    
    # altitude distribution
    elif args.plottype == 2:
        print(f"Viewing altitudes of spectral class {args.spectralclass}: {nspectra} spectra present\n")

        fig = plt.figure(figsize=(15,9))
        ax = fig.gca()
        im = ax.hist(spectralclasses[args.spectralclass]["alt"], bins=40)
        ax.set_title(f"Altitude distribution of {index[args.spectralclass]} soundings")
        ax.set_xlabel("Altitude")
        ax.set_ylabel("Frequency")
        
        if args.destination == "show":
            plt.show()

        elif args.destination == "save":
            try: os.makedirs(f"../class plots/altitude distribution/")
            except FileExistsError: pass

            plt.savefig(f"../class plots/altitude distribution/class {args.spectralclass}.png")
            print(f"Saved to ../class plots/altitude distribution/class {args.spectralclass}.png\n")
            plt.close()
            
        else:
            plt.savefig(args.destination)
            print(f"Saved to {args.destination}\n")
            plt.close()

    # seasonal distribution
    elif args.plottype == 3:
        print(f"Viewing seasonal distribution of spectral class {args.spectralclass}: {nspectra} spectra present\n")

        fig = plt.figure(figsize=(15,9))
        ax = fig.gca()
        im = ax.hist(spectralclasses[args.spectralclass]["ls"], bins=12)
        ax.set_title(f"Seasonal distribution of {index[args.spectralclass]} soundings")
        ax.set_xlabel("Solar longitude")
        ax.set_ylabel("Frequency")

        if args.destination == "show":
            plt.show()

        elif args.destination == "save":
            try: os.makedirs(f"../class plots/seasonal distribution/")
            except FileExistsError: pass

            plt.savefig(f"../class plots/seasonal distribution/class {args.spectralclass}.png")
            print(f"Saved to ../class plots/seasonal distribution/class {args.spectralclass}.png\n")
            plt.close()
            
        else:
            plt.savefig(args.destination)
            print(f"Saved to {args.destination}\n")
            plt.close()
            
# Occultation view
elif args.mode == 2:
    occ = occ_view[occ_view["file name"] == args.name]
    print(f"Composition of occultation {args.name}")
    print("Total soundings ".ljust(25), f"{occ['total'].iloc[0]}")
    for k, v in index.items(): print(f"{v.ljust(25)} {occ[str(k)].iloc[0]}")

    if args.view_soundings:
        with h5py.File(f"../data/level_1p0a/{args.name}.h5", "r") as f:
            y = np.array(f["Science/YMean"])
            x = np.array(f["Science/X"])[0]
            alt = np.mean(f["Geometry/Point0/TangentAltAreoid"], axis=1)
            fig = plt.figure(figsize=(15,9))
            ax = fig.gca()
            im = ax.plot(x, y.T)
            ax.set_ylim([0,1])
            
            if args.destination == "show":
                plt.show()

            elif args.destination == "save":
                try: os.makedirs(f"../occultations/")
                except FileExistsError: pass

                plt.savefig(f"../occultations/occultation {args.name}.png")
                print(f"\nSaved plot to ../occultations/occultation {args.name}.png\n")
                plt.close()
                
            else:
                plt.savefig(args.destination)
                print(f"Saved to {args.destination}\n")
                plt.close()