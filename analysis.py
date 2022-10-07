import numpy as np
import pandas as pd
import os

def find_wavelengths(wav, soundings):
    """
    Calculate wavelengths present in soundings by using the gradient to calculate peak to trough distances
    """
    dsoundings = np.gradient(soundings, axis=1, edge_order=2)
    sign = np.sign(dsoundings)
    wavs = []
    for i in range(soundings.shape[0]):
        count = np.array((np.roll(sign[i], -1) - sign[i])[:-1], dtype="bool")
        idx = np.argwhere(count).flatten()
        wavs.append((wav[idx] - np.roll(wav[idx], 1))[1:])
    
    wavs = np.array(wavs, dtype=object)*2
    wav1 = np.zeros(soundings.shape[0])
    wav2 = np.zeros(soundings.shape[0])
    
    for i in range(len(wavs)):
        try: wav1[i] = wavs[i][0]
        except IndexError: wav1[i] = np.NAN
        
        try: wav2[i] = wavs[i][1]
        except IndexError: wav2[i] = np.NAN
    
    wavs = np.vstack([wav1, wav2]).T
        
    return wavs

def create_occultation_view(time, finallabels):
    """
    Create nested dictionary where for each occultation the classes of spectra can be read along with their
    altitudes and mean transmission levels 
    """
    occ_view = []
    for occ in range(np.unique(time[:,-1]).shape[0]):
        occ_mask = time[:,-1] == np.unique(time[:,-1])[occ]
        occ_view.append({})
        occ_view[occ]["file name"] = np.unique(time[:,-1])[occ]
        for spectralclass in set(finallabels):
            class_mask = finallabels == spectralclass
            occ_view[occ][spectralclass] = (occ_mask & class_mask).sum()
        occ_view[occ]["total"] = occ_mask.sum()

    pd.DataFrame(occ_view).to_csv("../data/out/occ_view.csv", index=False)

def create_class_view(wav, soundings, space, time, finallabels):
    """
    Create nested dictionary where for each class the spatial and seasonal distribution as well as the mean transmission levels
    can be read
    """
    try: os.makedirs("../data/out/class_view")
    except FileExistsError: pass
    
    class_view = {}

    for spectralclass in set(finallabels):

        class_mask = finallabels == spectralclass
        
        alt = space[class_mask, 0]
        lat = space[class_mask, 1]
        lon = space[class_mask, 2]
        ls = time[class_mask, 3]
        mean_transmission = soundings[class_mask].mean(axis=1)
        occName = time[class_mask, -1]
        
        if spectralclass == 5:
            wavs = find_wavelengths(wav, soundings[class_mask])
            class_view[spectralclass] = np.vstack([ls, lat, lon, alt, mean_transmission, occName, wavs.T]).T
            
            pd.DataFrame(class_view[spectralclass]).to_csv(f"../data/out/class_view/class {spectralclass}.csv", 
                                                    index=False, header=["ls", "lat", "lon", "alt", "mean_transmission", 
                                                                         "filename", "primary wavelength", 
                                                                         "secondary wavelength"])
        
        else:
            class_view[spectralclass] = np.vstack([ls, lat, lon, alt, mean_transmission, occName]).T
            pd.DataFrame(class_view[spectralclass]).to_csv(f"../data/out/class_view/class {spectralclass}.csv", index=False,
                                                    header=["ls", "lat", "lon", "alt", "mean_transmission", "filename"])