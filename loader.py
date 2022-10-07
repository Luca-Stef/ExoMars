import numpy as np
from glob import glob
from datetime import datetime
from matplotlib import dates
import h5py
import os


def load_soundings(path="../data/level_1p0a/*.h5", clip_at=300, max_alt=120, min_alt=0, smoothing="poly", remove_flat=False, test=False, order=4):
    """
    Load soundings and location data from occultation files

    Inputs
    path          path to folder with occultation files, must include regex to distinguish occultation files of interest
    clip_at       wavelength to clip soundings at
    smoothing     type of smoothing to apply to soundings
    remove_flat   whether to remove flat soundings with a flat shape
    test          If true only one occultation file will be loaded

    Outputs
    wav           Array of shape (128,) with wavelengths of transmission values in spectra
    data          List of arrays of soundings, structured as data[occultation][sounding][wavelength]
    soundings     Array soundings structured as soundings[sounding][wavelength]
    shift_std     Standard deviation across wavelength of the change from one sounding to the next
    locations     locations of soundings, with soundings as rows and columns containing altitude, latitude, 
                  longitude, string date and time, datetime object, matplotlib date and time, solar longitude
    """
    occultationFiles = glob(path, recursive=True)
    data = []
    occLocations = []
    truncated = []
    occ_shift_std = []
    occ_shift_mean = []
    
    for file in occultationFiles:
        with h5py.File(file, "r") as f:
            wav = np.array(f["Science/X"])[0]
            
            if wav.shape[0] in [42, 466]:
                continue
               
            wav = compress(wav, wav.shape[0]/128)
            break
    
    for file in occultationFiles:
        with h5py.File(file, "r") as f:
            
            # Load in soundings and location data
            alt = np.mean(f["Geometry/Point0/TangentAltAreoid"], axis=1)
            occ = np.array(f["Science/YMean"])
            lat = np.mean(f["/Geometry/Point0/Lat"], axis=1)
            lon = np.mean(f["/Geometry/Point0/Lon"], axis=1)
            bytesUTC = np.array(f["/Geometry/ObservationDateTime"])
            stringUTC, datetimeUTC, mplUTC = create_date_objects(bytesUTC)
            ls = np.mean(np.array(f["/Geometry/LSubS"]), axis=1)
            occName = np.array([os.path.basename(file)[:-3]] * occ.shape[0])
            
            loc = np.vstack([alt, lat, lon, stringUTC, datetimeUTC, mplUTC, ls, occName]).T
            
            # Altitude mask attempts to remove below-surface and above-surface soundings
            msk_alt = (alt > min_alt) & (alt < max_alt)
            
            # Truncated occultation
            if occ.shape[1] in [466, 42]:
                truncated.append(occ[msk_alt])
                continue
            
            shift = np.nanstd(occ - np.roll(occ, 1, axis=0), axis=1)
            mean = np.nanmean(occ - np.roll(occ, 1, axis=0), axis=1)
            shift[0] = 0
            mean[0] = 0
            occ_shift_std.append(shift[msk_alt])
            occ_shift_mean.append(mean[msk_alt])
            data.append(occ[msk_alt])
            occLocations.append(loc[msk_alt])
    
        # Only load one occultation file for test runs
        if test:
            break
    
    print(f"{len(data)} occultations loaded")
    clip = wav_to_idx(clip_at, wav)
    wav = wav[clip:]
    soundings = []
    locations = []
    shift_mean = []
    shift_std = []

    # data[occultation][sounding][wavelength] -> soundings[sounding][wavelength]
    for i in range(len(data)):
        for j in range(data[i].shape[0]):
            
            # Remove unphysical transmissions
            if data[i][j].max() > 2 or data[i][j].std() > 0.5:
                continue
            
            # Compress to match on-board horizontal binning, clip to omit noisy data below 300nm
            sounding = compress(data[i][j], data[i][j].shape[0]/128)[clip:]
            # Interpolate nan values
            sounding = fill_nan(wav, sounding)
            soundings.append(sounding)
            locations.append(occLocations[i][j])
            shift_mean.append(occ_shift_mean[i][j])
            shift_std.append(occ_shift_std[i][j])
    
    soundings = np.vstack(soundings)
    locations = np.vstack(locations)
    shift_mean = np.vstack(shift_mean)
    shift_std = np.vstack(shift_std)
    print(f"{soundings.shape[0]} total soundings loaded.")
    
    if remove_flat:
        soundings, shift_std, shift_mean, locations = filter_flat_soundings(soundings, shift_std, shift_mean, locations)
        print("Flat soundings removed manually")
        
    if smoothing == "poly":
        soundings = poly_smooth(wav, soundings, order)
        print("Soundings smoothed using high-order polynomial fit")

    if smoothing == "moving_average":
        wav = moving_average(wav, 30)
        for i in range(soundings.shape[0]):
            soundings[i] = moving_average(soundings[i], 30) 

    if smoothing == "average":
        wav = compress(wav, 4)
        for i in range(soundings.shape[0]):
            soundings[i] = compress(soundings[i], 4)
        
    if smoothing == "rdp": # TODO 
        pass
    
    space = locations[:,:3]
    time = locations[:,3:]

    np.savetxt("../data/soundings.txt", soundings)
    np.savetxt("../data/wav.txt", wav)

    return wav, soundings, shift_std, shift_mean, space, time

def compress(arr, factor):
    """
    Compress an array by a given factor by taking averages of groups of points
    """
    if arr.shape[0] % factor:
        idx = arr.shape[0] % factor
        compressed = arr[:-idx].reshape(int(arr[:-idx].shape[0]/factor), int(factor)).sum(axis=1)/factor
        compressed = np.append(compressed, np.mean(arr[-idx:]))
    else:
        compressed = arr.reshape(int(arr.shape[0]/factor), int(factor)).sum(axis=1)/factor
    return compressed

def create_date_objects(bytesDates):
    """
    UTC date and time field is represented as a bytes object. This function converts from bytes to string literals,
    datetime objects, and matplotlib dates suitable for plotting
    
    Inputs
    bytesDates     Array of dimension (M,N). Dates and times of soundings represented as python bytes ojects
    
    Outputs
    stringDates    Same size array of dates but as strings
    datetimeDates  Python datetime objects
    mplDates       Matplotlib dates suitable for plotting (Averaged over every sounding)
    """
    stringDates = np.array([[bytesDates[i,j].decode("UTF-8") for j in range(bytesDates.shape[1])] 
                            for i in range(bytesDates.shape[0])])
    datetimeDates = np.array([[datetime.strptime(stringDates[i,j], "%Y %b %d %H:%M:%S.%f") 
           for j in range(bytesDates.shape[1])] for i in range(bytesDates.shape[0])])
    mplDates = dates.date2num(datetimeDates)
    
    mplDates = np.mean(mplDates, axis=1)
    datetimeDates = np.array(dates.num2date(mplDates))
    stringDates = np.array([datetimeDates[i].strftime("%Y %b %d %H:%M:%S.%f") for i in range(datetimeDates.shape[0])])
    
    return stringDates, datetimeDates, mplDates

def wav_to_idx(wl, wav):
    """
    Clip a sounding to remove data below 300 nm
    """
    return np.argmin(abs(wav - wl))

def fill_nan(wav, sounding):
    """
    Interpolate nan values
    """
    mask = np.isfinite(sounding)
    filledSounding = np.interp(wav, wav[mask], sounding[mask])
    return filledSounding

def poly_smooth(wav, soundings, order):
    """
    Use a high order polynomial fit to smooththe noise in soundings
    """
    smooth = np.zeros(soundings.shape)
    coeffs = np.polyfit(wav, soundings.T, order)
    for i in range(soundings.shape[0]):
        smooth[i] = np.poly1d(coeffs[:,i])(wav)
    return smooth

def filter_flat_soundings(soundings, shift_std, shift_mean, locations):
    """
    Remove soundings that are flat (determined by standard deviation) 
    """
    mean = np.reshape(soundings.mean(axis=1), (-1,1))
    std = np.reshape(soundings.std(axis=1), (-1,1))
    
    above_mask = (mean > 0.99).flatten() 
    below_mask = (mean < 0.01).flatten()
    flat_mask = (std < 0.001).flatten()
    
    soundings = soundings[~(above_mask|below_mask|flat_mask)]
    shift_std = shift_std[~(above_mask|below_mask|flat_mask)]
    shift_mean = shift_mean[~(above_mask|below_mask|flat_mask)]
    locations = locations[~(above_mask|below_mask|flat_mask)]
    
    return soundings, shift_std, shift_mean, locations

def moving_average(x, w):
    """
    Uses convolution with a step function to calculate moving average.
    """
    return np.convolve(x, np.ones(w), 'valid') / w