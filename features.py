import numpy as np
from sklearn.metrics import mean_squared_error as mse
from math import floor, ceil
np.set_printoptions(suppress=False)


def wav_to_idx(wl, wav):
    """
    Clip a sounding to remove data below 300 nm
    """
    return np.argmin(abs(wav - wl))

def filter_n(transform, n):
    """
    Takes a transformed signal as input and filters it such that only n non-zero frequencies remain
    """
    return np.where(abs(transform) >= np.sort(abs(transform))[-2*n], transform, 0+0j)

def fourier_smooth(sounding, **kwargs):
    """
    Smooths a sounding so only n_freq frequencies remain, alternatively use a threshold to filter frequency amplitudes.
    """
    transformed = np.fft.fft(sounding)
    if "n_freq" in kwargs.keys():
        cleanTransformed = filter_n(transformed, kwargs["n_freq"])
    elif "thresh" in kwargs.keys():
        cleanTransformed = np.where(abs(transformed) >= kwargs["thresh"], transformed, 0+0j)
    else:
        cleanTransformed = filter_n(transformed, 10)
    cleanSounding = np.fft.ifft(cleanTransformed)
    return abs(cleanSounding)

def fourier_extract(wav, soundings, n_freq):
    """
    Fourier transform the sounding, extract largest amplitude (excluding that corresponding to zero frequency),
    filter the sounding such that n_freq frequencies remain, calculate mse between filtered signal and original signal.
    
    Inputs
    sounding     Spectrum to be analysed 
    n_freq       Number of remaining frequencies after being filtered via fourier transform
    
    Outputs
    error        Mean squared error between original and smoothed spectra
    amp          Largest amplitude in fourier transform (excluding zero frequency)
    period       Period corresponding with the largest amplitude (excluding zero frequency)
    """
    fourier = np.zeros((soundings.shape[0], 3))

    for i in range(soundings.shape[0]):

        freq = np.fft.fftfreq(wav.shape[0])
        transformed = np.fft.fft(soundings[i])
        cleanTransformed = filter_n(transformed, n_freq)
        cleanSounding = np.fft.ifft(cleanTransformed)
        error = mse(soundings[i], abs(cleanSounding))
        idx = np.argmax(abs(transformed)[1:51]) + 1
        amp = abs(transformed[1:51]).max()
        period = 1/freq[idx]

        fourier[i] = np.array([error, amp, period])
    
    return fourier

def intersect_counter(f):
    """
    Counts the number of times f crosses x axis
    """
    sign = np.sign(f)
    count = np.array((np.roll(sign, -1) - sign)[:-1], dtype="bool").sum()
    return count

def optimum_counter(soundings, window):
    """
    Take moving average using window length given, take the gradient and count how many times 
    the gradient changes sign which should give a count of optima in original signal.
    """
    optc = np.zeros((soundings.shape[0], 1))
    for i in range(soundings.shape[0]):
        grad = np.gradient(moving_average(soundings[i], window), edge_order=2)
        optc[i] = intersect_counter(grad)
    return optc

def polyfit_residual(wav, soundings, deg):
    """
    Fits a polynomial of degree deg to the sounding and then calcualtes residuals with 
    original sounding
    """
    res = np.zeros((soundings.shape[0], 1))
    for i in range(soundings.shape[0]):
    
        coeffs = np.polyfit(wav, soundings[i], deg)
        fit = np.poly1d(coeffs)(wav)
        res[i] = np.abs(fit - soundings[i]).sum()
    
    return res

def idx_to_wav(wav, idx):
    """
    Calculates distance from wavelength at wav[idx] to wav[0]
    """
    return wav[idx] - wav[0]

def phase_coherence(soundings):
    """
    Finds phase coherence function of a sounding and returns ratio of maximum of function to minimum of function
    as well as the distance between these two points in the original sounding.
    """
    coh = np.zeros((soundings.shape[0], 1))
    for i in range(soundings.shape[0]):
        
        std = np.zeros(soundings[i].shape[0]-1)
        for j in range(soundings[i].shape[0]-1):
            std[j] = soundings[i][::j+1].std()
            
        idx_dist = np.abs(std.argmax() - std.argmin())
        max_min = (std.max() - std.min())

        coh[i] = max_min
    
    return coh

def moving_average(x, w):
    """
    Uses convolution with a step function to calculate moving average.
    """
    return np.convolve(x, np.ones(w), 'valid') / w

def moving_average_residual(soundings, window):
    """
    Uses convolution with a step function to calculate moving average, then calculates residual with original sounding.
    """
    mv_av_res = np.zeros((soundings.shape[0], 1))

    for i in range(soundings.shape[0]):
        mov_avg = moving_average(soundings[i], window)
        # clip sounding at the front and back to match shape of mov_avg
        orig_clipped = soundings[i][ceil(window/2)-1:-floor(window/2)]
        mv_av_res[i] = np.abs(mov_avg - orig_clipped).sum()

    return mv_av_res

def transmission_grad(wav, soundings, min, max):
    """
    Total change in transmission over region specified by min and max
    """
    minidx = wav_to_idx(min, wav)
    maxidx = wav_to_idx(max, wav)
    grad = soundings[:,maxidx] - soundings[:,minidx]
    grad = np.reshape(grad, (-1,1))
    return grad

def gradient_features(soundings):
    """
    Calculate derivative and second derivative of smoothed soundings and use these to calculate further features of soundings

    Inputs
    soundings     array of soundings of shape soundings[sounding,wavelength]

    Outputs 
    dminmax       minimum gradient multiplied by maximum in gradient
    dstd          standard deviation of gradient
    ddminmax      minimum and maximum of second derviative multiplied together
    ddstd         standard deviation of curvature
    """
    dsoundings = np.gradient(soundings, axis=1, edge_order=2)
    ddsoundings = np.gradient(dsoundings, axis=1, edge_order=2)
    dminmax = dsoundings.max(axis=1) * dsoundings.min(axis=1)
    dstd = dsoundings.std(axis=1)
    ddminmax = ddsoundings.max(axis=1) * ddsoundings.min(axis=1)
    ddstd = ddsoundings.std(axis=1)

    return np.vstack([dminmax, dstd, ddminmax, ddstd]).T

def curvature(wav, soundings):
    """
    Use the difference in total gradient over two regions to calculate a total curvature
    """
    grad1 = transmission_grad(wav, soundings, 300, 387.5)
    grad2 = transmission_grad(wav, soundings, 387.5, 475)
    grad3 = transmission_grad(wav, soundings, 475, 562.5)
    grad4 = transmission_grad(wav, soundings, 562.5, 650)
    curv1 = grad2 - grad1
    curv2 = grad4 - grad3
    curv3 = grad3 - grad2
    curv = np.hstack([curv1, curv2, curv3])
    return curv

def get_mask(*args):
    """
    Takes a list of contrainsts specifying a region in feature space and returns soundings satisfying those constraints
    """
    mn = args[0][1]
    mx = args[0][2]
    feature = args[0][0]
    mask = (mn < feature) & (feature < mx)
    
    if len(args) == 1:
        return mask
    
    for op, arg in zip(args[1::2], args[2::2]):
        mn = arg[1]
        mx = arg[2]
        feature = arg[0]
        if op == "and":
            mask = mask & (mn < feature) & (feature < mx)
        elif op == "or":
            mask = mask | ((mn < feature) & (feature < mx))
        
    return mask

def standardize(feature):
    """
    Standardise the feature space by dividing by standard deviation
    """
    return feature/feature.std(axis=0)

def normalize(feature):
    """
    Normalise the feature space by dividing by the maximum value for every feature
    """
    return feature/abs(feature).max(axis=0)

def create_features(wav, soundings, scale, *args):
    """
    Create a set of features from given soundings
    """
    return_list = []

    if "std" in args:
        std = np.reshape(soundings.std(axis=1), (-1,1))
        std = scale(std)
        return_list.append(std)

    if "fourier" in args:
        fourier = fourier_extract(wav, soundings, 2)
        fourier = scale(fourier)
        return_list.append(fourier)

    if "grad" in args:
        grad1 = transmission_grad(wav, soundings, 300, 387.5)
        grad2 = transmission_grad(wav, soundings, 387.5, 475)
        grad3 = transmission_grad(wav, soundings, 475, 562.5)
        grad4 = transmission_grad(wav, soundings, 562.5, 650)
        grad5 = transmission_grad(wav, soundings, 300, 500)
        grad = np.hstack([grad1, grad2, grad3, grad4, grad5])
        grad = scale(grad)
        return_list.append(grad)
        np.savetxt("../data/grad.txt", grad)

    if "coh" in args:
        coh = phase_coherence(soundings)
        coh = scale(coh)
        return_list.append(coh)

    if "polyres" in args:
        polyres1 = polyfit_residual(wav, soundings, 1)
        polyres2 = polyfit_residual(wav, soundings, 2)
        polyres3 = polyfit_residual(wav, soundings, 3)
        polyres = np.hstack([polyres1, polyres2, polyres3])
        polyres = scale(polyres)
        return_list.append(polyres)
        np.savetxt("../data/polyres.txt", polyres)

    if "optc" in args:
        optc = optimum_counter(soundings, 30)
        optc = scale(optc)
        return_list.append(optc)

    if "mv_av_res" in args:
        mv_av_res = moving_average_residual(soundings, ceil(soundings.shape[1]/4))
        mv_av_res = scale(mv_av_res)
        return_list.append(mv_av_res)

    if "poly_coeffs" in args:
        poly_coeffs = np.polyfit(wav, soundings.T, 3).T[:,:-1] # last coefficient is zeroth order coefficient
        poly_coeffs = scale(poly_coeffs)
        return_list.append(poly_coeffs)
        np.savetxt("../data/poly_coeffs.txt", poly_coeffs)

    if "curv" in args:
        curv = curvature(wav, soundings)
        curv = scale(curv)
        return_list.append(curv)
        np.savetxt("../data/curv.txt", curv)

    if "gradf" in args:
        gradf = gradient_features(soundings)
        gradf = scale(gradf)
        return_list.append(gradf)

    if "max_grad" in args:
        dsoundings = np.gradient(soundings, axis=1, edge_order=2)
        idx = abs(dsoundings).argmax(axis=1)
        max_grad = dsoundings[np.arange(dsoundings.shape[0]),idx]
        max_grad = scale(max_grad)
        return_list.append(max_grad)

    if "avg_grad" in args:
        avg_grad = np.gradient(soundings, axis=1, edge_order=2).mean(axis=1)
        avg_grad = scale(avg_grad)
        return_list.append(avg_grad)

    if "max_curv" in args:
        dsoundings = np.gradient(soundings, axis=1, edge_order=2)
        ddsoundings = np.gradient(dsoundings, axis=1, edge_order=2)
        idx = abs(ddsoundings).argmax(axis=1)
        max_curv = ddsoundings[np.arange(ddsoundings.shape[0]),idx]
        max_curv = scale(max_curv)
        return_list.append(max_curv)

    if "avg_curv" in args:
        dsoundings = np.gradient(soundings, axis=1, edge_order=2)
        avg_curv = np.gradient(dsoundings, axis=1, edge_order=2).mean(axis=1)
        avg_curv = scale(avg_curv)
        return_list.append(avg_curv)

    return tuple(return_list)
    
def centre(soundings):
    return soundings - np.reshape(soundings.mean(axis=1), (-1,1))

def dropout(wav, soundings, step):

    return wav[::step], soundings[:,::step]