#!/usr/bin/env python3
from __future__ import print_function

import time
import numpy as np
import scipy.signal
from scipy.interpolate import UnivariateSpline

# ----------------------------------------------------------------------------
# ----------------------------------------------------------------------------

# Original from: https://nbviewer.jupyter.org/github/demotu/BMC/blob/master/notebooks/DetectPeaks.ipynb
def detect_peaks(x, mph=None, mpd=1, threshold=0, edge='rising',
                 kpsh=False, valley=False, show=False, save=None, ax=None,
                 xdata=None, xlabel='Data #', ylabel='Amplitude'):
    """Detect peaks in data based on their amplitude and other features.

    Parameters
    ----------
    x : 1D array_like
        data.
    mph : {None, number}, optional (default = None)
        detect peaks that are greater than minimum peak height (if parameter
        `valley` is False) or peaks that are smaller than maximum peak height
         (if parameter `valley` is True).
    mpd : positive integer, optional (default = 1)
        detect peaks that are at least separated by minimum peak distance (in
        number of data).
    threshold : positive number, optional (default = 0)
        detect peaks (valleys) that are greater (smaller) than `threshold`
        in relation to their immediate neighbors.
    edge : {None, 'rising', 'falling', 'both'}, optional (default = 'rising')
        for a flat peak, keep only the rising edge ('rising'), only the
        falling edge ('falling'), both edges ('both'), or don't detect a
        flat peak (None).
    kpsh : bool, optional (default = False)
        keep peaks with same height even if they are closer than `mpd`.
    valley : bool, optional (default = False)
        if True (1), detect valleys (local minima) instead of peaks.
    show : bool, optional (default = False)
        if True (1), plot data in matplotlib figure.
    ax : a matplotlib.axes.Axes instance, optional (default = None).

    Returns
    -------
    ind : 1D array_like
        indices of the peaks in `x`.

    Notes
    -----
    The detection of valleys instead of peaks is performed internally by simply
    negating the data: `ind_valleys = detect_peaks(-x)`

    The function can handle NaN's

    See this IPython Notebook [1]_.

    References
    ----------
    .. [1] http://nbviewer.ipython.org/github/demotu/BMC/blob/master/notebooks/DetectPeaks.ipynb

    Examples
    --------
    >>> from detect_peaks import detect_peaks
    >>> x = np.random.randn(100)
    >>> x[60:81] = np.nan
    >>> # detect all peaks and plot data
    >>> ind = detect_peaks(x, show=True)
    >>> print(ind)

    >>> x = np.sin(2*np.pi*5*np.linspace(0, 1, 200)) + np.random.randn(200)/5
    >>> # set minimum peak height = 0 and minimum peak distance = 20
    >>> detect_peaks(x, mph=0, mpd=20, show=True)

    >>> x = [0, 1, 0, 2, 0, 3, 0, 2, 0, 1, 0]
    >>> # set minimum peak distance = 2
    >>> detect_peaks(x, mpd=2, show=True)

    >>> x = np.sin(2*np.pi*5*np.linspace(0, 1, 200)) + np.random.randn(200)/5
    >>> # detection of valleys instead of peaks
    >>> detect_peaks(x, mph=-1.2, mpd=20, valley=True, show=True)

    >>> x = [0, 1, 1, 0, 1, 1, 0]
    >>> # detect both edges
    >>> detect_peaks(x, edge='both', show=True)

    >>> x = [-2, 1, -2, 2, 1, 1, 3, 0]
    >>> # set threshold = 2
    >>> detect_peaks(x, threshold = 2, show=True)

    Version history
    ---------------
    '1.0.5':
        The sign of `mph` is inverted if parameter `valley` is True

    """
    x = np.atleast_1d(x).astype('float64')
    if x.size < 3: return np.array([], dtype=int)
    if valley:
        x = -x
        if mph is not None: mph = -mph
    # find indices of all peaks
    dx = x[1:] - x[:-1]
    # handle NaN's
    indnan = np.where(np.isnan(x))[0]
    if indnan.size:
        x[indnan] = np.inf
        dx[np.where(np.isnan(dx))[0]] = np.inf
    ine, ire, ife = np.array([[], [], []], dtype=int)
    if not edge:
        ine = np.where((np.hstack((dx, 0)) < 0) & (np.hstack((0, dx)) > 0))[0]
    else:
        if edge.lower() in ['rising', 'both']: ire = np.where((np.hstack((dx, 0)) <= 0) & (np.hstack((0, dx)) > 0))[0]
        if edge.lower() in ['falling', 'both']: ife = np.where((np.hstack((dx, 0)) < 0) & (np.hstack((0, dx)) >= 0))[0]
    ind = np.unique(np.hstack((ine, ire, ife)))
    # handle NaN's
    if ind.size and indnan.size:
        # NaN's and values close to NaN's cannot be peaks
        ind = ind[np.in1d(ind, np.unique(np.hstack((indnan, indnan-1, indnan+1))), invert=True)]
    # first and last values of x cannot be peaks
    if ind.size and ind[0] == 0: ind = ind[1:]
    if ind.size and ind[-1] == x.size-1: ind = ind[:-1]
    # remove peaks < minimum peak height
    if ind.size and mph is not None: ind = ind[x[ind] >= mph]
    # remove peaks - neighbors < threshold
    if ind.size and threshold > 0:
        dx = np.min(np.vstack([x[ind]-x[ind-1], x[ind]-x[ind+1]]), axis=0)
        ind = np.delete(ind, np.where(dx < threshold)[0])
    # detect small peaks closer than minimum peak distance
    if ind.size and mpd > 1:
        ind = ind[np.argsort(x[ind])][::-1]  # sort ind by peak height
        idel = np.zeros(ind.size, dtype=bool)
        for i in range(ind.size):
            if not idel[i]:
                # keep peaks with the same height if kpsh is True
                idel = idel | (ind >= ind[i] - mpd) & (ind <= ind[i] + mpd) & (x[ind[i]] > x[ind] if kpsh else True)
                idel[i] = 0  # Keep current peak
        # remove the small peaks and sort back the indices by their occurrence
        ind = np.sort(ind[~idel])
    if show or save:
        if indnan.size: x[indnan] = np.nan
        if valley:
            x = -x
            if mph is not None: mph = -mph
        _plot(x, mph, mpd, threshold, edge, valley, ax, ind, save, xdata, xlabel, ylabel)

    return ind

def _plot(x, mph, mpd, threshold, edge, valley, ax, ind, save, xdata, xlabel, ylabel):
    """Plot results of the detect_peaks function, see its help."""
    try: import matplotlib.pyplot as plt
    except ImportError: print('matplotlib is not available.')
    else:
        plt.style.use('default')
        if ax is None: _, ax = plt.subplots(1, 1, figsize=(18, 8))

        if xdata[0] > xdata[-1]: plt.gca().invert_xaxis()
        ax.plot(xdata, x, 'b', lw=1)
        if ind.size:
            label = 'valley' if valley else 'peak'
            if ind.size > 1: label += 's'
            ax.plot(xdata[ind], x[ind], '+', mfc=None, mec='r', mew=2, ms=8, label='%d %s' % (ind.size, label))
            ax.legend(loc='best', framealpha=.5, numpoints=1)
        if abs(xdata[1] - xdata[0]) > 150.: ax.set_xlim(xdata[0], xdata[-1])
        ymin, ymax = x[np.isfinite(x)].min(), x[np.isfinite(x)].max()
        yrange = ymax - ymin if ymax > ymin else 1
        ax.set_ylim(ymin - 0.1*yrange, ymax + 0.1*yrange)
        # ax.set_ylim(-0.05, 1.05)

        ax.set_xlabel(xlabel, fontsize=14)
        ax.set_ylabel(ylabel, fontsize=14)
        mode = 'Valley detection' if valley else 'Peak detection'
        ax.set_title("%s (mph=%s, mpd=%d, threshold=%s, edge='%s')"
                     % (mode, str(mph), mpd, str(threshold), edge))

        plt.grid(color='gray', alpha=0.4, linestyle='--', linewidth=0.5) # '#3D3D29'
        plt.tight_layout()
        plt.savefig(save) if save else plt.show()
        # plt.clf()
        plt.close()

# ----------------------------------------------------------------------------
# ----------------------------------------------------------------------------

def get_peak_width(data, x=None, fraction=2, smoothing=0, roots=False):
    if x == None: x = range(len(data))
    spline = UnivariateSpline(x, np.asarray(data)-np.max(data)/fraction, s=smoothing)
    print(spline)
    print(spline.roots())
    the_roots = spline.roots()
    r1, r2 = the_roots[:2]
    if roots: return r1, r2
    return r2 - r1

def GetSpline(data, xdata=None, smoothing=2):
    if xdata == None: xdata = range(len(data))
    spline = UnivariateSpline(xdata, data, s=smoothing)
    spline.set_smoothing_factor(smoothing)
    return spline

# ----------------------------------------------------------------------------
# ----------------------------------------------------------------------------

# Original from: https://gist.github.com/celliern/aff85af8ce527a6963ed8f01d79fc8a4
def new_find_peaks(y, x=None, slope_thresh=0., amp_thresh=None, medfilt_radius=5,
                   maxpeakn=10000, peakgroup=10, subchannel=True):
    """Find peaks along a 1D line.
    Function to locate the positive peaks in a noisy x-y data set.
    Detects peaks by looking for downward zero-crossings in the first derivative
    that exceed 'slope_thresh'.
    Returns an array containing position, height, and width of each peak.
    Sorted by position.
    'slope_thresh' and 'amp_thresh', control sensitivity: higher values will
    neglect smaller features.

    Parameters
    ---------
    y : array
        1D input array, e.g. a spectrum
    x : array (optional)
        1D array describing the calibration of y (must have same shape as y)
    slope_thresh : float (optional)
            1st derivative threshold to count the peak default is set to 0.5
            higher values will neglect smaller features.
    amp_thresh : float (optional)
            intensity threshold above which default is set to 10% of max(y)
            higher values will neglect smaller features.
    medfilt_radius : int (optional)
            median filter window to apply to smooth the data (see scipy.signal.medfilt)
            if 0, no filter will be applied. default is set to 5
    peakgroup : int (optional)
            number of points around the "top part" of the peak
            default is set to 10
    maxpeakn : int (optional)
            number of maximum detectable peaks default is set to 10000
    subchannel : bool (optional)
            default is set to True
    Returns
    -------
    P : structured array of shape (npeaks) and fields: position, width, height
        contains position, height, and width of each peak
    Examples
    --------
    >>> x = np.arange(0,50,0.01)
    >>> y = np.cos(x)
    >>> peaks = new_find_peaks(y, x, 0, 0)
    Notes
    -----
    Original code from T. C. O'Haver, 1995.
    Version 2 Last revised Oct 27, 2006. Converted to Python by Michael Sarahan, Feb 2011.
    Revised to handle edges better. MCS, Mar 2011
    """

    if x is None: x = np.arange(len(y), dtype=np.int64)
    if not amp_thresh: amp_thresh = 0.1 * y.max()
    peakgroup = np.round(peakgroup)
    if medfilt_radius: d = np.gradient(scipy.signal.medfilt(y, medfilt_radius))
    else: d = np.gradient(y)

    peak    = 0
    n       = np.round(peakgroup / 2 + 1)
    peak_dt = np.dtype([('position', np.float), ('height', np.float), ('width', np.float)])
    P       = np.array([], dtype=peak_dt)

    for j in range(len(y) - 4):
        if np.sign(d[j]) > np.sign(d[j + 1]): # Detects zero-crossing
            if np.sign(d[j + 1]) == 0: continue
            # if slope of derivative is larger than slope_thresh
            if d[j] - d[j + 1] > slope_thresh:
                # if height of peak is larger than amp_thresh
                if y[j] > amp_thresh:
                    # the next section is very slow, and actually messes things up for images (discrete
                    # pixels), so by default, don't do subchannel precision in the 1D peakfind step.
                    if subchannel:
                        s = 0
                        xx, yy = np.zeros(peakgroup), np.zeros(peakgroup)
                        for k in range(peakgroup):
                            groupindex = int(j + k - n + 1)
                            if groupindex < 1:
                                xx = xx[1:]
                                yy = yy[1:]
                                s += 1
                                continue
                            elif groupindex > y.shape[0] - 1:
                                xx = xx[:groupindex - 1]
                                yy = yy[:groupindex - 1]
                                break
                            xx[k - s] = x[groupindex]
                            yy[k - s] = y[groupindex]
                        avg   = np.average(xx)
                        stdev = np.std(xx)
                        xxf   = (xx - avg) / stdev
                        # Fit parabola to log10 of sub-group with centering and scaling
                        yynz  = yy != 0
                        coef  = np.polyfit(xxf[yynz], np.log10(np.abs(yy[yynz])), 2)
                        c1 = coef[2]
                        c2 = coef[1]
                        c3 = coef[0]
                        with np.errstate(invalid='ignore'):
                            width = np.linalg.norm(stdev * 2.35703 / (np.sqrt(2) * np.sqrt(-1 * c3)))
                        # if the peak is too narrow for least-squares technique to work well,
                        # just use the max value of y in the sub-group of points near peak.
                        if peakgroup < 7:
                            height   = np.max(yy)
                            position = xx[np.argmin(np.abs(yy - height))]
                        else:
                            position = -((stdev * c2 / (2 * c3)) - avg)
                            height   = np.exp(c1 - c3 * (c2 / (2 * c3)) ** 2)
                    # Fill results array P. One row for each peak detected, containing
                    # the peak position (x-value) and peak height (y-value)
                    else:
                        position = x[j]
                        height   = y[j]
                        # no way to know peak width without the above measurements.
                        width = 0
                    if (not np.isnan(position) and 0 < position < x[-1]):
                        P     = np.hstack((P, np.array([(position, height, width)], dtype=peak_dt)))
                        peak += 1
    # return only the part of the array that contains peaks
    # (not the whole maxpeakn x 3 array)
    if len(P) > maxpeakn:
        minh = np.sort(P['height'])[-maxpeakn]
        P = P[P['height'] >= minh]

    # Sorts the values as a function of position
    P.sort(0)

    return P

# ----------------------------------------------------------------------------
# ----------------------------------------------------------------------------

# Original from: https://gist.github.com/celliern/aff85af8ce527a6963ed8f01d79fc8a4
def original_find_peaks(y, x=None, slope_thresh=0., amp_thresh=None, medfilt_radius=5,
                        maxpeakn=30000, peakgroup=10, subchannel=True):
    """Find peaks along a 1D line.
    Function to locate the positive peaks in a noisy x-y data set.
    Detects peaks by looking for downward zero-crossings in the first
    derivative that exceed 'slope_thresh'.
    Returns an array containing position, height, and width of each peak.
    Sorted by position.
    'slope_thresh' and 'amp_thresh', control sensitivity: higher values will
    neglect smaller features.

    Parameters
    ---------
    y : array
        1D input array, e.g. a spectrum
    x : array (optional)
        1D array describing the calibration of y (must have same shape as y)
    slope_thresh : float (optional)
                   1st derivative threshold to count the peak
                   default is set to 0.5
                   higher values will neglect smaller features.
    amp_thresh : float (optional)
                 intensity threshold above which
                 default is set to 10% of max(y)
                 higher values will neglect smaller features.
    medfilt_radius : int (optional)
                     median filter window to apply to smooth the data
                     (see scipy.signal.medfilt)
                     if 0, no filter will be applied.
                     default is set to 5
    peakgroup : int (optional)
                number of points around the "top part" of the peak
                default is set to 10
    maxpeakn : int (optional)
              number of maximum detectable peaks
              default is set to 30000
    subchannel : bool (optional)
             default is set to True
    Returns
    -------
    P : structured array of shape (npeaks) and fields: position, width, height
        contains position, height, and width of each peak
    Examples
    --------
    >>> x = np.arange(0,50,0.01)
    >>> y = np.cos(x)
    >>> peaks = original_find_peaks(y, x, 0, 0)
    Notes
    -----
    Original code from T. C. O'Haver, 1995.
    Version 2  Last revised Oct 27, 2006 Converted to Python by
    Michael Sarahan, Feb 2011.
    Revised to handle edges better.  MCS, Mar 2011
    """

    if x is None:
        x = np.arange(len(y), dtype=np.int64)
    if not amp_thresh:
        amp_thresh = 0.1 * y.max()
    peakgroup = np.round(peakgroup)
    if medfilt_radius:
        d = np.gradient(scipy.signal.medfilt(y, medfilt_radius))
    else:
        d = np.gradient(y)
    n = np.round(peakgroup / 2 + 1)
    peak_dt = np.dtype([('position', np.float),
                        ('height', np.float),
                        ('width', np.float)])
    P = np.array([], dtype=peak_dt)
    peak = 0
    for j in range(len(y) - 4):
        if np.sign(d[j]) > np.sign(d[j + 1]):  # Detects zero-crossing
            if np.sign(d[j + 1]) == 0:
                continue
            # if slope of derivative is larger than slope_thresh
            if d[j] - d[j + 1] > slope_thresh:
                # if height of peak is larger than amp_thresh
                if y[j] > amp_thresh:
                    # the next section is very slow, and actually messes
                    # things up for images (discrete pixels),
                    # so by default, don't do subchannel precision in the
                    # 1D peakfind step.
                    if subchannel:
                        xx = np.zeros(peakgroup)
                        yy = np.zeros(peakgroup)
                        s = 0
                        for k in range(peakgroup):
                            groupindex = int(j + k - n + 1)
                            if groupindex < 1:
                                xx = xx[1:]
                                yy = yy[1:]
                                s += 1
                                continue
                            elif groupindex > y.shape[0] - 1:
                                xx = xx[:groupindex - 1]
                                yy = yy[:groupindex - 1]
                                break
                            xx[k - s] = x[groupindex]
                            yy[k - s] = y[groupindex]
                        avg = np.average(xx)
                        stdev = np.std(xx)
                        xxf = (xx - avg) / stdev
                        # Fit parabola to log10 of sub-group with
                        # centering and scaling
                        yynz = yy != 0
                        coef = np.polyfit(
                            xxf[yynz], np.log10(np.abs(yy[yynz])), 2)
                        c1 = coef[2]
                        c2 = coef[1]
                        c3 = coef[0]
                        with np.errstate(invalid='ignore'):
                            width = np.linalg.norm(stdev * 2.35703 /
                                                   (np.sqrt(2) * np.sqrt(-1 *
                                                                         c3)))
                        # if the peak is too narrow for least-squares
                        # technique to work  well, just use the max value
                        # of y in the sub-group of points near peak.
                        if peakgroup < 7:
                            height = np.max(yy)
                            position = xx[np.argmin(np.abs(yy - height))]
                        else:
                            position = - ((stdev * c2 / (2 * c3)) - avg)
                            height = np.exp(c1 - c3 * (c2 / (2 * c3)) ** 2)
                    # Fill results array P. One row for each peak
                    # detected, containing the
                    # peak position (x-value) and peak height (y-value).
                    else:
                        position = x[j]
                        height = y[j]
                        # no way to know peak width without
                        # the above measurements.
                        width = 0
                    if (not np.isnan(position) and 0 < position < x[-1]):
                        P = np.hstack((P,
                                       np.array([(position, height, width)],
                                                dtype=peak_dt)))
                        peak += 1
    # return only the part of the array that contains peaks
    # (not the whole maxpeakn x 3 array)
    if len(P) > maxpeakn:
        minh = np.sort(P['height'])[-maxpeakn]
        P = P[P['height'] >= minh]

    # Sorts the values as a function of position
    P.sort(0)

    return P

# ----------------------------------------------------------------------------
# ----------------------------------------------------------------------------
