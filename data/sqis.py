import numpy as np
import biosppy
import pyhrv
import matplotlib.pyplot as plt
import scipy.stats
import neurokit2 as nk


def orphanidou2015_sqi(ecg_window, sampling_rate, show=False):
    """C. Orphanidou, T. Bonnici, P. Charlton, D. Clifton, D. Vallance and L. Tarassenko, 
    "Signal quality indices for the electrocardiogram and photoplethysmogram: Derivation and applications to wireless monitoring", 
    IEEE J. Biomed. Health Informat., vol. 19, no. 3, pp. 832-838, May 2015.

    ecg_window = input ECG as a 1d numpy array
    sampling_rate = the hz of the input ECG signal
    show = a boolean value whether to show the obtained peaks

    returns average correlation coefficient (scipy.stats.pearsonr)
        that range from -1 to 1
    """

    ## names = ('ts', 'filtered', 'rpeaks', 'templates_ts', 'templates',
    #          'heart_rate_ts', 'heart_rate')

    try:
        out = biosppy.signals.ecg.ecg(signal=ecg_window, sampling_rate=sampling_rate, show=show)
    except Exception as e:
        return np.nan

    nni = pyhrv.tools.nn_intervals(rpeaks=out['rpeaks'])
    # nni is in ms, convert to s
    nni = nni / 1000

    # obtain median rr interval
    median_qrs_window = np.median(out['rpeaks'][1:] - out['rpeaks'][:-1]).astype(int)

    # check heart rate in reasonable range of [40,180]
    if np.any(out['heart_rate'] < 40) or np.any(180 < out['heart_rate']):
        return 1

    # if all nni are less than 3 seconds
    if np.any(nni > 3):
        return 1

    # check max_rr_interval / min_rr_interval < 2.2
    if (np.max(nni) / np.min(nni)) > 2.2:
        return 1

    templates = np.array([
        ecg_window[r_peak-median_qrs_window//2:r_peak+median_qrs_window//2] 
        for r_peak in out['rpeaks']
        if (r_peak-median_qrs_window//2 >= 0) and (r_peak+median_qrs_window//2 < len(ecg_window))
    ])

    average_template = np.mean(templates, axis=0)

    # scipy.stats.pearsonr returns r, p_value
    corrcoefs = [
        scipy.stats.pearsonr(x=templates[i], y=average_template)[0]
        for i in range(len(templates))
        ]

    return np.mean(corrcoefs)

def k_SQI(ecg_cleaned, kurtosis_method='fisher'):
    """Return the kurtosis of the signal, with Fisher's or Pearson's method.
    ecg_cleaned : np.array
        The cleaned ECG signal in the form of a vector of values.
    kurtosis_method : str
        Compute kurtosis (kSQI) based on "fisher" (default) or "pearson" definition.
    """
    if kurtosis_method == "fisher":
        return scipy.stats.kurtosis(ecg_cleaned, fisher=True)
    elif kurtosis_method == "pearson":
        return scipy.stats.kurtosis(ecg_cleaned, fisher=False)

def p_SQI(ecg_cleaned, sampling_rate, window, num_spectrum=[5, 15], dem_spectrum=[5, 40]):
    """Power Spectrum Distribution of QRS Wave.
    ecg_cleaned : np.array
        The cleaned ECG signal in the form of a vector of values.
    sampling_rate : int
        The sampling frequency of the signal (in Hz, i.e., samples/second).
    window : int
        Length of each window in seconds. See `signal_psd()`.
    """
    try:
        psd = nk.signal_power(
            ecg_cleaned,
            sampling_rate=sampling_rate,
            frequency_band=[num_spectrum, dem_spectrum],
            method="welch",
            normalize=False,
            window=window
            )

        num_power = psd.iloc[0][0]
        dem_power = psd.iloc[0][1]

        return num_power / dem_power
    except Exception as e:
        return np.nan

def bas_SQI(ecg_cleaned, sampling_rate, window, num_spectrum=[0, 1], dem_spectrum=[0, 40]):
    """Relative Power in the Baseline.
    ecg_cleaned : np.array
        The cleaned ECG signal in the form of a vector of values.
    sampling_rate : int
        The sampling frequency of the signal (in Hz, i.e., samples/second).
    window : int
        Length of each window in seconds. See `signal_psd()`.
    """

    try:
        psd = nk.signal_power(
            ecg_cleaned,
            sampling_rate=sampling_rate,
            frequency_band=[num_spectrum, dem_spectrum],
            method="welch",
            normalize=False,
            window=window
            )

        num_power = psd.iloc[0][0]
        dem_power = psd.iloc[0][1]

        return 1 - num_power / dem_power
    except Exception as e:
        return np.nan

def averageQRS_SQI(ecg_cleaned, sampling_rate):
    try:
        rating = nk.ecg_quality(ecg_cleaned=ecg_cleaned, rpeaks=None, sampling_rate=sampling_rate, method="averageQRS")

        if rating == "Excellent":
            return 2
        elif rating == "Unnacceptable":
            return 0
        else:
            return 1

    except Exception as e:
        # print(e)
        return np.nan

def zhao2018_SQI(ecg_cleaned, sampling_rate):
    try:
        rating = nk.ecg_quality(ecg_cleaned=ecg_cleaned, rpeaks=None, sampling_rate=sampling_rate, method="zhao2018", approach='fuzzy')
        if rating == "Excellent":
            return 2
        elif rating == "Unnacceptable":
            return 0
        else:
            return 1
    except Exception as e:
        # print(e)
        return np.nan
    
    