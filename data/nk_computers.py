'''
    Slimmed versions copied and culled from neurokit source
'''


# -*- coding: utf-8 -*-
import pandas as pd
from neurokit2 import signal_sanitize, ecg_peaks, as_vector, signal_interpolate, signal_power, signal_fixpeaks, signal_formatpeaks, signal_zerocrossings, find_consecutive, signal_rate
from neurokit2.complexity import entropy_approximate, entropy_shannon, entropy_sample
import numpy as np
import scipy

from numba import njit

def ecg_process(ecg_signal, sampling_rate=1000, method="neurokit"):
    """**Automated pipeline for preprocessing an ECG signal**
    This function runs different preprocessing steps. **Help us improve the documentation of
    this function by making it more tidy and useful!**
    Parameters
    ----------
    ecg_signal : Union[list, np.array, pd.Series]
        The raw ECG channel.
    sampling_rate : int
        The sampling frequency of ``ecg_signal`` (in Hz, i.e., samples/second). Defaults to 1000.
    method : str
        The processing pipeline to apply. Defaults to ``"neurokit"``.
    Returns
    -------
    signals : DataFrame
        A DataFrame of the same length as the ``ecg_signal`` containing the following columns:
        * ``"ECG_Raw"``: the raw signal.
        * ``"ECG_Clean"``: the cleaned signal.
        * ``"ECG_R_Peaks"``: the R-peaks marked as "1" in a list of zeros.
        * ``"ECG_Rate"``: heart rate interpolated between R-peaks.
        * ``"ECG_P_Peaks"``: the P-peaks marked as "1" in a list of zeros
        * ``"ECG_Q_Peaks"``: the Q-peaks marked as "1" in a list of zeros .
        * ``"ECG_S_Peaks"``: the S-peaks marked as "1" in a list of zeros.
        * ``"ECG_T_Peaks"``: the T-peaks marked as "1" in a list of zeros.
        * ``"ECG_P_Onsets"``: the P-onsets marked as "1" in a list of zeros.
        * ``"ECG_P_Offsets"``: the P-offsets marked as "1" in a list of zeros (only when method in
          ``ecg_delineate()`` is wavelet).
        * ``"ECG_T_Onsets"``: the T-onsets marked as "1" in a list of zeros (only when method in
          ``ecg_delineate()`` is wavelet).
        * ``"ECG_T_Offsets"``: the T-offsets marked as "1" in a list of zeros.
        * ``"ECG_R_Onsets"``: the R-onsets marked as "1" in a list of zeros (only when method in
          ``ecg_delineate()`` is wavelet).
        * ``"ECG_R_Offsets"``: the R-offsets marked as "1" in a list of zeros (only when method in
          ``ecg_delineate()`` is wavelet).
        * ``"ECG_Phase_Atrial"``: cardiac phase, marked by "1" for systole and "0" for diastole.
        * ``"ECG_Phase_Ventricular"``: cardiac phase, marked by "1" for systole and "0" for
          diastole.
        * ``"ECG_Atrial_PhaseCompletion"``: cardiac phase (atrial) completion, expressed in
          percentage
          (from 0 to 1), representing the stage of the current cardiac phase.
        * ``"ECG_Ventricular_PhaseCompletion"``: cardiac phase (ventricular) completion, expressed
          in percentage (from 0 to 1), representing the stage of the current cardiac phase.
        * **This list is not up-to-date. Help us improve the documentation!**
    info : dict
        A dictionary containing the samples at which the R-peaks occur, accessible with the key
        ``"ECG_Peaks"``, as well as the signals' sampling rate.
    See Also
    --------
    ecg_clean, ecg_peaks, ecg_delineate, ecg_phase, ecg_plot, .signal_rate
    Examples
    --------
    .. ipython:: python
      import neurokit2 as nk
      # Simulate ECG signal
      ecg = nk.ecg_simulate(duration=15, sampling_rate=1000, heart_rate=80)
      # Preprocess ECG signal
      signals, info = nk.ecg_process(ecg, sampling_rate=1000)
      # Visualize
      @savefig p_ecg_process.png scale=100%
      nk.ecg_plot(signals)
      @suppress
      plt.close()
    """
    # Sanitize input
    ecg_signal = signal_sanitize(ecg_signal)

    ecg_cleaned = ecg_clean(ecg_signal, sampling_rate=sampling_rate, method=method)
    # R-peaks
    instant_peaks, rpeaks, = ecg_peaks(
        ecg_cleaned=ecg_cleaned, sampling_rate=sampling_rate, method=method, correct_artifacts=True
    )

    rate = signal_rate(rpeaks, sampling_rate=sampling_rate, desired_length=len(ecg_cleaned))

    # quality = ecg_quality(ecg_cleaned, rpeaks=None, sampling_rate=sampling_rate)

    signals = pd.DataFrame(
        {"ECG_Raw": ecg_signal, "ECG_Clean": ecg_cleaned, "ECG_Rate": rate}
        #, "ECG_Rate": rate, "ECG_Quality": quality}
    )

    # # Additional info of the ecg signal
    # delineate_signal, delineate_info = ecg_delineate(
    #     ecg_cleaned=ecg_cleaned, rpeaks=rpeaks, sampling_rate=sampling_rate
    # )

    # cardiac_phase = ecg_phase(ecg_cleaned=ecg_cleaned, rpeaks=rpeaks, delineate_info=delineate_info)

    signals = pd.concat([signals, instant_peaks], axis=1)
    #, delineate_signal, cardiac_phase], axis=1)

    # Rpeaks location and sampling rate in dict info
    info = rpeaks
    info["sampling_rate"] = sampling_rate

    return signals, info

def ecg_analyze(data, sampling_rate=1000, method="auto", subepoch_rate=[None, None], withhrv=True):
    method = method.lower()
    if method in ["auto"]:

        if isinstance(data, dict):
            for i in data:
                duration = len(data[i]) / sampling_rate
            # if duration >= 10:
            features = ecg_intervalrelated(data, sampling_rate=sampling_rate)
            # else:
            #     # features = ecg_eventrelated(data, subepoch_rate=subepoch_rate)
            #     pass

        if isinstance(data, pd.DataFrame):
            # if duration >= 10:
            features = ecg_intervalrelated(data, sampling_rate=sampling_rate, withhrv=withhrv)
            # else: # features = ecg_eventrelated(data, subepoch_rate=subepoch_rate)
            #     pass

    return features

def ecg_intervalrelated(data, sampling_rate=1000, withhrv=True):
    intervals = {}

    # Format input
    if isinstance(data, pd.DataFrame):
        rate_cols = [col for col in data.columns if "ECG_Rate" in col]
        if len(rate_cols) == 1:
            intervals.update(_ecg_intervalrelated_formatinput(data))
            intervals.update(_ecg_intervalrelated_hrv(data, sampling_rate, withhrv))
        else:
            raise ValueError(
                "NeuroKit error: ecg_intervalrelated(): Wrong input,"
                "we couldn't extract heart rate. Please make sure"
                "your DataFrame contains an `ECG_Rate` column."
            )
        ecg_intervals = pd.DataFrame.from_dict(intervals, orient="index").T

    elif isinstance(data, dict):
        for index in data:
            intervals[index] = {}  # Initialize empty container

            # Add label info
            intervals[index]['Label'] = data[index]['Label'].iloc[0]

            # Rate
            intervals[index] = _ecg_intervalrelated_formatinput(data[index], intervals[index])

            # HRV
            intervals[index] = _ecg_intervalrelated_hrv(data[index], sampling_rate, intervals[index])

        ecg_intervals = pd.DataFrame.from_dict(intervals, orient="index")

    return ecg_intervals

def _ecg_intervalrelated_formatinput(data, output={}):

    # Sanitize input
    colnames = data.columns.values
    if len([i for i in colnames if "ECG_Rate" in i]) == 0:
        raise ValueError(
            "NeuroKit error: ecg_intervalrelated(): Wrong input,"
            "we couldn't extract heart rate. Please make sure"
            "your DataFrame contains an `ECG_Rate` column."
        )
    signal = data["ECG_Rate"].values
    output["ECG_Rate_Mean"] = np.mean(signal)

    return output

def _ecg_intervalrelated_hrv(data, sampling_rate, withhrv, output={}):
    # Sanitize input
    colnames = data.columns.values
    if len([i for i in colnames if "ECG_R_Peaks" in i]) == 0:
        raise ValueError(
            "NeuroKit error: ecg_intervalrelated(): Wrong input,"
            "we couldn't extract R-peaks. Please make sure"
            "your DataFrame contains an `ECG_R_Peaks` column."
        )

    # Transform rpeaks from "signal" format to "info" format.
    rpeaks = np.where(data["ECG_R_Peaks"].values)[0]
    rpeaks = {"ECG_R_Peaks": rpeaks}
    results = hrv(rpeaks, sampling_rate=sampling_rate, withhrv=withhrv)
    for column in results.columns:
        output[column] = float(results[column])

    return output

def hrv(peaks, sampling_rate=1000, withhrv=True, **kwargs):
    # Get indices
    out = []  # initialize empty container

    # Gather indices
    out.append(hrv_time(peaks, sampling_rate=sampling_rate))
    if (withhrv):
        out.append(hrv_frequency(peaks, sampling_rate=sampling_rate))
        out.append(hrv_nonlinear(peaks, sampling_rate=sampling_rate))

    # # Compute RSA if rsp data is available
    # if isinstance(peaks, pd.DataFrame):
    #     rsp_cols = [col for col in peaks.columns if "RSP_Phase" in col]
    #     if len(rsp_cols) == 2:
    #         rsp_signals = peaks[rsp_cols]
    #         rsa = hrv_rsa(peaks, rsp_signals, sampling_rate=sampling_rate)
    #         out.append(pd.DataFrame([rsa]))

    out = pd.concat(out, axis=1)


    return out


def ecg_clean(ecg_signal, sampling_rate=1000, method="neurokit"):
    ecg_signal = as_vector(ecg_signal)

    # Missing data
    n_missing = np.sum(np.isnan(ecg_signal))
    if n_missing > 0:
        ecg_signal = _ecg_clean_missing(ecg_signal)

    method = method.lower()  # remove capitalised letters
    if method in ["nk", "nk2", "neurokit", "neurokit2"]:
        clean = _ecg_clean_nk(ecg_signal, sampling_rate)
    # elif method in ["biosppy", "gamboa2008"]:
    #     clean = _ecg_clean_biosppy(ecg_signal, sampling_rate)
    # elif method in ["pantompkins", "pantompkins1985"]:
    #     clean = _ecg_clean_pantompkins(ecg_signal, sampling_rate)
    # elif method in ["hamilton", "hamilton2002"]:
    #     clean = _ecg_clean_hamilton(ecg_signal, sampling_rate)
    # elif method in ["elgendi", "elgendi2010"]:
    #     clean = _ecg_clean_elgendi(ecg_signal, sampling_rate)
    # elif method in ["engzee", "engzee2012", "engzeemod", "engzeemod2012"]:
    #     clean = _ecg_clean_engzee(ecg_signal, sampling_rate)
    # elif method in [
    #     "christov",
    #     "christov2004",
    #     "ssf",
    #     "slopesumfunction",
    #     "zong",
    #     "zong2003",
    #     "kalidas2017",
    #     "swt",
    #     "kalidas",
    #     "kalidastamil",
    #     "kalidastamil2017",
    # ]:
    #     clean = ecg_signal
    else:
        raise ValueError(
            "NeuroKit error: ecg_clean(): 'method' should be "
            "one of 'neurokit', 'biosppy', 'pantompkins1985',"
            " 'hamilton2002', 'elgendi2010', 'engzeemod2012'."
        )
    return clean

def _ecg_clean_nk(ecg_signal, sampling_rate=1000):

    # Remove slow drift and dc offset with highpass Butterworth.
    clean = signal_filter(
        signal=ecg_signal, sampling_rate=sampling_rate, lowcut=0.5, method="butterworth", order=5
    )

    clean = signal_filter(
        signal=clean, sampling_rate=sampling_rate, method="powerline", powerline=50
    )
    return clean

def signal_filter(
    signal,
    sampling_rate=1000,
    lowcut=None,
    highcut=None,
    method="butterworth",
    order=2,
    window_size="default",
    powerline=50,
    show=False,
):
    method = method.lower()

    if method in ["powerline"]:
        filtered = _signal_filter_powerline(signal, sampling_rate, powerline)
    else:

        # Sanity checks
        if lowcut is None and highcut is None:
            raise ValueError(
                "NeuroKit error: signal_filter(): you need to specify a 'lowcut' or a 'highcut'."
            )

        if method in ["butter", "butterworth"]:
            filtered = _signal_filter_butterworth(signal, sampling_rate, lowcut, highcut, order)
        else:
            raise ValueError(
                "NeuroKit error: signal_filter(): 'method' should be",
                " one of 'butterworth', 'butterworth_ba', 'bessel',",
                " 'savgol' or 'fir'.",
            )


    return filtered

def _signal_filter_powerline(signal, sampling_rate, powerline=50):
    """Filter out 50 Hz powerline noise by smoothing the signal with a moving average kernel with the width of one
    period of 50Hz."""

    if sampling_rate >= 100:
        b = np.ones(int(sampling_rate / powerline))
    else:
        b = np.ones(2)
    a = [len(b)]
    y = scipy.signal.filtfilt(b, a, signal, method="pad")
    return y

def _signal_filter_butterworth(signal, sampling_rate=1000, lowcut=None, highcut=None, order=5):
    """Filter a signal using IIR Butterworth SOS method."""
    freqs, filter_type = _signal_filter_sanitize(
        lowcut=lowcut, highcut=highcut, sampling_rate=sampling_rate
    )

    sos = scipy.signal.butter(order, freqs, btype=filter_type, output="sos", fs=sampling_rate)
    filtered = scipy.signal.sosfiltfilt(sos, signal)
    return filtered

def _signal_filter_sanitize(lowcut=None, highcut=None, sampling_rate=1000, normalize=False):

    # Sanity checks
    if isinstance(highcut, int):
        if sampling_rate <= 2 * highcut:
            print(
                "The sampling rate is too low. Sampling rate"
                " must exceed the Nyquist rate to avoid aliasing problem."
                f" In this analysis, the sampling rate has to be higher than {2 * highcut} Hz"
            )

    # Replace 0 by none
    if lowcut is not None and lowcut == 0:
        lowcut = None
    if highcut is not None and highcut == 0:
        highcut = None

    # Format
    if lowcut is not None and highcut is not None:
        if lowcut > highcut:
            filter_type = "bandstop"
        else:
            filter_type = "bandpass"
        freqs = [lowcut, highcut]
    elif lowcut is not None:
        freqs = [lowcut]
        filter_type = "highpass"
    elif highcut is not None:
        freqs = [highcut]
        filter_type = "lowpass"

    # Normalize frequency to Nyquist Frequency (Fs/2).
    # However, no need to normalize if `fs` argument is provided to the scipy filter
    if normalize is True:
        freqs = np.array(freqs) / (sampling_rate / 2)

    return freqs, filter_type


def _ecg_clean_missing(ecg_signal):

    ecg_signal = pd.DataFrame.pad(pd.Series(ecg_signal))

    return ecg_signal

def ecg_peaks(ecg_cleaned, sampling_rate=1000, method="neurokit", correct_artifacts=False, **kwargs):
    """Find R-peaks in an ECG signal.

    Find R-peaks in an ECG signal using the specified method. The method accepts unfiltered ECG signals
    as input, althought it is expected that a filtered (cleaned) ECG will result in better results.

    Parameters
    ----------
    ecg_cleaned : Union[list, np.array, pd.Series]
        The cleaned ECG channel as returned by `ecg_clean()`.
    sampling_rate : int
        The sampling frequency of `ecg_signal` (in Hz, i.e., samples/second).
        Defaults to 1000.
    method : string
        The algorithm to be used for R-peak detection. Can be one of 'neurokit' (default), 'pantompkins1985'
        'nabian2018', 'gamboa2008', 'zong2003', 'hamilton2002', 'christov2004', 'engzeemod2012', 'elgendi2010',
        'kalidas2017', 'martinez2003', 'rodrigues2021' or 'promac'.
    correct_artifacts : bool
        Whether or not to identify artifacts as defined by Jukka A. Lipponen & Mika P. Tarvainen (2019):
        A robust algorithm for heart rate variability time series artefact correction using novel beat
        classification, Journal of Medical Engineering & Technology, DOI: 10.1080/03091902.2019.1640306.
    **kwargs
        Additional keyword arguments, usually specific for each method.

    Returns
    -------
    signals : DataFrame
        A DataFrame of same length as the input signal in which occurences of R-peaks marked as "1"
        in a list of zeros with the same length as `ecg_cleaned`. Accessible with the keys "ECG_R_Peaks".
    info : dict
        A dictionary containing additional information, in this case the samples at which R-peaks occur,
        accessible with the key "ECG_R_Peaks", as well as the signals' sampling rate, accessible with
        the key "sampling_rate".

    See Also
    --------
    ecg_clean, ecg_findpeaks, ecg_process, ecg_plot, signal_rate,
    signal_fixpeaks

    Examples
    --------
    >>> import neurokit2 as nk
    >>>
    >>> ecg = nk.ecg_simulate(duration=10, sampling_rate=1000)
    >>> cleaned = nk.ecg_clean(ecg, sampling_rate=1000)
    >>> signals, info = nk.ecg_peaks(cleaned, correct_artifacts=True)
    >>> nk.events_plot(info["ECG_R_Peaks"], cleaned) #doctest: +ELLIPSIS
    <Figure ...>

    References
    ----------
    'neurokit'
        Unpublished. See this discussion for more information on the method:
        https://github.com/neuropsychology/NeuroKit/issues/476
    'pantompkins1985'
        - Pan, J., & Tompkins, W. J. (1985). A real-time QRS detection algorithm. IEEE transactions on
          biomedical engineering, (3), 230-236.
        From https://github.com/berndporr/py-ecg-detectors/
    'nabian2018'
        - Nabian, M., Yin, Y., Wormwood, J., Quigley, K. S., Barrett, L. F., Ostadabbas, S. (2018).
          An Open-Source Feature Extraction Tool for the Analysis of Peripheral Physiological Data.
          IEEE Journal of Translational Engineering in Health and Medicine, 6, 1-11.
          doi:10.1109/jtehm.2018.2878000
    'gamboa2008'
        - Gamboa, H. (2008). Multi-modal behavioral biometrics based on hci and electrophysiology.
          PhD ThesisUniversidade.
          From https://github.com/PIA-Group/BioSPPy/blob/e65da30f6379852ecb98f8e2e0c9b4b5175416c3/biosppy/signals/ecg.py
    'zong2003'
        - Zong, W., Heldt, T., Moody, G. B., & Mark, R. G. (2003). An open-source algorithm to
          detect onset of arterial blood pressure pulses. In Computers in Cardiology, 2003 (pp. 259-262). IEEE.
          From BioSPPy.
    'hamilton2002'
        - Hamilton, P. (2002, September). Open source ECG analysis. In Computers in cardiology (pp. 101-104). IEEE.
        From https://github.com/berndporr/py-ecg-detectors/
    'christov2004'
        - Ivaylo I. Christov, Real time electrocardiogram QRS detection using combined adaptive threshold,
          BioMedical Engineering OnLine 2004, vol. 3:28, 2004.
        From https://github.com/berndporr/py-ecg-detectors/
    'engzeemod2012'
        - C. Zeelenberg, A single scan algorithm for QRS detection and feature extraction, IEEE Comp.
          in Cardiology, vol. 6, pp. 37-42, 1979
        - A. Lourenco, H. Silva, P. Leite, R. Lourenco and A. Fred, "Real Time Electrocardiogram Segmentation
          for Finger Based ECG Biometrics", BIOSIGNALS 2012, pp. 49-54, 2012.
        From https://github.com/berndporr/py-ecg-detectors/
    'elgendi2010'
        - Elgendi, Mohamed & Jonkman, Mirjam & De Boer, Friso. (2010). Frequency Bands Effects on QRS Detection.
          The 3rd International Conference on Bio-inspired Systems and Signal Processing (BIOSIGNALS2010).
          428-431.
        From https://github.com/berndporr/py-ecg-detectors/
    'kalidas2017'
        - Vignesh Kalidas and Lakshman Tamil (2017). Real-time QRS detector using Stationary Wavelet Transform
          for Automated ECG Analysis. In: 2017 IEEE 17th International Conference on Bioinformatics and
          Bioengineering (BIBE). Uses the Pan and Tompkins thresolding.
        From https://github.com/berndporr/py-ecg-detectors/
    'martinez2003'
        TO BE DEFINED
    'rodrigues2021'
        - Gutiérrez-Rivas, R., García, J. J., Marnane, W. P., & Hernández, A. (2015). Novel real-time
          low-complexity QRS complex detector based on adaptive thresholding. IEEE Sensors Journal,
          15(10), 6036-6043.
        - Sadhukhan, D., & Mitra, M. (2012). R-peak detection algorithm for ECG using double difference
          and RR interval processing. Procedia Technology, 4, 873-877.
        - Rodrigues, Tiago & Samoutphonh, Sirisack & Plácido da Silva, Hugo & Fred, Ana. (2021).
          A Low-Complexity R-peak Detection Algorithm with Adaptive Thresholding for Wearable Devices.
    'promac'
        Unpublished. See this discussion for more information on the method:
        https://github.com/neuropsychology/NeuroKit/issues/222
    """
    rpeaks = ecg_findpeaks(ecg_cleaned, sampling_rate=sampling_rate, method=method, **kwargs)

    if correct_artifacts:
        _, rpeaks = signal_fixpeaks(rpeaks, sampling_rate=sampling_rate, iterative=True, method="Kubios")

        rpeaks = {"ECG_R_Peaks": rpeaks}

    instant_peaks = signal_formatpeaks(rpeaks, desired_length=len(ecg_cleaned), peak_indices=rpeaks)
    signals = instant_peaks
    info = rpeaks
    info['sampling_rate'] = sampling_rate  # Add sampling rate in dict info

    return signals, info

def ecg_findpeaks(ecg_cleaned, sampling_rate=1000, method="neurokit", show=False, **kwargs):
    # Try retrieving right column
    if isinstance(ecg_cleaned, pd.DataFrame):
        try:
            ecg_cleaned = ecg_cleaned["ECG_Clean"]
        except (NameError, KeyError):
            try:
                ecg_cleaned = ecg_cleaned["ECG_Raw"]
            except (NameError, KeyError):
                ecg_cleaned = ecg_cleaned["ECG"]

    # Sanitize input
    ecg_cleaned = signal_sanitize(ecg_cleaned)
    method = method.lower()  # remove capitalised letters

    # Run peak detection algorithm
    try:
        rpeaks = _ecg_findpeaks_neurokit(ecg_cleaned, sampling_rate=sampling_rate, show=show, **kwargs)
    except ValueError as error:
        raise error

    # Prepare output.
    info = {"ECG_R_Peaks": rpeaks}

    return info


def _ecg_findpeaks_neurokit(
    signal,
    sampling_rate=1000,
    smoothwindow=0.1,
    avgwindow=0.75,
    gradthreshweight=1.5,
    minlenweight=0.4,
    mindelay=0.3,
    show=False,
):
    """All tune-able parameters are specified as keyword arguments.

    The `signal` must be the highpass-filtered raw ECG with a lowcut of .5 Hz.

    """

    # Compute the ECG's gradient as well as the gradient threshold. Run with
    # show=True in order to get an idea of the threshold.
    grad = np.gradient(signal)
    absgrad = np.abs(grad)
    smooth_kernel = int(np.rint(smoothwindow * sampling_rate))
    avg_kernel = int(np.rint(avgwindow * sampling_rate))
    smoothgrad = signal_smooth(absgrad, kernel="boxcar", size=smooth_kernel)
    avggrad = signal_smooth(smoothgrad, kernel="boxcar", size=avg_kernel)
    gradthreshold = gradthreshweight * avggrad
    mindelay = int(np.rint(sampling_rate * mindelay))


    # Identify start and end of QRS complexes.
    qrs = smoothgrad > gradthreshold
    beg_qrs = np.where(np.logical_and(np.logical_not(qrs[0:-1]), qrs[1:]))[0]
    end_qrs = np.where(np.logical_and(qrs[0:-1], np.logical_not(qrs[1:])))[0]
    # Throw out QRS-ends that precede first QRS-start.
    end_qrs = end_qrs[end_qrs > beg_qrs[0]]

    # Identify R-peaks within QRS (ignore QRS that are too short).
    num_qrs = min(beg_qrs.size, end_qrs.size)
    min_len = np.mean(end_qrs[:num_qrs] - beg_qrs[:num_qrs]) * minlenweight
    peaks = [0]

    for i in range(num_qrs):

        beg = beg_qrs[i]
        end = end_qrs[i]
        len_qrs = end - beg

        if len_qrs < min_len:
            continue

        # Find local maxima and their prominence within QRS.
        data = signal[beg:end]
        locmax, props = scipy.signal.find_peaks(data, prominence=(None, None))

        if locmax.size > 0:
            # Identify most prominent local maximum.
            peak = beg + locmax[np.argmax(props["prominences"])]
            # Enforce minimum delay between peaks.
            if peak - peaks[-1] > mindelay:
                peaks.append(peak)

    peaks.pop(0)

    peaks = np.asarray(peaks).astype(int)  # Convert to int
    return peaks

def signal_smooth(signal, method="convolution", kernel="boxzen", size=10, alpha=0.1):
    if isinstance(signal, pd.Series):
        signal = signal.values

    length = len(signal)

    if isinstance(kernel, str) is False:
        raise TypeError("NeuroKit error: signal_smooth(): 'kernel' should be a string.")

    # Check length.
    size = int(size)
    if size > length or size < 1:
        raise TypeError("NeuroKit error: signal_smooth(): 'size' should be between 1 and length of the signal.")

    method = method.lower()

    # LOESS
    if method in ["loess", "lowess"]:
        smoothed = fit_loess(signal, alpha=alpha)

    # Convolution
    else:
        if kernel == "boxcar":
            # This is faster than using np.convolve (like is done in _signal_smoothing)
            # because of optimizations made possible by the uniform boxcar kernel shape.
            smoothed = scipy.ndimage.uniform_filter1d(signal, size, mode="nearest")

        # elif kernel == "boxzen":
        #     # hybrid method
        #     # 1st pass - boxcar kernel
        #     x = scipy.ndimage.uniform_filter1d(signal, size, mode="nearest")

        #     # 2nd pass - parzen kernel
        #     smoothed = _signal_smoothing(x, kernel="parzen", size=size)

        # elif kernel == "median":
        #     smoothed = _signal_smoothing_median(signal, size)

        # else:
        #     smoothed = _signal_smoothing(signal, kernel=kernel, size=size)

    return smoothed

def fit_loess(y, X=None, alpha=0.75, order=2):
    if X is None:
        X = np.linspace(0, 100, len(y))

    assert order in [1, 2], "Deg has to be 1 or 2"
    assert 0 < alpha <= 1, "Alpha has to be between 0 and 1"
    assert len(X) == len(y), "Length of X and y are different"

    X_domain = X

    n = len(X)
    span = int(np.ceil(alpha * n))

    y_predicted = np.zeros(len(X_domain))
    x_space = np.zeros_like(X_domain)

    for i, val in enumerate(X_domain):
        distance = abs(X - val)
        sorted_dist = np.sort(distance)
        ind = np.argsort(distance)

        Nx = X[ind[:span]]
        Ny = y[ind[:span]]

        delx0 = sorted_dist[span - 1]

        u = distance[ind[:span]] / delx0
        w = (1 - u ** 3) ** 3

        W = np.diag(w)
        A = np.vander(Nx, N=1 + order)

        V = np.matmul(np.matmul(A.T, W), A)
        Y = np.matmul(np.matmul(A.T, W), Ny)
        Q, R = scipy.linalg.qr(V)
        p = scipy.linalg.solve_triangular(R, np.matmul(Q.T, Y))

        y_predicted[i] = np.polyval(p, val)
        x_space[i] = val

    return y_predicted

def hrv_nonlinear(peaks, sampling_rate=1000, show=False, **kwargs):
    # Sanitize input
    peaks = _hrv_sanitize_input(peaks)
    if isinstance(peaks, tuple):  # Detect actual sampling rate
        peaks, sampling_rate = peaks[0], peaks[1]

    # Compute R-R intervals (also referred to as NN) in milliseconds
    rri, _ = _hrv_get_rri(peaks, sampling_rate=sampling_rate, interpolate=False)

    # Initialize empty container for results
    out = {}

    # Poincaré features (SD1, SD2, etc.)
    out = _hrv_nonlinear_poincare(rri, out)

    # Heart Rate Fragmentation
    # out = _hrv_nonlinear_fragmentation(rri, out)

    # Heart Rate Asymmetry
    # out = _hrv_nonlinear_poincare_hra(rri, out)

    # DFA
#    out = _hrv_dfa(peaks, rri, out, **kwargs)

    # Complexity
    tolerance = 0.2 * np.std(rri, ddof=1)
    out["ApEn"] = entropy_approximate(rri, delay=1, dimension=2, tolerance=tolerance)[0]
    out["SampEn"] = entropy_sample(rri, delay=1, dimension=2, tolerance=tolerance)[0]
    out["ShanEn"] = entropy_shannon(rri)[0]
    # out["FuzzyEn"] = entropy_fuzzy(rri, delay=1, dimension=2, tolerance=tolerance)[0]
    # out["MSE"] = entropy_multiscale(
    #     rri, dimension=2, tolerance=tolerance, composite=False, refined=False
    # )[0]
    # out["CMSE"] = entropy_multiscale(
    #     rri, dimension=2, tolerance=tolerance, composite=True, refined=False
    # )[0]
    # out["RCMSE"] = entropy_multiscale(
    #     rri, dimension=2, tolerance=tolerance, composite=True, refined=True
    # )[0]

    # out["CD"] = fractal_correlation(rri, delay=1, dimension=2, **kwargs)[0]
    out["HFD"] = fractal_higuchi(rri, **kwargs)[0]
    # out["KFD"] = fractal_katz(rri)[0]
    # out["LZC"] = complexity_lempelziv(rri, **kwargs)[0]
    out = pd.DataFrame.from_dict(out, orient="index").T.add_prefix("HRV_")
    return out

def fractal_higuchi(signal, k_max="default", show=False, **kwargs):
    # Sanity checks
    if isinstance(signal, (np.ndarray, pd.DataFrame)) and signal.ndim > 1:
        raise ValueError(
            "Multidimensional inputs (e.g., matrices or multichannel data) are not supported yet."
        )

    # Get k_max
    if isinstance(k_max, (str, list, np.ndarray, pd.Series)):
        # optimizing needed
        k_max, info = complexity_k(signal, k_max=k_max, show=False)
        idx = np.where(info["Values"] == k_max)[0][0]
        slope = info["Scores"][idx]
        intercept = info["Intercepts"][idx]
        average_values = info["Average_Values"][idx]
        k_values = np.arange(1, k_max + 1)
    else:
        # Compute higushi
        slope, intercept, info = _complexity_k_slope(k_max, signal)
        k_values = info["k_values"]
        average_values = info["average_values"]

    return slope, {
        "k_max": k_max,
        "Values": k_values,
        "Scores": average_values,
        "Intercept": intercept,
    }

def _hrv_get_rri(peaks=None, sampling_rate=1000, interpolate=False, **kwargs):

    rri = np.diff(peaks) / sampling_rate * 1000

    if interpolate is False:
        sampling_rate = None

    else:

        # Sanitize minimum sampling rate for interpolation to 10 Hz
        sampling_rate = max(sampling_rate, 10)

        # Compute length of interpolated heart period signal at requested sampling rate.
        desired_length = int(np.rint(peaks[-1]))

        rri = signal_interpolate(
            peaks[1:],  # Skip first peak since it has no corresponding element in heart_period
            rri,
            x_new=np.arange(desired_length),
            **kwargs
        )
    return rri, sampling_rate


def _hrv_sanitize_input(peaks=None):

    if isinstance(peaks, tuple):
        peaks = _hrv_sanitize_tuple(peaks)
    elif isinstance(peaks, (dict, pd.DataFrame)):
        peaks = _hrv_sanitize_dict_or_df(peaks)
    else:
        peaks = _hrv_sanitize_peaks(peaks)

    if peaks is not None:
        if isinstance(peaks, tuple):
            if any(np.diff(peaks[0]) < 0):  # not continuously increasing
                raise ValueError(
                    "NeuroKit error: _hrv_sanitize_input(): "
                    + "The peak indices passed were detected as non-consecutive. You might have passed RR "
                    + "intervals instead of peaks. If so, convert RRIs into peaks using "
                    + "nk.intervals_to_peaks()."
                )
        else:
            if any(np.diff(peaks) < 0):
                raise ValueError(
                    "NeuroKit error: _hrv_sanitize_input(): "
                    + "The peak indices passed were detected as non-consecutive. You might have passed RR "
                    + "intervals instead of peaks. If so, convert RRIs into peaks using "
                    + "nk.intervals_to_peaks()."
                )

    return peaks

# @njit
def complexity_k(signal, k_max="max", show=False):
    # Get the range of k-max values to be tested
    # ------------------------------------------
    # if isinstance(k_max, str):  # e.g., "default"
        # upper limit for k value (max possible value)
    k_max = int(np.floor(len(signal) / 2))  # so that normalizing factor is positive

    # if isinstance(k_max, int):
    kmax_range = np.arange(2, k_max + 1)
    # elif isinstance(k_max, (list, np.ndarray, pd.Series)):
    #     kmax_range = np.array(k_max)
    # else:
    #     print(
    #         "k_max should be an int or a list of values of kmax to be tested.",
    #     )

    # Compute the slope for each kmax value
    # --------------------------------------
    vectorized_k_slope = np.vectorize(_complexity_k_slope, excluded=[1])
    slopes, intercepts, info = vectorized_k_slope(kmax_range, signal)
    # k_values = [d["k_values"] for d in info]
    average_values = [d["average_values"] for d in info]

    # Find plateau (the saturation point of slope)
    # --------------------------------------------
    optimal_point = find_plateau(slopes, show=False)
    if optimal_point is not None:
        kmax_optimal = kmax_range[optimal_point]
    else:
        kmax_optimal = np.max(kmax_range)
        # print(
        #     "The optimal kmax value detected is 2 or less. There may be no plateau in this case. "
        #     + f"You can inspect the plot by set `show=True`. We will return optimal k_max = {kmax_optimal} (the max)."
        # )

    # Return optimal tau and info dict
    return kmax_optimal, {
        "Values": kmax_range,
        "Scores": slopes,
        "Intercepts": intercepts,
        "Average_Values": average_values,
    }

# @njit
def find_plateau(values, show=True):
    # find indices in increasing segments
    increasing_segments = np.where(np.diff(values) > 0)[0]

    # get indices where positive gradients are becoming less positive
    slope_change = np.diff(np.diff(values))
    gradients = np.where(slope_change < 0)[0]
    indices = np.intersect1d(increasing_segments, gradients)

    # exclude inverse peaks
    peaks = scipy.signal.find_peaks(-1 * values)[0]
    if len(peaks) > 0:
        indices = [i for i in indices if i not in peaks]

    # find greatest change in slopes amongst filtered indices
    largest = np.argsort(slope_change)[: int(0.1 * len(slope_change))]  # get top 10%
    optimal = [i for i in largest if i in indices]

    if len(optimal) >= 1:
        plateau = np.where(values == np.max(values[optimal]))[0][0]
    else:
        plateau = None

    return plateau

# @njit
def _complexity_k_slope(kmax, signal, k_number="max"):
    if k_number == "max":
        k_values = np.arange(1, kmax + 1)
    else:
        k_values = np.unique(np.linspace(1, kmax + 1, k_number).astype(int))

    # Step 3 of Vega & Noel (2015)
    vectorized_Lk = np.vectorize(_complexity_k_Lk, excluded=[1])

    # Compute length of the curve, Lm(k)
    average_values = vectorized_Lk(k_values, signal)

    # Slope of best-fit line through points (slope equal to FD)
    slope, intercept = -np.polyfit(np.log(k_values), np.log(average_values), 1)
    return slope, intercept, {"k_values": k_values, "average_values": average_values}

def _complexity_k_Lk(k, signal):
    n = len(signal)

    # Step 1: construct k number of new time series for range of k_values from 1 to kmax
    k_subrange = np.arange(1, k + 1)  # where m = 1, 2... k

    idx = np.tile(np.arange(0, len(signal), k), (k, 1)).astype(float)
    idx += np.tile(np.arange(0, k), (idx.shape[1], 1)).T
    mask = idx >= len(signal)
    idx[mask] = 0

    sig_values = signal[idx.astype(int)].astype(float)
    sig_values[mask] = np.nan

    # Step 2: Calculate length Lm(k) of each curve
    normalization = (n - 1) / (np.floor((n - k_subrange) / k).astype(int) * k)
    sets = (np.nansum(np.abs(np.diff(sig_values)), axis=1) * normalization) / k

    # Step 3: Compute average value over k sets of Lm(k)
    return np.sum(sets) / k

# =============================================================================
# Get SD1 and SD2
# =============================================================================
def _hrv_nonlinear_poincare(rri, out):
    """Compute SD1 and SD2.

    - Do existing measures of Poincare plot geometry reflect nonlinear features of heart rate
    variability? - Brennan (2001)

    """

    # HRV and hrvanalysis
    rri_n = rri[:-1]
    rri_plus = rri[1:]
    x1 = (rri_n - rri_plus) / np.sqrt(2)  # Eq.7
    x2 = (rri_n + rri_plus) / np.sqrt(2)
    sd1 = np.std(x1, ddof=1)
    sd2 = np.std(x2, ddof=1)

    out["SD1"] = sd1
    out["SD2"] = sd2

    # SD1 / SD2
    out["SD1SD2"] = sd1 / sd2

    # Area of ellipse described by SD1 and SD2
    out["S"] = np.pi * out["SD1"] * out["SD2"]

    # CSI / CVI
    T = 4 * out["SD1"]
    L = 4 * out["SD2"]
    out["CSI"] = L / T
    out["CVI"] = np.log10(L * T)
    out["CSI_Modified"] = L ** 2 / T

    return out


def _hrv_nonlinear_poincare_hra(rri, out):
    """Heart Rate Asymmetry Indices.

    - Asymmetry of Poincaré plot (or termed as heart rate asymmetry, HRA) - Yan (2017)
    - Asymmetric properties of long-term and total heart rate variability - Piskorski (2011)

    """

    N = len(rri) - 1
    x = rri[:-1]  # rri_n, x-axis
    y = rri[1:]  # rri_plus, y-axis

    diff = y - x
    decelerate_indices = np.where(diff > 0)[0]  # set of points above IL where y > x
    accelerate_indices = np.where(diff < 0)[0]  # set of points below IL where y < x
    nochange_indices = np.where(diff == 0)[0]

    # Distances to centroid line l2
    centroid_x = np.mean(x)
    centroid_y = np.mean(y)
    dist_l2_all = abs((x - centroid_x) + (y - centroid_y)) / np.sqrt(2)

    # Distances to LI
    dist_all = abs(y - x) / np.sqrt(2)

    # Calculate the angles
    theta_all = abs(np.arctan(1) - np.arctan(y / x))  # phase angle LI - phase angle of i-th point
    # Calculate the radius
    r = np.sqrt(x ** 2 + y ** 2)
    # Sector areas
    S_all = 1 / 2 * theta_all * r ** 2

    # Guzik's Index (GI)
    den_GI = np.sum(dist_all)
    num_GI = np.sum(dist_all[decelerate_indices])
    out["GI"] = (num_GI / den_GI) * 100

    # Slope Index (SI)
    den_SI = np.sum(theta_all)
    num_SI = np.sum(theta_all[decelerate_indices])
    out["SI"] = (num_SI / den_SI) * 100

    # Area Index (AI)
    den_AI = np.sum(S_all)
    num_AI = np.sum(S_all[decelerate_indices])
    out["AI"] = (num_AI / den_AI) * 100

    # Porta's Index (PI)
    m = N - len(nochange_indices)  # all points except those on LI
    b = len(accelerate_indices)  # number of points below LI
    out["PI"] = (b / m) * 100

    # Short-term asymmetry (SD1)
    sd1d = np.sqrt(np.sum(dist_all[decelerate_indices] ** 2) / (N - 1))
    sd1a = np.sqrt(np.sum(dist_all[accelerate_indices] ** 2) / (N - 1))

    sd1I = np.sqrt(sd1d ** 2 + sd1a ** 2)
    out["C1d"] = (sd1d / sd1I) ** 2
    out["C1a"] = (sd1a / sd1I) ** 2
    out["SD1d"] = sd1d  # SD1 deceleration
    out["SD1a"] = sd1a  # SD1 acceleration
    # out["SD1I"] = sd1I  # SD1 based on LI, whereas SD1 is based on centroid line l1

    # Long-term asymmetry (SD2)
    longterm_dec = np.sum(dist_l2_all[decelerate_indices] ** 2) / (N - 1)
    longterm_acc = np.sum(dist_l2_all[accelerate_indices] ** 2) / (N - 1)
    longterm_nodiff = np.sum(dist_l2_all[nochange_indices] ** 2) / (N - 1)

    sd2d = np.sqrt(longterm_dec + 0.5 * longterm_nodiff)
    sd2a = np.sqrt(longterm_acc + 0.5 * longterm_nodiff)

    sd2I = np.sqrt(sd2d ** 2 + sd2a ** 2)
    out["C2d"] = (sd2d / sd2I) ** 2
    out["C2a"] = (sd2a / sd2I) ** 2
    out["SD2d"] = sd2d  # SD2 deceleration
    out["SD2a"] = sd2a  # SD2 acceleration
    # out["SD2I"] = sd2I  # identical with SD2

    # Total asymmerty (SDNN)
    sdnnd = np.sqrt(0.5 * (sd1d ** 2 + sd2d ** 2))  # SDNN deceleration
    sdnna = np.sqrt(0.5 * (sd1a ** 2 + sd2a ** 2))  # SDNN acceleration
    sdnn = np.sqrt(sdnnd ** 2 + sdnna ** 2)  # should be similar to sdnn in hrv_time
    out["Cd"] = (sdnnd / sdnn) ** 2
    out["Ca"] = (sdnna / sdnn) ** 2
    out["SDNNd"] = sdnnd
    out["SDNNa"] = sdnna

    return out


def _hrv_nonlinear_fragmentation(rri, out):
    """Heart Rate Fragmentation Indices - Costa (2017)

    The more fragmented a time series is, the higher the PIP, IALS, PSS, and PAS indices will be.
    """

    diff_rri = np.diff(rri)
    zerocrossings = signal_zerocrossings(diff_rri)

    # Percentage of inflection points (PIP)
    out["PIP"] = len(zerocrossings) / len(rri)

    # Inverse of the average length of the acceleration/deceleration segments (IALS)
    accelerations = np.where(diff_rri > 0)[0]
    decelerations = np.where(diff_rri < 0)[0]
    consecutive = find_consecutive(accelerations) + find_consecutive(decelerations)
    lengths = [len(i) for i in consecutive]
    out["IALS"] = 1 / np.average(lengths)

    # Percentage of short segments (PSS) - The complement of the percentage of NN intervals in
    # acceleration and deceleration segments with three or more NN intervals
    out["PSS"] = np.sum(np.asarray(lengths) < 3) / len(lengths)

    # Percentage of NN intervals in alternation segments (PAS). An alternation segment is a sequence
    # of at least four NN intervals, for which heart rate acceleration changes sign every beat. We note
    # that PAS quantifies the amount of a particular sub-type of fragmentation (alternation). A time
    # series may be highly fragmented and have a small amount of alternation. However, all time series
    # with large amount of alternation are highly fragmented.
    alternations = find_consecutive(zerocrossings)
    lengths = [len(i) for i in alternations]
    out["PAS"] = np.sum(np.asarray(lengths) >= 4) / len(lengths)

    return out



def hrv_time(peaks, sampling_rate=1000, show=False, **kwargs):
    # Sanitize input
    peaks = _hrv_sanitize_input(peaks)
    if isinstance(peaks, tuple):  # Detect actual sampling rate
        peaks, sampling_rate = peaks[0], peaks[1]

    # Compute R-R intervals (also referred to as NN) in milliseconds
    rri, _ = _hrv_get_rri(peaks, sampling_rate=sampling_rate, interpolate=False)
    diff_rri = np.diff(rri)

    out = {}  # Initialize empty container for results

    # Deviation-based
    out["MeanNN"] = np.nanmean(rri)
    out["SDNN"] = np.nanstd(rri, ddof=1)
    # for i in [1, 2, 5]:
    #     out["SDANN" + str(i)] = _sdann(rri, window=i)
    #     out["SDNNI" + str(i)] = _sdnni(rri, window=i)

    # Difference-based
    out["RMSSD"] = np.sqrt(np.nanmean(diff_rri ** 2))
    out["SDSD"] = np.nanstd(diff_rri, ddof=1)

    # # Normalized
    # out["CVNN"] = out["SDNN"] / out["MeanNN"]
    # out["CVSD"] = out["RMSSD"] / out["MeanNN"]

    # Robust
    # out["MedianNN"] = np.nanmedian(rri)
    # out["MadNN"] = mad(rri)
    # out["MCVNN"] = out["MadNN"] / out["MedianNN"]  # Normalized
    # out["IQRNN"] = scipy.stats.iqr(rri)

    # Extreme-based
    nn50 = np.sum(np.abs(diff_rri) > 50)
    nn20 = np.sum(np.abs(diff_rri) > 20)
    out["pNN50"] = nn50 / len(rri) * 100
    out["pNN20"] = nn20 / len(rri) * 100

    # # Geometrical domain
    # if "binsize" in kwargs.keys():
    #     binsize = kwargs["binsize"]
    # else:
    #     binsize = (1 / 128) * 1000
    # bins = np.arange(0, np.max(rri) + binsize, binsize)
    # bar_y, bar_x = np.histogram(rri, bins=bins)
    # # HRV Triangular Index
    # out["HTI"] = len(rri) / np.max(bar_y)
    # # Triangular Interpolation of the NN Interval Histogram
    # out["TINN"] = _hrv_TINN(rri, bar_x, bar_y, binsize)


    out = pd.DataFrame.from_dict(out, orient="index").T.add_prefix("HRV_")
    return out

def hrv_frequency(
    peaks,
    sampling_rate=1000,
    ulf=(0, 0.0033),
    vlf=(0.0033, 0.04),
    lf=(0.04, 0.15),
    hf=(0.15, 0.4),
    vhf=(0.4, 0.5),
    psd_method="welch",
    show=False,
    silent=True,
    normalize=True,
    order_criteria=None,
    **kwargs
):
    # Sanitize input
    peaks = _hrv_sanitize_input(peaks)
    if isinstance(peaks, tuple):  # Detect actual sampling rate
        peaks, sampling_rate = peaks[0], peaks[1]

    # Compute R-R intervals (also referred to as NN) in milliseconds (interpolated at 1000 Hz by default)
    rri, sampling_rate = _hrv_get_rri(
        peaks, sampling_rate=sampling_rate, interpolate=True, **kwargs
    )

    frequency_band = [ulf, vlf, lf, hf, vhf]
    power = signal_power(
        rri,
        frequency_band=frequency_band,
        sampling_rate=sampling_rate,
        method=psd_method,
        max_frequency=0.5,
        show=False,
        normalize=normalize,
        order_criteria=order_criteria,
        **kwargs
    )

    power.columns = ["ULF", "VLF", "LF", "HF", "VHF"]

    out = power.to_dict(orient="index")[0]
    out_bands = out.copy()  # Components to be entered into plot

    if silent is False:
        for frequency in out.keys():
            if out[frequency] == 0.0:
                print(
                    "The duration of recording is too short to allow"
                    " reliable computation of signal power in frequency band " + frequency + "."
                    " Its power is returned as zero."
                )

    # Normalized
    total_power = np.nansum(power.values)
    out["LFHF"] = out["LF"] / out["HF"]
    out["LFn"] = out["LF"] / total_power
    out["HFn"] = out["HF"] / total_power

    # Log
    out["LnHF"] = np.log(out["HF"])  # pylint: disable=E1111

    out = pd.DataFrame.from_dict(out, orient="index").T.add_prefix("HRV_")

    return out

def _hrv_sanitize_tuple(peaks):

    # Get sampling rate
    info = [i for i in peaks if isinstance(i, dict)]
    sampling_rate = info[0]["sampling_rate"]

    # Get peaks
    if isinstance(peaks[0], (dict, pd.DataFrame)):
        try:
            peaks = _hrv_sanitize_dict_or_df(peaks[0])
        except NameError:
            if isinstance(peaks[1], (dict, pd.DataFrame)):
                try:
                    peaks = _hrv_sanitize_dict_or_df(peaks[1])
                except NameError:
                    peaks = _hrv_sanitize_peaks(peaks[1])
            else:
                peaks = _hrv_sanitize_peaks(peaks[0])

    return peaks, sampling_rate


def _hrv_sanitize_dict_or_df(peaks):

    # Get columns
    if isinstance(peaks, dict):
        cols = np.array(list(peaks.keys()))
        if "sampling_rate" in cols:
            sampling_rate = peaks["sampling_rate"]
        else:
            sampling_rate = None
    elif isinstance(peaks, pd.DataFrame):
        cols = peaks.columns.values
        sampling_rate = None

    cols = cols[["Peak" in s for s in cols]]

    if len(cols) > 1:
        cols = cols[[("ECG" in s) or ("PPG" in s) for s in cols]]

    if len(cols) == 0:
        raise NameError(
            "NeuroKit error: hrv(): Wrong input, ",
            "we couldn't extract R-peak indices. ",
            "You need to provide a list of R-peak indices.",
        )

    peaks = _hrv_sanitize_peaks(peaks[cols[0]])

    if sampling_rate is not None:
        return peaks, sampling_rate
    else:
        return peaks


def _hrv_sanitize_peaks(peaks):

    if isinstance(peaks, pd.Series):
        peaks = peaks.values

    if len(np.unique(peaks)) == 2:
        if np.all(np.unique(peaks) == np.array([0, 1])):
            peaks = np.where(peaks == 1)[0]

    if isinstance(peaks, list):
        peaks = np.array(peaks)

    return peaks
