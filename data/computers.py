
import heartpy as hp
import numpy as np
from scipy.stats import variation, iqr
from scipy.fft import fft, next_fast_len
import matplotlib.pyplot as plt
import neurokit2 as nk
def getb2bFeatures(beat2BeatSequence):
    return {
        'b2b_var': variation(beat2BeatSequence),
        'b2b_iqr': iqr(beat2BeatSequence),
        'b2b_range': np.ptp(beat2BeatSequence),
        'b2b_std': np.std(beat2BeatSequence)
    }


def getSignalFeatures(mvSequence):
    n = len(mvSequence)
    nn = next_fast_len(n)
    x = np.pad(np.array(mvSequence), (0, nn-n), 'constant')
    f = fft(x)
    '''
Fs = 150                         # sampling rate
Ts = 1.0/Fs                      # sampling interval
t = np.arange(0,1,Ts)            # time vector
ff = 5                           # frequency of the signal
y = np.sin(2 * np.pi * ff * t)

plt.subplot(2,1,1)
plt.plot(t,y,'k-')
plt.xlabel('time')
plt.ylabel('amplitude')

plt.subplot(2,1,2)
n = len(y)                       # length of the signal
k = np.arange(n)
T = n/Fs
frq = k/T # two sides frequency range
freq = frq[range(n/2)]           # one side frequency range

Y = np.fft.fft(y)/n              # fft computing and normalization
Y = Y[range(n/2)]

plt.plot(freq, abs(Y), 'r-')
plt.xlabel('freq (Hz)')
plt.ylabel('|Y(freq)|')

plt.show()      
    '''
    Fs = 500 # samplearate
    t = np.arange(0, 60, 1.0/Fs)
    plt.subplot(2, 1, 1)
    plt.plot(t, mvSequence)
    plt.subplot(2, 1, 2)
    k = np.arange(nn)
    T = nn / Fs
    frq = k / T
    freq = frq[range(n//2)]

    Y = f / nn
    Y = Y[range(nn//2)]
    plt.plot(freq, abs(Y))
    plt.savefig(
        'yo.png'
    )   
    print(1/0)
    #print(-2j * np.pi * 0 * np.arange(n)/n)
    fft1 = np.sum(x * np.exp(-2j * np.pi * 0 * np.arange(nn)/nn))
    fft2 = np.sum(x * np.exp(-2j * np.pi * 1 * np.arange(nn)/nn))
    #print(fft1)
    #print(fft2)
    # print(iqr(mvSequence))
    return {
        # 'fft_1': np.abs(f[0]),
        # 'fft_2': np.abs(f[1]),
        'fft_1': np.abs(fft1),
        'fft_2': np.abs(fft2),
        #'hfd': hfda.measure(mvSequence, len(mvSequence)//2),
        # 'beat_iqr': iqr(mvSequence)
    }

def featurize_longertimewindow(dataSlice, samplerate):
    filtered = hp.remove_baseline_wander(dataSlice, samplerate)
    # filtered_scaled = hp.scale_data(filtered)
    # features = getSignalFeatures(dataSlice)
    ecg_cleaned = nk.ecg_clean(filtered, sampling_rate=samplerate)
    sigs, info = nk.ecg_process(ecg_cleaned, sampling_rate=samplerate)
    ecg = sigs['ECG_Clean']
    peaks, info = nk.ecg_peaks(ecg, sampling_rate=samplerate)
    hrv = nk.hrv_frequency(peaks, sampling_rate=samplerate)
    nkFeatsToKeep = [
        'HRV_LF',
        'HRV_HF',
        'HRV_VHF',
        'HRV_LFHF',
        'HRV_LFn',
        'HRV_HFn',
        'HRV_LnHF'
    ]
    m = hrv
    features = dict()
    for feat in nkFeatsToKeep:
        features[feat] = m[feat][0] if isinstance(m[feat][0], float) else None
        # if (not isinstance(m[feat], float)):
        #     features[feat] = None
        # else:
        #     features[feat] = m[feat]
    return features
    # try:
        # w, m = hp.process(filtered_scaled, samplerate, clean_rr=False)
        # heartpyFeatsToKeep = ['sd1', 'sd2', 'sd1/sd2']
        # for feat in heartpyFeatsToKeep:
    # except:
    #     return None

def featurize(dataSlice, samplerate):
    detrended = hp.remove_baseline_wander(dataSlice, samplerate)
    ecg_cleaned = nk.ecg_clean(detrended, sampling_rate=samplerate)
    sigs, info = nk.ecg_process(ecg_cleaned, sampling_rate=samplerate)
    ecg = sigs['ECG_Clean']
    peaks, info = nk.ecg_peaks(ecg, sampling_rate=samplerate)
    hrv = nk.hrv_time(peaks, sampling_rate=samplerate)
    # rrs = nk.utils._hrv_get_rri(peaks, sampling_rate=samplerate, interpolate=False)
    # print(info)
    # print('yo')
    # print(np.diff(info['ECG_R_Peaks']) / 
    # print(hrv)
    # print(sigs)
    # print(np.mean(sigs['ECG_Quality']))
    # try:
    # filtered_scaled = hp.scale_data(detrended)
    # w, m = hp.process(filtered_scaled, samplerate, clean_rr=False)
    # rrIntervals = list()
    # beat2beatIntervals = list()
    # for rrIntervalMS in w['RR_list_cor']:
    #     beat2beatIntervals.append(60 / (rrIntervalMS/1000.0))
    features = dict() #getb2bFeatures(beat2beatIntervals)
    heartpyFeatsToKeep = ['bpm', 'pnn20', 'pnn50', 'rmssd', 'ibi', 'sdnn', 'sdsd']
    nkFeatsToKeep = [
        'HRV_MeanNN',
        'HRV_SDNN',
        'HRV_RMSSD',
        'HRV_SDSD',
        'HRV_CVNN',
        'HRV_CVSD',
        'HRV_MedianNN',
        'HRV_MadNN',
        'HRV_MCVNN',
        'HRV_IQRNN',
        'HRV_pNN50',
        'HRV_pNN20',
        'HRV_HTI',
        'HRV_TINN'
    ]
    '''HRV_MeanNN  HRV_SDNN  HRV_SDANN1  HRV_SDNNI1  HRV_SDANN2  HRV_SDNNI2  HRV_SDANN5  HRV_SDNNI5  HRV_RMSSD  HRV_SDSD  HRV_CVNN  HRV_CVSD  HRV_MedianNN  HRV_MadNN  HRV_MCVNN  HRV_IQRNN  HRV_pNN50  HRV_pNN20   HRV_HTI  HRV_TINN'''
    m = hrv
    for feat in nkFeatsToKeep:
        features[feat] = m[feat][0] if isinstance(m[feat][0], float) else None
        # print(feat, m[feat])
        # if (not isinstance(m[feat], float)):
        #     features[feat] = None
        # else:
        #     features[feat] = m[feat]
    return features
    # except:
    #     return None