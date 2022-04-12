
import heartpy as hp
import numpy as np
from scipy.stats import variation, iqr
from scipy.fft import fft, next_fast_len
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
    try:
        filtered = hp.remove_baseline_wander(dataSlice, samplerate)
        filtered_scaled = hp.scale_data(filtered)
        features = getSignalFeatures(dataSlice)
        w, m = hp.process(filtered_scaled, samplerate, clean_rr=False)
        heartpyFeatsToKeep = ['sd1', 'sd2', 'sd1/sd2']
        for feat in heartpyFeatsToKeep:
            if (not isinstance(m[feat], float)):
                features[feat] = None
            else:
                features[feat] = m[feat]
        return features
    except:
        return None

def featurize(dataSlice, samplerate):
    try:
        filtered = hp.remove_baseline_wander(dataSlice, samplerate)
        filtered_scaled = hp.scale_data(filtered)
        w, m = hp.process(filtered_scaled, samplerate, clean_rr=False)
        rrIntervals = list()
        beat2beatIntervals = list()
        for rrIntervalMS in w['RR_list_cor']:
            beat2beatIntervals.append(60 / (rrIntervalMS/1000.0))
        features = getb2bFeatures(beat2beatIntervals)
        heartpyFeatsToKeep = ['bpm', 'pnn20', 'pnn50', 'rmssd', 'ibi', 'sdnn', 'sdsd']
        for feat in heartpyFeatsToKeep:
            if (not isinstance(m[feat], float)):
                features[feat] = None
            else:
                features[feat] = m[feat]
        return features
    except:
        return None
