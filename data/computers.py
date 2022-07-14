
import heartpy as hp
import hfda
import numpy as np
from scipy.stats import variation, iqr
from scipy.fft import fft, next_fast_len
import matplotlib.pyplot as plt
import neurokit2 as nk
import pandas as pd
from pyclustertend import hopkins
from sklearn.metrics import silhouette_score
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from .nk_computers import ecg_process, ecg_analyze
import logging

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
    '''
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

'''
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

'''
def featurize_longertimewindow(dataSlice, samplerate):
    sigs, info = ecg_process(dataSlice, sampling_rate=samplerate)
    print(sigs.head())
    ecg = sigs['ECG_Clean']
    rPeaks = sigs['ECG_R_Peaks'].to_numpy().nonzero()[0]
    rrIntervals = list()
    for i in range(len(rPeaks)-1):
        rrIntervals.append(((rPeaks[i+1] - rPeaks[i]) / samplerate)*1000)
    print(rrIntervals)
    #w, m = hp.process(ecg, samplerate, clean_rr=False)
    analysis_outputs = nk.ecg_analyze(sigs, sampling_rate=samplerate)
    print(sigs.columns)
    print(analysis_outputs.columns)
    try:
        features = getRRIntervalStatistics(rrIntervals)#getSignalFeatures(filtered)
        for feat in ['sd1', 'sd2', 'sd1/sd2', 'pnn20', 'pnn50']:
            features[feat] = m[feat] if isinstance(m[feat], float) else None
        hfd = hfda.measure(w['RR_list'], 5) #https://www.ncbi.nlm.nih.gov/pmc/articles/PMC6110872/
        features['hfd'] = hfd if isinstance(hfd, float) else None
        sampEn, _ = nk.entropy_sample(rrIntervals, dimension=1)
        print(sampEn, analysis_outputs['HRV_SampEn'])
        features['sample_entropy'] = sampEn
        peaks, peakinfo = nk.ecg_peaks(ecg, sampling_rate=samplerate)
        hrv_indices = nk.hrv_frequency(peaks, sampling_rate=samplerate, show=False)
        m = hrv_indices
        for feat in ['HRV_LF', 'HRV_HF', 'HRV_LFHF']:
            features[feat] = m[feat][0] if isinstance(m[feat][0], float) else None
        return features
    except:
        return None

def featurize_longertimewindow_nk(dataSlice, samplerate):
    try:
        sigs, info = ecg_process(dataSlice, sampling_rate=samplerate)
        measures = ecg_analyze(sigs, sampling_rate=samplerate)
        sigs.columns = sigs.columns.str.lower()
        measures.columns = measures.columns.str.lower()
        rPeaks = sigs['ecg_r_peaks'].to_numpy().nonzero()[0]
        rrIntervals = list()
        for i in range(len(rPeaks)-1):
            rrIntervals.append(((rPeaks[i+1] - rPeaks[i]) / samplerate)*1000)
        features = getRRIntervalStatistics(rrIntervals)#getSignalFeatures(filtered)
        m = measures
        for feat in ['hrv_hfd', 'hrv_lf', 'hrv_hf', 'hrv_lfhf', 'hrv_sd1', 'hrv_sd2', 'hrv_sd1sd2', 'hrv_pnn20', 'hrv_pnn50', 'hrv_hfd', 'hrv_sampen', 'hrv_shanen', 'hrv_apen']:
            features[feat] = m[feat][0] if isinstance(m[feat][0], float) else None
        return features
    except Exception:
        return logging.exception('featurize_nk longer time window')

def getRRIntervalStatistics(rrIntervals):
    #first, make rr, rr_{n+1} pairs
    rrIntervalPairs = list(zip(rrIntervals[:-1], rrIntervals[1:]))
    scaledDF = StandardScaler().fit_transform(pd.DataFrame(rrIntervalPairs))
    hopkinsStat = hopkins(scaledDF, len(rrIntervalPairs))
    kMeanses = [KMeans(n_clusters=i, max_iter=300, random_state=16) for i in range(1, 11)]
    [kMeans.fit(scaledDF) for kMeans in kMeanses]
    maxSilScore = maxSilhouetteScore(scaledDF, kMeanses[1:]) #only pass clusters 2 through 10 to sil score
    sse_1 = kMeanses[0].inertia_
    sse_2 = kMeanses[1].inertia_
    return {
        'hopkins_statistic': hopkinsStat,
        'max_sil_score': maxSilScore,
        'sse_1_clusters': sse_1,
        'sse_2_clusters': sse_2
    }

#implemented by Dr. Rooney
def maxSilhouetteScore(rrIntervalPairs, kMeanses):
    maxSilScore = -2 #silhouette score ranges between -1 and 1
    for cluster in kMeanses:
        score = silhouette_score(rrIntervalPairs, cluster.labels_)
        maxSilScore = max(score, maxSilScore)
    return maxSilScore

def hopkinsStatistic(X):
    #X=X.values
    sample_size = int(X.shape[0]*0.05)
    X_uniform_random_sample = uniform(X.min(axis=0), X.max(axis=0) ,(sample_size , X.shape[1]))
    random_indices=sample(range(0, X.shape[0], 1), sample_size)
    X_sample = X[random_indices]
    neigh = NearestNeighbors(n_neighbors=2)
    nbrs=neigh.fit(X)
    u_distances , u_indices = nbrs.kneighbors(X_uniform_random_sample , n_neighbors=2)
    u_distances = u_distances[: , 0] #distance to the first (nearest) neighbour
    w_distances , w_indices = nbrs.kneighbors(X_sample , n_neighbors=2)
    w_distances = w_distances[: , 1]
    u_sum = np.sum(u_distances)
    w_sum = np.sum(w_distances)
    H = u_sum/ (u_sum + w_sum)
    return H

def featurize(dataSlice, samplerate):
    try:
        sigs, info = ecg_process(dataSlice, sampling_rate=samplerate)
        ecg = sigs['ECG_Clean']
        w, m = hp.process(ecg, samplerate, clean_rr=False)
        beat2beatIntervals = list()
        for rrIntervalMS in w['RR_list']:
            beat2beatIntervals.append(60 / (rrIntervalMS/1000.0))
        features = getb2bFeatures(beat2beatIntervals)
        heartpyFeatsToKeep = ['bpm', 'rmssd', 'ibi', 'sdnn', 'sdsd']
        for feat in heartpyFeatsToKeep:
            features[feat] = m[feat] if isinstance(m[feat], float) else None
        return features
    except:
        return None

def featurize_nk(dataSlice, samplerate):
    try:
        sigs, info = ecg_process(dataSlice, sampling_rate=samplerate)
        m_nk = ecg_analyze(sigs, sampling_rate=samplerate, withhrv=False)
        m_nk.columns = m_nk.columns.str.lower()
        rPeaks = sigs['ECG_R_Peaks'].to_numpy().nonzero()[0]
        beat2beatIntervals = list()
        for i in range(len(rPeaks)-1):
            rrIntervalS = ((rPeaks[i+1] - rPeaks[i]) / samplerate)
            beat2beatIntervals.append(60 / rrIntervalS)
        features = getb2bFeatures(beat2beatIntervals)
        nkFeatsToKeep = ['ecg_rate_mean', 'hrv_rmssd', 'hrv_sdnn', 'hrv_sdsd']
        for feat in nkFeatsToKeep:
            features[feat] = m_nk[feat][0] if isinstance(m_nk[feat][0], float) else None
        return features
    except Exception:
        return logging.exception('featurize_nk')

def featurize_2(dataSlice, samplerate):
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
