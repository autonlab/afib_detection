import datetime as dt
import functools
from pathlib import Path
import pandas as pd
import heartpy as hp
import matplotlib.pyplot as plt
import os
from pprint import PrettyPrinter, pprint
from tqdm import tqdm

import data.utilities as datautils
# import utilities as datautils
from scipy.stats.mstats import winsorize
from scipy.stats import kstest
import numpy as np

import neurokit2 as nk
from data.sqis import k_SQI, p_SQI, bas_SQI, orphanidou2015_sqi

def plotDFSegments(df, dst):
    for _, row in df.iterrows():
        fin, start, stop = row['fin_study_id'], row['start'], row['stop']
        plotSegment(fin, start, stop, extrainfo=row, dst=dst)

def plotSegment(fin, start, stop, searchDirectory='/home/rkaufman/workspace/remote', extrainfo=None, dst=None):
    file = datautils.findFileByFIN(str(fin), searchDirectory)
    dataslice, samplerate = datautils.getSlice(file, datautils.HR_SERIES, start, stop)
    plotSlice(dataslice, samplerate, fin, start, extrainfo, dst=dst)



def plotSlice(dataSlice, samplerate, fin=None, start=None, extrainfo=None, dst=None):
    detrended = hp.remove_baseline_wander(dataSlice, samplerate)
    detrended_scaled = hp.scale_data(detrended)
    w, m = hp.process(detrended_scaled, samplerate, clean_rr=False)
    hp.plotter(w, m, figsize=(12, 4), show=False)
    # # plt.plot(hp.scale_data(hp.filter_signal(detrended, cutoff=[.75, 3.5], sample_rate=samplerate, filtertype='bandpass')))
    modelconfig = mu.getModelConfig()
    if (not(extrainfo is None)):
        text = f'ECG Quality: {extrainfo["ecg_quality"]}\nBPM_new: {m["bpm"]}\n'
        text += '\n'.join([f'{feat:10}: {extrainfo[feat]:5.4}' for feat in modelconfig.features])
        plt.figtext(.004, .2, text, ha="left")
    if (dst):
        plt.savefig(
            dst / f'{fin}_{start.strftime("%m-%d-%Y_%H:%M:%S")}_detrended_withbeats.png'
        )
    else:
        plt.savefig(
            Path(__file__).parent / 'results' / 'assets' / f'{fin}_{start.strftime("%m-%d-%Y_%H:%M:%S")}_detrended_withbeats.png'
        )
    plt.close()
    plt.clf()
    ### nk quality / processing
    # # print(data)
    # # ecg_cleaned = nk.ecg_clean(dataSlice, sampling_rate=samplerate)
    # sigs, info = nk.ecg_process(dataSlice, sampling_rate=samplerate)
    # # cleaned = hp.remove_baseline_wander(hp.scale_data(dataSlice), samplerate)
    # # bandpassed = nk.signal_filter(cleaned, lowcut=20, highcut=80)
    # # nk.ecg_plot(sigs)
    # ecg = sigs['ECG_Clean']
    # quality = nk.ecg_quality(ecg, sampling_rate=samplerate, method="zhao2018")


    # w, m = hp.process(hp.scale_data(ecg, samplerate), samplerate, clean_rr=False)
    # hp.plotter(w, m, figsize=(12, 4), show=False)
    # if (not(extrainfo is None)):
    #     text = f'ECG Quality: {quality}\n'
    #     text += '\n'.join([f'{feat:10}: {extrainfo[feat]:5.4}' for feat in modelconfig.features])
    #     plt.figtext(.004, .2, text, ha="left")
    # if (dst):
    #     plt.savefig(
    #         dst / f'{fin}_{start.strftime("%m-%d-%Y_%H:%M:%S")}_neurokit_withbeats.png'
    #     )
    # else:
    #     plt.savefig(
    #         Path(__file__).parent / 'results' / 'assets' / f'{fin}_{start.strftime("%m-%d-%Y_%H:%M:%S")}_neurokit_default.png'
    #     )
    # plt.close()
    # plt.clf()

def quantifyNoise(df, searchDirectory='/home/rkaufman/workspace/remote'):
    resk, respower, resbaseline, restemplate = list(), list(), list(), list()
    for _, row in df.iterrows():
        fin, start, stop = row['fin_study_id'], row['start'], row['stop']
        file = datautils.findFileByFIN(str(fin), searchDirectory)
        dataslice, samplerate = datautils.getSlice(file, datautils.HR_SERIES, start, stop)
        ecg_cleaned = nk.ecg_clean(dataslice, sampling_rate=samplerate, method='neurokit')
        resk.append(k_SQI(ecg_cleaned))
        respower.append(p_SQI(ecg_cleaned, samplerate, 1))
        resbaseline.append(bas_SQI(ecg_cleaned, samplerate, 1))
        restemplate.append(orphanidou2015_sqi(ecg_cleaned, samplerate))

    df['ecg_quality_k'] = resk
    df['ecg_quality_power'] = respower#df.apply(lambda r: getNoiseReading(r, 'power'), axis='columns')
    df['ecg_quality_baseline'] = resbaseline#df.apply(lambda r: getNoiseReading(r, 'baseline'), axis='columns')
    df['ecg_quality_template'] = restemplate#df.apply(lambda r: getNoiseReading(r, 'templateMatch'), axis='columns')
    return df

def detectedBeatPlotter(df, searchDirectory="/home/rkaufman/workspace/remote", beatDetector="heartpy"):
    """Plot segments showing detected beats

    Args:
        df (_type_): df of segments, must contain columns: [`start`, `stop`, `fin_study_id`]
        beatDetector (str):  Type of beat detector
    """
    hr_series = datautils.HR_SERIES
    b2b_series = datautils.B2B_SERIES

    for _, row in df.iterrows():
        start, stop = row['start'], row['stop']
        fin = row['fin_study_id']

        file = datautils.findFileByFIN(str(fin), searchDirectory)
        dataslice, samplerate = datautils.getSlice(file, hr_series, start, stop)
        plotSlice(dataslice, samplerate, fin, start, )

import model.utilities as mu

def compareFeatureSets():
    modelconfig = mu.getModelConfig()
    trainData = pd.read_csv(
        Path(__file__).parent / 'data' / 'assets' / modelconfig.trainDataFile
    )
    testData = pd.read_csv(
        Path(__file__).parent / 'data' / 'assets' / modelconfig.goldDataFile
    )


    out = str()
    for feat in modelconfig.features:
        kst = kstest(trainData[trainData['label'] == 'SINUS'][feat], testData[testData['label'] == 'SINUS'][feat])
        print(f'For {feat}, p-value from kstest: {kst.pvalue}')
        out += f'{feat},{kst.pvalue}\n'
        fig, ax = plt.subplots()
        trainFeatures = winsorize(trainData[feat], limits=[.05, .05])
        testFeatures = winsorize(testData[feat], limits=[.05, .05])
        ax.hist(trainFeatures, bins=200, label=f"Train set {feat}", histtype="step", density=True)
        ax.hist(testFeatures, bins=200, label=f"Eval set {feat}", histtype="step", density=True)
        plt.legend()
        plt.savefig(
            Path(__file__).parent / 'results' / 'assets' / f'{feat.replace(os.sep, "_")}_comparison.svg'
        )
        # plt.show()
    print(out)

def showConfidentlyIncorrects(df, pos_label='ATRIAL_FIBRILLATION', threshold=0.8):
    """Requires df columns:
    - 'fin_study_id'
    - 'start'
    - 'stop',
    - 'label'
    - 'probabilities'

    Args:
        df (_type_): _description_
        pos_label (str, optional): _description_. Defaults to 'ATRIAL_FIBRILLATION'.
        threshold (float, optional): _description_. Defaults to 0.8.
    """
    df = df.reset_index()
    labels = list(df['label'])
    for i, prob in enumerate(df['probabilities']):
        if prob > threshold and labels[i] != 'ATRIAL_FIBRILLATION':
            plotSegment(df['fin_study_id'][i], df['start'][i], df['stop'][i], extrainfo=df.iloc[i, :])

def outputRRIntervals(df, dstCsv, winLength_s=120, searchDirectory='/home/rkaufman/workspace/remote'):
    dfResult = {
        'fin_study_id': list(),
        'peak_time': list(),
        'rr_interval_ms': list(),
        'is_segment_of_interest': list(),
        'clinician_label': list(),
        'clinician_notes': list(),
        'model_prediction': list()
    }
    for _, row in tqdm(df.iterrows(), total=len(df)):
        fin, start, stop = row['fin_study_id'], row['start'], row['stop']
        file = datautils.findFileByFIN(str(fin), searchDirectory)
        mid = start + (stop-start)/2
        newStart, newStop = mid - dt.timedelta(seconds=winLength_s//2), mid+dt.timedelta(seconds=winLength_s//2)
        dataslice, samplerate = datautils.getSlice(file, datautils.HR_SERIES, newStart, newStop)
        detrended = hp.remove_baseline_wander(dataslice, samplerate)
        detrended_scaled = hp.scale_data(detrended)
        w, m = hp.process(detrended_scaled, samplerate, clean_rr=False)
        peaks = w['peaklist']
        correctPeaks = w['binary_peaklist']
        for i, peakIdx in enumerate(peaks):
            if (i == len(peaks)-1):
                continue
            if (correctPeaks[i] + correctPeaks[i+1] != 2):
                # print(correctPeaks[i])
                continue
            rrIntervalIndices = peaks[i], peaks[i+1]
            rrInterval = (start + dt.timedelta(seconds=peaks[i]/samplerate), start+dt.timedelta(seconds=peaks[i+1]/samplerate))
            rrInterval_ms = (rrInterval[1]-rrInterval[0]).total_seconds() * 1000
            # print(rrInterval_ms)
            timeElapsedSinceStart_s = peakIdx / samplerate
            peakTime = newStart + dt.timedelta(seconds=timeElapsedSinceStart_s)
            dfResult['fin_study_id'].append(fin)
            dfResult['peak_time'].append(peakTime)
            dfResult['rr_interval_ms'].append(rrInterval_ms)
            if (start < peakTime and peakTime < stop):
                dfResult['is_segment_of_interest'].append(True)
                dfResult['clinician_label'].append(row['label'])
                dfResult['model_prediction'].append(row['model_prediction'])
                dfResult['clinician_notes'].append(row['notes'])
            else:
                dfResult['is_segment_of_interest'].append(False)
                dfResult['clinician_label'].append('N/A')
                dfResult['model_prediction'].append('N/A')
                dfResult['clinician_notes'].append('N/A')
    pd.DataFrame(dfResult).to_csv(dstCsv, index=False)

from itertools import cycle

if __name__ == '__main__':
    pass
    # detectedBeatPlotter(pd.read_csv(
    #     Path(__file__).parent / 'data' / 'assets' / 'sinus_segments_gold.csv',
    #     parse_dates=['start', 'stop']))
    #compareFeatureSets()

    # print('Doin evalset')
    # withnoise = quantifyNoise(pd.read_csv(
    #     './evalset_withpredictions.csv',
    #     parse_dates=['start', 'stop']))
    # withnoise.to_csv('evalset_withpredictions.csv')
    # print('Done evalset')

    # print('Doin testset')
    # withnoise = quantifyNoise(pd.read_csv(
    #     './testset_withpredictions.csv',
    #     parse_dates=['start', 'stop']))
    # withnoise.to_csv('testset_withpredictions.csv')
    # print('Done testset')
    # print('Doin trainset')
    # withnoise = quantifyNoise(pd.read_csv(
    #     Path(__file__).parent / 'data' / 'assets' / 'trainset_10000_featurized_withfilter.csv',
    #     parse_dates=['start', 'stop']))
    # withnoise.to_csv('trainset_withfilters')
    # print('Done trainset')

    # withnoise = quantifyNoise(pd.read_csv(
    #     Path(__file__).parent / 'data' / 'assets' / 'testset_featurized_withextras.csv',
    #     parse_dates=['start', 'stop']))
    # withnoise.to_csv('testset_featurized_withextras.csv')


    ### ecg quality plotting
    # df = pd.read_csv('./data/assets/trainset_10000_featurized_withfilter.csv', parse_dates=['start', 'stop'])
    # df1, df2, df3 = df[df['ecg_quality'] == 'Excellent'], df[df['ecg_quality'] == 'Barely acceptable'], df[df['ecg_quality'] == 'Unacceptable']
    # dirs = cycle(['excellent', 'barely_acceptable', 'unacceptable'])
    # for d in [df1, df2, df3]:
    #     dst = next(dirs)
    #     print(dst)
    #     randos = d.sample(n=20)
    #     plotDFSegments(randos, Path(f'/home/rkaufman/workspace/afib_detection/results/assets/{dst}'))

    ### RR-interval csv-saving
    # df = pd.read_csv('./testset_withpredictions.csv', parse_dates=['start', 'stop'])
    # confidentlyIncorrects = df[(df['afib_confidence'] > .7) & (df['label'] == 'SINUS')]
    # confidentCorrects = df[(df['afib_confidence'] > .7) & (df['label'] == 'ATRIAL_FIBRILLATION')]
    # showRRIntervals(confidentlyIncorrects, 'confidently_incorrectAFIB_withintervals.csv')
    # showRRIntervals(confidentCorrects, 'confidently_correctAFIB_withintervals.csv')
