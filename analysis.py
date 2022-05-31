import datetime as dt
import functools
from pathlib import Path
import pandas as pd
import heartpy as hp
import matplotlib.pyplot as plt
import os
from pprint import PrettyPrinter, pprint
from tqdm import tqdm
from typing import List, Union

import data.utilities as datautils
from sklearn.preprocessing import OneHotEncoder
from sklearn.inspection import permutation_importance
from sklearn.metrics import roc_auc_score, make_scorer
from scipy.stats.mstats import winsorize
from scipy.stats import kstest
import numpy as np
from data.custom_hp_plotters import plotter, plot_poincare

import neurokit2 as nk
from data.sqis import k_SQI, p_SQI, bas_SQI, orphanidou2015_sqi

def plotDFSegments(df, dst):
    for _, row in df.iterrows():
        fin, start, stop = row['fin_study_id'], row['start'], row['stop']
        plotSegment(fin, start, stop, extrainfo=row, dst=dst)

def plotSegment(fin, start, stop, searchDirectory='/home/rkaufman/workspace/remote', extrainfo=None, dst=None):
    file = datautils.findFileByFIN(str(fin), searchDirectory)
    dataslice, samplerate = datautils.getSlice(file, datautils.HR_SERIES, start, stop)
    plotSlice(dataslice, samplerate, searchDirectory, fin, start, extrainfo, dst=dst)

def plotSlice_butterworthFiltered(dataSlice, samplerate, searchDir, fin=None, start=None, extrainfo=None, dst=None):
    sigs, info = nk.ecg_process(dataSlice, sampling_rate=samplerate)
    ecg = sigs['ECG_Clean']
    filteredSlice = nk.signal_filter(dataSlice, sampling_rate=samplerate, order=3, lowcut=.5, highcut=40)
    w, m = hp.process(hp.scale_data(ecg, samplerate), samplerate, clean_rr=False)
    hp.plotter(w, m, figsize=(12, 4), show=False)
    if (dst):
        plt.savefig(
            dst / f'{fin}_{start.strftime("%m-%d-%Y_%H:%M:%S")}_2.png'
        )

    plt.clf()

def plotSlice_morphology(dataSlice, samplerate, searchDir, fin=None, start=None, extrainfo=None, dst=None):
    sigs, info = nk.ecg_process(dataSlice, sampling_rate=samplerate)
    ecg = sigs['ECG_Clean']
    _, rpeaks = nk.ecg_peaks(ecg, sampling_rate=samplerate)
    _, waves_peak = nk.ecg_delineate(ecg, rpeaks, sampling_rate=samplerate, method="peak", show=True, show_type="bounds_P")
    text = ''
    thingsToCollect = {
        'type'
    }
    for sqi in thingsToCollect:
        text += f'{sqi}: {extrainfo[sqi]}\n'
    plt.figtext(.1, .75, text, ha="left",
        bbox = {'facecolor': 'red', 'alpha': 0.5, 'pad': 10})
    if (dst):
        plt.savefig(
            dst / f'{fin}_{start.strftime("%m-%d-%Y_%H:%M:%S")}.png'
        )

    plt.clf()

def plotSlice_hrv(dataSlice, samplerate, searchDir, fin=None, start=None, extrainfo=None, dst=None):
    # plt.figure(figsize=(12, 16))
    stop = start + dt.timedelta(seconds=10)
    newStart, newStop = start - dt.timedelta(seconds=25), stop + dt.timedelta(seconds=25)
    file = datautils.findFileByFIN(str(fin), searchDir)
    dataslice, samplerate_n = datautils.getSlice(file, datautils.HR_SERIES, newStart, newStop)
    sigs, info = nk.ecg_process(dataslice, sampling_rate=samplerate)
    # # cleaned = hp.remove_baseline_wander(hp.scale_data(dataSlice), samplerate)
    # # bandpassed = nk.signal_filter(cleaned, lowcut=20, highcut=80)
    # # nk.ecg_plot(sigs)
    ecg = sigs['ECG_Clean']
    peaks, peakinfo = nk.ecg_peaks(ecg, sampling_rate=samplerate)
    print(peaks[peaks['ECG_R_Peaks'] != 0])
    hrv_indices = nk.hrv_frequency(peaks, sampling_rate=samplerate, show=True)
    print(hrv_indices)
    plt.show()
    if (dst):
        plt.savefig(
            dst / f'{fin}_{start.strftime("%m-%d-%Y_%H:%M:%S")}.svg'
        )
    
    plt.clf()



def plotSlice(dataSlice, samplerate, searchDirectory, fin=None, start=None, extrainfo=None, dst=None):
    # fig, (ax1, ax2) = plt.subplots(2, 2)
    plt.subplot(221)

    # equivalent but more general
    plt.figure(figsize=(12, 8))
    # plt.subplots(2, 2)

    plt.subplots_adjust(wspace= 0.25, hspace= 0.25)

    # plt.subplot(2, 1, 1)
    stop = start + dt.timedelta(seconds=10)
    newStart, newStop = start - dt.timedelta(seconds=25), stop + dt.timedelta(seconds=25)
    file = datautils.findFileByFIN(str(fin), searchDirectory)
    dataslice, samplerate_n = datautils.getSlice(file, datautils.HR_SERIES, newStart, newStop)
    detrended = hp.remove_baseline_wander(dataslice, samplerate_n)
    detrended_scaled_og = hp.scale_data(detrended)
    w, m = hp.process(detrended_scaled_og, samplerate_n, clean_rr=False)
    ax1 = plt.subplot(2, 2, 2)
    plot_poincare(w, m, ax1, show=False, title="Poincare: 60 second window centered about segment")
    # ax2.plot(p)
    # print(type(p))


    ax2 = plt.subplot(2, 2, (3, 4))
    # fig.add_subplot(2, 2, (3, 4))
    detrended = hp.remove_baseline_wander(dataSlice, samplerate)
    detrended_scaled = hp.scale_data(detrended)
    w, m = hp.process(detrended_scaled, samplerate, clean_rr=False)
    plotter(w, m, ax2, show=False)
    # # plt.plot(hp.scale_data(hp.filter_signal(detrended, cutoff=[.75, 3.5], sample_rate=samplerate, filtertype='bandpass')))
    modelconfig = mu.getModelConfig()
    if (not(extrainfo is None)):
        # ax3 = plt.subplot(222)
        thingsToCollect = [
            'ecg_quality_k',
            'ecg_quality_power',
            'ecg_quality_template',
        ]
        text = ''
        text += f"Annotation: {extrainfo['label']}\n"
        # text += f"Model prediction: {extrainfo['model_prediction']}\n"
        # text += f"AFIB confidence: {extrainfo['afib_confidence']:.2}\n"
        ecg_cleaned = detrended_scaled
        kurtosis = k_SQI(ecg_cleaned)
        # https://sapienlabs.org/lab-talk/factors-that-impact-power-spectrum-density-estimation/
        power = p_SQI(ecg_cleaned, samplerate, 1)
        template = orphanidou2015_sqi(ecg_cleaned, samplerate)
        noiseCycle = cycle([kurtosis, power, template])
        for sqi in thingsToCollect:
            text += f'{sqi}: {float(next(noiseCycle)):.3}\n'
        # text += '\n'.join([f'{feat:10}: {extrainfo[feat]:5.4}' for feat in modelconfig.features])
        # plt.figtext(.66, .58, text, ha="left",
        plt.figtext(.1, .75, text, ha="left",
            bbox = {'facecolor': 'red', 'alpha': 0.5, 'pad': 10})
    if (dst):
        plt.savefig(
            dst / f'{fin}_{start.strftime("%m-%d-%Y_%H:%M:%S")}.svg'
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

def compareFeatureSets(df1Path, df2Path, dst=Path('results/assets')):
    modelconfig = mu.getModelConfig()
    # trainData = pd.read_csv(
    #     Path(__file__).parent / 'data' / 'assets' / modelconfig.trainDataFile
    # )
    # testData = pd.read_csv(
    #     Path(__file__).parent / 'data' / 'assets' / modelconfig.goldDataFile
    # )
    trainData = pd.read_csv(
        Path(__file__).parent / 'data' / 'assets' / df1Path
    )
    testData = pd.read_csv(
        Path(__file__).parent / 'data' / 'assets' / df2Path
    )
    # trainData, testData = pd.read_csv('./evalset_withpredictions.csv'), pd.read_csv('./testset_withpredictions')


    out = str()
    newfeats = ['ecg_quality_k', 'ecg_quality_power', 'ecg_quality_template']
    for feat in modelconfig.features:#modelconfig.features:
        kst = kstest(trainData[feat], testData[feat])
        print(f'For {feat}, p-value from kstest: {kst.pvalue}')
        out += f'{feat},{kst.pvalue}\n'
        fig, ax = plt.subplots()
        trainFeatures = winsorize(trainData[feat], limits=[.05, .05])
        testFeatures = winsorize(testData[feat], limits=[.05, .05])
        ax.hist(trainFeatures, bins=200, label=f"{df1Path.split('.')[0]} {feat}", histtype="step", density=True)
        ax.hist(testFeatures, bins=200, label=f"{df2Path.split('.')[0]} {feat}", histtype="step", density=True)
        plt.legend()
        # plt.savefig(
        #     Path(__file__).parent / 'results' / 'assets' / f'{feat.replace(os.sep, "_")}_comparison.svg'
        # )
        plt.savefig(
            Path(dst) / f'{feat.replace(os.sep, "_")}_comparison.svg'
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

def permutationFeatureImportance(model, X_test: pd.DataFrame, y_test: pd.DataFrame, n_repeats=1, feature_subset=None):
    """Function that uses permutation feature importance to give a model-agnostic view of feature importance

    Args:
        model (_type_): _description_
        X_test (pd.DataFrame): _description_
        y_test (pd.DataFrame): _description_
        n_repeats (int, optional): _description_. Defaults to 1.
        feature_subset (_type_, optional): _description_. Defaults to None.
    """
    if (feature_subset is None):
        feature_subset = X_test.columns
    feature_subset = [f.lower() for f in feature_subset]

    X_test_OG = X_test.copy().reset_index(drop=True)

    featureImportances = dict()
    enc = OneHotEncoder(categories='auto')
    y_test = enc.fit_transform(y_test.to_numpy().reshape(-1, 1)).toarray()

    originalScore = roc_auc_score(y_test, model.predict_proba(X_test))
    #go through each feature
    for feat in feature_subset:
        permutedError = 0
        for _ in range(n_repeats):
            X_test = X_test_OG.copy() #collect copy with all other features constant
            sampled =  X_test[feat].sample(frac=1).reset_index(drop=True) #sample entire row without replacement
            X_test[feat] = sampled
            permutedPredictions = model.predict_proba(X_test)

            # print(permutedPredictions)
            permutedScore = roc_auc_score(y_test, permutedPredictions)
            permutedError += originalScore - permutedScore
        permutedError = permutedError / n_repeats
        
        featureImportances[feat] = permutedError
    # model.predict = model.predict_proba
    # x = permutation_importance(model, X_test_OG, y_test, scoring=make_scorer(roc_auc_score), n_repeats=n_repeats)
    # for i, val in enumerate(X_test_OG.columns):
    #     print(f'{val}: {x["importances_mean"][i]:.3} | {featureImportances[val]:.3}')
    # print(list(zip(x['importances_mean'], featureImportances.values())))
    fiSorted = sorted(featureImportances.items(), key=lambda x: x[1], reverse=True)
    return featureImportances, fiSorted
import altair as alt
import altair_viewer as altview
def compareFeatures_plot(dfs: List[pd.DataFrame], features: List[str], dfTitles: Union[List[str], None] = None):
    if dfTitles == None:
        dfTitles = [f'dataset {i}' for i in range(len(dfs))]
    for i, df in enumerate(dfs):
        alt.data_transformers.disable_max_rows()
        c = alt.Chart(df.loc[:, features+['label']]).mark_circle(size=60).encode(
            x=features[0],
            y=features[1],
            color="label",
            tooltip=features,
        ).properties(
            title=dfTitles[i]
        )
        altview.display(c)
        input('y')
    




from itertools import cycle

if __name__ == '__main__':
    df_trainset = pd.read_csv(
        Path(__file__).parent / 'data' / 'assets' / 'trainset_featurized.csv',
        parse_dates=['start', 'stop']
        )
    df_evalset = pd.read_csv(
        Path(__file__).parent / 'data' / 'assets' / 'evalset_featurized.csv',
        parse_dates=['start', 'stop']
        )
    df_testset = pd.read_csv(
        Path(__file__).parent / 'data' / 'assets' / 'testset_featurized.csv',
        parse_dates=['start', 'stop']
        )
    compareFeatures_plot(
        [df_trainset, df_evalset, df_testset],
        ['sse_1_clusters', 'pnn20'],
        ['trainset', 'evalset', 'testset']
    )
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
    # compareFeatureSets('trainset_10000_featurized_withextras.csv', 'testset_featurized_withextras.csv', dst='./results/assets/train_vs_test')
    # compareFeatureSets('evalset_final_cleaned.csv','testset_featurized_withextras.csv',dst='./results/assets/eval_vs_test')
    # compareFeatureSets('trainset_10000_featurized_withextras.csv', 'evalset_final_cleaned.csv', dst='./results/assets/train_vs_eval')
    # withnoise = quantifyNoise(pd.read_csv(
    #     Path(__file__).parent / 'data' / 'assets' / 'trainset_10000_featurized_withfilter.csv',
    #     parse_dates=['start', 'stop']))
    # withnoise.to_csv('trainset_withfilters')
    # print('Done trainset')

    # withnoise = quantifyNoise(pd.read_csv(
    #     Path(__file__).parent / 'data' / 'assets' / 'testset_featurized_withextras.csv', 
    #     parse_dates=['start', 'stop']),
    #     searchDirectory='/home/romman/workspace/remote')
    # withnoise.to_csv('testset_featurized_withextras.csv')


    ## ecg quality plotting
    # df = pd.read_csv('./data/assets/testset_withpredictions.csv', parse_dates=['start', 'stop'])
    # df_confident_incorrects = df[(df['afib_confidence'] > .7) & (df['label'] != 'ATRIAL_FIBRILLATION')]
    # df_correctly_afibs = df[(df['label'] == 'ATRIAL_FIBRILLATION') & (df['model_prediction'] == 'ATRIAL_FIBRILLATION')]
    # df_sinuses = df[(df['sinus_confidence'] > .7) & (df['label'] != 'ATRIAL_FIBRILLATION')]
    # dirs = cycle(['confident_incorrects', 'correctly_afibs', 'correctly_not_afibs'])
    # df1, df2, df3 = df_confident_incorrects, df_correctly_afibs, df_sinuses
    # outputRRIntervals(df_sinuses, './results/confident_sinuses.csv')
    # for d in [df1, df2, df3]:
    #     dst = next(dirs)
    #     print(dst)
    #     randos = d#.sample(n=20)
    #     plotDFSegments(randos, Path(f'/home/rkaufman/workspace/afib_detection/results/assets/'))

    # plotDFSegments(
    #     pd.read_csv('./data/assets/evalset_featurized.csv', parse_dates=['start', 'stop']),
    #     Path('/home/rkaufman/workspace/afib_detection/results/assets/evalset_plots')
    # )
