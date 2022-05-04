import datetime as dt
from pathlib import Path
import pandas as pd
import heartpy as hp
import matplotlib.pyplot as plt
import os
from pprint import PrettyPrinter 

import data.utilities as datautils
# import utilities as datautils
from scipy.stats.mstats import winsorize
from scipy.stats import kstest
from data.custom_hp_plotters import plotter, plot_poincare

import neurokit2 as nk
from data.sqis import k_SQI, p_SQI, bas_SQI, orphanidou2015_sqi

def plotDFSegments(df, dst):
    for _, row in df.iterrows():
        fin, start, stop = row['fin_study_id'], row['start'], row['stop']
        plotSegment(fin, start, stop, extrainfo=row, dst=dst)
def plotSegment(fin, start, stop, searchDirectory='/home/romman/workspace/remote', extrainfo=None, dst=None):
    file = datautils.findFileByFIN(str(fin), searchDirectory)
    dataslice, samplerate = datautils.getSlice(file, datautils.HR_SERIES, start, stop)
    plotSlice(dataslice, samplerate, searchDirectory, fin, start, extrainfo, dst=dst)

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
        text += f"Model prediction: {extrainfo['model_prediction']}\n"
        text += f"AFIB confidence: {extrainfo['afib_confidence']:.2}\n"
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
    def getNoiseReading(row, noiseType):
        fin, start, stop = row['fin_study_id'], row['start'], row['stop']
        file = datautils.findFileByFIN(str(fin), searchDirectory)
        dataslice, samplerate = datautils.getSlice(file, datautils.HR_SERIES, start, stop)
        ecg_cleaned = nk.ecg_clean(dataslice, sampling_rate=samplerate, method='neurokit')
        try:
            if (noiseType == 'kurtosis'):
                noiseSQI = k_SQI(ecg_cleaned)
            elif (noiseType == 'power'):
                # https://sapienlabs.org/lab-talk/factors-that-impact-power-spectrum-density-estimation/
                noiseSQI = p_SQI(ecg_cleaned, samplerate, 1)
            elif (noiseType == 'baseline'):
                # https://sapienlabs.org/lab-talk/factors-that-impact-power-spectrum-density-estimation/
                noiseSQI = bas_SQI(ecg_cleaned, samplerate, 1)
            elif (noiseType == 'templateMatch'):
                noiseSQI = orphanidou2015_sqi(ecg_cleaned, samplerate)
            # return nk.ecg_quality(detrended_scaled, sampling_rate=samplerate, method="zhao2018")
            return nk.ecg_quality(detrended_scaled, sampling_rate=samplerate)
        except:
            return 'failed_check'

    df['ecg_quality_k'] = df.apply(lambda r: getNoiseReading(r, 'kurtosis'), axis='columns')
    df['ecg_quality_power'] = df.apply(lambda r: getNoiseReading(r, 'power'), axis='columns')
    df['ecg_quality_baseline'] = df.apply(lambda r: getNoiseReading(r, 'baseline'), axis='columns')
    df['ecg_quality_template'] = df.apply(lambda r: getNoiseReading(r, 'templateMatch'), axis='columns')
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

from itertools import cycle

if __name__ == '__main__':
    # detectedBeatPlotter(pd.read_csv(
    #     Path(__file__).parent / 'data' / 'assets' / 'sinus_segments_gold.csv', 
    #     parse_dates=['start', 'stop']))
    # compareFeatureSets('trainset_10000_featurized_withextras.csv', 'testset_featurized_withextras.csv', dst='./results/assets/train_vs_test')
    # compareFeatureSets('evalset_final_cleaned.csv','testset_featurized_withextras.csv',dst='./results/assets/eval_vs_test')
    # compareFeatureSets('trainset_10000_featurized_withextras.csv', 'evalset_final_cleaned.csv', dst='./results/assets/train_vs_eval')
    # withnoise = quantifyNoise(pd.read_csv(
    #     Path(__file__).parent / 'data' / 'assets' / 'trainset_10000_featurized_withfilter.csv', 
    #     parse_dates=['start', 'stop']))
    # withnoise.to_csv('trainset_10000_featurized_withfilter.csv')

    # withnoise = quantifyNoise(pd.read_csv(
    #     Path(__file__).parent / 'data' / 'assets' / 'testset_featurized_withextras.csv', 
    #     parse_dates=['start', 'stop']),
    #     searchDirectory='/home/romman/workspace/remote')
    # withnoise.to_csv('testset_featurized_withextras.csv')


    ## ecg quality plotting
    df = pd.read_csv('./data/assets/testset_withpredictions.csv', parse_dates=['start', 'stop'])
    df_confident_incorrects = df[(df['afib_confidence'] > .7) & (df['label'] != 'ATRIAL_FIBRILLATION')]
    df_correctly_afibs = df[(df['label'] == 'ATRIAL_FIBRILLATION') & (df['model_prediction'] == 'ATRIAL_FIBRILLATION')]
    df_sinuses = df[(df['sinus_confidence'] > .7) & (df['label'] != 'ATRIAL_FIBRILLATION')]
    dirs = cycle(['confident_incorrects', 'correctly_afibs', 'correctly_not_afibs'])
    df1, df2, df3 = df_confident_incorrects, df_correctly_afibs, df_sinuses
    for d in [df1, df2, df3]:
        dst = next(dirs)
        print(dst)
        randos = d#.sample(n=20)
        plotDFSegments(randos, Path(f'/home/romman/workspace/afib_detection/results/assets/{dst}'))