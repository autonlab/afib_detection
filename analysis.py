from pathlib import Path
import pandas as pd
import heartpy as hp
import matplotlib.pyplot as plt
from pprint import PrettyPrinter 

import data.utilities as datautils
# import utilities as datautils
from scipy.stats.mstats import winsorize
from scipy.stats import kstest

import neurokit2 as nk

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
    def getNoiseReading(row):
        fin, start, stop = row['fin_study_id'], row['start'], row['stop']
        file = datautils.findFileByFIN(str(fin), searchDirectory)
        dataslice, samplerate = datautils.getSlice(file, datautils.HR_SERIES, start, stop)
        detrended_scaled = hp.scale_data(hp.remove_baseline_wander(dataslice, samplerate))
        try:
            # return nk.ecg_quality(detrended_scaled, sampling_rate=samplerate, method="zhao2018")
            return nk.ecg_quality(detrended_scaled, sampling_rate=samplerate)
        except:
            return 'failed_check'

    df['ecg_quality'] = df.apply(getNoiseReading, axis='columns')
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



    for feat in modelconfig.features:
        kst = kstest(trainData[trainData['label'] == 'SINUS'][feat], testData[testData['label'] == 'SINUS'][feat])
        print(f'For {feat}, p-value from kstest: {kst.pvalue}')
    #     fig, ax = plt.subplots()
    #     trainFeatures = winsorize(trainData[feat], limits=[.05, .05])
    #     testFeatures = winsorize(testData[feat], limits=[.05, .05])
    #     ax.hist(trainFeatures , bins=200, label=f"Train set {feat}", histtype="step", density=True)
    #     ax.hist(testFeatures , bins=200, label=f"Test set {feat}", histtype="step", density=True)
    #     plt.legend()
    #     plt.show()
        # ax.hist()

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
    pass
    # detectedBeatPlotter(pd.read_csv(
    #     Path(__file__).parent / 'data' / 'assets' / 'sinus_segments_gold.csv', 
    #     parse_dates=['start', 'stop']))
    # compareFeatureSets()
    # withnoise = quantifyNoise(pd.read_csv(
    #     Path(__file__).parent / 'data' / 'assets' / 'trainset_10000_featurized_withfilter.csv', 
    #     parse_dates=['start', 'stop']))
    # withnoise.to_csv('trainset_10000_featurized_withfilter.csv')

    withnoise = quantifyNoise(pd.read_csv(
        Path(__file__).parent / 'data' / 'assets' / 'testset_featurized_withextras.csv', 
        parse_dates=['start', 'stop']))
    withnoise.to_csv('testset_featurized_withextras.csv')


    ### ecg quality plotting
    # df = pd.read_csv('./data/assets/trainset_10000_featurized_withfilter.csv', parse_dates=['start', 'stop'])
    # df1, df2, df3 = df[df['ecg_quality'] == 'Excellent'], df[df['ecg_quality'] == 'Barely acceptable'], df[df['ecg_quality'] == 'Unacceptable']
    # dirs = cycle(['excellent', 'barely_acceptable', 'unacceptable'])
    # for d in [df1, df2, df3]:
    #     dst = next(dirs)
    #     print(dst)
    #     randos = d.sample(n=20)
    #     plotDFSegments(randos, Path(f'/home/rkaufman/workspace/afib_detection/results/assets/{dst}'))

