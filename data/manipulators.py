from heartpy.preprocessing import scale_data
from joblib import Parallel, delayed
import math
import numpy as np
import neurokit2 as nk
import os
import pandas as pd
import pickle
from pathlib import Path
from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import StandardScaler
import time
from tqdm import tqdm
from . import utilities as datautils

def loadScaler():
    with open(Path(__file__).parent / 'assets' / 'scaler_physionet.pkl', 'rb') as readfile:
        scaler = pickle.load(readfile)
    return scaler

def applyScaler(df, featuresToScale, scaler):
    # print(df[featuresToScale])
    # scaledDf = pd.DataFrame(scaler.transform(df[featuresToScale]), columns=featuresToScale)
    df[featuresToScale] = scaler.transform(df[featuresToScale])
    # print(df[featuresToScale])
    return df
from scipy.stats.mstats import winsorize
def winsorizeDF(df, featuresToWinsorize):
    for feat in featuresToWinsorize:
        limits = [.05, .05]
        df[feat] = winsorize(df[feat], limits=limits)
    return df
def computeAndApplyScaler(df, featuresToScale, scalerType="standard"):
    """_summary_

    Args:
        df (_type_): _description_
        featuresToScale (_type_): _description_
        scalerType (str, optional): Type of scaler to use from sklearn.preprocessing lib. Defaults to "standard".
    """

    if scalerType=='standard':
        scaler = StandardScaler()
    else:
        raise ValueError(f'scalerType [{scalerType}] unsupported!')
    df[featuresToScale] = scaler.fit_transform(df[featuresToScale])
    # with open(Path(__file__).parent / 'assets' / 'scaler_physionet.pkl', 'wb') as writefile:
    #     pickle.dump(scaler, writefile)
    return df, scaler

def applySplits(df, prespecifiedTestSet=None, test_size=.2, group_identifier='fin_study_id'):
    """Split given dataframe into train and test sets, either by group_identifier or given prespecified test entries

    Args:
        df (pd.DataFrame): dataframe to split 
        prespecifiedTestSet (List[Num], optional): List of `fin_study_id`'s to keep in testset. Defaults to None.

    Returns:
        tuple(pd.DataFrame, pd.DataFrame): trainset, testset
    """
    if (prespecifiedTestSet is None):
        trainIndices, testIndices = next(GroupShuffleSplit(n_splits=1, test_size=.2, random_state=7).split(df, groups=df[group_identifier]))
        trainset, testset = df.iloc[trainIndices], df.iloc[testIndices]
    else:
        trainset = df[~df[group_identifier].isin(prespecifiedTestSet)]
        testset = df[df[group_identifier].isin(prespecifiedTestSet)]
    return trainset, testset

def oversample(df, classColumn, samplingLikelihoodColumn):
    """Oversample classes less represented than max represented class

    Args:
        df (_type_): _description_
        classColumn (str): class to oversample from
        samplingLikelihoodColumn (str): associated likelihood to oversample given data point
    """
    maxClassSize = df[classColumn].value_counts().max()
    allDFs = [df]
    for idx, classSubset in df.groupby(classColumn):
        samplesForClass = classSubset.sample(maxClassSize - len(classSubset), replace=True, weights=classSubset[samplingLikelihoodColumn])
        allDFs.append(samplesForClass)
    return pd.concat(allDFs)

def filterAndNormalize(df, features, preexistingScaler=None):
    # Filter
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df = df.dropna()

    # Normalize
    if (preexistingScaler):
        df_normalized = applyScaler(df, features, preexistingScaler)
    else:
        df_normalized = computeAndApplyScaler(df, features)
    return df_normalized

def remapLabels(df, labelColumn, labelmap):
    df[labelColumn] = df[labelColumn].map(lambda x: labelmap[x] if x in labelmap else x)
    return df

def padToDense(m: np.array, maxLength: int) -> np.array:
    #https://stackoverflow.com/questions/37676539/numpy-padding-matrix-of-different-row-size
    result = np.zeros((m.shape[0], maxLength))
    for idx, row in enumerate(m):
        result[idx, :len(row)] += row
    return result

def pickleDump(any: object, dst: str) -> None:
    with open(dst, 'wb') as writefile:
        pickle.dump(any, writefile)

def pickleLoad(src: str) -> object:
    with open(src, 'rb') as readfile:
        res = pickle.load(readfile)
    return res

def slicer(row, searchDir):
    fin_study_id, start, stop = int(row['fin_study_id']), row['start'], row['stop']
    dataslice, samplerate = datautils.getSliceFIN(fin_study_id, datautils.HR_SERIES, start, stop, searchDir)
    try:
        sigs, info = nk.ecg_process(dataslice, sampling_rate=samplerate)
        ecg = sigs['ECG_Clean']
        filteredSlice = ecg
    except:
        #in empty vector return case, use butterworth order 3 filter
        filteredSlice = nk.signal_filter(dataslice, sampling_rate=samplerate, order=3, lowcut=.5, highcut=40)
    dataslice = scale_data(filteredSlice) #scale to ensure consistency
    return (dataslice, samplerate)

def packForNN(x: np.array, identifiers: pd.DataFrame, searchDir: str = '/home/rkaufman/workspace/remote') -> np.array:
    """ Given 2 dimensional array of features, collect underlying ECG series and pack it to front with zero padding

    Args:
        x (np.array): calculated features
        identifiers (pd.DataFrame): df containing fin_study_id, start, and stop that correspond to `x` input

    Returns:
        np.array: two dimensional with signal packed at front
    """
    assert(len(identifiers) == x.shape[0])
    pklFile = './data/assets/10000_segments_signals.pkl'
    # if (False and x.shape[0] > 7000 and os.path.exists(pklFile)):
    #     print('Loading source scaled segments instead of computing...')
    #     slices = pickleLoad(pklFile)
    # else:
    print('Computing source scaled segments instead of loading...')
    start = time.time()
    slices = Parallel(n_jobs=7)(delayed(slicer)(row, searchDir) for _, row in identifiers.iterrows())
    stop = time.time()
    print(f'Took {(stop - start) / 60} seconds to extract source signals')
    pickleDump(slices,
        pklFile)
    print(f'Dumped to {pklFile}')
    maxDataSliceLen = 0
    signalArrays = list()
    feats = list()
    resultIndices = list()
    for idx, (dataslice, samplerate) in tqdm(enumerate(slices), total=len(slices)):
        if (len(dataslice) == 0):
            continue
        if (samplerate < 500):
            if (not math.isclose(samplerate, 250)):
                continue
            # print(samplerate)
            dataslice = np.interp(np.arange(5000), np.arange(5000)[::2], dataslice)
        signalArrays.append(np.array(dataslice))
        feats.append(x[idx])
        resultIndices.append(idx)
        maxDataSliceLen = max(len(dataslice), maxDataSliceLen)
    # pickleDump(signalArrays, str(
    #     Path(__file__).parent / 'assets' / 'signalArrays.pkl'))
    signalArrays = np.stack(signalArrays, axis=0)
    x = np.stack(feats, axis=0)
    padded = padToDense(signalArrays, maxDataSliceLen)

    #concatenate along rows so we have padded ecg signal followed by features
    result = np.concatenate((padded, x), axis=1)
    return result, resultIndices