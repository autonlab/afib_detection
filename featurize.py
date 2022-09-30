import audata as aud
import datetime as dt
from joblib import Parallel, delayed
import numpy as np
import pandas as pd
from pathlib import Path
import pickle as pkl
import time
from tqdm import tqdm
import logging

import data.computers as dc
import data.manipulators as dm
import data.utilities as datautils

def innerCollectSeries(fin, group, dfColumns, searchDirectory):
    shortSeries = {
        'fin_study_id': list(),
        'start': list(),
        'stop': list(),
        'signal': list(),
        'samplerate': list(),
        # 'label': list()
    }
    longSeries = {
        'fin_study_id': list(),
        'start': list(),
        'stop': list(),
        'signal': list(),
        'samplerate': list()
    }



    file = datautils.findFileByFIN(str(fin), searchDirectory)
    if (not file):
        print(f'Could not find fin_study_id: {fin} in {searchDirectory}. Skipping.')
        return pd.DataFrame(shortSeries), pd.DataFrame(longSeries)
    hr_series = datautils.HR_SERIES

    auf = aud.File.open(file)
    for idx, row in group.iterrows():

        start, stop = row['start'], row['stop']
        dataslice, samplerate = datautils.getSlice(file, hr_series, start, stop)

        newStart, newStop = start-dt.timedelta(seconds=55), stop+dt.timedelta(seconds=55)
        longerdataslice, sampleratelong = datautils.getSlice(file, hr_series, newStart, newStop)

        shortSeries['fin_study_id'].append(fin)
        shortSeries['start'].append(start)
        shortSeries['stop'].append(stop)
        shortSeries['signal'].append(dataslice)
        shortSeries['samplerate'].append(samplerate)
        # label = '_'.join(row['rhythm_label'].split()).upper()
        # label = row['label']
        # shortSeries['label'].append(label)

        longSeries['fin_study_id'].append(fin)
        longSeries['start'].append(newStart)
        longSeries['stop'].append(newStop)
        longSeries['signal'].append(longerdataslice)
        longSeries['samplerate'].append(sampleratelong)
    return pd.DataFrame(shortSeries), pd.DataFrame(longSeries)

def featurizeSubDF(fin, group, dfColumns, searchDirectory, features):
    warnings.filterwarnings("ignore")
    logging.basicConfig(filename='./data/logs/log.log')
    featurizedResult = {
        'fin_study_id': list(),
        'start': list(),
        'stop': list(),
    }

    if ('label' in dfColumns):
        featurizedResult['label'] = list()
    if ('rhythm_label' in dfColumns):
        featurizedResult['label'] = list()
    if ('confidence' in dfColumns):
        featurizedResult['labelmodel_confidence'] = list()

    for feature in features:
        featurizedResult[feature] = list()
    label = None
    for idx, row in group.iterrows():
        if ('label' in dfColumns):
            label = row['label']
        start, stop = row['start'], row['stop']
        if ('confidence' in dfColumns):
            lm_confidence = row['confidence']
        if (label == 'NOISE'): continue
        dataslice, samplerate = row['signal'], row['samplerate']
        longdataslice, longsamplerate = row['longsignal'], row['longsamplerate']

        if (len(dataslice) < 15):
            continue
        # newStart, newStop = start-dt.timedelta(seconds=25), stop+dt.timedelta(seconds=25)
        # longerdataslice, samplerate = datautils.getSlice(file, hr_series, newStart, newStop)

        shortSegmentFeatures = dc.featurize_nk(dataslice, samplerate)
        longSegmentFeatures = dc.featurize_longertimewindow_nk(longdataslice, longsamplerate)
        if (shortSegmentFeatures and longSegmentFeatures):
            for feature in shortSegmentFeatures:
                featurizedResult[feature].append(shortSegmentFeatures[feature])
            for feature in longSegmentFeatures:
                featurizedResult[feature].append(longSegmentFeatures[feature])
            featurizedResult['start'].append(start)
            featurizedResult['stop'].append(stop)
            featurizedResult['fin_study_id'].append(fin)
            if (label):
                featurizedResult['label'].append(label)
            if ('confidence' in dfColumns):
                featurizedResult['labelmodel_confidence'].append(lm_confidence)
    return pd.DataFrame(featurizedResult)

def featurize_parallel(src=None, dst=None, load=False):
    print('Featurizing in parallel...')
    dataconfig = datautils.getDataConfig()
    rawDataFile = src if src else dataconfig.rawDataFile
    outFile = dst if dst else dataconfig.featurizedDataOutput

    infile = Path(__file__).parent / 'data' / 'assets' / rawDataFile
    print(f' Reading from {infile}')
    df = pd.read_csv(
        infile,
        parse_dates=['start', 'stop'])
    df.columns = df.columns.str.lower()
    searchDirectory = dataconfig.finSearchDirectory
    print(f' Searching from {searchDirectory}')

    print(f'Will attempt to write result to {outFile}')
    if (load):
        shortSeries = dm.pickleLoad('./data/assets/10000_signal_segments.pkl')
        longSeries = dm.pickleLoad('./data/assets/10000_longsignal_segments.pkl')
    else:
        start = time.time()
        shortSeries, longSeries = zip(*Parallel(n_jobs=7)(delayed(innerCollectSeries)(fin, group, df.columns, searchDirectory) for fin, group in df.groupby('fin_study_id')))
        stop = time.time()
        shortSeries = pd.concat(shortSeries)
        longSeries = pd.concat(longSeries)
        # if (len(shortSeries) > 1000):
        #     dm.pickleDump(shortSeries, './data/assets/10000_signal_segments.pkl')
        #     dm.pickleDump(longSeries, './data/assets/10000_longsignal_segments.pkl')
        print(f'Took {(stop - start)/60:.2f} minutes to collect {len(df)} signals')
    # print(shortSeries.columns, longSeries.columns)
    shortSeries['longsignal'] = longSeries['signal']
    shortSeries['longsamplerate'] = longSeries['samplerate']
    for col in df.columns:
        if 'label' in col:
            shortSeries[col] = df[col]
        if 'confidence' in col:
            shortSeries[col] = df[col]
    print(' Successfully read source signals, now featurizing...')

    start = time.time()
    del longSeries
    pds = Parallel(n_jobs=7)(delayed(innerLoopFor)(int(fin), group, shortSeries.columns, searchDirectory, dataconfig.features_nk) for fin, group in shortSeries.groupby('fin_study_id'))
    stop = time.time()
    print(f'Took {(stop - start) / 60 :.2f} minutes to compute features')
    result = pd.concat(pds)
    print(f'Attempting to write result to {outFile}')
    try:
        result.to_csv(
            Path(__file__).parent / 'data' / 'assets' / outFile
        )
        print('Success!')
    except:
        print('Falling back to local directory, featurizedOutput.csv')
        result.to_csv('./featurizedOutput.csv')

def featurize_parallel_all(src=None, dst=None, load=False):
    print('Featurizing in parallel...')
    dataconfig = datautils.getDataConfig()
    rawDataFile = src if src else dataconfig.rawDataFile
    outFile = dst if dst else dataconfig.featurizedDataOutput

    infile = Path(__file__).parent / 'data' / 'assets' / rawDataFile
    print(f' Reading from {infile}')
    df = pd.read_csv(
        infile,
        parse_dates=['start', 'stop'])
    df.columns = df.columns.str.lower()
    searchDirectory = dataconfig.finSearchDirectory
    print(f' Searching from {searchDirectory}')

    print(f'Will attempt to write result to {outFile}')
    start = time.time()


    shortSeries, longSeries = zip(*Parallel(n_jobs=7)(delayed(innerCollectSeries)(fin, group, df.columns, searchDirectory)
        for fin, group in df.groupby('fin_study_id')))


    stop = time.time()
    shortSeries = pd.concat(shortSeries)
    longSeries = pd.concat(longSeries)
    print(f'Took {(stop - start)/60:.2f} minutes to collect {len(df)} signals')
    # print(shortSeries.columns, longSeries.columns)
    shortSeries['longsignal'] = longSeries['signal']
    shortSeries['longsamplerate'] = longSeries['samplerate']
    for col in df.columns:
        if 'label' in col:
            shortSeries[col] = df[col]
        if 'confidence' in col:
            shortSeries[col] = df[col]
    print(' Successfully read source signals, now featurizing...')

    start = time.time()
    del longSeries

    pds = Parallel(n_jobs=7)(delayed(featurizeSubDF)(int(fin), group, shortSeries.columns, searchDirectory, dataconfig.features_nk) 
        for fin, group in shortSeries.groupby('fin_study_id'))


    stop = time.time()
    print(f'Took {(stop - start) / 60 :.2f} minutes to compute features')
    result = pd.concat(pds)
    print(f'Attempting to write result to {outFile}')
    try:
        result.to_csv(
            Path(__file__).parent / 'data' / 'assets' / outFile
        )
        print('Success!')
    except:
        print('Falling back to local directory, featurizedOutput.csv')
        result.to_csstopv('./featurizedOutput.csv')

def featurize_with_parallelized_io():
    """Featurize dataframe.
     PRECONDITIONS:
     - `df` contains `fin_study_id`, `start`, and `stop` columns
     - `searchDirectory` contains fins referenced in df

    Args:
        df (_type_): _description_
        searchDirectory (_type_): _description_
    """

    print('Featurizing...')
    dataconfig = datautils.getDataConfig()
    infile = Path(__file__).parent / 'data' / 'assets' / dataconfig.rawDataFile
    print(f' Reading from {infile}')
    df = pd.read_csv(
        infile,
        parse_dates=['start', 'stop'])
    df.columns = df.columns.str.lower()
    searchDirectory = dataconfig.finSearchDirectory
    print(f' Searching from {searchDirectory}')
    hr_series = datautils.HR_SERIES

    print(f'Will attempt to write result to {dataconfig.featurizedDataOutput}')
    featurizedResult = {
        'fin_study_id': list(),
        'start': list(),
        'stop': list(),
    }

    if ('label' in df.columns):
        featurizedResult['label'] = list()
    if ('rhythm_label' in df.columns):
        featurizedResult['label'] = list()
    if ('confidence' in df.columns):
        featurizedResult['labelmodel_confidence'] = list()

    for feature in dataconfig.features_nk:
        featurizedResult[feature] = list()

    start = time.time()
    shortSeries, longSeries = zip(*Parallel(n_jobs=7, prefer="threads")(delayed(innerCollectSeries)(fin, group, df.columns, searchDirectory) for fin, group in df.groupby('fin_study_id')))
    stop = time.time()
    print(stop-start)
    shortSeries = pd.concat(shortSeries)
    longSeries = pd.concat(longSeries)
    shortSeries['longsignal'] = longSeries['signal']
    shortSeries['longsamplerate'] = longSeries['samplerate']
    del longSeries
    progress = tqdm(total=len(shortSeries))
    for fin, group in shortSeries.groupby('fin_study_id'):
        label = None
        for idx, row in group.iterrows():
            start, stop = row['start'], row['stop']
            dataslice, samplerate = row['signal'], row['samplerate']
            longerdataslice, samplerate = row['longsignal'], row['longsamplerate']

            shortSegmentFeatures = dc.featurize_nk(dataslice, samplerate)
            longSegmentFeatures = dc.featurize_longertimewindow_nk(longerdataslice, samplerate)
            if (shortSegmentFeatures and longSegmentFeatures):
                for feature in shortSegmentFeatures:
                    featurizedResult[feature].append(shortSegmentFeatures[feature])
                for feature in longSegmentFeatures:
                    featurizedResult[feature].append(longSegmentFeatures[feature])
                featurizedResult['start'].append(start)
                featurizedResult['stop'].append(stop)
                featurizedResult['fin_study_id'].append(fin)
                # featurizedResult['label'].append(label)
                if ('confidence' in df.columns):
                    featurizedResult['labelmodel_confidence'].append(lm_confidence)
            progress.update(1)
    progress.close()
    result = pd.DataFrame(featurizedResult)
    print(f'Attempting to write result to {dataconfig.featurizedDataOutput}')
    try:
        result.to_csv(
            Path(__file__).parent / 'data' / 'assets' / dataconfig.featurizedDataOutput
        )
        print('Success!')
    except:
        print('Falling back to local directory, featurizedOutput.csv')
        result.to_csv('./featurizedOutput.csv')

def featurize():
    """Featurize dataframe.
     PRECONDITIONS:
     - `df` contains `fin_study_id`, `start`, and `stop` columns
     - `searchDirectory` contains fins referenced in df

    Args:
        df (_type_): _description_
        searchDirectory (_type_): _description_
    """

    print('Featurizing...')
    dataconfig = datautils.getDataConfig()
    infile = Path(__file__).parent / 'data' / 'assets' / dataconfig.rawDataFile
    print(f' Reading from {infile}')
    df = pd.read_csv(
        infile,
        parse_dates=['start', 'stop'])
    df.columns = df.columns.str.lower()
    searchDirectory = dataconfig.finSearchDirectory
    print(f' Searching from {searchDirectory}')
    hr_series = datautils.HR_SERIES

    print(f'Will attempt to write result to {dataconfig.featurizedDataOutput}')
    featurizedResult = {
        'fin_study_id': list(),
        'start': list(),
        'stop': list(),
    }

    if ('label' in df.columns):
        featurizedResult['label'] = list()
    if ('rhythm_label' in df.columns):
        featurizedResult['label'] = list()
    if ('confidence' in df.columns):
        featurizedResult['labelmodel_confidence'] = list()

    for feature in dataconfig.features_nk:
        featurizedResult[feature] = list()

    progress = tqdm(total=len(df))
    for fin, group in df.groupby('fin_study_id'):
        file = datautils.findFileByFIN(str(fin), searchDirectory)
        if (not file):
            print(f'Could not find fin_study_id: {fin} in {searchDirectory}. Skipping.')
            progress.update(len(group))
            continue

        auf = aud.File.open(file)
        label = None
        for idx, row in group.iterrows():

            start, stop = row['start'], row['stop']
            if ('label' in df.columns):
                label = row['label']
            if ('confidence' in df.columns):
                lm_confidence = row['confidence']
            # if ('rhythm_label' in df.columns):
            #     label = '_'.join(row['rhythm_label'].split()).upper()
            if (label and label == 'NOISE'): continue
            dataslice, samplerate = datautils.getSlice(file, hr_series, start, stop)

            if (len(dataslice) < 15):
                progress.update(1)
                continue
            newStart, newStop = start-dt.timedelta(seconds=55), stop+dt.timedelta(seconds=55)
            longerdataslice, samplerate = datautils.getSlice(file, hr_series, newStart, newStop)

            shortSegmentFeatures = dc.featurize_nk(dataslice, samplerate)
            longSegmentFeatures = dc.featurize_longertimewindow_nk(longerdataslice, samplerate)
            if (shortSegmentFeatures and longSegmentFeatures):
                for feature in shortSegmentFeatures:
                    featurizedResult[feature].append(shortSegmentFeatures[feature])
                for feature in longSegmentFeatures:
                    featurizedResult[feature].append(longSegmentFeatures[feature])
                featurizedResult['start'].append(start)
                featurizedResult['stop'].append(stop)
                featurizedResult['fin_study_id'].append(fin)
                featurizedResult['label'].append(label)
                if ('confidence' in df.columns):
                    featurizedResult['labelmodel_confidence'].append(lm_confidence)
                # for k, v in featurizedResult.items():
                #     print(k, len(v))
            progress.update(1)
    progress.close()
    result = pd.DataFrame(featurizedResult)
    print(f'Attempting to write result to {dataconfig.featurizedDataOutput}')
    try:
        result.to_csv(
            Path(__file__).parent / 'data' / 'assets' / dataconfig.featurizedDataOutput
        )
        print('Success!')
    except:
        print('Falling back to local directory, featurizedOutput.csv')
        result.to_csv('./featurizedOutput.csv')

from data.process_physionet import loadPhysionet
import numpy as np
def featurize_physionet(index_pass=False):
    samplerate = 300
    xSignals, yLabels = loadPhysionet()
    dfDict = {
        'label': list()
    }

    allFeats = dict()
    did = False
    if (index_pass):
        signal_indices = []

    for i in tqdm(range(len(xSignals))):
        signal = np.squeeze(np.asarray(xSignals[i]))
        #only featurize first 10 seconds
        signal = signal[:10*samplerate]
        feats = dc.featurize_nk(signal, samplerate)

        label = yLabels[i]
        if (feats is None):
            continue
        if (index_pass):
            signal_indices.append(i)
            continue
        dfDict['label'].append(label)
        if (i == 0):
            for f in feats:
                dfDict[f] = list()
        for f in feats:
            dfDict[f].append(feats.get(f, None))
    if (index_pass):
        with open('./data/assets/physionet/indices.pkl', 'wb') as writefile:
            pkl.dump(signal_indices, writefile)
        return signal_indices
    pd.DataFrame(dfDict).to_csv('physionet_featurized.csv', index=False)

from data.computers import featurize_nk, featurize_longertimewindow_nk
import matplotlib.pyplot as plt
from prediction.europace.loadData import getJustRecordByID
import neurokit2 as nk
# beforeMinutesOfInterest = [15, 30, 60, 120]
# beforeMinutesOfInterest = [5, 10, 15, 30]
beforeMinutesOfInterest = [30, 60]
beforeMinutesOfInterest = [10, 15]
def featurize_europace(df: pd.DataFrame, window_hold=False):
    #intervals back define the points of comparison for trend analysis
    # trendWindowLength_minutes = 3
    trendWindowLength_minutes = 5
    forgottenSegments = 0 # :(
    dfDict = {
        'patient_id': list(),
        'time': list(),
        'basetime': list()
    }
    labelColumns = list()
    for col in df.columns.str.lower():
        if col.startswith('afib_in'):
            dfDict[col] = list()
            labelColumns.append(col)
    # for idx, point in tqdm(df.iterrows(), total=len(df)):
    for idx, point in df.iterrows():
        forgetSegment = False
        id, time, basetime = point['patient_id'], point['time'], point['basetime']
        record = getJustRecordByID(id)

        #lambda to convert file time to signal index
        getIdxForTime = lambda datetime: int((datetime - basetime).total_seconds() * record.fs)
        startIdx, stopIdx = getIdxForTime(time - dt.timedelta(minutes=trendWindowLength_minutes)), getIdxForTime(time)
        sigToUse = 0 #(could also be 1), see record.n_sig
        signal, samplerate = record.p_signal[startIdx:stopIdx, sigToUse], record.fs
        #take slice of time preceding 5 minutes, featurize
        features, processedSigs = featurize_nk(signal, samplerate, sendProcessed=True)
        if (type(features) == type(None)):
            #skip point if featurization fails
            continue
        longerFeats = featurize_longertimewindow_nk(signal, samplerate, processedSignals=processedSigs)
        if (type(longerFeats) == type(None)):
            continue
        features.update(longerFeats)
        features, longerFeats = dict(), dict()
        #, then the same for each beforeMinutesOfInterest
        initialTime = time
        features_new = None
        for minutesBefore in beforeMinutesOfInterest:
            time = initialTime - dt.timedelta(minutes=minutesBefore)
            if (time < basetime):
                forgetSegment=True
                break
            # startIdx, stopIdx = getIdxForTime(time - dt.timedelta(minutes=trendWindowLength_minutes)), getIdxForTime(time)
            startIdx, stopIdx = getIdxForTime(time), getIdxForTime(initialTime)
            signal, samplerate = record.p_signal[startIdx:stopIdx, sigToUse], record.fs
            features_new, processedSigs = featurize_nk(signal, samplerate, sendProcessed=True)
            if (type(features_new) == type(None)):
                #skip point if featurization fails
                break
            longerFeats = featurize_longertimewindow_nk(signal, samplerate, processedSignals=processedSigs)
            if (type(longerFeats) == type(None)):
                break
            features_new.update(longerFeats)
            for feat in features_new:
                features[f"{feat}_{minutesBefore}"] = features_new[feat]
        if (type(features_new) == type(None) or type(longerFeats) == type(None)):
            continue
        if (forgetSegment):
            forgottenSegments += 1
            # case where point is too near to beginning of signal to featurize prior trends
            continue
        if (len(dfDict.keys()) < 10):
            for feat in features:
                dfDict[feat] = list()
        dfDict['patient_id'].append(id)
        dfDict['time'].append(initialTime)
        dfDict['basetime'].append(basetime)
        for labelCol in labelColumns:
            dfDict[labelCol].append(point[labelCol])
        for feat in features:
            dfDict[feat].append(features[feat])
    # pd.DataFrame(dfDict).to_csv('featurized_afib_predictor_data.csv', index=False)
    return pd.DataFrame(dfDict)

if __name__ == '__main__':
    # df = pd.read_csv(str(Path(__file__).parent / 'data/assets/europace' / 'afib_predictor_training_times.csv'), parse_dates=['time', 'basetime'], dtype={'patient_id': str})
    # df = pd.read_csv(str(Path(__file__).parent /  'afib_predictor_additional_negatives.csv'), parse_dates=['time', 'basetime'], dtype={'patient_id': str})
    df = pd.read_csv(str(Path(__file__).parent /  'modded_predictor_data.csv'), parse_dates=['time', 'basetime'], dtype={'patient_id': str})
    # df = pd.read_csv(str(Path(__file__).parent /  'afib_episode_samples_30.csv'), parse_dates=['time', 'basetime'], dtype={'patient_id': str})
    # df = pd.read_csv(str(Path(__file__).parent /  'preceding_afib_episodes.csv'), parse_dates=['time', 'basetime'], dtype={'patient_id': str})
    # featurize_europace(df)
    # df = df[:1000:5]
    res = Parallel(n_jobs=7)(delayed(featurize_europace)(group) for id, group in df.groupby('patient_id'))
    concatenated = pd.concat(res)
    concatenated.to_csv('featurized_afib_predictor_data_lengthened.csv', index=False)
    print('SUCCESS')
    # import warnings
    # warnings.filterwarnings("ignore")

    # from pstats import SortKey
    # import cProfile
    # featurizeOutput = datautils.getDataConfig().featurizedDataOutput
    # featurize_physionet()
    # print(f'Will save profile results to {featurizeOutput}')
    # cProfile.run("featurize()", Path(__file__).parent / 'results' / 'profiles' / f'{featurizeOutput[:-4]}_numba.prof', sort=SortKey.CUMULATIVE)
    # cProfile.run("featurize_parallel(load=True)", Path(__file__).parent / 'results' / 'profiles' / f'featurizeRun2_{featurizeOutput}.prof', sort=SortKey.CUMULATIVE)
    # featurize()
    # featurize_parallel(load=False)
    # featurize_parallel(src='5000_segments.csv', dst='5000_featurized_nk.csv', load=False)