import audata as aud
import datetime as dt
from joblib import Parallel, delayed
import numpy as np
import pandas as pd
from pathlib import Path
import pickle as pkl
import time
from tqdm import tqdm

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
        'label': list()
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
        if ('rhythm_label' in row):
            shortSeries['label'].append('_'.join(row['rhythm_label'].split()).upper())
        elif ('label' in row):
            shortSeries['label'].append(row['label'])

        longSeries['fin_study_id'].append(fin)
        longSeries['start'].append(newStart)
        longSeries['stop'].append(newStop)
        longSeries['signal'].append(longerdataslice)
        longSeries['samplerate'].append(sampleratelong)
    return pd.DataFrame(shortSeries), pd.DataFrame(longSeries)

def innerLoopFor(fin, group, dfColumns, searchDirectory, features):
    warnings.filterwarnings("ignore")
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
    for idx, row in group.iterrows():
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

        shortSegmentFeatures = dc.featurize(dataslice, samplerate)
        longSegmentFeatures = dc.featurize_longertimewindow(longdataslice, longsamplerate)
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

def featurize_parallel(load=False):
    print('Featurizing in parallel...')
    dataconfig = datautils.getDataConfig()
    infile = Path(__file__).parent / 'data' / 'assets' / dataconfig.rawDataFile
    print(f' Reading from {infile}')
    df = pd.read_csv(
        infile,
        parse_dates=['start', 'stop'])
    df.columns = df.columns.str.lower()
    searchDirectory = dataconfig.finSearchDirectory
    print(f' Searching from {searchDirectory}')

    print(f'Will attempt to write result to {dataconfig.featurizedDataOutput}')
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
    pds = Parallel(n_jobs=7)(delayed(innerLoopFor)(int(fin), group, df.columns, searchDirectory, dataconfig.features) for fin, group in shortSeries.groupby('fin_study_id'))
    stop = time.time()
    print(f'Took {(stop - start) / 60 :.2f} minutes to compute features')
    result = pd.concat(pds)
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

    for feature in dataconfig.features:
        featurizedResult[feature] = list()

    progress = tqdm(total=len(df))
    for fin, group in df.groupby('fin_study_id'):
        file = datautils.findFileByFIN(str(fin), searchDirectory)
        if (not file):
            print(f'Could not find fin_study_id: {fin} in {searchDirectory}. Skipping.')
            progress.update(len(group))
            continue

        auf = aud.File.open(file)
        for idx, row in group.iterrows():

            start, stop = row['start'], row['stop']
            if ('label' in df.columns):
                label = row['label']
            if ('confidence' in df.columns):
                lm_confidence = row['confidence']
            if ('rhythm_label' in df.columns):
                label = '_'.join(row['rhythm_label'].split()).upper()
            if (label == 'NOISE'): continue
            dataslice, samplerate = datautils.getSlice(file, hr_series, start, stop)

            if (len(dataslice) < 15):
                progress.update(1)
                continue
            newStart, newStop = start-dt.timedelta(seconds=55), stop+dt.timedelta(seconds=55)
            longerdataslice, samplerate = datautils.getSlice(file, hr_series, newStart, newStop)

            shortSegmentFeatures = dc.featurize(dataslice, samplerate)
            longSegmentFeatures = dc.featurize_longertimewindow(longerdataslice, samplerate)
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

if __name__ == '__main__':
    import warnings
    warnings.filterwarnings("ignore")
    # from pstats import SortKey
    # import cProfile
    # featurizeOutput = datautils.getDataConfig().featurizedDataOutput
    # print(f'Will save profile results to {featurizeOutput}')
    # cProfile.run("featurize()", Path(__file__).parent / 'results' / 'profiles' / f'featurizeRun_{featurizeOutput}.prof', sort=SortKey.CUMULATIVE)
    # featurize()
    featurize_parallel(load=False)