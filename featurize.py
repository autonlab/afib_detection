import audata as aud
import datetime as dt
import numpy as np
import pandas as pd
from pathlib import Path
import pytz
from tqdm import tqdm

import data.computers as dc
import data.utilities as datautils

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

    featurizedResult = {
        'fin_study_id': list(),
        'start': list(),
        'stop': list(),
        'label': list(),
        #'labelmodel_confidence': list()
    }
    for feature in dataconfig.features:
        featurizedResult[feature] = list()

    progress = tqdm(total=len(df))
    for fin, group in df.groupby('fin_study_id'):
        file = datautils.findFileByFIN(str(fin), searchDirectory)
        if (not file):
            # print(f'Could not find fin_study_id: {fin} in {searchDirectory}. Skipping.')
            progress.update(len(group))
            continue

        auf = aud.File.open(file)
        basetime = auf.time_reference
        basetime = auf[hr_series][0]['time'].item().to_pydatetime()
        basetime = basetime.replace(tzinfo=pytz.UTC)
        for idx, row in group.iterrows():

            start, stop = row['start'], row['stop']
            label = row['label']
            # label = '_'.join(row['rhythm_label'].split()).upper()
            # if (label == 'NOISE'):
            #     continue
            # print(row['rhythm_label'], label)
        #    lm_confidence = row['confidence']

            dataslice, samplerate = datautils.getSlice(file, hr_series, start, stop)

            if (len(dataslice) < 15):
                progress.update(1)
                continue
            features = dc.featurize(dataslice, samplerate)
            newStart, newStop = start-dt.timedelta(seconds=25), stop+dt.timedelta(seconds=25)
            longerdataslice, samplerate = datautils.getSlice(file, hr_series, newStart, newStop)
            others = dc.featurize_longertimewindow(longerdataslice, samplerate)

            if (features and others):
                for feature in features:
                    featurizedResult[feature].append(features[feature])
                for feature in others:
                    featurizedResult[feature].append(others[feature])
                featurizedResult['start'].append(start)
                featurizedResult['stop'].append(stop)
                featurizedResult['fin_study_id'].append(fin)
                featurizedResult['label'].append(label)
                #featurizedResult['labelmodel_confidence'].append(lm_confidence)
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
    featurize()
