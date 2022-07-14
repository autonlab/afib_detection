"""
   Application of upstream classifiers to stitch together atrial fibrillation events (10 second segments classified as afib) (10 second segments classified as afib) (10 second segments classified as afib) (10 second segments classified as afib) (10 second segments classified as afib) (10 second segments classified as afib) (10 second segments classified as afib) (10 second segments classified as afib) (10 second segments classified as afib) and episodes
"""
import audata as aud
import datetime as dt
from data.utilities import findFileByFIN, getSlice, HR_SERIES, getDataConfig
from data.computers import featurize_nk, featurize_longertimewindow_nk
import pandas as pd
from pathlib import Path
import pytz
from tqdm import tqdm



def identifyEventsForFIN(fin: int, searchDirectory: Path, dst: Path = Path('./')):
    """Given fin and directory within to find its signal, returns all events present as well as sufficient statistics and features computed

    Args:
        fin (Union[int, str]): fin_study_id, identifying patient
        searchDirectory (pathlib.Path): directory to find h5 file corresponding to given fin
    """
    file = findFileByFIN(fin, searchDirectory)
    if (file is None):
        print(f"Couldn't find fin {fin}.")
        return
    auf = aud.File.open(file)
    basetime = auf[HR_SERIES][0]['time'].item().to_pydatetime().replace(tzinfo=pytz.UTC)
    finaltime = auf[HR_SERIES][-1]['time'].item().to_pydatetime().replace(tzinfo=pytz.UTC)


    dataconfig = getDataConfig()
    resultDF = {
        'start': list(),
        'stop': list()
    }
    for feat in dataconfig.features_nk:
        resultDF[feat] = list()

    stoptime = basetime + dt.timedelta(seconds=110)
    #iterate through patient's signal, loading 10 minutes into memory at a time
    while (stoptime < finaltime):
        starttime, stoptime = stoptime, stoptime+dt.timedelta(minutes=10)

        # get a 12 minute (ish) slice so we can get 2 minute window behind first 10 second segment
        signalSubset, signalSampleRate = getSlice(file, HR_SERIES, starttime - dt.timedelta(seconds=110), stoptime) 
        #iterate through signalSubset, featurizing portion for downstream classification
        for startIdx in tqdm(range(int(110*signalSampleRate), len(signalSubset), int(10*signalSampleRate))):
            # stop is 10 seconds after start
            stopIdx = min(int(startIdx + 10 * signalSampleRate), len(signalSubset))
            # long series start is 2 minutes before stop
            longSeriesStartIdx = max(0, int(stopIdx - 120 * signalSampleRate))

            shortSeries, longSeries = signalSubset[startIdx:stopIdx].to_numpy(), signalSubset[longSeriesStartIdx:stopIdx].to_numpy()
            #featurize and store
            shortSegmentFeatures = featurize_nk(shortSeries, signalSampleRate)
            longSegmentFeatures = featurize_longertimewindow_nk(longSeries, signalSampleRate)
            if (shortSegmentFeatures and longSegmentFeatures):
                for feature in shortSegmentFeatures:
                    resultDF[feature].append(shortSegmentFeatures[feature])
                for feature in longSegmentFeatures:
                    resultDF[feature].append(longSegmentFeatures[feature])
                resultDF['start'].append(starttime + dt.timedelta(seconds = startIdx / signalSampleRate))
                resultDF['stop'].append(starttime + dt.timedelta(seconds = stopIdx / signalSampleRate))
        pd.DataFrame(resultDF).to_csv(dst / f"{fin}.csv", index=False)

if __name__=='__main__':
    fins = pd.read_csv('/zfsauton2/home/rkaufman/misc_data/chronicoranyaf.csv')
    fins_nonchronic = fins[fins['chronicaf'] == 0]
    print(f"Culled from {len(fins)} to {len(fins_nonchronic)} patients for analysis.")
    fins.columns = fins.columns.str.lower()
    fins = fins['fin_study_id'].apply(int).to_numpy()

    src = Path('/zfsmladi/originals')
    dst = Path('/zfsauton2/home/rkaufman/afib_stitching/')

    from joblib import Parallel, delayed
    Parallel(n_jobs=80)(delayed(identifyEventsForFIN)(fin, src, dst) for fin in fins)
    # identifyEventsForFIN(1331644, Path('/home/rkaufman/workspace/remote'))
