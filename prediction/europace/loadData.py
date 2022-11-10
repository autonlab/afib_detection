#parsing data from .atr, .dat files into signals, sample rates, and annotations
import datetime as dt
import pandas as pd
from pathlib import Path
from tqdm import tqdm
from typing import Tuple, Union, List
import wfdb

def getAllRecordIDs(europaceDownloadDir = Path(__file__).parent.parent.parent / 'data' / 'assets' / 'europace/physionet.org/files/ltafdb/1.0.0/') -> List[str]:
    return open(str(europaceDownloadDir / 'RECORDS'), 'r').read().splitlines()

def parse(src=Path(__file__).parent.parent.parent / 'data' / 'assets' / 'europace'):
    searchDir = src / 'physionet.org/files/ltafdb/1.0.0/'
    for datFile in searchDir.glob('*.dat'):
        curRecord = str(searchDir / datFile.name.split('.')[0])
        break
        record = wfdb.rdrecord(curRecord)
        ann = wfdb.rdann(curRecord, 'atr')
        # print(record.fs)
        # print(record.d_signal)
        # print(record.physical)
        print(len(record.p_signal))
        print(len(ann.symbol))
        print(len(ann.aux_note))
        print(len(ann.sample))
        print((len(record.p_signal) / record.fs) / 60 / 60)
        print(record.base_datetime)
        # print(ann.symbol)
        # print(ann.aux_note)

def getRecordByID(id: str, searchDir: Path = Path(__file__).parent.parent.parent / 'data' / 'assets' / 'europace/physionet.org/files/ltafdb/1.0.0/') -> Tuple[wfdb.Record, wfdb.Annotation]:
    return wfdb.rdrecord( str(searchDir / id) ), wfdb.rdann( str(searchDir / id), 'atr')

def getJustRecordByID(id: str, searchDir: Path = Path(__file__).parent.parent.parent / 'data' / 'assets' / 'europace/physionet.org/files/ltafdb/1.0.0/') -> Tuple[wfdb.Record, wfdb.Annotation]:
    return wfdb.rdrecord( str(searchDir / id) )

def getHeaderById(id: str, searchDir: Path = Path(__file__).parent.parent.parent / 'data' / 'assets' / 'europace/physionet.org/files/ltafdb/1.0.0/') -> Tuple[wfdb.Record, wfdb.Annotation]:
    return wfdb.rdheader( str(searchDir / id) )

def getRhythmAnnotations(patientID: str, rhythmToCollect: str = '(AFIB', dst: Union[None, Path] = None) -> pd.DataFrame:
    """Collect all the rhythm annotations of specified type for given patient, putting their start, stops, and patient id into dataframe

    Args:
        patientID (str): _description_
        rhythmToCollect (str, optional): _description_. Defaults to '(AFIB'.
        dst (Union[None, Path], optional): Directory to store csv, if you leave as None csv won't be saved. Defaults to None.

    Returns:
        pd.DataFrame: _description_
    """
    record, annotation = getRecordByID(patientID)
    #extract information we'll need from record
    recordBaseTime: dt.datetime = record.base_datetime
    samplingRate: int = record.fs

    #dictionary we'll turn into dataframe for result
    dfDict = {
        'start': list(),
        'stop': list(),
        'id': list(),
        'basetime': list()
    }
    totalTime = dt.timedelta(seconds = 0)
    # iterate through auxillary notes (where the rhythm annotations are present), finding all samples of rhythm to collect and collecting them accordingly
    auxNotes = annotation.aux_note
    ## AUXNOTE('qrs' param) ['', '', ]
    ## AUXNOTE: ['rhy', '', '(rhy', ...]
    ## SAMPLE:  [2000, '', 2800, ..., ]
    for i, ann in enumerate(auxNotes):

        if ann == rhythmToCollect:
            start: float = annotation.sample[i] / samplingRate
            startTime: dt.datetime = recordBaseTime + dt.timedelta(seconds = start)
            j = i+1
            while ((j < len(auxNotes)) and auxNotes[j] == ''):
                # print(auxNotes[j])
                j += 1
            if (j < len(auxNotes)):
                end: float = annotation.sample[j] / samplingRate
            else:
                end: float = len(record.p_signal) / samplingRate
            endTime: dt.datetime = recordBaseTime + dt.timedelta(seconds = end)
            dfDict['id'].append(patientID)
            dfDict['start'].append(startTime)
            dfDict['stop'].append(endTime)
            dfDict['basetime'].append(record.base_datetime)
            totalTime += endTime - startTime
    result = pd.DataFrame(dfDict)
    if (not (dst is None)):
        result.to_csv(str( dst / f"{patientID}_events.csv"), index=False)
    return result

def collectEpisodesFromEvents(patientID: str, searchDir: Path = Path(__file__).parent.parent.parent / 'data' / 'assets'/ 'europace') -> pd.DataFrame:
    df = pd.read_csv(str(searchDir / f'{patientID}_events.csv'), parse_dates=['start', 'stop', 'basetime'], dtype={'id': str})
    dfDict = {
        'start': list(),
        'stop': list(),
        'id': list(),
        'basetime': list()
    }
    i = 0
    minsNext = 0
    minsTotal =0
    while ( i < len(df) ):
        event = df.iloc[i, :]
        eventStart, eventStop = event['start'], event['stop']
        sumOfEvents_halfHour: dt.timedelta = min(eventStart+dt.timedelta(minutes=30), eventStop) - eventStart
        j = i+1 #upcoming event index

        # Collect upcoming events while they begin within 30 upcoming minutes of current event
        while ( (j < len(df)) and ((df.iloc[j, :]['start'] - eventStart) <= dt.timedelta(minutes=30))):
            eventEnd, item = eventStart + dt.timedelta(minutes=30), df.iloc[j, :]
            eventLength = min(eventEnd, item['stop']) - item['start']
            sumOfEvents_halfHour += eventLength
            j += 1

        if (sumOfEvents_halfHour >= dt.timedelta(minutes=5)):
            #if there are more than 5 minutes of events within these 30 minutes, then we have an afib episode.
            ## episode ends when there is no afib event within 30 minutes
            curEndIdx = j-1
            while ((curEndIdx+1 < len(df)) and (df.iloc[curEndIdx+1, :]['start'] - df.iloc[curEndIdx, :]['stop'] <= dt.timedelta(minutes=30))):
                curEndIdx += 1
            startAfib, endAfib = event['start'], df.iloc[curEndIdx, :]['stop']

            dfDict['start'].append(startAfib)
            dfDict['stop'].append(endAfib)
            dfDict['id'].append(event['id'])
            dfDict['basetime'].append(event['basetime'])
            j = curEndIdx + 1
            minsTotal += sumOfEvents_halfHour.total_seconds()//60
            minsNext += 1

        i = j
    # if (minsNext > 0):
    #     print('Avg (minutes) of episodes starting within first 30 minutes: ' + str(minsTotal / minsNext))
    #     print('Percentage of time in afib' + str(((minsTotal / minsNext)/30)*100))
    return pd.DataFrame(dfDict), minsTotal, minsNext

def collectEpisodesForID(patientID: str, dst: Path):
    collectEvents(patientID, dst=dst)
    return collectEpisodesFromEvents(patientID, searchDir=dst)

def collectAllEventsAndEpisodes():
    dst = Path(__file__).parent.parent.parent / 'data' / 'assets' / 'europace'
    allIDs = getAllRecordIDs()
    minsTotal, minsNext = 0, 0
    for id in tqdm(allIDs, total=len(allIDs)):
        episodesDF, mt, mn = collectEpisodesForID(id, dst / 'events')
        minsTotal += mt; minsNext += mn
        episodesDF.to_csv( str( dst / 'episodes' / f'{id}_episodes.csv' ), index=False)
    if (minsNext > 0): 
        print('Avg (minutes) of episodes starting within first 30 minutes: ' + str(minsTotal / minsNext))
        print('Percentage of time in afib' + str(((minsTotal / minsNext)/30)*100))
    return None

def readAllEpisodes():
    dst = Path(__file__).parent.parent.parent / 'data' / 'assets' / 'europace'
    allIDs = getAllRecordIDs()
    allDFs = []
    for id in allIDs:
        df = pd.read_csv( str( dst / 'episodes' / f'{id}_episodes.csv' ), parse_dates=['start', 'stop', 'basetime'], dtype={'id': str})
        allDFs.append(df)
    everyEpisode = pd.concat(allDFs)
    firstThreeHours = 0
    firstTwoHours = 0
    firstThirtyMins=0
    within3HoursOfAnother = 0
    allOthers = 0
    durSum = dt.timedelta(seconds=0)
    minDur, maxDur = dt.timedelta(hours=69), dt.timedelta(microseconds=69)
    for idx, ep in tqdm(everyEpisode.iterrows(), total=len(everyEpisode)):
        dur = ep['stop'] - ep['start']
        minDur, maxDur = min(minDur, dur), max(maxDur, dur)
        durSum += dur
        id = str(ep['id'])
        durationSinceStart = ep['start'] - ep['basetime']
        if (idx > 0 and everyEpisode.iloc[idx-1]['id'] == id):
            durationSincePrior = ep['start'] - everyEpisode.iloc[idx-1]['stop']
        else:
            durationSincePrior = None
        happened = False
        if (durationSinceStart < dt.timedelta(hours=3)):
            happened = True
            firstThreeHours += 1
        if (durationSincePrior and durationSincePrior < dt.timedelta(hours=3)):
            happened = True
            within3HoursOfAnother += 1
        if (durationSinceStart < dt.timedelta(hours=2)):
            happened=True
            firstTwoHours += 1
        if (durationSinceStart < dt.timedelta(minutes=30)):
            happened=True
            firstThirtyMins += 1
        if (happened==False):
            allOthers += 1
    import numpy as np
    # print('Mean episode length: ' + str(durSum / len(everyEpisode)))
    # print('Shortest episode: ' + str(minDur))
    # print('Longest episode: '+ str(maxDur))
    # print('Avg time in afib per patient: ' + str(durSum / len(allIDs)))
    # print('Avg episodes per file: ' + str(len(everyEpisode) / len(allIDs)))

    # print(f"First thirty minutes: {firstThirtyMins}")
    # print(f"First two hours: {firstTwoHours}")
    # print(f"First three hours: {firstThreeHours}")
    # print(f"Within three hours: {within3HoursOfAnother}")

    # print(f"All rest: {allOthers}")
    return everyEpisode
import random
def endTimeForPatient(id):
    # record = getJustRecordByID(id)
    header = getHeaderById(id)
    # duration_s = len(record.p_signal) / record.fs # seconds in signal
    duration_s = header.sig_len / header.fs

    base = header.base_datetime
    # print(duration_s, header.sig_len / header.fs)
    return base + dt.timedelta(seconds=duration_s)
timeIntervals = [(15, 30), (30, 60), (60, 90), (90, 120)]

def sampleEveryPatient():
    allIds = getAllRecordIDs()
    dfDict = {
        'time': list(),
        'patient_id': list(),
        'basetime': list(),
    }
    for id in allIds:
        #get a point in time for every 5 minutes
        header = getHeaderById(id)
        signalStart = header.base_datetime
        signalEnd = endTimeForPatient(id)
        minutesInSignal = (signalEnd - signalStart).total_seconds() / 60
        ## split signal into 5 minute increments and go
        iterations = int(minutesInSignal // 5)
        for i in range(iterations):
            t = signalStart + dt.timedelta(minutes=5)*(i+1)
            print(t, signalEnd - t)
            dfDict['time'].append(t)
            dfDict['patient_id'].append(id)
            dfDict['basetime'].append(signalStart)
    return pd.DataFrame(dfDict)
def sampleDataModded(episodes: pd.DataFrame, only_negatives=False) -> pd.DataFrame:
    dfDict = {
        'time': list(),
        'patient_id': list(),
        'basetime': list(),
        'event': list()
    }
    ## collect positive samples from episodes present
    for idx, episode in episodes.iterrows():
        id, episodeStart, basetime = episode['id'], episode['start'], episode['basetime']
        basetime = episode['basetime']
        # for each episode, sample points up to four hours before event:
        ## - 15-30 minutes before event
        ## - 30-60 minutes before event
        ## - 60-90 minutes before event
        ## - 90-120 minutes before event
        for minutesBefore in range(1, 120):
            timeBeforeStart = episodeStart - dt.timedelta(minutes=minutesBefore)
            if (timeBeforeStart < basetime):
                continue
            dfDict['time'].append(timeBeforeStart)
            dfDict['basetime'].append(basetime)
            dfDict['patient_id'].append(id)
            dfDict['event'].append(1)
    for i, group in episodes.groupby('id'):
        id = group['id'][0]
        lastEp = group.sort_values('start').iloc[-1, :]
        header = getHeaderById(id)
        remainingTime = endTimeForPatient(id) - lastEp['stop']
        if (remainingTime <= dt.timedelta(minutes=15)):
            continue #skip sampling negative cases from this patient
        for minutesBefore in range(10, 130):
            timeBeforeEnd = endTimeForPatient(id) - dt.timedelta(minutes=minutesBefore)
            if (timeBeforeEnd <= (lastEp['stop']+dt.timedelta(hours=2))):
                continue
            dfDict['time'].append(timeBeforeEnd)
            dfDict['basetime'].append(header.base_datetime)
            dfDict['patient_id'].append(id)
            dfDict['event'].append(0)
    nonAfibbyPatients = set(getAllRecordIDs()) - set(episodes['id'].unique())
    for nonAfibPatientId in nonAfibbyPatients:
        #collect two hours before end of signal and middle of signal for non afib episode patients
        #end
        header = getHeaderById(nonAfibPatientId)
        base, end = header.base_datetime, endTimeForPatient(nonAfibPatientId)
        for minutesBefore in range(10, 130):
            timeBeforeEnd = endTimeForPatient(id) - dt.timedelta(minutes=minutesBefore)
            dfDict['time'].append(timeBeforeEnd)
            dfDict['basetime'].append(header.base_datetime)
            dfDict['patient_id'].append(id)
            dfDict['event'].append(0)
        midPoint = base + (end - base) / 2
        for minutesBefore in range(10, 130):
            timeBeforeEnd = midPoint - dt.timedelta(minutes=minutesBefore)
            dfDict['time'].append(timeBeforeEnd)
            dfDict['basetime'].append(header.base_datetime)
            dfDict['patient_id'].append(id)
            dfDict['event'].append(0)
    return pd.DataFrame(dfDict)




def sampleData(episodes: pd.DataFrame, only_negatives=False) -> pd.DataFrame:
    dfDict = {
        'time': list(),
        'patient_id': list(),
        'basetime': list()
    }
    for start, stop in timeIntervals:
        dfDict[f"afib_in_{start}_to_{stop}"] = list()
    ## collect positive samples from episodes present
    for idx, episode in episodes.iterrows():
        if (only_negatives):
            break
        id, episodeStart, basetime = episode['id'], episode['start'], episode['basetime']
        basetime = episode['basetime']
        # for each episode, featurize segments at following intervals:
        ## - 15-30 minutes before event
        ## - 30-60 minutes before event
        ## - 60-90 minutes before event
        ## - 90-120 minutes before event
        for start, stop in timeIntervals:
            for minutesBefore in range(start, stop):
                timeBeforeStart = episodeStart - dt.timedelta(minutes=minutesBefore)
                if (timeBeforeStart < basetime):
                    continue
                dfDict['time'].append(timeBeforeStart)
                dfDict['basetime'].append(basetime)
                dfDict['patient_id'].append(id)
                dfDict[f"afib_in_{start}_to_{stop}"].append(True)
                for otherStart, otherStop in (set(timeIntervals) - set([(start, stop)])):
                    dfDict[f"afib_in_{otherStart}_to_{otherStop}"].append(False)

    ## collect negative samples, ensuring they are properly distant from episodes present
    # First, collect randomly from patients who have no afib episodes
    nonAfibbyPatients = set(getAllRecordIDs()) - set(episodes['id'].unique())
    print(len(nonAfibbyPatients), len(episodes['id'].unique()), len(getAllRecordIDs()))
    for nonAfibPatientId in nonAfibbyPatients:
        break
        # collect 20 random points in time from these patients
        negSamplesPerPatient = 50
        record, annotation = getRecordByID(nonAfibPatientId)
        basetime = record.base_datetime
        end = basetime + dt.timedelta(seconds = len(record.p_signal) / record.fs)

        for i in range(negSamplesPerPatient):
            #collect random point in time after first three hours of signal so we can featurized
            randomTime = basetime + (random.random() * (end - (basetime+dt.timedelta(hours=3))))
            dfDict['time'].append(randomTime)
            dfDict['patient_id'].append(nonAfibPatientId)
            dfDict['basetime'].append(basetime)
            for start, stop in timeIntervals:
                # no afib upcoming for all time intervals
                dfDict[f"afib_in_{start}_to_{stop}"].append(False)
    # Then, sample from patients who have episodes, ensuring proper distance to and from episodes
    for i, group in episodes.groupby('id'):
        id = group.iloc[0, :]['id']
        negSamplesPerPatient = 30
        record, annotation = getRecordByID(id)
        basetime = record.base_datetime
        end = basetime + dt.timedelta(seconds = len(record.p_signal) / record.fs)

        basetime_adjusted = basetime + dt.timedelta(hours=2)
        for i in range(negSamplesPerPatient):
            #collect random point in time, avoiding first two hours and last two hours of signal (first two so we can featurize, last two because we don't know if episode began after recording ended)
            randomTime = basetime_adjusted + (random.random() * (end - (basetime+dt.timedelta(hours=4))))

            # we wish to use negative samples that are two hours after an episode and more than two hours before another episode
            def timeAvoidsEpisodes(time: dt.datetime, df: pd.DataFrame, distance_hours=2):
                for _, row in df.iterrows():
                    start, stop = row['start'] - dt.timedelta(hours=2), row['stop'] + dt.timedelta(hours=2)
                    if (start <= time and time <= stop):
                        print(start, time, stop)
                        return False
                return True
            if timeAvoidsEpisodes(randomTime, group, distance_hours=2):
                dfDict['time'].append(randomTime)
                dfDict['patient_id'].append(id)
                dfDict['basetime'].append(basetime)
                for start, stop in timeIntervals:
                    # no afib upcoming for all time intervals
                    dfDict[f"afib_in_{start}_to_{stop}"].append(False)

    return pd.DataFrame(dfDict)

class Struct:
    def __init__(self, **entries):
        self.__dict__.update(entries)
import yaml
def getModelConfig():
    with open(Path('/home/rkaufman/workspace/afib_detection/model') / 'config.yml', 'r') as yamlfile:
        configContents = yaml.safe_load(yamlfile)
    return Struct(**configContents)
def getAllFeatsGivenMode(mode, filterCurrent=False):
    if (mode == 'overlapping'):
        beforeMinutesOfInterest = [10, 15]
        # overlapping featurization doesn't have vanilla feats, all are suffixed
        possibleFeatureSuffixes =  [ f"_{m}" for m in beforeMinutesOfInterest ]
    elif (mode == 'lengthened'):
        beforeMinutesOfInterest = [30, 60]
        possibleFeatureSuffixes = [''] + [ f"_{m}" for m in beforeMinutesOfInterest ]
    elif (mode == 'acute'):
        beforeMinutesOfInterest = [5, 10, 15, 30]
        possibleFeatureSuffixes = [''] + [ f"_{m}" for m in beforeMinutesOfInterest ]

    if (filterCurrent):
        possibleFeatureSuffixes = [possibleFeatureSuffixes[0]]
    # beforeMinutesOfInterest = []
    # beforeMinutesOfInterest = [15, 30, 60, 120]
    allFeats = list()
    allFeatBases = getModelConfig().features_nk
    for featBase in allFeatBases:
        for featureSuffix in possibleFeatureSuffixes:
            allFeats.append(f"{featBase}{featureSuffix}")
    return allFeats

def getTimePrecedingFirstAfib():
    eps = readAllEpisodes()
    dfDict = {
        'patient_id': list(),
        'time': list(),
        'basetime': list()
    }
    j = 0
    maxIds = set(['105', '64', '55', '45'])
    for i, group in eps.groupby('id'):
        if (j == 10):
            break
        firstEpisode = group.sort_values('start').iloc[0, :]
        id = firstEpisode['id']
        if (not (id in maxIds)):
            continue
        print(group)
        header = getHeaderById(id)
        base = header.base_datetime
        timeToStart_m = int((firstEpisode['start'] - base).total_seconds() // 60)
        # iterate through minutes, skipping interval 15-120 minutes before since they are already present
        for minutesBefore in range(timeToStart_m):
            if (minutesBefore >= 15) and (minutesBefore <= 120):
                continue
            dfDict['patient_id'].append(id)
            dfDict['time'].append(firstEpisode['start'] - dt.timedelta(minutes=minutesBefore))
            dfDict['basetime'].append(base)
        # j += 1
    # together = zip(max_ids, timesToStart)
    # s = sorted(together, key= lambda x: x[1], reverse=True)
    # print('\n'.join([str(j) for j in s]))
    return pd.DataFrame(dfDict)

# df = getTimePrecedingFirstAfib()
# df.to_csv('preceding_afib_episodes.csv', index=False)
# 1/0
# def loadDataInSurvivalFormat(src='all_predictor_data_bucket.csv'):
# def loadDataInSurvivalFormat(src='featurized_afib_predictor_data.csv', inDFForm=False):
def loadDataInSurvivalFormat(src='featurized_afib_predictor_data.csv', filterCurrent=False, acute=False, lengthened=False, overlapping=False, inDFForm=False):
    X, t, E = [], [], []
    #to be done after featurization and population of all features
    eps = readAllEpisodes()
    if (acute):
        allFeats = getAllFeatsGivenMode('acute', filterCurrent=filterCurrent)
        src = src.split('.')[0] + "_acute" + ".csv"
    elif (lengthened):
        allFeats = getAllFeatsGivenMode('lengthened', filterCurrent=filterCurrent)
        src = src.split('.')[0] + "_lengthened" + ".csv"
    elif (overlapping):
        allFeats = getAllFeatsGivenMode('overlapping', filterCurrent=filterCurrent)
        src = src.split('.')[0] + "_overlapping" + ".csv"

    df = pd.read_csv(Path(__file__).parent.parent.parent / 'data' / 'assets' / src, parse_dates=['time', 'basetime'], dtype={'patient_id': str})
    nonCensoredData = set()
    if (inDFForm):
        subIndices = list()
        patientIDs = list()

    positives = list()


    #iterate through src, find time to event for each and event type for each
    # print(eps['id'], df['patient_id'])
    for i, point in tqdm(df.iterrows(), total=len(df)):
        # define features for point, time-to-event (tte), and event
        x, tte, e = None, None, None
        def timeToEvent(p) -> Tuple[dt.timedelta, int]:
            e = None
            episodesForID = eps[eps['id'] == p['patient_id']]
            if len(episodesForID) == 0:
                return None, None
                tte = endTimeForPatient(p['patient_id']) - p['time']
                e = 0
            else:
                #sort episodes in patient by time and iterate, if we find a match we know it's the soonest episode which also comes after point
                # firstEpisode, lastEpisode = episodesForID.sort_values('start').iloc[0, :], episodesForID.sort_values('start').iloc[-1,:]
                # if (p['time'] < firstEpisode['start']):
                #     tte = (firstEpisode['start'] - p['time']).to_pytimedelta()
                #     #is match if time isn't in prior episode
                #     e = 1
                # elif (p['time'] > lastEpisode['stop']):
                #     tte = (endTimeForPatient(p['patient_id']) - p['time']).to_pytimedelta()
                #     e = 0
                # else: #point is after first episode and before another
                #     tte, e = None, None
                episodesForID = episodesForID.sort_values('start')
                for j, ep in episodesForID.iterrows():
                    ## uncomment below to skip non-first afib episodes
                    # if (j > 0):
                    #     return None, None
                    if ep['start'] > p['time']:
                        #disquality points that are within two hours after prior episode
                        if ((j>0) and ((p['time']-episodesForID.iloc[j-1, :]['stop']) < dt.timedelta(minutes=120))):
                            return None, None
                        tte = ep['start'] - p['time']
                        #is match if time isn't in prior episode
                        e = 1
                        positives.append(tte)
                        return tte.to_pytimedelta(), e
                if e is None:
                    tte = endTimeForPatient(p['patient_id']) - p['time']
                    e = 0
            return tte, e

        x = point[allFeats]
        tte, e = timeToEvent(point)
        if (type(tte) != type(None)):
            # if (acute):
            #     #if tte is greater than 30 minutes away for afib event then we ignore
            #     if ((e == 1) and (tte > dt.timedelta(minutes=30))):
            #         continue
            if (inDFForm):
                subIndices.append(i)
                patientIDs.append(point['patient_id'])
            X.append(x); t.append(tte), E.append(e)
    # print('num negatives:')
    # print(sum(E) - len(E))
    # print('num positives:')
    # print(sum(E))
    # print(f'Max tte for event: {max(positives)}')

    X = pd.DataFrame.from_records(X, columns=allFeats)
    import numpy as np
    X.replace([-np.inf, np.inf], np.nan, inplace=True)
    old = set(X.index)
    nanValued = list(old - set(X.index))
    t = np.delete(t, nanValued)
    E = np.delete(E, nanValued)
    patientIDs = np.delete(patientIDs, nanValued)
    subIndices = np.delete(subIndices, nanValued)
    if (inDFForm):
        df = df[df.index.isin(subIndices)]
        df['patient_id'] = patientIDs
        df.reset_index(inplace=True)
        return (X, t, E), df
    return X, t, np.array(E)





def sampleFromAfibEpisodes(collectionsPerEpisode=30):
    eps = readAllEpisodes()
    dfDict = {
        'patient_id': list(),
        'time': list(),
        'basetime': list()
    }
    for _, ep in eps.iterrows():
        print(ep)
        id, start, stop, basetime = ep['id'], ep['start'], ep['stop'], ep['basetime']
        allNewTimes = pd.date_range(
            start=start,
            end=stop,
            periods=collectionsPerEpisode).to_numpy()
        for time in allNewTimes:
            dfDict['patient_id'].append(id)
            dfDict['basetime'].append(basetime)
            dfDict['time'].append(time)
        print(id, allNewTimes)
    return pd.DataFrame(dfDict)

# go through, finding first events for patient. If it is an afib rhythm, and there are 

if __name__ == '__main__':
    # collectAllEventsAndEpisodes()
    # eps = readAllEpisodes()
    # df = sampleDataModded(eps)
    # print(df)
    # print(df[df['event']==0])
    # print(df[df['event']==1])
    # df.to_csv('modded_predictor_data.csv', index=False)
    # res = {
    #     'patient_id': list(),
    #     'tt_first_episode_minutes': list()
    # }


    # for i, group in eps.groupby('id'):
    #     print(len(group))
    #     firstEpisode = group.sort_values('start').iloc[0, :]
    #     id = firstEpisode['id']
    #     header = getHeaderById(id)
    #     base = header.base_datetime
    #     timeToStart_m = int((firstEpisode['start'] - base).total_seconds() // 60)
    #     res['patient_id'].append(id)
    #     res['tt_first_episode_minutes'].append(timeToStart_m)
    # print(res)
    # pd.DataFrame(res).to_csv('timeToFirstEpisode.csv', index=False)
    # df = sampleFromAfibEpisodes()#sampleData(eps, only_negatives=True)
    # print(df)
    # df.to_csv('afib_episode_samples_30.csv', index=False)

    # x, t, e = loadDataInSurvivalFormat()
    # print(t, e)
    # print(len(x), len(t), len(e))
    # print(sum(t) / len(t))

    # parse()
    # collectEpisodesForID('122', dst)
    # print(collectEvents('01', dst=dst))
    # collectEpisodesFromEvents('01', searchDir=dst)
    # print(collectEvents('122'))
    # print(collectEvents('00'))
    df = sampleEveryPatient()
    df.to_csv('europace_everybody_5min.csv')