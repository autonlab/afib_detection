import pandas as pd
import numpy as np
import random
import audata as aud
from tqdm import tqdm
# from pptree import print_tree
from utilities import findFileByFIN, getFINFromPath, isValid, getRandomSlice, getSlice, HR_SERIES
import heartpy as hp
import scipy as sp
from scipy.stats import variation, iqr
import pytz
import warnings

class Node(object):
    def __init__(self, value,children=None, extrainfo=None):
        self.extrainfo = extrainfo
        if children:
            self.children = children
        else:
            self.children = list()
        # if (name):
        #     self.name = name
        # else:
        if (isinstance(value, pd.DataFrame)):
            self.name = str(len(value))
        else:
            self.name = str(value)
        self.value = value

    def addChild(self, child):
        self.children.append(child)
    
    def collectLeaves(self, chainsofar=None):
        '''
        Function returning list of all leaves in tree'''
        if len(self.children) == 0:
            return [self]

        result = list()
        for c in self.children:
            result += c.collectLeaves()
        return result 

    def __repr__(self, depth=0):
        eiStr = self.extrainfo if self.extrainfo else ''
        res = depth*"\t" + f'{self.name} {eiStr}\n'
        for child in self.children:
            res += child.__repr__(depth+1)
        return res

def constructTreeFromCategoriesAndValues(categories, valuesForEach, t=None):
    if (len(categories) == 0):
        return 0
    if (t == None):
        t = Node('root', [Node(categoryVal) for categoryVal in valuesForEach[0]])
    else:
        for categoryVal in valuesForEach[0]:
            t.addChild(Node(categoryVal))
    for child in t.children:
        constructTreeFromCategoriesAndValues(categories[1:], valuesForEach[1:], child)
    return t

def printTreePartitionCardinalityFromDataframe(t, df, chainsofar=None, depth=0, runningSum = 0):
    if (chainsofar==None):
        chainsofar = list()
    for child in t.children:
        if (child.children):
            localChain = chainsofar + [child.value]
            printTreePartitionCardinalityFromDataframe(child, df, localChain, depth+1)
        else:
            localDF = df
            localChain = chainsofar + [child.value]
            for idx, item in enumerate(localChain):
                localDF = localDF[localDF[buckets[idx]] == item]
            child.addChild(Node(localDF, extrainfo=(len(localDF) / len(df))*100 % 1000))
    return t

def getSum(t):
    if (t.extrainfo):
        return t.extrainfo
    s = 0
    for c in t.children:
        s += getSum(c)
    return s

def formTreeFromDF(columnsToPartitionBy, csvSrc="./1.28.22demoallplusmeds.csv"):
    df = pd.read_csv(csvSrc)
    df.columns = df.columns.str.lower()
    # print(df.head())

    #organize buckets by list of hierarchy
    # columnsToPart = ['age_score', 'sex', 'racecategories']
    possibleValuesForEach = list()
    for bucket in columnsToPartitionBy:
        possibleValuesForBucket = df[bucket].unique()
        possibleValuesForEach.append(list(possibleValuesForBucket))
        # print(f'{bucket}: {possibleValuesForBucket}')
        for possibleVal in possibleValuesForBucket:
            valuesInBucket = df[df[bucket] == possibleVal]
            # print(f'\t{possibleVal}: {len(valuesInBucket)}')

    categoryTree = constructTreeFromCategoriesAndValues(buckets, possibleValuesForEach)
    # print(categoryTree)
    t = printTreePartitionCardinalityFromDataframe(categoryTree, df)
#    for child in t.children:
#        print_tree(child, nameattr="name", horizontal=False)
    # print(t)
    # ensure partition percentages add to 100
    assert(np.isclose(100, getSum(t)))
    return t

def getbpm(series, samplingFrequency):
    # want: RR_list with given filter
    if (len(series) < 4000):
        #print('AAH')
        return -1
    #print(len(series), min(series), max(series))
    try:
        w, m = hp.process(series, samplingFrequency)
    except:
        return -1
    beats = sum(w['binary_peaklist'])
    bpm = beats * 6
    #print(bpm, m['bpm'])
    
    return bpm

def getb2bIQR(series, samplingFrequency):
    # want: RR_list with given filter
    if (len(series) < 4000):
        #print('AAH')
        return -1
    #print(len(series), min(series), max(series))
    try:
        w, m = hp.process(series, samplingFrequency)
        w, m = hp.process_rr(w['RR_list'], threshold_rr=True, clean_rr=True, calc_freq=False)
    except:
        return -1
    
    RR_list = np.array(w['RR_list_cor'])
    beat2beatIntervals = list()
    for rrIntervalMS in RR_list:
        beat2beatIntervals.append(60 / (rrIntervalMS/1000.0))
    b2b_iqr = sp.stats.iqr(np.array(beat2beatIntervals))
    return b2b_iqr
    #rrsToKeep = list()
    #binaryPeaklist = np.array(w['binary_peaklist'])
    #pprint(w.keys())
    
    #for i, e in enumerate(w['RR_list']):
    #    if binaryPeaklist[i+1] == 0 or binaryPeaklist[i] == 0:
    #        continue
    #    rrsToKeep.append(e)
    #RR_list = np.array(rrsToKeep)
        # print(w['peaklist'])
        # print(w['binary_peaklist'])
        # RR_list = np.array(w['RR_list'])[np.invert(np.array(w['RR_masklist']).astype(bool))]
    # binaryPeaklist = np.array(w['binary_peaklist'])
    # beatSequence = np.array(w['peaklist'])[binaryPeaklist.astype(bool)]
    # print(RR_list, beatSequence, binaryPeaklist)
    # hp.plotter(w, m,title=filterType)
    # plt.show()

    #print(RR_list)
    #print(beat2beatIntervals)

def satisfiesIQRForAfib(fin, start, stop, searchDir, afib_threshold=15.0):
    if (not start) or (not stop):
        return False
    realistic_max = 100.0
    series, sf = getSlice(findFileByFIN(fin, searchDir), HR_SERIES, start, stop)
    b2b_iqr = getb2bIQR(series, sf)
    if (b2b_iqr) < 0:
        return False
    #print(b2b_iqr)
    return (b2b_iqr > afib_threshold) and (b2b_iqr < realistic_max)

def satisfiesIQRForSinus(fin, start, stop, searchDir, afib_threshold=15.0):
    if (not start) or (not stop):
        return False
    series, sf = getSlice(findFileByFIN(fin, searchDir), HR_SERIES, start, stop)
    b2b_iqr = getb2bIQR(series, sf)
    if (b2b_iqr) < 0:
        return False
    #print(b2b_iqr)
    return b2b_iqr < afib_threshold

def satisfiesBPMThreshold(fin, start, stop, searchDir, bpm_threshold=140):
    if (not start) or (not stop):
        return False
    series, sf = getSlice(findFileByFIN(fin, searchDir), HR_SERIES, start, stop)
    bpm = getbpm(series, sf)
    if (bpm) < 0:
        return False
    #print(b2b_iqr)
    return bpm > bpm_threshold

def sampleCollector_random(df, numSamplesDesired, patientSeriesSearchDirectory, progress=None):
    df.columns = df.columns.str.lower()
    dfDict = {
        'fin_study_id': list(),
        'start': list(),
        'stop': list(),
    }

    # afibSubset = df[df['afib'] > 0]
#    if len(afibSubset) == 0 or len(nonAfibSubset) == 0:
    #go through patients
    fins = list(df['fin_study_id'])
    fins = [str(fin) for fin in fins]
    while numSamplesDesired > 0:
        fin = random.choice(fins)
        file = findFileByFIN(fin, patientSeriesSearchDirectory)
        while (not file):
            fin = random.choice(fins)
            file = findFileByFIN(fin, patientSeriesSearchDirectory)
        fins.remove(fin)

        #select 4 segments per patient
        numToSelect = 10
        # numToSelect = max(2, round(numSamplesDesired * .10))
        for i in range(numToSelect):
            start, stop = getRandomSlice(fin, 10, patientSeriesSearchDirectory)

            if (start and stop):
                dfDict['fin_study_id'].append(fin)
                dfDict['start'].append(start)
                dfDict['stop'].append(stop)
                numSamplesDesired -= 1
                if (progress):
                    progress.update(1)
    return pd.DataFrame(dfDict)

def sampleCollector_testset(df, numSamplesDesired, patientSeriesSearchDirectory):
    dfDict = {
        'fin_study_id': list(),
        'start': list(),
        'stop': list(),
    }
    #for test set, we wish to select 50% randomly, and 50% only if the b2b iqr is over afib_threshold
    numSamplesSinusLeft, numSamplesAfibLeft = round(.4*numSamplesDesired), round(.6*numSamplesDesired)

    afibSubset = df
    nonAfibSubset = df
#    if len(afibSubset) == 0 or len(nonAfibSubset) == 0:

    #go through patients with afib, collecting two sinus and two afibs from each until complete
    fins = list(afibSubset['fin_study_id'])
    fins = [str(fin) for fin in fins]
    while numSamplesAfibLeft > 0:
        fin = random.choice(fins)
        while (not findFileByFIN(fin, patientSeriesSearchDirectory)):
            fin = random.choice(fins)
        for i in range(min(2, numSamplesAfibLeft)):
            itrCount = 0
            start, stop = getRandomSlice(fin, 10, patientSeriesSearchDirectory)
#            while (not start): #in case get random slice returned an error, try again
#                start, stop = getRandomSlice(fin, 10, patientSeriesSearchDirectory)
#                itrCount += 1
#                if (itrCount > 50):
#                    break
#            if (itrCount > 50):
#                break
            #get slice, feed it into iqr featurizer, True if over False if under
            itrCount = 0
            while (not satisfiesIQRForAfib(fin, start, stop, patientSeriesSearchDirectory)):
                start, stop = getRandomSlice(fin, 10, patientSeriesSearchDirectory)
                itrCount += 1
                if (itrCount > 50):
                    break
            if (itrCount > 50):
                break

            #now that we've found sample that satisfies iqr requirement, add it and continue
            dfDict['fin_study_id'].append(fin)
            dfDict['start'].append(start)
            dfDict['stop'].append(stop)
            numSamplesAfibLeft -= 1

    fins = list(nonAfibSubset['fin_study_id'])
    fins = [str(fin) for fin in fins]
    target = numSamplesSinusLeft
    while numSamplesSinusLeft > 0:
        fin = random.choice(fins)
        while (not findFileByFIN(fin, patientSeriesSearchDirectory)):
            fin = random.choice(fins)
        for i in range(min(2, numSamplesSinusLeft)):
            itrCount = 0
            #get slice, feed it into iqr featurizer, False if over True if under
            itrCount = 0
            start, stop = getRandomSlice(fin, 10, patientSeriesSearchDirectory)
            while (not satisfiesIQRForSinus(fin, start, stop, patientSeriesSearchDirectory)):
                start, stop = getRandomSlice(fin, 10, patientSeriesSearchDirectory)
                itrCount += 1
                if (itrCount > 25):
                    break
            if (itrCount > 25):
                break

            #now that we've found sample that is beneath afib iqr threshold, add it and continue
            dfDict['fin_study_id'].append(fin)
            dfDict['start'].append(start)
            dfDict['stop'].append(stop)
            numSamplesSinusLeft -= 1
    return pd.DataFrame(dfDict)

def stratifiedSample(sampleCountTotal, columnsToStratifyBy, csvDF, sampleCollectorFn, searchDir):
    t = formTreeFromDF(columnsToStratifyBy, csvDF)
    
    partitionedDFs = t.collectLeaves()
    samples = list()
    print(f'Sampling {sampleCountTotal} segments from via {sampleCollectorFn.__name__}.')
    # samples = ProgressParallel(n_jobs=4)(delayed(sampleCollectorFn)(
    #     partition.value, #partitionedDF
    #     round( sampleCountTotal * partition.extrainfo / 100 ), #samplesToCollect
    #     searchDir
    # ) for partition in partitionedDFs)
    progress = tqdm(total=sampleCountTotal)
    for partition in partitionedDFs:
        partitionedDF = partition.value
        percentageInThisPartition = partition.extrainfo
        samplesToCollect = round(sampleCountTotal * percentageInThisPartition / 100.0)
        # runningSum += samplesToCollect
        # print('\t' + str(samplesToCollect))
        samples.append(sampleCollectorFn(partitionedDF, samplesToCollect, searchDir, progress))
    progress.close()

    return pd.concat(samples, ignore_index=True)

def filter_segments(srcCsv):
    df = pd.read_csv(srcCsv, parse_dates=['start', 'stop'])
    res = {
    'fin_study_id': list(),
    'start': list(), 'stop': list(), 'b2b_iqr': list()
    }
    patientSeriesSearchDirectory = '/zfs2/mladi/viewer/projects/mladi/originals'
    for idx, row in tqdm(df.iterrows(), total=len(df)):
        filepath = findFileByFIN(row['fin_study_id'], patientSeriesSearchDirectory)
        f = aud.File.open(filepath)
        start = row['start']
        stop = row['stop']
        if (isValid(f, start, stop)):
            res['fin_study_id'].append(row['fin_study_id'])
            res['start'].append(start)
            res['stop'].append(stop)
            series, sf = getSlice(filepath, HR_SERIES, start, stop)
            b2b_iqr = getb2bIQR(series, sf)
            res['b2b_iqr'].append(b2b_iqr)
    d = pd.DataFrame(res)
    d.to_csv('filtered_shared_segments.csv')
    print(d.head())


def test_segments():
    df = pd.read_csv('./stratified_sampled_shared.csv', parse_dates=['start', 'stop'])
    seriesOfInterest = '/data/waveforms/II:value'
    seriesOfInterest =          '/data/waveforms/II:value'
    frow, srow = df.iloc[0,:], df.iloc[2,:]
    patientSeriesSearchDirectory = '/zfs2/mladi/viewer/projects/mladi/originals'

    filepath = findFileByFIN(srow['fin_study_id'], patientSeriesSearchDirectory)
    f = aud.File.open(filepath)
    seriesOfInterest = HR_SERIES
    print(seriesOfInterest)
    fileStartTime = f[seriesOfInterest][0]['time'].item().to_pydatetime()
    fileEndTime = f[seriesOfInterest][-1]['time'].item().to_pydatetime()
    fileStartTime, fileEndTime = fileStartTime.replace(tzinfo=pytz.UTC), fileEndTime.replace(tzinfo=pytz.UTC)

    prospectiveStart = srow['start']
    prospectiveEnd = srow['stop']
    prospectiveStart, prospectiveEnd = prospectiveStart.replace(tzinfo=pytz.UTC), prospectiveEnd.replace(tzinfo=pytz.UTC)
    print(fileStartTime, fileEndTime)
    print(prospectiveStart, prospectiveEnd)
    print(isValid(f, prospectiveStart, prospectiveEnd))

if __name__ == '__main__':
    warnings.filterwarnings('ignore')
    buckets = ['age_score', 'sex', 'racecategories']
#    filter_segments('150_stratified_samples.csv')
#    test_segments()
    # samples = stratifiedSample(10000, buckets, "./assets/1.28.22demoallplusmeds.csv", sampleCollector_random, '/home/rkaufman/workspace/remote')
    progress = tqdm(total=5000)
    samples = sampleCollector_random(pd.read_csv('./assets/inDemoButNotTrainset.csv'), 5000, '/home/rkaufman/workspace/remote', progress)
    progress.close()
    samples.to_csv('./assets/5000_segments.csv')
#    print(samples)
#    samples.to_csv('150_stratified_samples.csv')
    # randoms = sampleCollector_random(pd.read_csv('./1.28.22demoallplusmeds.csv'), 5, '/zfs2/mladi/viewer/projects/mladi/originals')
    # randoms.to_csv('randoms.csv')
#    samples = pd.read_csv('150_stratified_samples.csv')
#    samples.to_csv('stratified_gold_2.csv')
#    samples = pd.read_csv('stratified_gold_2.csv')
#    samplesRandomized = samples.sample(frac=1).reset_index(drop=True)
#    first50, rest = samplesRandomized.iloc[:50,:], samplesRandomized.iloc[50:,:]
#    first50.to_csv('stratified_gold_shared2.csv')
#    rest.to_csv('stratified_gold_rest2.csv')
#    filter_segments('stratified_gold_shared2.csv')
    # formTreeFromDF(copy.copy(buckets))
