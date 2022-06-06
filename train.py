from joblib import Memory
import numpy as np
import os
import pandas as pd
from pathlib import Path
from scipy.stats.mstats import winsorize
import sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from model.df4tsc.resnet import Classifier_RESNET

import data.manipulators as dm
import data.utilities as du
from model.labelmodel import LabelModelCustom
from snorkel.labeling.model import LabelModel
import model.utilities as mu

def trainlm(modifyLabels=False) -> LabelModelCustom:
    modelconfig = mu.getModelConfig()
    print(f'Loading features: {modelconfig.features}')
    print(f'Loading data set: {modelconfig.trainDataFile}')
    df = pd.read_csv(
        Path(__file__).parent / 'data' / 'assets' / modelconfig.trainDataFile,
        parse_dates=['start', 'stop'])

    df.columns = df.columns.str.lower()

    fitModel = LabelModelCustom()
    fitModel.fit(df)
    predictedLabels = fitModel.predict(df)
    predictedProbas = [max(e) for e in fitModel.predict_proba(df)]
    df['label'] = predictedLabels
    df['confidence'] = predictedProbas
    if (modifyLabels):
        print(f'Writing labelmodel votes to: {modelconfig.trainDataFile} ...')
        df.to_csv(
            Path(__file__).parent / 'data' / 'assets' / modelconfig.trainDataFile,
            index=False
        )
        print('Done.')
    return fitModel

def p(string, v):
    """Print if verbose on
    """
    if (v):
        print(string)

def train(model='RandomForestSK', winsorize=False, load=False, usesplits=True, verbose=False, filterUnreasonableValues=False, filterGold=False, overwriteTrainset=None, overwriteTestset=None, reduceDimension=False):
    """Train and return specified model.
     PRECONDITIONS:
    - CSV of featurized data in `data/assets`, csv title specified in `model/config.yml` in `trainDataFile` field
    - Features enumerated under `features` field of `model/config.yml`

    Args:
        load (bool, optional): _description_. Defaults to False.
    """
    goldData = pd.DataFrame()

    dataConfig = du.getDataConfig()
    mem = Memory(dataConfig.cacheDirectory)

    ## Load necessary configuration from model
    modelconfig = mu.getModelConfig()
    modelconfig.features = [f.lower() for f in modelconfig.features]
    if (reduceDimension):
        # modelconfig.features = set(modelconfig.features) - set(['hfd', 'hrv_hf', 'hrv_lfhf', 'sd1', 'sample_entropy', 'max_sil_score', 'hrv_lf', 'b2b_var', 'rmssd', 'sd1/sd2', 'sd2', 'hopkins_statistic', 'b2b_std'])
        # modelconfig.features = list(modelconfig.features)

        modelconfig.features = ['hrv_hf', 'hrv_lf', 'sse_2_clusters', 'sse_1_clusters', 'rmssd', 'pnn20', 'sample_entropy', 'max_sil_score', 'sdsd']
        print(f'Features after reduction: {modelconfig.features}')

    trainDataFile = overwriteTrainset if overwriteTrainset else modelconfig.trainDataFile
    goldDataFile = overwriteTestset if overwriteTestset else modelconfig.goldDataFile
    p(f'Loading features: {modelconfig.features}', verbose)
    p(f'Loading data set: {trainDataFile}', verbose)
    if (goldDataFile.endswith('csv')):
        p(f'Loading gold set: {goldDataFile}', verbose)
        goldData = pd.read_csv(
            Path(__file__).parent / 'data' / 'assets' / goldDataFile,
            parse_dates=['start', 'stop']
        )
        goldData.columns=goldData.columns.str.lower()
        if (modelconfig.labelCorrectionMap):
            goldData = dm.remapLabels(goldData, 'label', modelconfig.labelCorrectionMap)
    df = pd.read_csv(
        Path(__file__).parent / 'data' / 'assets' / trainDataFile,
        parse_dates=['start', 'stop'])
    df.columns = df.columns.str.lower()
    if (modelconfig.labelCorrectionMap):
        df = dm.remapLabels(df, 'label', modelconfig.labelCorrectionMap)
    ## Filter then normalize the data
    p('Filtering and normalizing...', verbose)
    #count occurrences of infinity in dataframe and mark them for dropping if existent
    numInfs = np.isinf(df.select_dtypes('float').stack()).groupby(level=1).sum().sum()
    if (numInfs > 0):
        p(f'\tFound {numInfs} entries with value infinity, replacing them with nan', verbose)
        df.replace([np.inf, -np.inf], np.nan, inplace=True)

    before = len(df); df = df.dropna(); after = len(df)
    p(f'\tDropped {before - after} rows with nan values present.', verbose)
    if (winsorize):
        df = dm.winsorizeDF(df, modelconfig.features)
    if (filterUnreasonableValues):
        df = df[df['b2b_iqr'] < 30]
        goldData = goldData[goldData['b2b_iqr'] < 30]
    df_normalized, scaler = dm.computeAndApplyScaler(df, modelconfig.features)

    ## Split for testing and evaluation
    if (usesplits): # can only avoid using splits if you input gold data set
        p('Splitting into train and test sets...', verbose)
        if (goldData.empty):
            train, test = dm.applySplits(df_normalized)
        else:
            train, test = dm.applySplits(df_normalized, prespecifiedTestSet=goldData['fin_study_id'].unique())
    else:
        p('Not splitting at all :-)', verbose)
        train = df_normalized
    if (not goldData.empty):
        if (winsorize):
            test = dm.winsorizeDF(test, modelconfig.features)
        test = dm.filterAndNormalize(goldData, modelconfig.features, preexistingScaler=scaler)
        if (filterGold):
            p('Filtering gold values too many standard devs away from training mean...', verbose)
            minStdDev, maxStdDev = -5, 5
            before = len(test)
            beforeDF = test.copy()
            for feat in modelconfig.features:
                test = test[(minStdDev <= test[feat]) & (test[feat] <= maxStdDev)]
            diff = beforeDF[~beforeDF.index.isin(test.index)]
            diff.to_csv('removed_segments.csv')
            after = len(test)
            p(f'In total, dropped {before - after} features from gold set', verbose)

    ## Oversample, extract only features (removing identifiers on data)
    shouldOversample = input('Oversample? (y/N): ')
    if (shouldOversample == 'y'):
        p('Oversampling...', verbose)
        countsBeforehand = [(subset['label'].iloc[0], len(subset)) for idx, subset in train.groupby('label')]
        train = dm.oversample(train, 'label', 'confidence')
        countsAfterward = [(subset['label'].iloc[0], len(subset)) for idx, subset in train.groupby('label')]
        for i in range(len(countsAfterward)):
            p(f'\t Class {countsAfterward[i][0]} grew from {countsBeforehand[i][1]:5} to {countsAfterward[i][1]:5} elements', verbose)
    else:
        p('Skipping oversample.', verbose)
    # print(train['label'])
    trainData, trainLabels = train[modelconfig.features], train['label']
    testData, testLabels = test[modelconfig.features], test['label']
    # t2Data, t2Labels = testOnLMLabels[modelconfig.features], testOnLMLabels['label']

    ## train the model, return the model, test the model
    modelname = model
    if (model == 'RandomForestSK'):
        p('Training randomforest...', verbose)
        model = RandomForestClassifier(max_depth=12, n_estimators=1000, class_weight={'ATRIAL_FIBRILLATION': .15, 'NOT_AFIB': .85}, random_state=66)
        # fitModel = RandomForestClassifier(max_depth=5, n_estimators=1000, class_weight='balanced', random_state=66)
        model.fit(trainData, trainLabels)
        modelPredictions = model.predict(testData)
        modelProbabilities = model.predict_proba(testData)
    elif (model == 'LabelModel'):
        fitModel = LabelModelCustom()
        fitModel.fit(trainData)
        modelPredictions = fitModel.predict(testData)
        # modelProbabilities = [max(e) for e in fitModel.predict_proba(testData)]
        modelProbabilities = fitModel.predict_proba(testData)
        res = list()
        for modelPred in modelPredictions:
            if modelPred == 'OTHER' or modelPred == 'SINUS':
                res.append('NOT_AFIB')
            else:
                res.append(modelPred)
        modelPredictions = res

    elif (model == 'LogisticRegression'):
        model = LogisticRegression(random_state = 66)
        model.fit(trainData, trainLabels)
        modelPredictions = model.predict(testData)
        modelProbabilities = model.predict_proba(testData)
        w = model.coef_[0]
    elif (model == 'ResNet'):
        # create input shape from num features
        packForNN = mem.cache(dm.packForNN)

        x_train = trainData.to_numpy()
        x_train, indices = packForNN(x_train, train)
        x_train = x_train.reshape((x_train.shape[0], x_train.shape[1], 1))
        y_train = trainLabels.to_numpy().reshape((-1, 1))
        y_train = y_train[indices]
        # from keras.models import load_model
        # model = load_model(
        #     str(Path(__file__).parent / 'model' / 'assets') + os.sep + 'best_model.hdf5'
        # )
        model = Classifier_RESNET(
            str(Path(__file__).parent / 'model' / 'assets') + os.sep, #outputDir
            x_train.shape[1:], #inputShape
            2, #numClasses
            verbose=True
        )
        x_test = testData.to_numpy()
        x_test, indices = packForNN(x_test, test)
        x_test = x_test.reshape((x_test.shape[0], x_test.shape[1], 1))
        y_test = testLabels.to_numpy().reshape((-1, 1))
        y_test = y_test[indices]
        # enc = sklearn.preprocessing.OneHotEncoder(categories=['ATRIAL_FIBRILLATION', 'SINUS'])
        enc = sklearn.preprocessing.OneHotEncoder(categories='auto')
        # enc = sklearn.preprocessing.OneHotEncoder(categories=['ATRIAL_FIBRILLATION', 'SINUS', 'OTHER'])
        enc.fit(y_train.reshape(-1, 1))
        y_train = enc.transform(y_train.reshape(-1, 1)).toarray()
        model = model.fit(x_train, y_train, None, None, None)

        modelProbabilities = model.predict(x_test)
        modelPredictions = [['ATRIAL_FIBRILLATION', 'SINUS', 'OTHER'][np.argmax(a)] for a in modelProbabilities]
        # print(modelPredictions)
        # modelPredictions = enc.inverse_transform(modelProbabilities)
        newModelPredictions = list()
        for pred in modelPredictions:
            if pred in modelconfig.labelCorrectionMap.keys():
                newModelPredictions.append(modelconfig.labelCorrectionMap[pred])
            else:
                newModelPredictions.append(pred)
        modelPredictions = newModelPredictions
        testLabels = testLabels.to_numpy()[indices]

    p('Done', verbose)
    cacheddata = {
        'testData': testData,
        'testLabels': testLabels,
        'testPredictions': modelPredictions,
        'testPredProbabilities': modelProbabilities,# if model.predict_proba else None,
        'testIdentifiers': goldData[goldData.index.isin(test.index)],
        'trainData': trainData,
        'trainLabels': trainLabels,
        'features': modelconfig.features
    }
    if (modelname == 'LogisticRegression'):
        cacheddata['w'] = w
    return model, cacheddata

if __name__ == "__main__":
    trainlm(modifyLabels=True)
    # train(model='LabelModel', usesplits=False)
    # train(usesplits=False)
    # lrModel, cacheddata = train(usesplits=False, model="LogisticRegression", verbose=False)
    # rfModel = train(usesplits=False, model="RandomForestSK", verbose=False)
