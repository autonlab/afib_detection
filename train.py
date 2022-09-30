from joblib import Memory
import numpy as np
import os
import pandas as pd
from pathlib import Path
import pickle
from scipy.stats.mstats import winsorize
import sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from model.df4tsc.resnet import Classifier_RESNET

import random

import data.manipulators as dm
import data.utilities as du
from model.labelmodel import LabelModelCustom
from snorkel.labeling.model import LabelModel
import model.utilities as mu

def trainlm(modifyLabels=False) -> LabelModelCustom:
    modelconfig = mu.getModelConfig()
    print(f'Loading features: {modelconfig.features_nk}')
    print(f'Loading data set: {modelconfig.trainDataFile}')
    df = pd.read_csv(
        Path(__file__).parent / 'data' / 'assets' / modelconfig.trainDataFile,
        parse_dates=['start', 'stop'])

    df.columns = df.columns.str.lower()

    fitModel = LabelModelCustom()
    fitModel.fit(df)
    if (modifyLabels):
        predictedLabels = fitModel.predict(df)
        predictedProbas = [max(e) for e in fitModel.predict_proba(df)]
        df['label'] = predictedLabels
        df['confidence'] = predictedProbas
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
    modelconfig.features = [f.lower() for f in modelconfig.features_nk]
    if (reduceDimension):
        # modelconfig.features = set(modelconfig.features) - set(['hfd', 'hrv_hf', 'hrv_lfhf', 'sd1', 'sample_entropy', 'max_sil_score', 'hrv_lf', 'b2b_var', 'rmssd', 'sd1/sd2', 'sd2', 'hopkins_statistic', 'b2b_std'])
        # modelconfig.features = list(modelconfig.features)

        modelconfig.features = ['b2b_range', 'b2b_var', 'sse_2_clusters', 'sse_1_clusters', 'hrv_pnn20', 'hrv_pnn50', 'hrv_shanen', 'ecg_rate_mean', 'hrv_sd1']
        modelconfig.features = [
            'b2b_iqr',
            'b2b_std',
            'sse_1_clusters',
            'sse_diff',
            'hrv_sampen',
            'hrv_pnn50'
        ]
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
    if (filterUnreasonableValues):
        df = df[df['b2b_iqr'] > 0]
        goldData = goldData[goldData['b2b_iqr'] > 0]
    if (winsorize):
        df = dm.winsorizeDF(df, modelconfig.features)
    df_normalized, scaler = dm.computeAndApplyScaler(df, modelconfig.features)
    # scaler = dm.loadScaler()
    # df_normalized = dm.applyScaler(df, modelconfig.features, scaler)

    ## Split for testing and evaluation
    if (usesplits): # can only avoid using splits if you input gold data set
        p('Splitting into train and test sets...', verbose)
        print(len(goldData))
        if (goldData.empty):
            train, test = dm.applySplits(df_normalized)
        else:
            train, test = dm.applySplits(df_normalized, prespecifiedTestSet=goldData['fin_study_id'].unique())
        print(len(test))
    else:
        p('Not splitting at all :-)', verbose)
        train = df_normalized
        test = goldData
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
    withPCA = False
    if (withPCA):
        from sklearn import decomposition
        pca = decomposition.PCA(n_components=5)
        pca.fit(trainData.to_numpy())
        trainData = pca.transform(trainData.to_numpy())
        testData = pca.transform(testData.to_numpy())
    ## train the model, return the model, test the model
    modelname = model
    if (model == 'TRANSFORMER'):
        #load transformer model
        import torch
        from transformer.model import load_model, fit_data
        from featurize import featurize_physionet
        transformer, device = load_model()
        #get series signal
        xSignals, yLabels = loadPhysionet()
        # xSignals, yLabels = np.array(xSignals), np.array(yLabels)
        with open('./data/assets/physionet/indices.pkl', 'rb') as readfile:
            X_test = pickle.load(readfile)
        # X_test = featurize_physionet(index_pass=True)
        X_test = random.sample(X_test, int(len(X_test) / 4))
        X_test, Y_test = [torch.from_numpy(xSignals[i]).float() for i in X_test], [yLabels[i] for i in X_test]
        # [ y.to(device) for y in Y_test ]
        #feed it into model transformer once refit
        X_data = [ torch.from_numpy(x).float() for x in fit_data(X_test, transformer)]
        [ x.to(device) for x in X_data ]
        modelIn = torch.stack(X_data).to(device)
        modelPredictions = transformer(modelIn)
        modelProbabilities = [[x.item(), 1-x.item()] for x in modelPredictions]
        thresh = torch.tensor([.5])
        # thresh.to('cpu')
        modelPredictions = modelPredictions.to('cpu')
        preds = (modelPredictions>thresh).float()*1
        modelPredictions = [['NOT_AFIB', 'ATRIAL_FIBRILLATION'][int(predIndex.item())] for predIndex in preds]

    elif (model == 'RandomForestSK'):
        p('Training randomforest...', verbose)
        # model = RandomForestClassifier(max_depth=12, n_estimators=1000, class_weight={'ATRIAL_FIBRILLATION': .15, 'NOT_AFIB': .85}, random_state=66)
        # model = RandomForestClassifier(n_estimators=1000, random_state=66)
        model = RandomForestClassifier(max_depth=10, random_state=66)
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

        #load and transform data

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
from data.process_physionet import loadPhysionet

def trainPhysionet(model):
    LABEL_CODE = {'N': 0,  # normal
                'A': 1, # afib
                'O': 2, # other rhythm
                '~': 3} # noisy
    LABEL_CODE = {0: 'NOT_AFIB',  # normal
                1: 'ATRIAL_FIBRILLATION', # afib
                2: 'NOT_AFIB', # other rhythm
                3: 'NOISE'} # noisy
    featurizedDF = pd.read_csv('./data/assets/physionet/physionet_featurized.csv')
    #filter out the 10 seconds or less segments
    # featurizedDF = featurizedDF[~featurizedDF['hrv_shanen'].isna()]
    featurizedDF['label'] = featurizedDF['label'].apply(lambda x: LABEL_CODE[x])
    features = mu.getModelConfig().features_trunc
    # features = ['b2b_iqr', 'hrv_pnn20', 'hrv_pnn50', 'hrv_shanen', 'sse_2_clusters', 'sse_diff', 'ecg_rate_mean']
    df = featurizedDF
    numInfs = np.isinf(df.select_dtypes('float').stack()).groupby(level=1).sum().sum()
    if (numInfs > 0):
        p(f'\tFound {numInfs} entries with value infinity, replacing them with nan', True)
        df.replace([np.inf, -np.inf], np.nan, inplace=True)
    before = len(df); df = df.dropna(); after = len(df)
    p(f'\tDropped {before - after} rows with nan values present.', True)
    df = dm.winsorizeDF(df, features)
    featurizedNormalizedDF, scaler = dm.computeAndApplyScaler(df, features)
    featurizedNormalizedDF = featurizedNormalizedDF[featurizedNormalizedDF['label'] != 'NOISE']
    x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(
        featurizedNormalizedDF[features],
        featurizedNormalizedDF['label'],
        test_size=.10,
        random_state=66
    )
    withPCA = False
    if (withPCA):
        from sklearn import decomposition
        pca = decomposition.PCA(n_components=4)
        pca.fit(x_train.to_numpy())
        x_train = pca.transform(x_train.to_numpy())
        x_test = pca.transform(x_test.to_numpy())
    ## train the model, return the model, test the model
    if (model == 'RandomForestSK'):
        # m = RandomForestClassifier(max_depth=7, n_estimators=500, random_state=66)
        m = RandomForestClassifier( random_state=66)
        # fitModel = RandomForestClassifier(max_depth=5, n_estimators=1000, class_weight='balanced', random_state=66)
        m.fit(x_train, y_train)
        modelPredictions = m.predict(x_test)
        modelProbabilities = m.predict_proba(x_test)
        with open('./data/assets/physionet_rf.pkl', 'wb') as writefile:
            pickle.dump(m, writefile)
    elif (model == 'LabelModel'):
        fitModel = LabelModelCustom()
        fitModel.fit(x_train)
        modelPredictions = fitModel.predict(x_test)
        # modelProbabilities = [max(e) for e in fitModel.predict_proba(testData)]
        modelProbabilities = fitModel.predict_proba(x_test)
        res = list()
        for modelPred in modelPredictions:
            if modelPred == 'OTHER' or modelPred == 'SINUS':
                res.append('NOT_AFIB')
            else:
                res.append(modelPred)
        modelPredictions = res

    elif (model == 'LogisticRegression'):
        m = LogisticRegression(random_state = 66)
        m.fit(x_train, y_train)
        modelPredictions = m.predict(x_test)
        modelProbabilities = m.predict_proba(x_test)
    print(model)
    print(m.classes_)
    print(modelPredictions[0])
    print(modelProbabilities[0])
    cacheddata = {
        'testData': x_test,
        'testLabels': y_test,
        'testPredictions': modelPredictions,
        'testPredProbabilities': modelProbabilities,# if model.predict_proba else None,
        # 'testIdentifiers': goldData[goldData.index.isin(test.index)],
        'trainData': x_train,
        'trainLabels': y_train,
        'features': features,
        'model': m
    }
    return m, cacheddata

from featurize import beforeMinutesOfInterest
possibleFeatureSuffixes = [''] + [ f"_{m}" for m in beforeMinutesOfInterest ]
allFeats = list()
allFeatBases = mu.getModelConfig().features_nk
for featBase in allFeatBases:
    for featureSuffix in possibleFeatureSuffixes:
        allFeats.append(f"{featBase}{featureSuffix}")
from prediction.europace.loadData import timeIntervals, loadDataInSurvivalFormat
def trainEuropaceBABY(df: pd.DataFrame, additionDF: pd.DataFrame):
    global allFeats
    # Collect features
    # print(allFeats)
    # Remove nans and infs
    numInfs = np.isinf(df.select_dtypes('float').stack()).groupby(level=1).sum().sum()
    if (numInfs > 0):
        print(f'\tFound {numInfs} entries with value infinity, replacing them with nan')
        df.replace([np.inf, -np.inf], np.nan, inplace=True)

        additionDF.replace([np.inf, -np.inf], np.nan, inplace=True)

    before = len(df); df = df.dropna(); after = len(df)
    additionDF = additionDF.dropna()
    print(f'\tDropped {before - after} rows with nan values present.')
    # Filter physiologically unreasonable values
    for featureSuffix in possibleFeatureSuffixes:
        df = df[df[f'b2b_iqr{featureSuffix}'] > 0]
        df = df[df[f'ecg_rate_mean{featureSuffix}'] > 10]

    
    # winsorize    
    df = dm.winsorizeDF(df, allFeats)
    additionDF = dm.winsorizeDF(additionDF, allFeats)

    #Add diff columns
    def addDifferences(df):
        global allFeats
        allFeats = set(allFeats)
        for feat in allFeatBases:
            for i, m in enumerate(beforeMinutesOfInterest):
                otherFeat = featBase + f"_{m}"
                allFeats.add("{}_diff_from_{}".format(feat, m))
                df["{}_diff_from_{}".format(feat, m)] = df[feat] - df[otherFeat]
        allFeats = list(allFeats)
        print('--')
        print('\n'.join(sorted(allFeats)))
        print('--')
        return df
    df = addDifferences(df)
    additionDF = addDifferences(additionDF)
    # print('\n'.join(df.columns))

    # Normalize and center about mean
    df, scaler = dm.computeAndApplyScaler(df, allFeats)
    additionDF = dm.applyScaler(additionDF, allFeats, scaler)



    # Split into test and train
    from sklearn.model_selection import train_test_split
    survivorIndices = df.index
    additionIndices = additionDF.index
    x_all, y_all = df[allFeats], df.index
    df = df.reset_index()
    trainset, testset = dm.applySplits(df, test_size=.3, group_identifier='patient_id')
    #train_test_split(df[allFeats], np.array(df.index), test_size=.5)
    x_train, x_test, Y_train, Y_test = trainset[allFeats], testset[allFeats], trainset.index, testset.index
    # print(df.iloc[Y_train, :]['patient_id'].unique())
    results = dict()

    for start, stop in timeIntervals:
        suffix = f"{start}_to_{stop}"
        label = f"afib_in_{suffix}"
        allOthers = [f"afib_in_{start}_to_{stop}" for start, stop in set(timeIntervals) - set([(start, stop)])]
        y_train = list()
        count = 0
        allElse = 0
        for idx in Y_train:
            #filter out indices that are negative and have a positive instance of another class, to avoid muddying the waters
            # print(sum(df.iloc[idx, :][allOthers]), df.iloc[idx, :][label])
            if (df.iloc[idx, :][label] == 0) and (sum(df.iloc[idx, :][allOthers]) > 0):
                count += 1
                continue
            allElse  += 1
            y_train.append(idx)
        y_test = list()
        for idx in Y_test:
            #filter out indices that are negative and have a positive instance of another class, to avoid muddying the waters
            if df.iloc[idx][label] == 0 and sum(df.iloc[idx, :][allOthers]) > 0:
                count += 1
                continue
            allElse += 1
            y_test.append(idx)

        # y_all_sub_indices = list(), list()
        # for idx in y_all:
        #     #filter out indices that are negative and have a positive instance of another class, to avoid muddying the waters
        #     if df.iloc[idx][label] == 0 and sum(df.iloc[idx, :][allOthers]) > 0:
        #         count += 1
        #         continue
        #     allElse += 1
        #     y_all_sub.append(idx)

        getDataGivenIndices = lambda indices: df.iloc[indices, :][allFeats]
        getLabelGivenIndices = lambda indices: df.iloc[indices, :][label]
        # x_all_sub, y_all_sub = getDataGivenIndices(y_all_sub_indices), getLabelGivenIndices(y_all_sub_indices)
        # print(f"Discarded {count}")
        # print(f"Kept {allElse}")
        
        
                

        x_train, x_test = getDataGivenIndices(y_train), getDataGivenIndices(y_test)
        y_train, y_test = getLabelGivenIndices(y_train), getLabelGivenIndices(y_test)

        rfForLabel = RandomForestClassifier(max_depth=7, random_state=66)

        withPCA = False
        if (withPCA):
            from sklearn import decomposition
            pca = decomposition.PCA(n_components=6)
            pca.fit(x_train.to_numpy())
            x_train = pca.transform(x_train.to_numpy())
            x_test = pca.transform(x_test.to_numpy())
        rfForLabel.fit(x_train, y_train)
        # from sklearn.model_selection import GridSearchCV 
        # rfGrid = GridSearchCV(estimator=RandomForestClassifier(), scoring='roc_auc', n_jobs=7, param_grid=
        # {
        #     'criterion': ['gini', 'entropy'],
        #     'max_depth': [5, 7, 9, 11, 13],
        #     'max_features': ['sqrt', 'log2', None, 3, 5, 7, 9, 11],
        #     'bootstrap': [False, True],
        #     'random_state': [66],
        #     'max_samples': [None, .3, .5, .7]
        #     })
        # rfGrid.fit(x_train.append(x_test),y_train.append(y_test))
        # # print(rfGrid.cv_results_)
        # res = rfGrid.cv_results_
        # with open('bestScores.pkl', 'wb+') as writefile:
        #     pickle.dump(res, writefile)
        # print(rfGrid.best_score_)
        # rfForLabel = rfGrid.best_estimator_
        test_pred = rfForLabel.predict(x_test)
        test_pred_probabilities = rfForLabel.predict_proba(x_test)
        results[label] = {
            'testData': x_test,
            'testLabels': y_test,
            'testPredictions': test_pred,
            'testPredProbabilities': test_pred_probabilities,# if model.predict_proba else None,
            'allDataPredProbabilities': rfForLabel.predict_proba(x_all),
            'allDataIdentifiers': survivorIndices,
            'additionPredProbabilities': rfForLabel.predict_proba(additionDF[allFeats]),
            'additionIdentifiers': additionIndices,
            # 'testIdentifiers': goldData[goldData.index.isin(test.index)],
            'trainData': x_train,
            'trainLabels': y_train,
            'features': allFeats,
            'm': rfForLabel   
        }
    return results

def trainEuropaceSurvival(model_type: str = 'cph', acute=False):
    dataConfig = du.getDataConfig()
    mem = Memory(dataConfig.cacheDirectory)
    loadData = mem.cache(loadDataInSurvivalFormat)
    a, dfOG = loadData(acute=acute, inDFForm=True)
    features, ttes, events = a
    # features, ttes, events = loadData()
    from sklearn.model_selection import train_test_split
    from model.auton_survival.auton_survival import preprocessing
    from model.auton_survival.auton_survival.models import cph
    from model.auton_survival.auton_survival.metrics import survival_regression_metric
    from model.auton_survival.auton_survival.estimators import SurvivalModel
    # print(features)
    # newCols = ['ecg_rate_mean_30', 'hrv_sd2_60', 'hrv_pnn50_30', 'hrv_pnn50_15', 'b2b_var_120', 'sse_1_clusters_30', 'hrv_pnn50_60', 'b2b_var_30', 'hrv_sd2_15', 'hrv_sdnn']
    # newCols =['hrv_sdnn_30', 'hrv_sdnn_15', 'ecg_rate_mean', 'b2b_range_30', 'sse_1_clusters_15', 'hrv_sd2', 'hrv_pnn50_30', 'b2b_std_15', 'ecg_rate_mean_15', 'hrv_sd2_30', 'hrv_sdnn', 'b2b_std_30'] 
    # newCols =['b2b_var', 'b2b_var_30', 'b2b_var_60', 'b2b_iqr', 'ecg_rate_mean_30', 'hrv_pnn50_60', 'sse_2_clusters_120', 'sse_diff', 'sse_diff_30', 'sse_diff_60', 'hrv_sdnn', 'hrv_sd2_15'] 
    # newCols = ['b2b_std', 'hrv_sd2_60', 'ecg_rate_mean_60', 'b2b_var_30', 'hrv_sd1sd2_60', 'hrv_sdnn_15', 'sse_1_clusters_60', 'ecg_rate_mean_15', 'b2b_var', 'hrv_sd2', 'hrv_sd2_15', 'b2b_std_30']
    # newCols = ['hrv_sdnn_10', 'b2b_std', 'hrv_sd2_5', 'ecg_rate_mean_10', 'hrv_sd2_30', 'hrv_sd2', 'hrv_sd2_15', 'b2b_var', 'hrv_sdnn_5', 'ecg_rate_mean_5', 'b2b_std_5', 'hrv_sdnn_30']
    newCols = ['ecg_rate_mean', 'b2b_std_30', 'hrv_pnn50_5', 'hrv_sdsd_15', 'hrv_sd1_15', 'hrv_rmssd_15', 'hrv_pnn50', 'ecg_rate_mean_5', 'ecg_rate_mean_30', 'hrv_pnn20_5', 'hrv_sdnn_15', 'b2b_std']
    features = features[newCols]
    features: pd.DataFrame = preprocessing.Preprocessor().fit_transform(features, cat_feats=[], num_feats=list(features.columns))
    # features.replace([np.inf, -np.inf], np.nan, inplace=True)
    # features.dropna(inplace=True)
    if (model_type == 'dcph'):
        model = SurvivalModel(model_type, layers=[150], random_seed=62)
    else:
        model = SurvivalModel(model_type, random_seed=62)

    # model = cph.DeepCoxPH(layers=[100])
    # features = features.to_numpy(dtype=np.float64)
    ttes = np.array([tte.total_seconds()/60 for tte in ttes])
    events = np.array(events)
    x_train, x_test, tte_train, tte_test, e_train, e_test, indices_train, indices_test = train_test_split(features, ttes, events, dfOG.index,test_size=.2, random_state=25)
    # print('yo')
    # print((len(events) - sum(events)) / 60)
    # print(sum(events) / 60)
    # print(np.where(events == 0) / 60)
    outcomes_train = pd.DataFrame({
        'time': tte_train,
        'event': e_train
    })
    outcomes = pd.DataFrame({
        'time': tte_test, 'event': e_test
    })
    together = x_train.join(outcomes_train)
    together.replace([np.inf, -np.inf], np.nan, inplace=True)
    together.dropna(inplace=True)
    model.fit(together.drop(columns=['time', 'event']), together[['time', 'event']], val_data=(x_test, outcomes))
    times = [15, .5*60, 1.0*60, 2.0*60]
    times = [10, 15, 20, 25, 30]
    # times = [.5*60*60, 1*60*60, .75*60*60]
    preds = model.predict_risk(x_test, times)
    # print(preds)
    # Compute Brier Score, Integrated Brier Score
    # Area Under ROC Curve and Time Dependent Concordance Index
    metrics = ['brs', 'ibs', 'auc', 'ctd']
    # score = survival_regression_metric(metric='auc', outcomes_train=outcomes_train, 
    #                                 outcomes=outcomes, predictions=preds,
    #                                 times=times)
    from model.auton_survival.examples.estimators_demo_utils import plot_performance_metrics
    results = dict()
    outcomes_train.dropna(inplace=True);outcomes.dropna(inplace=True)
    results['AUC'] = survival_regression_metric(metric='auc', outcomes_train=outcomes_train, 
                                    outcomes=outcomes, predictions=preds,
                                    times=times)
    results['Brier Score'] = survival_regression_metric(metric='brs', outcomes_train=outcomes_train, 
                                    outcomes=outcomes, predictions=preds,
                                    times=times)
    results['Concordance Index'] = survival_regression_metric(metric='ctd', outcomes_train=outcomes_train, 
                                    outcomes=outcomes, predictions=preds,
                                    times=times)
    import matplotlib.pyplot as plt
    plt.figure(figsize=(12, 4))
    plot_performance_metrics(results, times)
    plt.savefig(f'./results/assets/survival_results_{model_type.upper()}{"_acute_culled" if acute else ""}.png')
    plt.clf()
    print(model_type.upper())
    preds = model.predict_risk(features, [20])
    dfOG['model_confidence'] = preds[:,0]
    dfOG['testset'] = dfOG.index.isin(indices_test)
    dfOG['event'] = events
    dfOG.to_csv(f'withConfidence_survival20_{model_type}.csv', index=False)
    if (model_type == 'cph'):
        print('Plotting cph coefficients...')
        dst = './results/assets/cph_confidences.png'
        plt.figure(figsize=(12, 16))
        model._model.plot(hazard_ratios=True)
        with open('./results/assets/cph_confidences.txt', 'w+') as writefile:
            ps = model._model.params_.sort_values()
            first = ps.iloc[:6].index
            second = ps.iloc[-6:].index
            together = list(list(first)+list(second))
            print(f"5 lowest and 5 highest cph coefficients: {together}")
            writefile.write(ps.to_string())
            # pickle.dump(ps, writefile)
        plt.savefig(dst)
        print(f'Done! Saved to {dst}')

    # print(score)


if __name__ == "__main__":

    allLabels = [f"afib_in_{start}_to_{stop}" for start, stop in timeIntervals]
    dtypeDict = {
        'patient_id': str
    }
    for label in allLabels:
        dtypeDict[label] = bool
    # trainEuropaceBABY(pd.read_csv('featurized_afib_predictor_data.csv', parse_dates=['time', 'basetime'], dtype=dtypeDict))
    # trainEuropaceBABY(pd.read_csv('featurized_afib_preceding_times.csv', parse_dates=['time', 'basetime'], dtype=dtypeDict))
    # trainEuropaceSurvival('dsm')
    trainEuropaceSurvival('cph', acute=True)
    trainEuropaceSurvival('dcph', acute=True)
    trainEuropaceSurvival('rsf', acute=True)
    # trainlm(modifyLabels=True)
    # trainPhysionet('RandomForestSK')
    # train(model='LabelModel', usesplits=False)
    # train(usesplits=False)
    # lrModel, cacheddata = train(usesplits=False, model="LogisticRegression", verbose=False)
    # rfModel = train(usesplits=False, model="RandomForestSK", verbose=False)
