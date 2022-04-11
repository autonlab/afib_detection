
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.stats.mstats import winsorize
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from model.df4tsc.resnet import Classifier_RESNET


import data.manipulators as dm
from model.labelmodel import LabelModelCustom
import model.utilities as mu

def trainlm():
    modelconfig = mu.getModelConfig()
    print(f'Loading features: {modelconfig.features}')
    # print(f'Loading data set: {modelconfig.trainDataFile}')
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
    print(f'Writing labelmodel votes to: {modelconfig.trainDataFile} ...')
    df.to_csv(
        Path(__file__).parent / 'data' / 'assets' / modelconfig.trainDataFile,
    )
    print('Done.')
    return

def p(string, v):
    """Print if verbose on
    """
    if (v):
        print(string)

def train(model='RandomForestSK', load=False, usesplits=True, verbose=False, filterGold=False, overwriteTrainset=None, overwriteTestset=None):
    """Train and return specified model.
     PRECONDITIONS:
    - CSV of featurized data in `data/assets`, csv title specified in `model/config.yml` in `trainDataFile` field
    - Features enumerated under `features` field of `model/config.yml`

    Args:
        load (bool, optional): _description_. Defaults to False.
    """
    goldData = pd.DataFrame()

    ## Load necessary configuration from model
    modelconfig = mu.getModelConfig()
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
        goldData = dm.remapLabels(goldData, 'label', modelconfig.labelCorrectionMap)
    df = pd.read_csv(
        Path(__file__).parent / 'data' / 'assets' / trainDataFile,
        parse_dates=['start', 'stop'])
    df.columns = df.columns.str.lower()
    df = dm.remapLabels(df, 'label', modelconfig.labelCorrectionMap)
    ## Filter then normalize the data
    p('Filtering and normalizing...', verbose)
    #count occurrences of infinity in dataframe and mark them for dropping if existent
    numInfs = np.isinf(df.select_dtypes('float').stack()).groupby(level=1).sum().sum()
    if (numInfs > 0):
        p(f'\tFound {numInfs} entries with value infinity, replacing them with nan.', verbose)
        df.replace([np.inf, -np.inf], np.nan, inplace=True)

    before = len(df); df = df.dropna(); after = len(df)
    p(f'\tDropped {before - after} rows with nan values present.', verbose)
    if (model == 'LabelModel'):
        p('Training labelmodel...', verbose)
        fitModel = LabelModelCustom()
        fitModel.fit(df[modelconfig.features])
        # predictedLabels = fitModel.predict(testData)
        # predictedProbas = [max(e) for e in fitModel.predict_proba(df)]
        # p(classification_report(y_true=goldData['label'], y_pred=fitModel.predict(goldData[modelconfig.features])), verbose)
        return

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
    trainData, trainLabels = train[modelconfig.features], train['label']
    testData, testLabels = test[modelconfig.features], test['label']
    # t2Data, t2Labels = testOnLMLabels[modelconfig.features], testOnLMLabels['label']

    ## train the model, return the model, test the model
    modelname = model
    if (model == 'RandomForestSK'):
        p('Training randomforest...', verbose)
        model = RandomForestClassifier(max_depth=5, n_estimators=1000, class_weight={'ATRIAL_FIBRILLATION': .1, 'SINUS': .9}, random_state=66)
        # fitModel = RandomForestClassifier(max_depth=5, n_estimators=1000, class_weight='balanced', random_state=66)
        model.fit(trainData, trainLabels)
    elif (model == 'LogisticRegression'):
        model = LogisticRegression(random_state = 66)
        model.fit(trainData, trainLabels)
        w = model.coef_[0]
    elif (model == 'ResNet'):
        model = Classifier_RESNET(
            Path(__file__).parent / 'model' / 'assets', #outputDir
            len(modelconfig.features), #inputShape
            2, #numClasses
        )
        model.fit(trainData, trainLabels, None, None, None)

    # print(classification_report(y_true=testLabels, y_pred=model.predict(testData)))
    modelPredictions = model.predict(testData)
    if (model.predict_proba):
        modelProbabilities = model.predict_proba(testData)
    
    p('Done', verbose)
    print(goldData[goldData['fin_study_id'] == 1972442])
    cacheddata = {
        'testData': testData,
        'testLabels': testLabels,
        'testPredictions': modelPredictions,
        'testPredProbabilities': modelProbabilities if model.predict_proba else None,
        'testIdentifiers': goldData[goldData.index.isin(test.index)],
        'trainData': trainData,
        'trainLabels': trainLabels,
        'features': modelconfig.features
    }
    if (modelname == 'LogisticRegression'):
        cacheddata['w'] = w
    return model, cacheddata

if __name__ == "__main__":
    trainlm()
    # train(model='LabelModel', usesplits=False)
    # train(usesplits=False)
    # lrModel, cacheddata = train(usesplits=False, model="LogisticRegression", verbose=False)
    # rfModel = train(usesplits=False, model="RandomForestSK", verbose=False)