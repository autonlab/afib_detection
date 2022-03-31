
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pathlib import Path

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import PrecisionRecallDisplay, classification_report


import data.manipulators as dm
from model.labelmodel import LabelModelCustom
import model.utilities as mu

def trainlm():
    modelconfig = mu.getModelConfig()
    print(f'Loading features: {modelconfig.features}')
    # print(f'Loading data set: {modelconfig.trainDataFile}')
    df = pd.read_csv(
        Path(__file__).parent / 'data' / 'assets' / 'old_featurized_lm_segments.csv',
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


def train(model='RandomForestSK', load=False, usesplits=True):
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
    print(f'Loading features: {modelconfig.features}')
    print(f'Loading data set: {modelconfig.trainDataFile}')
    if (modelconfig.goldDataFile.endswith('csv')):
        print(f'Loading gold set: {modelconfig.goldDataFile}')
        goldData = pd.read_csv(
            Path(__file__).parent / 'data' / 'assets' / modelconfig.goldDataFile,
            parse_dates=['start', 'stop']
        )
        goldData.columns=goldData.columns.str.lower()
        goldData = dm.remapLabels(goldData, 'label', modelconfig.labelCorrectionMap)
    df = pd.read_csv(
        Path(__file__).parent / 'data' / 'assets' / modelconfig.trainDataFile,
        parse_dates=['start', 'stop'])
    df.columns = df.columns.str.lower()
    df = dm.remapLabels(df, 'label', modelconfig.labelCorrectionMap)
    ## Filter then normalize the data
    print('Filtering and normalizing...')
    #count occurrences of infinity in dataframe and mark them for dropping if existent
    numInfs = np.isinf(df.select_dtypes('float').stack()).groupby(level=1).sum().sum()
    if (numInfs > 0):
        print(f'\tFound {numInfs} entries with value infinity, replacing them with nan.')
        df.replace([np.inf, -np.inf], np.nan, inplace=True)

    before = len(df); df = df.dropna(); after = len(df)
    print(f'\tDropped {before - after} rows with nan values present.')
    
    df_normalized, scaler = dm.computeAndApplyScaler(df, modelconfig.features)

    ## Split for testing and evaluation
    if (usesplits and (not goldData.empty)): # can only avoid using splits if you input gold data set
        print('Splitting into train and test sets...')
        if (goldData.empty):
            train, test = dm.applySplits(df_normalized)
        else:
            train, testOnLMLabels = dm.applySplits(df_normalized, prespecifiedTestSet=goldData['fin_study_id'].unique())
    else:
        print('Not splitting at all :-)')
        train = df_normalized
        test = dm.filterAndNormalize(goldData, modelconfig.features, preexistingScaler=scaler)

    ## Oversample, extract only features (removing identifiers on data)
    shouldOversample = input('Oversample? (y/N): ')
    if (shouldOversample):
        print('Oversampling...')
        countsBeforehand = [(subset['label'].iloc[0], len(subset)) for idx, subset in train.groupby('label')]
        train = dm.oversample(train, 'label', 'confidence')
        countsAfterward = [(subset['label'].iloc[0], len(subset)) for idx, subset in train.groupby('label')]
        for i in range(len(countsAfterward)):
            print(f'\t Class {countsAfterward[i][0]} grew from {countsBeforehand[i][1]:5} to {countsAfterward[i][1]:5} elements')
    trainData, trainLabels = train[modelconfig.features], train['label']
    testData, testLabels = test[modelconfig.features], test['label']
    # t2Data, t2Labels = testOnLMLabels[modelconfig.features], testOnLMLabels['label']

    ## train the model, return the model, test the model
    if (model == 'RandomForestSK'):
        print('Training randomforest...')
        fitModel = RandomForestClassifier(max_depth=5, n_estimators=1000, class_weight={'ATRIAL_FIBRILLATION': .1, 'SINUS': .9}, random_state=66)
        # fitModel = RandomForestClassifier(max_depth=5, n_estimators=1000, class_weight='balanced', random_state=66)
        fitModel.fit(trainData, trainLabels)
    
    print(classification_report(y_true=testLabels, y_pred=fitModel.predict(testData)))
    # probas = fitModel.predict_proba(testData)
    # probas = [max(e) for e in probas]
    # disp = PrecisionRecallDisplay.from_predictions(y_true=testLabels, y_pred=probas, pos_label="ATRIAL_FIBRILLATION", name="random forest")
    # plt.show()
    disp = PrecisionRecallDisplay.from_estimator(fitModel, testData, testLabels, pos_label="ATRIAL_FIBRILLATION", name="random forest")
    plt.show()
    # print(classification_report(y_true=t2Labels, y_pred=fitModel.predict(t2Data)))

    # print(train['fin_study_id'].unique())
    # print(test['fin_study_id'].unique())

    print('Done')

if __name__=="__main__":
    # trainlm()
    train(usesplits=False)