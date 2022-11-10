from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import PrecisionRecallDisplay, classification_report, precision_recall_curve
from analysis import plotSegment, showConfidentlyIncorrects, permutationFeatureImportance
from data.roc import roc
from itertools import cycle
import model.utilities as mu
from train import train, trainPhysionet, trainEuropaceBABY


def applyModelToPatient(src='featurized_all_5min.csv')
    pass
def showResults(trainingResults: dict, title=None, posLabel=None, feature_importance=True):
    resultantModel = trainingResults['m']
    newDecisionThreshold = .5
    # print(resultantModel.classes_)

    # features = [f.lower() for f in mu.getModelConfig().features_nk]
    from train import allFeats
    features = allFeats

    def bestThreshold(labels, probabilities):
        precision, recall, thresholds = precision_recall_curve(labels, probabilities, pos_label=posLabel)
        f1Scores = 2*precision*recall/(precision+recall)
        bestThresh = thresholds[np.argmax(f1Scores)]
        bestF1Score = max(f1Scores)
        return bestThresh, bestF1Score
    thres, f1 = bestThreshold(trainingResults['testLabels'], trainingResults['testPredProbabilities'][:, 1])
    # print(thres, f1)
    bestThreshPreds =   [1 if i else 0 for i in (trainingResults['testPredProbabilities'][:, 1] >= thres).astype(bool)]
 
    cr = classification_report(
        y_true=trainingResults['testLabels'],
        y_pred=trainingResults['testPredictions']
        )
    print(f'{title if title else "nameless"} classification report:\n{cr}')
    cr = classification_report(
        y_true=trainingResults['testLabels'],
        y_pred=bestThreshPreds
        )
    print(f'{title if title else "nameless"} optimized threshold classification report:\n{cr}')
    if (feature_importance):
        featImport, featImportSorted = permutationFeatureImportance(resultantModel, trainingResults['testData'], trainingResults['testLabels'], feature_subset=features, n_repeats=10)
        print('\n\n----- Feature importances -----\n\n')
        newlinetab = "\n\t"
        fiSorted = [f'{name}: {importance:.2}' for name, importance in featImportSorted]
        print(f'Top features:{newlinetab}{newlinetab.join(fiSorted)}')
    yTest = trainingResults['testLabels']
    yScore = trainingResults['testPredProbabilities'][:, 1]
    roc(yTest, yScore, f'ROC {title if title else "plot"}', dstTitle=f'roc_{title if title else "nameless"}.png', posLabel=posLabel)

if __name__ == "__main__":
    showClassificationReport = True
    showFeatureImportance = False
    newDecisionThreshold = None
    import warnings
    warnings.filterwarnings("ignore")

    # features = [f.lower() for f in mu.getModelConfig().features_nk]
    features = [f.lower() for f in mu.getModelConfig().features_nk]
    trainData = pd.read_csv('featurized_afib_predictor_data_joined.csv', parse_dates=['time', 'basetime'], dtype={'patient_id': str})
    additionData = pd.read_csv('featurized_afib_preceding_times.csv', parse_dates=['time', 'basetime'], dtype={'patient_id': str})
    additionData['preceding_afib'] = [True for i in range(len(additionData))]
    res = trainEuropaceBABY(trainData, additionData)
    for learningTask, learningTaskData in res.items():
        showResults(learningTaskData, title=learningTask, feature_importance=False)
    #     trainData[f"{learningTask}_confidence"] = [None for i in range(len(trainData))]
    #     trainData[f"{learningTask}_confidence"][learningTaskData['allDataIdentifiers']] = learningTaskData['allDataPredProbabilities'][:,1]
    #     additionData[f"{learningTask}_confidence"] = [None for i in range(len(additionData))]
    #     additionData[f"{learningTask}_confidence"][learningTaskData['additionIdentifiers']] = learningTaskData['additionPredProbabilities'][:,1]
    # additionData.to_csv('preceding_afib_episodes_3moremaximal_withmodelconf.csv')
    # features = ['b2b_iqr', 'hrv_pnn20', 'hrv_pnn50', 'hrv_shanen', 'sse_2_clusters', 'sse_diff', 'ecg_rate_mean']
    # features = ['b2b_range', 'b2b_var', 'sse_2_clusters', 'sse_1_clusters', 'hrv_pnn20', 'hrv_pnn50', 'hrv_shanen', 'ecg_rate_mean', 'hrv_sd1']
    # features = set(features) - set(['hfd', 'hrv_hf', 'hrv_lfhf', 'sd1', 'sample_entropy', 'max_sil_score', 'hrv_lf', 'b2b_var', 'rmssd', 'sd1/sd2', 'sd2', 'hopkins_statistic', 'b2b_std'])
    # features = list(features)
    # resnet, resnet_data = train(
    #     filterGold=False,
    #     usesplits=True,
    #     model="ResNet",
    #     verbose=True,
    #     )
    # lm, lm_data = trainPhysionet(
    #     model="LabelModel",
    #     )
    # rf_sk, rf_sk_data = trainPhysionet(
    #     model="RandomForestSK",
    #     )
    # lr, lr_data = trainPhysionet(
    #     model="LogisticRegression",
    #     )
    # tr, tr_data = train(
    #     filterGold=False,
    #     usesplits=True,
    #     model="TRANSFORMER",
    #     verbose=True,
    #     # filterUnreasonableValues=True
    #     )
    # lm, lm_data = train(
    #     filterGold=False,
    #     usesplits=True,
    #     model="LabelModel",
    #     verbose=True,
    #     # filterUnreasonableValues=True
    #     )
    # rf_sk, rf_sk_data = train(
    #     filterGold=False,
    #     usesplits=False,
    #     model="RandomForestSK",
    #     verbose=True,
    #     filterUnreasonableValues=True,
    #     reduceDimension=False,
    #     winsorize=True
    #     )
    # lr, lr_data = train(
    #     filterGold=False,
    #     usesplits=False,
    #     model="LogisticRegression",
    #     verbose=True,
    #     filterUnreasonableValues=True,
    #     reduceDimension=False,
    #     winsorize=True
    #     )




    1 / 0





    rf_sk_data = res['afib_in_15_to_30']
    rf_sk = rf_sk_data['m']
    lr_data = res['afib_in_30_to_60']
    lr = lr_data['m']
    # rf_sk_data = res['afib_in_15_to_30']
    # lr_data = res['afib_in_30_to_60']
    # rf_sk, rf_sk_data = trainPhysionet('RandomForestSK')
    # lr, lr_data = trainPhysionet('LogisticRegression')
    df = pd.read_csv('./data/assets/testset_featurized_w_phillips.csv')
    # Filter
    import numpy as np
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df = df.dropna()
    phillipsDF = df
    phillips_true, phillips_pred = phillipsDF['label'].apply(lambda x: 'ATRIAL_FIBRILLATION' if x=='ATRIAL_FIBRILLATION' else 'SINUS'), phillipsDF['philafibmarker'].apply(lambda x: 'ATRIAL_FIBRILLATION' if x else 'SINUS')
    if (showClassificationReport):
        # resnet_cr = classification_report(
        #     y_true=resnet_data['testLabels'], y_pred=resnet_data['testPredictions']
        #     )
        if (newDecisionThreshold):
            # print(lm_data['testPredictions'])
            # print((lm_data['testPredProbabilities'] >= newDecisionThreshold).astype(bool))
            # print(np.where((lm_data['testPredProbabilities'] >= newDecisionThreshold).astype(bool)))
            # print(np.where((lm_data['testPredProbabilities'] >= newDecisionThreshold).astype(bool)[0]))
            # lmPreds =   [['NOT_AFIB', 'NOT_AFIB', 'ATRIAL_FIBRILLATION'][np.where(i[0][0])] for i in (lm_data['testPredProbabilities'] >= newDecisionThreshold).astype(bool)]
            print(rf_sk.classes_)
            rfSkIdx = list(rf_sk.classes_).index('ATRIAL_FIBRILLATION')
            lrIdx = list(lr.classes_).index('ATRIAL_FIBRILLATION')
            print(rfSkIdx, lrIdx)
            lrPreds =   [['NOT_AFIB', 'ATRIAL_FIBRILLATION'][1 if i else 0] for i in (lr_data['testPredProbabilities'][:, lrIdx] >= newDecisionThreshold).astype(bool)]
            rfSkPreds = [['NOT_AFIB', 'ATRIAL_FIBRILLATION'][1 if i else 0] for i in (rf_sk_data['testPredProbabilities'][:, rfSkIdx] >= newDecisionThreshold).astype(bool)]
        def bestThreshold(labels, probabilities):
            precision, recall, thresholds = precision_recall_curve(labels, probabilities, pos_label='ATRIAL_FIBRILLATION')
            f1Scores = 2*precision*recall/(precision+recall)
            bestThresh = thresholds[np.argmax(f1Scores)]
            bestF1Score = max(f1Scores)
            return bestThresh, bestF1Score
        # print('30 to 60:')
        # # print('LR:')
        # thres, f1 = bestThreshold(lr_data['testLabels'], lr_data['testPredProbabilities'][:, lrIdx])
        # print(f'thresh: {thres}', '\n', f'f1: {f1}')
        # # print('RF:')
        # print('15 to 30:')
        # thres, f1 = bestThreshold(rf_sk_data['testLabels'], rf_sk_data['testPredProbabilities'][:, rfSkIdx])
        # print(f'thresh: {thres}', '\n', f'f1: {f1}')
        # df_test = pd.read_csv('./data/assets/final_annotations_featurized_nk.csv', parse_dates=['start', 'stop'])
        # df_test = df_test[df_test['b2b_iqr'] > 0]
        # df_test['afib_confidence'] = rf_sk_data['testPredProbabilities'][:,0]
        # df_test.to_csv('./data/assets/final_annotations_featurized_nka.csv', index=False)
 
        # lm_cr = classification_report(
        #     y_true=lm_data['testLabels'],
        #     y_pred=lmPreds#lm_data['testPredictions']
        #     )
        lr_cr = classification_report(
            y_true=lr_data['testLabels'],
            y_pred=lr_data['testPredictions']
            # y_pred=lrPreds#lr_data['testPredictions']
            )
        # tr_cr = classification_report(
        #     y_true=tr_data['testLabels'],
        #     y_pred=tr_data['testPredictions']
        #     )
        randForestSK_cr = classification_report(
            y_true=rf_sk_data['testLabels'],
            y_pred=rf_sk_data['testPredictions']
            # y_pred=rfSkPreds#rf_sk_data['testPredictions']
            )
        phil_cr = classification_report(
            y_true=phillips_true,
            y_pred=phillips_pred
            )
        # print(f'Transformer classification report:\n{tr_cr}')
        print(f'LogisticRegressor classification report:\n{lr_cr}')
        print(f'RandomForest (sklearn) classification report:\n{randForestSK_cr}')
        # print(f'ResNet classification report:\n{resnet_cr}')
        print(f'RandomForest (autonlab) classification report:\n{"... in progress ..."}')
        # print(f'Labelmodel classification report:\n{lm_cr}')
        print(f'Phillips alerts classification report:\n{phil_cr}')
    if (showFeatureImportance):
        rf_sk_featureImportance, rfSK_fiSorted = permutationFeatureImportance(rf_sk, rf_sk_data['testData'], rf_sk_data['testLabels'], feature_subset=features, n_repeats=10)
        lr_featureImportance, lr_fiSorted = permutationFeatureImportance(lr, lr_data['testData'], lr_data['testLabels'], feature_subset=features, n_repeats=10)
        print('\n\n----- Feature importances -----\n\n')
        newlinetab = "\n\t"
        lr_fiSorted = [f'{name}: {importance:.2}' for name, importance in lr_fiSorted]
        rfSK_fiSorted = [f'{name}: {importance:.2}' for name, importance in rfSK_fiSorted]
        print(f'LogisticRegressor top features:{newlinetab}{newlinetab.join(lr_fiSorted)}')
        print(f'RandomForest (sklearn) top features:{newlinetab}{newlinetab.join(rfSK_fiSorted)}')
        print(f'ResNet top 5 features:\n\t{"... in progress ..."}')
        print(f'RandomForest (autonlab) top 5 features:{newlinetab}{"... in progress ..."}')
    yTests = list()
    yScores = list()
    # titles = cycle(['Transformer', 'LogisticRegressor', 'RandomForest (sklearn)', 'ResNet'])
    titles = cycle(['LogisticRegressor', 'RandomForest (sklearn)', 'LabelModel', 'ResNet'])
    titles = cycle(['30 to 60', '15 to 30', 'blah', 'blah'])
    # titles = cycle(['Transformer', 'LogisticRegressor', 'RandomForest (sklearn)', 'LabelModel', 'ResNet'])
    # afibIndex = list(lr.classes_).index('ATRIAL_FIBRILLATION')
    for cacheddata in [lr_data, rf_sk_data]:#, lm_data]:#resnet_data
    # for cacheddata in [tr_data, lr_data, rf_sk_data]:#resnet_data
        yTests.append(cacheddata['testLabels'])
        print(cacheddata['testPredProbabilities'])
        singleProbs = [prob[1] for prob in cacheddata['testPredProbabilities']]
        yScores.append((next(titles),singleProbs))
    roc(yTests, yScores, 'ROC Comparison', dstTitle='roc_final.png')
    '''
    for modelName, model in models:
        # fpr, tpr = dict(), dict()
        # roc_auc = dict()
        # fpr[0], tpr[0], _ = roc_curve(cacehddata['testLabels'], cacheddata['testPredProbabilities'])
        # model, cacheddata = train(filterGold=True, usesplits=False, model=modelName, verbose=True)
        disp = PrecisionRecallDisplay.from_estimator(model, cacheddata['testData'], cacheddata['testLabels'], pos_label="ATRIAL_FIBRILLATION", name=modelName)
        plt.savefig(
            Path(__file__).parent / 'results' / 'assets' / f'{modelName}_prCurve.png'
        )
        plt.clf()
        if (modelName == 'LogisticRegression'):
            # thanks to [this](https://sefiks.com/2021/01/06/feature-importance-in-logistic-regression/)
            w = cacheddata['w']
            fi = pd.DataFrame(cacheddata['features'], columns=['feature'])
            fi['importance'] = pow(math.e, w)
            fi = fi.sort_values(by = ['importance'])
            fig, ax = plt.subplots()
            ax = fi.plot.barh(x='feature', y='importance')
            plt.suptitle('Feature importances (logistic regression model)')
            plt.savefig(
                Path(__file__).parent / 'results' / 'assets' / f'feature_importances.png'
            )
            plt.clf()

    #select top 4 most important features for plotting
    fi = fi.iloc[-4:, :]
    for feature in fi['feature']:
        fig, ax = plt.subplots()
        ax.hist(cacheddata['trainData'][feature], label="Train data", histtype="step", density=True)
        ax.hist( cacheddata['testData'][feature], label="Test data", histtype="step", density=True)
        plt.legend()
        plt.suptitle(f'{feature} distribution')
        plt.xlabel('Standard deviations from mean, centered at 0')
        plt.ylabel('Portion of data in this range')
        plt.savefig(
            Path(__file__).parent / 'results' / 'assets' / f'{feature}Histogram.png'
        )
        plt.clf()

    '''