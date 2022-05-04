from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import math
from sklearn.metrics import PrecisionRecallDisplay, classification_report
from analysis import plotSegment, showConfidentlyIncorrects
from data.roc import roc

from train import train

if __name__ == "__main__":
    # trainlm()
    # train(model='LabelModel', usesplits=False)
    # train(usesplits=False)
    #lrModel, cacheddata = train(filterGold=True, usesplits=True, model="LogisticRegression", verbose=True)
    lrModel, cacheddata_testset = train(
        filterGold=False,
        usesplits=True,
        model="LogisticRegression",
        verbose=True,
        #overwriteTrainset='trainset_10000_featurized_withextras.csv',
        overwriteTestset='testset_featurized.csv'
        )
    lrModel, cacheddata_evalset = train(
        filterGold=False,
        usesplits=True,
        model="LogisticRegression",
        verbose=True,
        overwriteTrainset='trainset_10000_featurized_withextras.csv',
        overwriteTestset='testset_featurized_withextras.csv'
    )
    pd.DataFrame({
        'afib': cacheddata_newtest['testPredProbabilities'][:, 0],
        'sinus': cacheddata_newtest['testPredProbabilities'][:, 1],
        'true_class': cacheddata_newtest['testLabels']
    }).to_csv('lrpredictions.csv', index=False)
    '''
    lrModel, cacheddata_oldTrainNewTest = train(
        filterGold=True,
        usesplits=True,
        model="LogisticRegression",
        verbose=True,
        overwriteTrainset='trainset_featurized_withfilter.csv',
        overwriteTestset='testset_featurized_withfilter.csv'
        )
    lrModel, cacheddata_oldTrainOldTest = train(
        filterGold=True,
        usesplits=True,
        model="LogisticRegression",
        verbose=True,
        overwriteTrainset='trainset_featurized_withfilter.csv'
        )
    '''
    # lrModel, cacheddata = train(filterGold=True, usesplits=True, model="LogisticRegression", verbose=True)
    # rfModel, cacheddata2 = train(filterGold=True, usesplits=True, model="RandomForestSK", verbose=True)
    # print(cacheddata)
    '''
    probs = [
        ('Test set selected by Dr. Rooney', cacheddata['testPredProbabilities'][:,0]),
        ('Test set selected by Dr. Rooney, oldtrainset', cacheddata_oldTrainOldTest['testPredProbabilities'][:,0]),
        ('Test set oversampled via b2b iqr', cacheddata_newtest['testPredProbabilities'][:,0]),
        ('Test set oversampled via b2b iqr, old trainset', cacheddata_oldTrainNewTest['testPredProbabilities'][:,0]),
    ]
    '''
    #showConfidentlyIncorrects(df)
    '''roc((cacheddata['testLabels'],
        cacheddata_oldTrainOldTest['testLabels'],
        cacheddata_newtest['testLabels'],
        cacheddata_oldTrainNewTest['testLabels']), probs, "Trained on new trainset")
    print(classification_report(y_true=cacheddata['testLabels'],
        y_pred=cacheddata['testPredictions']
        ))
    print(classification_report(y_true=cacheddata['testLabels'],
        y_pred=cacheddata_oldTrainOldTest['testPredictions']
        ))
        '''
    '''print(classification_report(y_true=cacheddata_newtest['testLabels'],
        y_pred=cacheddata_oldTrainNewTest['testPredictions']
        ))'''
    models = [
        ('LogisticRegression', lrModel),
        ('RandomForest', rfModel)]
    #print(classification_report(y_true=cacheddata['testLabels'], y_pred=cacheddata['testPredictions']))
    #print(classification_report(y_true=cacheddata2['testLabels'], y_pred=cacheddata2['testPredictions']))
    # (clf.predict_proba(X_test)[:,1] >= 0.3).astype(bool)
    # probs = list()
    # for i, label in enumerate(cacheddata['testLabels']):
    #     pred = cacheddata['testPredictions'][i]
    #     print(cacheddata['testPredProbabilities'][i], cacheddata['testPredictions'][i])
    #     if pred != 'ATRIAL_FIBRILLATION':
    #         prob = min(cacheddata['testPredProbabilities'][i])
    #         # prob = 1 - prob
    #     else:
    #         prob = max(cacheddata['testPredProbabilities'][i])
    #     probs.append(prob)

    # plt.hist(goldLabelsAndProbabilities[goldLabelsAndProbabilities['label'] == 'ATRIAL_FIBRILLATION']['probas'], alpha=.8, histtype='step', label='AFib', bins=15)
    # plt.hist(goldLabelsAndProbabilities[goldLabelsAndProbabilities['label'] != 'ATRIAL_FIBRILLATION']['probas'], alpha=.8, histtype='step', label='Not AFib', bins=20)
    # plt.xlabel('Model probability of AFib')
    # plt.ylabel('Quantity of data points')
    # plt.legend()
    # plt.savefig(
    #     Path(__file__).parent / 'results' / 'assets' / f'probaPlots.png'
    # )
    # plt.clf()
    # print(1/0)

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
