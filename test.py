from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import math
from sklearn.metrics import PrecisionRecallDisplay, classification_report
from analysis import plotSegment, showConfidentlyIncorrects, permutationFeatureImportance
from data.roc import roc
from itertools import cycle
import model.utilities as mu
from train import train

if __name__ == "__main__":
    showClassificationReport = True
    showFeatureImportance = False

    features = [f.lower() for f in mu.getModelConfig().features_nk]
    # features = ['b2b_range', 'b2b_var', 'sse_2_clusters', 'sse_1_clusters', 'hrv_pnn20', 'hrv_pnn50', 'hrv_shanen', 'ecg_rate_mean', 'hrv_sd1']
    # features = set(features) - set(['hfd', 'hrv_hf', 'hrv_lfhf', 'sd1', 'sample_entropy', 'max_sil_score', 'hrv_lf', 'b2b_var', 'rmssd', 'sd1/sd2', 'sd2', 'hopkins_statistic', 'b2b_std'])
    # features = list(features)
    # resnet, resnet_data = train(
    #     filterGold=False,
    #     usesplits=True,
    #     model="ResNet",
    #     verbose=True,
    #     )
    lm, lm_data = train(
        filterGold=False,
        usesplits=True,
        model="LabelModel",
        verbose=True,
        # filterUnreasonableValues=True
        )
    rf_sk, rf_sk_data = train(
        filterGold=False,
        usesplits=True,
        model="RandomForestSK",
        verbose=True,
        # filterUnreasonableValues=True
        reduceDimension=True,
        # winsorize=True
        )
    lr, lr_data = train(
        filterGold=False,
        usesplits=True,
        model="LogisticRegression",
        verbose=True,
        # filterUnreasonableValues=True
        reduceDimension=True,
        # winsorize=True
        )
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
        lm_cr = classification_report(
            y_true=lm_data['testLabels'],
            y_pred=lm_data['testPredictions']
            )
        lr_cr = classification_report(
            y_true=lr_data['testLabels'],
            y_pred=lr_data['testPredictions']
            )
        randForestSK_cr = classification_report(
            y_true=rf_sk_data['testLabels'],
            y_pred=rf_sk_data['testPredictions']
            )
        phil_cr = classification_report(
            y_true=phillips_true,
            y_pred=phillips_pred
            )
        print(f'LogisticRegressor classification report:\n{lr_cr}')
        print(f'RandomForest (sklearn) classification report:\n{randForestSK_cr}')
        # print(f'ResNet classification report:\n{resnet_cr}')
        print(f'RandomForest (autonlab) classification report:\n{"... in progress ..."}')
        print(f'Labelmodel classification report:\n{lm_cr}')
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
    titles = cycle(['LogisticRegressor', 'RandomForest (sklearn)', 'LabelModel', 'ResNet'])
    afibIndex = list(lr.classes_).index('ATRIAL_FIBRILLATION')
    for cacheddata in [lr_data, rf_sk_data, lm_data]:#resnet_data
        yTests.append(cacheddata['testLabels'])
        singleProbs = [prob[afibIndex] for prob in cacheddata['testPredProbabilities']]
        yScores.append((next(titles),singleProbs))
    roc(yTests, yScores, 'ROC Comparison')
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