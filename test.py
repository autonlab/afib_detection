from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import math
from sklearn.metrics import PrecisionRecallDisplay, classification_report

from train import train

if __name__ == "__main__":
    # trainlm()
    # train(model='LabelModel', usesplits=False)
    # train(usesplits=False)
    lrModel, cacheddata = train(filterGold=False, usesplits=False, model="LogisticRegression", verbose=True)
    rfModel, cacheddata2 = train(filterGold=False, usesplits=False, model="RandomForestSK", verbose=True)
    # print(cacheddata)

    models = [
        ('LogisticRegression', lrModel),
        ('RandomForest', rfModel)]
    print(classification_report(y_true=cacheddata['testLabels'], y_pred=cacheddata['testPredictions']))
    print(classification_report(y_true=cacheddata2['testLabels'], y_pred=cacheddata2['testPredictions']))
    for modelName, model in models:
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