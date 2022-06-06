from snorkel.labeling.model import LabelModel
from snorkel.labeling import LFAnalysis
import numpy as np

from .assets.labelmodel_heuristics import get_vote_vector, numberToLabelMap

def getHeuristicVotes(featurizedData):
    L_train = list()
    for i, row in featurizedData.iterrows():
        L_train.append(get_vote_vector(**row))
    return np.array(L_train)

class LabelModelCustom:
    def __init__(self, **kwargs):
        self.lm = LabelModel(cardinality=3, verbose=False)
        self.l_train = None

    def fit(self, featurizedData):
        self.l_train = getHeuristicVotes(featurizedData)
        self.lm.fit(L_train=self.l_train, n_epochs=500, log_freq=100, seed=42)

    def predict(self, featurizedData):
        hVotes = getHeuristicVotes(featurizedData)
        predictions = [numberToLabelMap[prediction] for prediction in self.lm.predict(L=hVotes)]
        return predictions

    def predict_proba(self, featurizedData):
        hVotes = getHeuristicVotes(featurizedData)
        return self.lm.predict_proba(L=hVotes)
    
    def getAnalysis(self) -> LFAnalysis:
        return LFAnalysis(self.l_train)
