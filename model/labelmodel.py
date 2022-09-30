from snorkel.labeling.model import LabelModel
from snorkel.labeling import LFAnalysis
import numpy as np
from tqdm import tqdm
import data.utilities as du
from joblib import Memory

from .assets.labelmodel_heuristics import get_vote_vector, get_vote_vector_nk, numberToLabelMap

def getHeuristicVotes(featurizedData):
    L_train = list()
    for i, row in tqdm(featurizedData.iterrows(), total=len(featurizedData)):
        L_train.append(get_vote_vector_nk(**row))
    return np.array(L_train)

class LabelModelCustom:
    def __init__(self, **kwargs):
        dataConfig = du.getDataConfig()
        mem = Memory(dataConfig.cacheDirectory)
        self.getHeuristicVotes = mem.cache(getHeuristicVotes)
        self.lm = LabelModel(cardinality=3, verbose=False)
        self.l_train = None

    def fit(self, featurizedData):
        self.l_train = self.getHeuristicVotes(featurizedData)
        self.lm.fit(L_train=self.l_train, n_epochs=500, log_freq=100, seed=42)

    def predict(self, featurizedData, returnHeuristicVotes=False):
        hVotes = self.getHeuristicVotes(featurizedData)
        predictions = [numberToLabelMap[prediction] for prediction in self.lm.predict(L=hVotes)]
        if (returnHeuristicVotes):
            return predictions, [[numberToLabelMap[v] for v in hVoteSet] for hVoteSet in hVotes]
        else:
            return predictions

    def predict_proba(self, featurizedData):
        hVotes = self.getHeuristicVotes(featurizedData)
        return self.lm.predict_proba(L=hVotes)
    
    def getAnalysis(self) -> LFAnalysis:
        return LFAnalysis(self.l_train)
