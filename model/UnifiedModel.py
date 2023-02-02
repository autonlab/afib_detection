from abc import ABC, abstractmethod

import datetime as dt
import pandas as pd
import numpy as np
from torch import nn

class UnifiedModel(ABC):

    @abstractmethod
    def apply(self, df: pd.DataFrame) -> pd.DataFrame:
        """Given dataframe with columns patient_id, start, and stop: Return df of model confidences

        Args:
            df (pd.DataFrame): _description_
        """

from data.physionet_utilities import collectSignalFromPatient
from kyle.bundle_data import knitBundles
from kyle.DataPorting import collectSignalInForm
import prediction.loadData as ld
class ConvNetMMD(UnifiedModel):

    def __init__(self, model: nn.Module):
        self.model = model
    
    def apply(self, df: pd.DataFrame, t_evals = np.array([7.5*60, 15*60, 30*60, 60*60])):
        #collect all patient signal data
        dataBundle, label_dict = knitBundles()
        res = list()

        #convert patient_id, starts, stops into data expected by convnet model
        for patient_id, id_group in df.groupby('patient_id'):
            signalForPatient = dataBundle[patient_id]
            basetime = ld.getHeaderById(patient_id)[0].base_datetime
            #extract corresponding to timestamps present in id group
            correspondingPortions = collectSignalInForm(signalForPatient, id_group['stop'].tolist(), basetime)
            getScoresFor = lambda datum: self.model._classifierScores(self.model(datum), t_evals)
            modelConfidences = np.array(getScoresFor(d) for d in correspondingPortions)
            for i, t in enumerate(t_evals):
                id_group[f'model_confidence_{t}'] = modelConfidences[:,i]
            res.append(id_group)
        return pd.concat(res)