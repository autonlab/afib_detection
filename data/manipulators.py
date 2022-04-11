from pyexpat import features
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import StandardScaler

def loadScaler():
    return

def applyScaler(df, featuresToScale, scaler):
    # print(df[featuresToScale])
    # scaledDf = pd.DataFrame(scaler.transform(df[featuresToScale]), columns=featuresToScale)
    df[featuresToScale] = scaler.transform(df[featuresToScale])
    # print(df[featuresToScale])
    return df

def computeAndApplyScaler(df, featuresToScale, scalerType="standard"):
    """_summary_

    Args:
        df (_type_): _description_
        featuresToScale (_type_): _description_
        scalerType (str, optional): Type of scaler to use from sklearn.preprocessing lib. Defaults to "standard".
    """

    if scalerType=='standard':
        scaler = StandardScaler()
    else:
        raise ValueError(f'scalerType [{scalerType}] unsupported!')

    df[featuresToScale] = scaler.fit_transform(df[featuresToScale])
    return df, scaler

def applySplits(df, prespecifiedTestSet=None):
    """Split given dataframe into train and test sets, either by fin_study_id or given prespecified test entries

    Args:
        df (pd.DataFrame): dataframe to split 
        prespecifiedTestSet (List[Num], optional): List of `fin_study_id`'s to keep in testset. Defaults to None.

    Returns:
        tuple(pd.DataFrame, pd.DataFrame): trainset, testset
    """
    if (prespecifiedTestSet is None):
        trainIndices, testIndices = next(GroupShuffleSplit(n_splits=1, test_size=.2, random_state=7).split(df, groups=df['fin_study_id']))
        trainset, testset = df.iloc[trainIndices], df.iloc[testIndices]
    else:
        trainset = df[~df['fin_study_id'].isin(prespecifiedTestSet)]
        testset = df[df['fin_study_id'].isin(prespecifiedTestSet)]
    return trainset, testset

def oversample(df, classColumn, samplingLikelihoodColumn):
    """Oversample classes less represented than max represented class

    Args:
        df (_type_): _description_
        classColumn (str): class to oversample from
        samplingLikelihoodColumn (str): associated likelihood to oversample given data point
    """
    maxClassSize = df[classColumn].value_counts().max()
    allDFs = [df]
    for idx, classSubset in df.groupby(classColumn):
        samplesForClass = classSubset.sample(maxClassSize - len(classSubset), replace=True, weights=classSubset[samplingLikelihoodColumn])
        allDFs.append(samplesForClass)
    return pd.concat(allDFs)

def filterAndNormalize(df, features, preexistingScaler=None):
    # Filter
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df = df.dropna()

    # Normalize
    if (preexistingScaler):
        df_normalized = applyScaler(df, features, preexistingScaler)
    else:
        df_normalized = computeAndApplyScaler(df, features)
    return df_normalized

def remapLabels(df, labelColumn, labelmap):
    df[labelColumn] = df[labelColumn].map(lambda x: labelmap[x] if x in labelmap else x)
    return df