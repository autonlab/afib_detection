import pickle

thresholds = {
    'cov_high'  :    .08,
    'cov_low'   :    .05,
    'iqr_high'  :  14.,
    'iqr_low'   :   6.,
    'std_high'  :   8.,
    'std_low'   :   5.,
    'range_high':  10.,
    'range_low' :   5.,
    'pnn20_high':    .80,
    'pnn20_low' :    .71,
    'pnn50_high':    .70,
    'pnn50_mid' :    .50,
    'pnn50_low' :    .30,
    'rmssd_high': 185.,
    'rmssd_low' : 100.,
    'sdnn_high' : 120.,
    'sdnn_mid'  :  60.,
    'sdnn_low'  :  50.
}

ABSTAIN= -1
ATRIAL_FIBRILLATION = 0
SINUS = 1
# OTHER = SINUS
OTHER = 2 # depending on 3-class or binary problem choice
physionetRF = None
# scaler = None

numberToLabelMap = {
    -1: 'ABSTAIN',
    0: 'ATRIAL_FIBRILLATION',
    1: 'SINUS',
    2: 'OTHER',
}
def get_vote_vector_nk( **kwargs):
    return [
        variation_afib(kwargs['b2b_var']),
        variation_other(kwargs['b2b_var']),
        variation_sinus(kwargs['b2b_var']),
        iqr_afib(kwargs['b2b_iqr']),
        iqr_other(kwargs['b2b_iqr']),
        iqr_sinus(kwargs['b2b_iqr']),
        range_afib(kwargs['b2b_range']),
        range_other(kwargs['b2b_range']),
        range_sinus(kwargs['b2b_range']),
        std_afib(kwargs['b2b_std']),
        std_other(kwargs['b2b_std']),
        std_sinus(kwargs['b2b_std']),
        pnn20_afib(kwargs['hrv_pnn20']),
        pnn20_other(kwargs['hrv_pnn20']),
        pnn20_sinus(kwargs['hrv_pnn20']),
        #pnn50_afib(kwargs['pnn50']),
        pnn50_other(kwargs['hrv_pnn50']),
        pnn50_sinus(kwargs['hrv_pnn50']),
        rmssd_afib(kwargs['hrv_rmssd']),
        rmssd_other(kwargs['hrv_rmssd']),
        rmssd_sinus(kwargs['hrv_rmssd']),
        sdnn_afib(kwargs['hrv_sdnn']),
        sdnn_other(kwargs['hrv_sdnn']),
        sdnn_sinus(kwargs['hrv_sdnn']),
        hopkins_sinus(kwargs['hopkins_statistic']),
        hopkins_other(kwargs['hopkins_statistic']),
        sil_coef_sinus(kwargs['max_sil_score']),
        sil_coef_other(kwargs['max_sil_score']),
        sse_afib_other(kwargs['sse_1_clusters']),
        sse_afib_sinus(kwargs['sse_1_clusters']),
        sse_diff_afib(kwargs['sse_1_clusters'], kwargs['sse_2_clusters']),
        # trainedPhysionet(kwargs)
        # mitbih_model_afib(kwargs['']),
        # mitbih_model_sinus(kwargs['']),
        # mitbih_model_other(kwargs[''])
    ]
def get_vote_vector( **kwargs):
    return [
        variation_afib(kwargs['b2b_var']),
        variation_other(kwargs['b2b_var']),
        variation_sinus(kwargs['b2b_var']),
        iqr_afib(kwargs['b2b_iqr']),
        iqr_other(kwargs['b2b_iqr']),
        iqr_sinus(kwargs['b2b_iqr']),
        range_afib(kwargs['b2b_range']),
        range_other(kwargs['b2b_range']),
        range_sinus(kwargs['b2b_range']),
        std_afib(kwargs['b2b_std']),
        std_other(kwargs['b2b_std']),
        std_sinus(kwargs['b2b_std']),
        pnn20_afib(kwargs['pnn20']),
        pnn20_other(kwargs['pnn20']),
        pnn20_sinus(kwargs['pnn20']),
        #pnn50_afib(kwargs['pnn50']),
        pnn50_other(kwargs['pnn50']),
        pnn50_sinus(kwargs['pnn50']),
        rmssd_afib(kwargs['rmssd']),
        rmssd_other(kwargs['rmssd']),
        rmssd_sinus(kwargs['rmssd']),
        sdnn_afib(kwargs['sdnn']),
        sdnn_other(kwargs['sdnn']),
        sdnn_sinus(kwargs['sdnn']),
        hopkins_sinus(kwargs['hopkins_statistic']),
        hopkins_other(kwargs['hopkins_statistic']),
        sil_coef_sinus(kwargs['max_sil_score']),
        sil_coef_other(kwargs['max_sil_score']),
        sse_afib_other(kwargs['sse_1_clusters']),
        sse_afib_sinus(kwargs['sse_1_clusters']),
        sse_diff_afib(kwargs['sse_1_clusters'], kwargs['sse_2_clusters'])
        # mitbih_model_afib(kwargs['']),
        # mitbih_model_sinus(kwargs['']),
        # mitbih_model_other(kwargs[''])
    ]

def sil_coef_sinus(maxSilScore):
    if maxSilScore > .65:
        return SINUS
    else:
        return ABSTAIN
def sil_coef_other(maxSilScore):
    if maxSilScore > .65:
        return OTHER
    else:
        return ABSTAIN

def hopkins_sinus(hopkinsStatistic):
    if hopkinsStatistic < .20:
        return SINUS
    else:
        return ABSTAIN
def hopkins_other(hopkinsStatistic):
    if hopkinsStatistic < .20:
        return OTHER
    else:
        return ABSTAIN

def sse_afib_other(sseSingleCluster):
    if sseSingleCluster > 400:
        return ATRIAL_FIBRILLATION
    else:
        return OTHER
def sse_afib_sinus(sseSingleCluster):
    if sseSingleCluster > 400:
        return ATRIAL_FIBRILLATION
    else:
        return SINUS

def sse_diff_afib(sseSingle, sseDouble):
    if (sseSingle - sseDouble) > 200:
        return ATRIAL_FIBRILLATION
    else:
        return ABSTAIN
from data.manipulators import loadScaler, applyScaler
from model.utilities import getModelConfig
import pandas as pd
def trainedPhysionet(features):
    global physionetRF, scaler
    if (physionetRF is None):
        print('once')
        with open('./data/assets/physionet_rf.pkl', 'rb') as readfile:
            physionetRF = pickle.load(readfile)
        scaler = loadScaler()
    #load model

    #predict
    # try:
    features = pd.DataFrame(features, index=[0])
    features_nk = getModelConfig().features_trunc
    features = applyScaler(features, features_nk, scaler)
    res = physionetRF.predict(features[features_nk])[0]
    if (res == 'ATRIAL_FIBRILLATION'):
        return ATRIAL_FIBRILLATION
    elif (res == 'NOT_AFIB'):
        return SINUS
    else:
        return ABSTAIN
    # except:
    #     return ABSTAIN
    # return res

def variation_afib(cov):
    if (cov > thresholds['cov_high']):
        return ATRIAL_FIBRILLATION
    else:
        return ABSTAIN

def variation_other(cov):
    if (cov > thresholds['cov_high']):
        return OTHER
    else:
        return ABSTAIN

def variation_sinus(cov):
    if (cov < thresholds['cov_low']):
        return SINUS
    else:
        return ABSTAIN

def iqr_afib(iqr):
    if iqr > thresholds['iqr_low']:
        return ATRIAL_FIBRILLATION
    else:
        return ABSTAIN

def iqr_other(iqr):
    if iqr < thresholds['iqr_high']:
        return OTHER
    else:
        return ABSTAIN

def iqr_sinus(iqr):
    if iqr < thresholds['iqr_low']:
        return SINUS
    else:
        return ABSTAIN

def std_afib(std):
    if std > thresholds['std_high']:
        return ATRIAL_FIBRILLATION
    else:
        return ABSTAIN

def std_other(std):
    if std > thresholds['std_low']:
        return OTHER
    else:
        return ABSTAIN

def std_sinus(std):
    if std < thresholds['std_low']:
        return SINUS
    else:
        return ABSTAIN

def range_afib(r):
    if r > thresholds['range_high']:
        return ATRIAL_FIBRILLATION
    else:
        return ABSTAIN

def range_sinus(r):
    if r < thresholds['range_high']:
        return SINUS
    else:
        return ABSTAIN

def range_other(r):
    if r > thresholds['range_low']:
        return OTHER
    else:
        return ABSTAIN

def pnn20_afib(pnn20):
    if (pnn20 > thresholds['pnn20_high']):
        return ATRIAL_FIBRILLATION
    else:
        return ABSTAIN

def pnn20_sinus(pnn20):
    if (pnn20 < thresholds['pnn20_low']):
        return SINUS
    else:
        return ABSTAIN

def pnn20_other(pnn20):
    if (pnn20 < thresholds['pnn20_low']):
        return OTHER
    else:
        return ABSTAIN

def pnn50_other(pnn50):
    if (pnn50 < .55):
        return OTHER
    else:
        return ABSTAIN
def pnn50_sinus(pnn50):
    if (pnn50 < .55):
        return SINUS
    else:
        return ABSTAIN

'''
def pnn50_afib(pnn50):
    if (pnn50 > thresholds['pnn50_mid']):
        return ATRIAL_FIBRILLATION
    else:
        return ABSTAIN
def pnn50_other(pnn50):
    if ((thresholds['pnn50_low'] < pnn50) and (pnn50 < thresholds['pnn50_high'])):
        return OTHER
    else:
        return ABSTAIN

def pnn50_sinus(pnn50):
    if (pnn50 < thresholds['pnn50_low']):
        return SINUS
    else:
        return ABSTAIN
'''

def rmssd_afib(rmssd):
    if (rmssd > thresholds['rmssd_low']):
        return ATRIAL_FIBRILLATION
    else:
        return ABSTAIN

def rmssd_sinus(rmssd):
    if (rmssd < thresholds['rmssd_low']):
        return SINUS
    else:
        return ABSTAIN

def rmssd_other(rmssd):
    if ((thresholds['rmssd_low'] < rmssd) and (rmssd < thresholds['rmssd_high'])):
        return OTHER
    else:
        return ABSTAIN

def sdnn_afib(sdnn):
    if (sdnn > thresholds['sdnn_mid']):
        return ATRIAL_FIBRILLATION
    else:
        return ABSTAIN

def sdnn_other(sdnn):
    if ((thresholds['sdnn_low'] < sdnn) and (sdnn < thresholds['sdnn_high'])):
        return OTHER
    else:
        return ABSTAIN

def sdnn_sinus(sdnn):
    if (sdnn < thresholds['sdnn_low']):
        return SINUS
    else:
        return ABSTAIN
