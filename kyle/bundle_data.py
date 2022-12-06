import wfdb
import numpy as np
from os.path import join as pathjoin
import pickle

dir_ = '../data/assets/europace/physionet.org/files/ltafdb/1.0.0'

def TimeSince(rec,dir=dir_):
    '''
        Grab ECG data and record time-till-event and time-since-event
        Inputs:
            rec - string record to read
            dir - string director to read from
        Outputs:
            x - float np.ndarray with shape Tx2. The ECG signal
            y - bool np.ndarray with shape Tx2. y[:,0] is time until next change (either entering or exiting afib). y[:,1] is time since last change (either entering or exiting afib).
            i - int8 np.ndarray with shape T, indicates the kind of rhythm the point is currently in. See TypeDict for definitions
            delta - bool np.ndarray with shape Tx2. Indicates whether the time until/since change is observed (True) or not (i.e. BOF/EOF) (False)
    '''
    global TypeDict
    rec = pathjoin(dir,rec)
    record = wfdb.rdrecord(rec) # read data for record 00
    x = record.p_signal
    T, fs = x.shape[0], record.fs
    #qrs_annotations = wfdb.rdann(rec,'qrs') # read reference beat and rhythm annotations
    atr_annotations = wfdb.rdann(rec,'atr') # read unaudited beat annotations generaged by sqrs, with AF terminations (T) 
    #
    # annotation symbol '+' = rhythm change
    # type of rhythm is in the aux_note
    sym = np.array(atr_annotations.symbol)
    rhythm = np.array(atr_annotations.aux_note)[sym=='+']
    t = atr_annotations.sample[sym=='+']
    intervals = [(t[i],t[i+1],rhythm[i]) for i in range(len(t)-1)]+[(t[-1],T,rhythm[-1])]
    #
    t = np.arange(T)/fs # time in seconds
    y = np.empty((T,4)) #y[:,0] = time till afib, y[:,1] = time since afib, y[:,2] = time till next rhythm change, y[:,3] = time since last rhym change
    delta = np.zeros((T,4),dtype=bool)
    i = -np.ones(T,dtype=np.int8)
    y[:,0] = np.flip(t)
    y[:,1] = t
    y[:,2] = np.flip(t)
    y[:,3] = t
    for s,e,kind in intervals: # iterate through all intervals
        k = kind[1:]
        if k not in TypeDict: 
            TypeDict[k] = len(TypeDict)
            print("New type: %s"%(k))
        i[s:e] = TypeDict[k]
        #if kind!='(AFIB': continue
        t_until = s/fs-t
        t_since = -t_until
        # record time until start-of-interval 
        for j in ([0,2] if kind=='(AFIB' else [2]):
            idx = (y[:,j]>t_until) & (t_until>=0)
            y[ idx, j] = t_until[idx]
            delta[ idx, j] = True
            # record time since start-of-interval
            idx = (y[:,j+1]>t_since) & (t_since>=0)
            y[ idx, j+1] = t_since[idx]
            delta[ idx, j+1] = True
            if e<T:
                t_until = e/fs-t
                t_since = -t_until
                # record time until end-of-interval
                idx = (y[:,j]>t_until) & (t_until>=0)
                y[ idx, j] = t_until[idx]
                delta[ idx, j] = True
                # record time since end-of-interval
                idx = (y[:,j+1]>t_since) & (t_since>=0)
                y[ idx, j+1] = t_since[idx]
                delta[ idx, j+1] = True
    return x,y,np.array(i),delta
def innerLoad(line):
    dat = dict()
    rec = line.rstrip()
    print(rec)
    x,y,i,delta = TimeSince(rec)
    dat[rec] = (x,y,i,delta)
    # cnt += 1
    pickle.dump((dat,{}),open(f'./bundled/xybundle_{rec}.pkl','wb'))
def knitBundles():
    knitData, knitLabels = dict(), dict()
    for line in open(pathjoin(dir_,'RECORDS'),'r'):
        rec = line.rstrip()
        data, label_code =  pickle.load(open(f'./bundled/xybundle_{rec}.pkl', 'rb'))
        knitData, knitLabels = {**knitData, **data}, {**knitLabels, **label_code}
    return knitData, {'N': 3, 'AFIB': 0, 'VT': 6, 'AB': 4, 'SVTA': 8, 'T': 5, 'B': 7, 'SBR': 1, 'IVR': 2}

from joblib import Parallel, delayed
if __name__ == '__main__':
    # Dat = {}
    TypeDict = {'N': 3, 'AFIB': 0, 'VT': 6, 'AB': 4, 'SVTA': 8, 'T': 5, 'B': 7, 'SBR': 1, 'IVR': 2}
    cnt = 0
    Parallel(n_jobs=7)(delayed(innerLoad)(line) for line in open(pathjoin(dir_, 'RECORDS'), 'r'))
    # for line in open(pathjoin(dir_,'RECORDS'),'r'):
    #     rec = line.rstrip()
    #     x,y,i,delta = TimeSince(rec)
    #     Dat[rec] = (x,y,i,delta)
    #     cnt += 1
    #     print(cnt)
    #     pickle.dump((Dat,TypeDict),open('xybundle.pkl','wb'))


