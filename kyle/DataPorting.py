
import datetime as dt
import numpy as np
from typing import Iterable

def collectSignalInForm(tup, timePoints: Iterable[dt.datetime], basetime: dt.datetime):
    """Follows form of PrepareData (in train_cnn*.py files). Returns shaped data points given patient signal data and points in time

    Args:
        tup (_type_): _description_
        label_code (_type_): _description_
        timePoint (_type_): _description_
    """

    x,y,i,delta = tup
    w = 60*128 # 1 min window (3min x 60s x 128Hz)
    nlags = 3
    lag_spacing = 1*60*128 # 1 min spacing
    dilation = 1
    stride = 30*128 # 3 min stride (3min x 60s x 128Hz)
    T = x.shape[0]
    start = w+(nlags-1)*lag_spacing

    def stack(x):
        xlist = []
        for i in range(nlags):
            idx0, idx1 = int(i*lag_spacing), int(i*lag_spacing+w)
            x_ = x[idx0:idx1:dilation,:]
            mn = x.mean(0)
            q25 = np.quantile(x,0.25,axis=0)
            q75 = np.quantile(x,0.75,axis=0)
            x_ = (x_-mn[None,:])/(q75-q25+0.1)[None,:]
            xlist.append( x_ )
        return np.concatenate(xlist,1)
    timeToIndex = lambda x: int((x - basetime).total_seconds() * 128)
    dat_ = [(np.float32(stack(x[(i-start):i,:])))
             for i in map(timeToIndex, timePoints)]
    return dat_