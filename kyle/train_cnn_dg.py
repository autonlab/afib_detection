# https://physionet.org/content/challenge-2017/1.0.0/
# fetched data using this command:
# wget -r -N -c -np https://physionet.org/files/challenge-2017/1.0.0/

print('Note: to set gpu, run with $ CUDA_VISIBLE_DEVICES=X python train.py')

from os import path, getcwd
import numpy as np
from math import floor,log,sqrt
import pickle

from sklearn.metrics import roc_auc_score

from numba import njit
from numba.types import bool_
from collections import Counter


import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader

from PairedDataLoader import PairedDataLoader

# mlflow to log experiment
import mlflow
mlflow.start_run()
myname = path.basename(__file__)
mlflow.log_artifact(myname)
#mlflow.pytorch.autolog(log_models=True)

# Set seeds for reproducibility
np.random.seed(3)
torch.manual_seed(3)

#######################################################################
# Helper functions

def ClassBalance(dataloader,H=120): # H in minutes
    ttes = []
    for x,y,d in dataloader:
        tte = np.round( y.numpy()/(5*60) )*5 # tte to the nearest 5 minutes
        tte[ tte>H ] = H
        good = (tte>=H) | (d.numpy())
        ttes.append(tte[good])
    ttes = np.concatenate(ttes)
    cntr = Counter(ttes)
    n = sum([v for v in cntr.values()])
    for k in cntr.keys(): cntr[k] = np.round( cntr[k]/n, 4)
    cntr = [(int(k),cntr[k]) for k in cntr.keys()]
    cntr.sort()
    return cntr


@njit
def TimeTillNext(x,FREQ=128.):
    N = len(x)
    tte = np.empty(N)
    delta = np.empty(N,dtype=bool_)
    t, d, dt = 0., False, 1./FREQ
    tte[N-1], delta[N-1] = dt, d
    for i in range(N-2,-1,-1):
        if x[i]!=x[i+1] :
            t, d = dt, True
        else:
            t += dt
        tte[i] = t
        delta[i] = d
    return tte, delta


def PrepareData(tup,LABEL_CODE):
    '''
    Inputs:
        x - float np.ndarray with shape Tx2. The ECG signal
        y - bool np.ndarray with shape Tx4. y[:,0] is time until next afib change (either entering or exiting afib). y[:,1] is time since last change afib change. y[:,2] is time until next change of any rhythm. y[:,3] is time since last change of any rhythm.
        i - int8 np.ndarray with shape T, indicates the kind of rhythm the point is currently in. See TypeDict for definitions
        delta - bool np.ndarray with shape Tx4. Indicates whether the time until/since change is observed (True) or not (i.e. BOF/EOF) (False)
    Outputs:
        list of x,tte,delta tuples
    '''
    x,y,i,delta = tup
    w = 60*128 # 1 min window (3min x 60s x 128Hz)
    nlags = 3
    lag_spacing = 1*60*128 # 1 min spacing
    dilation = 1
    stride = 30*128 # 3 min stride (3min x 60s x 128Hz)
    T = x.shape[0]
    start = w+(nlags-1)*lag_spacing

    # afib >= 5min in length
    good_afib = (y[:,:2].sum(1) >= 5*60) & (i==LABEL_CODE['AFIB'])
    tte_ga, delta_ga = TimeTillNext(good_afib)
    #print(tte_ga.min(),tte_ga.max())
    # normal rhythm >= 15 min in length and >=5 min from last change
    # normal rhythm >= 15 min in length and >=5 min from last change
    good = (y[:,2:].sum(1) >= 5*60) & (y[:,3] >= 5*60) & (i==LABEL_CODE['N'])
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
    dat_ = [(np.float32(stack(x[(i-start):i,:])),
             tte_ga[i-1],
             delta_ga[i-1])
             for i in np.arange(start,T,stride) if good[i]]
    return dat_

#######################################################################
# device set up

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Using device:', device)
print()

#################################################################
# create model

#########################
# source: https://github.com/jindongwang/transferlearning/blob/master/code/distance/mmd_pytorch.py
class MMD_loss(nn.Module):
    def __init__(self, kernel_type='rbf', kernel_mul=2.0, kernel_num=5):
        super(MMD_loss, self).__init__()
        self.kernel_num = kernel_num
        self.kernel_mul = kernel_mul
        self.fix_sigma = None
        self.kernel_type = kernel_type

    def guassian_kernel(self, source, target, kernel_mul=2.0, kernel_num=5, fix_sigma=None):
        n_samples = int(source.size()[0]) + int(target.size()[0])
        total = torch.cat([source, target], dim=0)
        total0 = total.unsqueeze(0).expand(
            int(total.size(0)), int(total.size(0)), int(total.size(1)))
        total1 = total.unsqueeze(1).expand(
            int(total.size(0)), int(total.size(0)), int(total.size(1)))
        L2_distance = ((total0-total1)**2).sum(2)
        if fix_sigma:
            bandwidth = fix_sigma
        else:
            bandwidth = torch.sum(L2_distance.data) / (n_samples**2-n_samples)
        bandwidth /= kernel_mul ** (kernel_num // 2)
        bandwidth_list = [bandwidth * (kernel_mul**i)
                          for i in range(kernel_num)]
        kernel_val = [torch.exp(-L2_distance / bandwidth_temp)
                      for bandwidth_temp in bandwidth_list]
        return sum(kernel_val)

    def linear_mmd2(self, f_of_X, f_of_Y):
        loss = 0.0
        delta = f_of_X.float().mean(0) - f_of_Y.float().mean(0)
        loss = delta.dot(delta.T)
        return loss

    def forward(self, source, target):
        if self.kernel_type == 'linear':
            return self.linear_mmd2(source, target)
        elif self.kernel_type == 'rbf':
            batch_size = int(source.size()[0])
            kernels = self.guassian_kernel(
                source, target, kernel_mul=self.kernel_mul, kernel_num=self.kernel_num, fix_sigma=self.fix_sigma)
            XX = torch.mean(kernels[:batch_size, :batch_size])
            YY = torch.mean(kernels[batch_size:, batch_size:])
            XY = torch.mean(kernels[:batch_size, batch_size:])
            YX = torch.mean(kernels[batch_size:, :batch_size])
            loss = torch.mean(XX + YY - XY - YX)
            return loss

##########################

class SoftClip(nn.Module):
    def __init__(self, lower, upper):
        super(SoftClip, self).__init__()
        self.lower = lower
        self.upper = upper
        self.n = nn.Sigmoid()
    def forward(self,x):
        return self.lower+(self.upper-self.lower)*self.n(x)

class CNNdetector(nn.Module):

    def __init__(self, sequence_dim, hidden_dim, embed_dim, kernel_size, stride=1, padding=0, dilation=1):
        super(CNNdetector, self).__init__()
        self.padding = padding
        self.dilation = dilation
        self.kernel_size = kernel_size
        self.stride = stride
        self.sequence_dim = sequence_dim
        self.net = nn.Sequential(
            nn.Conv1d(sequence_dim, hidden_dim, kernel_size, stride, padding, dilation),
            nn.ReLU(),
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size, stride, padding, dilation),
            nn.ReLU(),
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size, stride, padding, dilation),
            nn.ReLU(),
            nn.Conv1d(hidden_dim, embed_dim, kernel_size, stride, padding, dilation)
            )

    def forward(self, x): # x is batch x time x sequence_dim
        return self.net(x.permute(0,2,1)).permute(0,2,1)


class Net(nn.Module):
    def __init__(self,input_shape,time_scale):
        super(Net, self).__init__()

        n = 1 # number of mixture components
        self.time_scale = time_scale
        self.n = n
        # limits
        self.clip = SoftClip(-5,3)

        sequence_dim = input_shape[1]
        hidden_dim = 64
        embed_dim = 32
        kernel_size = 5
        stride = 4
        padding = 0
        dilation = 1
        self.cnn = CNNdetector(sequence_dim, hidden_dim, embed_dim, kernel_size, stride, padding, dilation)

        self.mmd = MMD_loss()

        self.head = nn.Sequential(
            nn.Linear(embed_dim,32),
            nn.ReLU(),
            nn.Linear(32,32),
            nn.ReLU(),
            nn.Linear(32,4),
            nn.ReLU(),
            nn.Linear(4,3*n)
            )


    def forward(self, x_in): # x is batch x time x sequence_dim
        if x_in.dim()<3: x = x_in.unsqueeze(0)
        else: x = x_in
        out = self.head( self.cnn(x).mean(-2) )
        # normalize log mixture weights
        log_w = out[:,:self.n]-torch.logsumexp(out[:,:self.n], 1)[:,None]
        # soft clip scale
        log_scale = self.clip(out[:,self.n:-self.n])+log(self.time_scale)
        # soft clip shape
        log_shape = self.clip(out[:,-self.n:])
        return log_w, log_scale, log_shape

    def log_probs(self, t, a):
        '''
        Compute log(P(t)) and log(1-CDF(t)) for given coefficients a
        '''
        n = self.n
        log_w, log_scale, log_shape = a # unpack
        scale = torch.exp(log_scale)
        shape = torch.exp(log_shape)
        if hasattr(t, "__len__"):
            lc = -(t[:,None]/scale)**shape
            lp = log_shape-log_scale+(shape-1.)*(torch.log(t+1e-2)[:,None]-log_scale)+lc
        else:
            lc = -(t/scale)**shape
            lp = log_shape-log_scale+(shape-1.)*(log(t+1e-2)-log_scale)+lc
        lprob = torch.logsumexp(lp+log_w,1)
        l1mcdf = torch.logsumexp(lc+log_w,1)
        return lprob, l1mcdf

    def RiskTrajectoryLoss(self,tte,delta,a,H):
        '''
        Compute loss for a 'risk trajectory'
        Inputs:
            tte - time-till-event tensor of shape N, the time till afib or EOF
            delta - bool tensor of shape N, indicates whether the event is an observed onset of afib (True) or EOF (False)
            H - time horizon (in seconds), beyond which we treat everything as censored at H
            a - network output; a = net(X)
        Outputs:
            total log-probability of tte given fhat
        '''
        censored = ( ( tte > H ) | ( ~delta ) ).float() # if the event is beyond the prediction horizon, we treat it as censored at time (T-1)*dt
        tte[tte>H] = H
        lprob, l1mcdf = self.log_probs(tte,a)
        return -( lprob*(1-censored)+l1mcdf*censored ).mean()

    def ClassifierScores(self,tte,delta,a,t):
        N, M = len(tte), len(t)
        scores = np.empty((N,M))
        labels = np.empty((N,M),dtype=bool)
        good = np.empty((N,M),dtype=bool)
        for i in range(M):
            _, l1mcdf = self.log_probs(t[i],a)
            scores[:,i] = l1mcdf.cpu().numpy()
            labels[:,i] = tte>t[i]
            good[:,i] = (tte>t[i]) | (delta)
        return scores, labels, good

    def _classifierScores(self, a, t):
        N, M = len(a), len(t)
        scores = np.empty((N,M))
        for i in range(M):
            _, l1mcdf = self.log_probs(t[i],a)
            scores[:,i] = l1mcdf.cpu().numpy()
        return scores

    def PatientGeneralizationLoss(self,source, target):
        src = self.cnn(source).mean(-2)
        tgt = self.cnn(target).mean(-2)
        return self.mmd(src, tgt)


#######################################################################
# read data

print('Loading data...')

# data_path = path.join(getcwd(),'xybundle.pkl')

# data, LABEL_CODE = pickle.load(open(data_path,'rb'))
from bundle_data import knitBundles
data, LABEL_CODE = knitBundles()
print(LABEL_CODE)
print('   load complete.')
print('Preparing data...')
patients = np.array([k for k in data.keys()])
# print(patients)
idx = np.random.random(len(patients))>=0.5
train_patients = patients[idx]
test_patients = patients[~idx]
mlflow.log_metric('train_patients', train_patients)
mlflow.log_metric('test_patients', test_patients)

train_data = [PrepareData(data[pat],LABEL_CODE) for pat in train_patients]
def y_categories(t,d):
    if t>=2*60*60: return 2
    if t>=1*60*60 and d: return 1
    if d: return 0
    return -1

train_data_y = sum([[y_categories(t,d) for x,t,d in dat] for dat in train_data],[])
train_data_x = sum([[x for x,t,d in dat] for dat in train_data],[])
train_data_pat = np.repeat(np.arange(len(train_data)),[len(dat) for dat in train_data])
train_data_x = [train_data_x[i] for i in range(len(train_data_x)) if train_data_y[i]!=-1]
train_data_pat = [train_data_pat[i] for i in range(len(train_data_pat)) if train_data_y[i]!=-1]
train_data_y = [train_data_y[i] for i in range(len(train_data_y)) if train_data_y[i]!=-1]

pair_loader = PairedDataLoader(train_data_x,buckets=train_data_pat,conditioned_on=train_data_y,batch_size=32)

train_dataloader = DataLoader(sum(train_data,[]), batch_size=32, shuffle=True)
print("Train class balance:")
print( ClassBalance(train_dataloader) )

test_dataloader = DataLoader(sum([PrepareData(data[pat],LABEL_CODE) for pat in test_patients],[]), batch_size=32, shuffle=True)
print("Test class balance:")
print( ClassBalance(test_dataloader) )

print('   prep complete.')
#################################################################
# train model


print('Creating model...')
H = 2*60*60 # 2 hours
'''
dataEntry = data[train_patients[1]]
print(len(dataEntry))
print(f'X shape: {dataEntry[0].shape}')
print(f'Y shape: {dataEntry[1].shape}')
print(f'i shape: {dataEntry[2].shape}')
print(f'd shape: {dataEntry[3].shape}')
print(data[train_patients[0]])
'''
dat_ = PrepareData(data[train_patients[0]],LABEL_CODE)
print(dat_)
x,y,d = dat_[0]
print(x.shape)
model = Net(x.shape,H)
model.to(device)
print('   model created.')

print('Training...')

torch.cuda.empty_cache()

best_auc = -np.inf

REPORT_FREQ = 1
optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
#optimizer = torch.optim.Adam(model.parameters(), lr=0.0001, weight_decay=1e-1)
for epoch in range(5000):
    model.train() # set model to training mode
    avg_loss, c = 0., 0.
    for x,y,d in train_dataloader:
        x = x.to(device)
        y = y.to(device)
        d = d.to(device)
        xl, xr = pair_loader.sample()
        xl = xl.to(device)
        xr = xr.to(device)
        optimizer.zero_grad()
        a = model(x)
        loss = model.RiskTrajectoryLoss(y,d,a,H) + 0.01* model.PatientGeneralizationLoss(xl, xr)
        loss.backward()
        optimizer.step()
        avg_loss += loss.item()
        c += 1.
    avg_loss /= c
    # print statistics
    if epoch%REPORT_FREQ == 0:
        model.eval()
        with torch.no_grad():
            print("Epoch %i: training loss: %f"%(epoch,avg_loss))
            t_eval = np.array([H/8,H/4,H/2,H]) # 15min, 30min, 1hr, 2hr
            eval_aucs = {}
            for loader_name,loader in [("train",train_dataloader),("test",test_dataloader)]:
                scores, labels, good = [], [], []
                for x,y,d in loader:
                    x = x.to(device)
                    a = model(x)
                    scores_, labels_, good_ = model.ClassifierScores(y,d,a,t_eval)
                    scores.append(scores_)
                    labels.append(labels_)
                    good.append(good_)
                scores = np.concatenate(scores)
                labels = np.concatenate(labels)
                good = np.concatenate(good)
                aucs = []
                for i in range(len(t_eval)):
                    aucs.append( roc_auc_score(labels[good[:,i],i],scores[good[:,i],i]) )
                print(loader_name+" AUCs: "+', '.join(["%.3f"%v for v in aucs]))
                eval_aucs[loader_name] = aucs
            if np.mean(eval_aucs["test"])>best_auc:
                best_auc = np.mean(eval_aucs["test"])
                torch.save(model.state_dict(), './bestModel.pt')
                for i in range(len(t_eval)):
                    mlflow.log_metric("TrainAUC%i"%int(t_eval[i]/60), eval_aucs["train"][i])
                    mlflow.log_metric("TestAUC%i"%int(t_eval[i]/60), eval_aucs["test"][i])




print('')
print('done.')




