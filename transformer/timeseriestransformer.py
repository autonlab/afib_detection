import math
import numpy as np
from scipy.interpolate import interp1d
import heapq

import torch
import torch.nn as nn
import torch.nn.functional as F

def standardize(x):
    if x.ndim==1:
        x -= np.median(x)
        x /= np.median(heapq.nlargest(10, x))
    else:
        for i in range(x.shape[1]):
            x[:,i] -= np.median(x[:,i])
            x[:,i] /= np.median(heapq.nlargest(10,x[:,i]))
    return x
        
# time delay embedding
def time_delay(x, tau, d):
    '''
    maps x(t) -> [x(t),x(t+tau),x(t+2*tau),...,x(t+(d-1)*tau)]
    '''
    assert x.ndim in {2,3}
    if x.ndim==2:
        L = x.shape[0]
        td_x = np.empty((L-tau*(d-1),d))
        for i in range(d):
            td_x[:,i] = x[i*tau : L-((d-1)*tau-i*tau),0]
    else:
        L = x.shape[1]
        td_x = np.empty((x.shape[0],L-tau*(d-1),d))
        for i in range(d):
            td_x[:,:,i] = x[:,i*tau : L-((d-1)*tau-i*tau),0]
    return td_x

# attention based network that summarizes relative positional information to make binary classifications
class SimpleTransformer(nn.Module):
    
    def __init__(self, max_sequence_length, sequence_dim, embed_dim, set_dim, num_classes, vdim=1, mode='sigmoid', parametric_set_fnc=True):
        super(SimpleTransformer, self).__init__()
        # params
        self.max_sequence_length = max_sequence_length
        self.sequence_dim = sequence_dim
        self.embed_dim = embed_dim
        self.set_dim = set_dim
        self.num_classes = num_classes
        self.vdim = vdim
        self.mode = mode # mode in 'softmax', 'cosine', 'sigmoid'
        self.parametric_set_fnc = parametric_set_fnc
        self.device = 'cpu'
        # networks
        # Q & K map series vector x(t) into new coordinates and compare for attention
        self.Q = nn.Linear(sequence_dim,embed_dim,bias=False)
        self.K = nn.Linear(sequence_dim,embed_dim,bias=False)
        # V computes weights for linear combination of time series of set vectors
        #self.V = nn.Linear(sequence_dim,vdim,bias=False)
        self.V = nn.Sequential(
            nn.Linear(sequence_dim,sequence_dim),
            nn.ReLU(),
            nn.Linear(sequence_dim,sequence_dim),
            nn.ReLU(),
            nn.Linear(sequence_dim,vdim),
            nn.Sigmoid()
            )
        # pre-computed indexes for relative-positions
        self.Tidx = torch.arange(max_sequence_length)
        self.Tidx = torch.abs(self.Tidx[:,None]-self.Tidx[None,:])
        
        # Phi network encodes set information in vector of dimension set_dim
        if not parametric_set_fnc:
            self.SetPhi = nn.Parameter(torch.rand(max_sequence_length,set_dim,requires_grad=True))
        else:
            self.t = torch.linspace(0,1,max_sequence_length)
            self.Phi = nn.Sequential(
                nn.Linear(1,2*set_dim),
                nn.ReLU(),
                nn.Linear(2*set_dim,2*set_dim),
                nn.ReLU(),
                nn.Linear(2*set_dim,2*set_dim),
                nn.ReLU(),
                nn.Linear(2*set_dim,set_dim),
                )
        
        if mode=='softmax':
            self.Iinf = torch.diag(-float('inf')*torch.ones(max_sequence_length))
        else:
            self.Iinf = 1.-torch.eye(max_sequence_length)
            if mode=='sigmoid': 
                self.offset = nn.Parameter(torch.tensor(-10.,requires_grad=True))
        
        # head does the final classification
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(vdim*set_dim,vdim*set_dim),
            nn.ReLU(),
            nn.Linear(vdim*set_dim,vdim*set_dim),
            nn.ReLU(),
            nn.Linear(vdim*set_dim,num_classes),
            nn.Softmax(dim=-1) if num_classes>1 else nn.Sigmoid()
            )
    
    def to(self, *args, **kwargs):
        self = super().to(*args, **kwargs) 
        self.Tidx = self.Tidx.to(*args, **kwargs) 
        if self.parametric_set_fnc:
            self.t = self.t.to(*args, **kwargs)
        self.Iinf = self.Iinf.to(*args, **kwargs) 
        if self.mode=='sigmoid': self.offset = self.offset.to(*args, **kwargs) 
        return self
    
    def transform(self,x, tau, d, freq, T=10, lrc='c', SAMPLING_FREQUENCY=300, kind='cubic'): # applies resampling and time delay
        # dimensions are (time x feature)
        if x.ndim==1: x = x[:,np.newaxis] # assuming x is 1d time series (time)
        # resample freq
        if freq!=SAMPLING_FREQUENCY:
            L = x.shape[0]
            t = np.arange(0,L/freq,1/freq)
            Lq = max(t)*SAMPLING_FREQUENCY
            tq = np.arange(0,Lq/SAMPLING_FREQUENCY,1/SAMPLING_FREQUENCY)
            x_ = np.empty((Lq,x.shape[1]))
            for k in range(x.shape[1]):
                x_[:,k] = interp1d(t,x[:,k],kind=kind)(tq)
        else: x_=x
        L = x_.shape[0]
        x_ = standardize(x_)
        if lrc=='c':
            t = (L-1)/SAMPLING_FREQUENCY
            w0 = int( (t-T)/2*SAMPLING_FREQUENCY )
            if w0<0: w0=0
            w1 = w0+int( T*SAMPLING_FREQUENCY )
            if w1<=L: x_ = x_[w0:w1]
        elif lrc=='l':
            w = int( T*SAMPLING_FREQUENCY )
            if w<L: x_ = x_[:w]
        else: # lrc=='r':
            w = int( T*SAMPLING_FREQUENCY )
            if w<L: x_ = x_[-w:]
        x_ = time_delay(x_, tau, d)
        return x_[::10,:]
     
    def forward(self, x): # x is batch x time x sequence_dim
        if x.dim()<3: x = x.unsqueeze(0)
        L = x.shape[1] # length of time series
        
        # first map series into new coords for attention
        q = self.Q(x) # batch x time x embed_dim
        k = self.K(x) # batch x time x embed_dim
        
        # base of attention is the dot products: q_i^T*k_j.
        a = torch.einsum('bik,bjk->bij',q,k) # batch x time x time
        if self.mode=='softmax':
            # softmax: normalize a using softmax
            d_k = q.size()[-1]
            a = a / math.sqrt(d_k)
            attention = F.softmax(a+self.Iinf[:L,:L], dim=-1) # batch x time x time, +Iinf to force diag(attention)=0
        elif self.mode=='cosine':
            # cosine similarity: normalize a by product of norms of q and k
            qnorm2 = (q*q).sum(-1) # batch x time x 1
            knorm2 = (k*k).sum(-1) # batch x time x 1
            norms2 = torch.einsum('bi,bj->bij',qnorm2,knorm2) # batch x time x time
            norms2 = F.relu(norms2-1e-8)+1e-8 # prevent division by zero
            attention = a/torch.sqrt(norms2) # batch x time x time
            attention = attention*self.Iinf[:L,:L] # force diag(attention)=0
        elif self.mode=='sigmoid':
            d_k=q.size()[-1]
            attention = torch.sigmoid(a/math.sqrt(d_k) + self.offset)
            attention = attention*self.Iinf[:L,:L]
        else:
            raise ValueError("mode must be one of {'softmax','cosine','sigmoid'}")
        
        if self.parametric_set_fnc:
            self.SetPhi = self.Phi(self.t[:L,None])
        # take weighted sum of SetPhi vectors, which encode relative position information in set vector form.
        set_values = torch.einsum('bij,ijk->bik',attention,self.SetPhi[self.Tidx[:L,:L]]) # batch x time x set_dim
        
        # summarize set representations
        v = self.V(x) # batch x time x vdim
        summarized = torch.einsum('bij,bik->bjk',set_values,v) # batch x set_dim x vdim
        
        # make final prediction
        return self.head(summarized) if self.num_classes>1 else self.head(summarized).flatten()



