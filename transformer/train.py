# https://physionet.org/content/challenge-2017/1.0.0/
# fetched data using this command:
# wget -r -N -c -np https://physionet.org/files/challenge-2017/1.0.0/

print('Note: to set gpu, run with $ CUDA_VISIBLE_DEVICES=X python train.py')

from os import path, getcwd
import scipy.io
import numpy as np
import timeseriestransformer as tt

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

SAMPLING_FREQUENCY = 300 #hz

# Set seeds for reproducibility
np.random.seed(3)
torch.manual_seed(3)

#######################################################################
# device set up

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Using device:', device)
print()

#################################################################
# create model

print('Creating model...')

max_sequence_length = SAMPLING_FREQUENCY#*10
sequence_dim = 20
embed_dim = sequence_dim
set_dim = 10
num_classes = 1 #len(LABEL_CODE)
vdim = 1

model = tt.SimpleTransformer(max_sequence_length, sequence_dim, embed_dim, set_dim, num_classes, mode='cosine',parametric_set_fnc=True,vdim=vdim)

for w in model.Q.parameters(): nn.init.eye_(w)
for w in model.K.parameters(): nn.init.eye_(w)

model = model.to(device)

#######################################################################
# read data

print('Loading data...')

data_dir = path.join(getcwd(),'physionet.org/files/challenge-2017/1.0.0')

LABEL_CODE = {'N': 0,  # normal
              'A': 1, # afib
              'O': 2, # other rhythm
              '~': 3} # noisy

data = []
for folder in ['training','validation']:
    for line in open(path.join(data_dir,folder,'REFERENCE.csv'), 'r'):
        line = line.rstrip().split(',')
        obs_path = path.join(data_dir,folder,line[0]+'.mat')
        obs_label = line[1]
        data.append( (obs_path,folder,obs_label) )

cntr = 0
Xall, Yall = [], []
for fname, folder, label in data:
    print('   %.1f%% complete'%(100.*cntr/len(data)),end='\r')
    cntr+=1
    x = 1.*scipy.io.loadmat(fname)['val'].T
    if x.shape[0]<SAMPLING_FREQUENCY*10: continue
    x = model.transform(x, 5, 20, SAMPLING_FREQUENCY, T=10, lrc='c')
    Xall.append(x)
    Yall.append(LABEL_CODE[label])

print('   100.0% complete',end='\r')
print('')

X, Xval, Y, Yval = [], [], [], []
for i in range(len(Xall)):
    if np.random.random()<=0.8:
        X.append(torch.from_numpy(Xall[i]).float())
        Y.append(Yall[i])
    else:
        Xval.append(torch.from_numpy(Xall[i]).float())
        Yval.append(Yall[i])
        
train_dataloader = DataLoader([(X[i],Y[i]) for i in range(len(X))], batch_size=16, shuffle=True)
val_dataloader = DataLoader([(Xval[i],Yval[i]) for i in range(len(Xval))], batch_size=16, shuffle=True)

#################################################################
# train model

from sklearn.metrics import roc_auc_score

print('Training...')

model.train() # set model to training mode
torch.cuda.empty_cache()

best = -np.inf
SAVE_PATH = './binary_afib_model.pt'

REPORT_FREQ = 5
criterion = nn.BCELoss()#nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
for epoch in range(5000):
    avg_loss, c = 0., 0.
    for x,y in train_dataloader:
        x = x.to(device)
        y = y.to(device)
        optimizer.zero_grad()
        yhat = model(x)
        loss = criterion(yhat, (y==1).float())
        loss.backward()
        optimizer.step()
        avg_loss += loss.item()
        c += 1.
    avg_loss /= c
    # print statistics
    if epoch%REPORT_FREQ == 0:
        with torch.no_grad():
            val_loss, n = 0., 0.
            scores, ytrue = [], []
            for x,y in val_dataloader:
                x = x.to(device)
                y = y.to(device)
                yhat = model(x) #torch.argmax(model(x), dim=-1)
                scores += [yhat[i].item() for i in range(yhat.shape[0])]
                ytrue += [y[i].item()==1 for i in range(yhat.shape[0])]
                val_loss += criterion(yhat, (y==1).float())
                #val_acc += (yhat==y).float().sum().item()
                n += 1#x.shape[0]
        val_loss /= n
        ytrue,scores = np.array(ytrue), np.array(scores)
        mxacc = max([(ytrue==(scores>s)).mean() for s in scores])
        testauc = roc_auc_score(ytrue,scores)
        print("Epoch %i: vacc=%.3f, vauc=%.3f, vloss=%.3f, loss=%.3f"%(epoch,mxacc,testauc,val_loss,avg_loss) )#,end='\r' )
        if testauc>best:
            best = testauc
            torch.save(model.state_dict(), SAVE_PATH)

print('')
print('done.')




