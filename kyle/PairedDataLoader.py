import numpy as np
import torch

class PairedDataLoader():
    '''
    Samples random pairs, optionaly conditioned on predefined groupings
    '''
    def __init__(self,data_list,buckets=None,conditioned_on=None,batch_size=1,strict_batch_size=False):
        self.batch_size = batch_size
        self.no_bucket_mode = True if buckets is None else False
        self.data_store = {}
        for i in range(len(data_list)):
            if conditioned_on is None: condition=0
            else: condition = conditioned_on[i]
            if buckets is None: bucket=0
            else: bucket = buckets[i]
            if condition not in self.data_store: self.data_store[condition] = {}
            if bucket not in self.data_store[condition]: self.data_store[condition][bucket] = []
            self.data_store[condition][bucket].append( torch.from_numpy(data_list[i]).unsqueeze(0) )
        # concat tensors
        for condition in self.data_store.keys():
            for bucket in self.data_store[condition].keys():
                self.data_store[condition][bucket] = torch.cat( self.data_store[condition][bucket] )
    
    def sample(self,condition=None):
        if condition is None:
            keys = [k for k in self.data_store.keys()]
            condition = np.random.choice(keys)
        data = self.data_store[condition]
        if self.no_bucket_mode:
            # sampling in no-bucket mode
            bucket_idx = np.array([0,0])
            b = np.random.choice(data[bucket_idx[0]].shape[0], size=2*self.batch_size, replace=False)
            b0 = b[:self.batch_size]
            b1 = b[self.batch_size:]
        else: 
            # sampling in bucket mode
            keys = [k for k in data.keys()]
            bucket_p = np.array([data[k].shape[0] for k in keys])
            bucket_p = bucket_p/bucket_p.sum()
            bucket_idx = np.random.choice(keys, size=2, replace=False, p=bucket_p)
            size = min(min(self.batch_size,data[bucket_idx[0]].shape[0]),data[bucket_idx[1]].shape[0])
            b0 = np.random.choice(data[bucket_idx[0]].shape[0], size=size, replace=False)
            b1 = np.random.choice(data[bucket_idx[1]].shape[0], size=size, replace=False)
        dat0 = data[bucket_idx[0]][b0]
        dat1 = data[bucket_idx[1]][b1]
        return dat0, dat1


if __name__ == "__main__":
    N = 12
    dat = [np.random.random((3,4)) for i in range(N)]
    buckets = np.tile([1,2,3],N//3)
    conditions = np.tile([0,1],N//2)
    # no buckets, no conditions
    loader = PairedDataLoader(dat,batch_size=2)
    print(loader.sample())
    # no buckets
    loader = PairedDataLoader(dat,conditioned_on=conditions,batch_size=2)
    print(loader.sample())
    # no conditions
    loader = PairedDataLoader(dat,buckets=buckets,batch_size=2)
    print(loader.sample())
    # both buckets and conditions
    loader = PairedDataLoader(dat,buckets=buckets,conditioned_on=conditions,batch_size=2)
    print(loader.sample())

