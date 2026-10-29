"""Full author fluorescence acquisition and native valid-frame intervals."""
import numpy as np
def load_observations(path):
    with np.load(path,allow_pickle=False) as z:return z['intensity'][0].copy(),z['interval_start'][0].copy(),z['interval_stop'][0].copy()
def observation_sequences(intensity,interval_start,interval_stop):
    result=[]
    for trace,lo,hi in zip(intensity,interval_start,interval_stop):
        selected=trace[int(lo):int(hi)]
        if len(selected)<1 or np.any(selected.sum(axis=1)==0):raise ValueError('Invalid native observation interval')
        value=selected[:,1]/selected.sum(axis=1)
        if not np.isfinite(value).all():raise ValueError('Nonfinite apparent fluorescence ratio')
        result.append(value)
    return result
def pack_sequences(sequences):
    lengths=np.array([len(x) for x in sequences],dtype=np.int64)
    data=np.zeros((len(sequences),int(lengths.max())),dtype=float)
    for i,x in enumerate(sequences):data[i,:len(x)]=x
    valid=np.arange(data.shape[1])[None,:]<lengths[:,None]
    return data,valid,lengths
