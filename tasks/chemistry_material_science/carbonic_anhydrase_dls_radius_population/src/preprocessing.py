import numpy as np

def load_data(path):
    with np.load(path,allow_pickle=False) as a:
        lag_us=a['lag_us'][0,1:].copy();t=lag_us/1e6;g=a['correlation'][0,:,1:].T.copy();r=a['radius_nm'][0].copy()
    return t,g,r,lag_us

def transform(t,g,lag_us):
    early=np.asarray(lag_us)<5.0
    beta=np.exp(np.polyfit(t[early],np.log(g[early]-1),2)[-1])
    with np.errstate(invalid='ignore'):field=np.sqrt((g-1)/beta)
    valid=np.array([np.flatnonzero(g[:,j]<1.0)[0] if np.any(g[:,j]<1.0) else len(t) for j in range(g.shape[1])])
    return beta,field,valid
