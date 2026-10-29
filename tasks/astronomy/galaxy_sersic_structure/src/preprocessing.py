import numpy as np
SHAPE=(120,120)
def load_data(path='data/raw_data.npz'):
    with np.load(path,allow_pickle=False)as f:a={k:f[k].copy()for k in('images','inverse_variance','psfs','mask')}
    shapes=dict(images=(1,3,120,120),inverse_variance=(1,3,120,120),psfs=(1,3,63,63),mask=(1,1,120,120))
    for k,v in a.items():
        if v.shape!=shapes[k]or not np.isfinite(v).all():raise ValueError('Invalid '+k)
    if(a['inverse_variance']<=0).any():raise ValueError('Invalid inverse variance')
    return{k:v[0]for k,v in a.items()}
