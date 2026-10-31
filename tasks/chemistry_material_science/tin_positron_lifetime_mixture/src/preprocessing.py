import numpy as np

def load_data(path):
    with np.load(path,allow_pickle=False) as a:counts=a['counts'][0].copy();prompt=a['prompt_counts'][0].copy();channel=a['channel'][0].copy()
    if counts.shape!=(6,10000) or prompt.shape!=(10000,) or not np.array_equal(channel,np.arange(10000)):raise ValueError('Invalid native measurement axes')
    if np.any(counts<0) or np.any(prompt<0):raise ValueError('Negative counts')
    binned=counts.reshape(6,1000,10).sum(2).astype('float64');return counts,prompt,binned

def prepare(binned):
    area=binned[:,33:949].sum(1);background=binned[:,749:949].mean(1)
    return area,background
