"""Load original-precision acquisitions and fixed independent calibration."""
from pathlib import Path
import numpy as np
T2_MS=np.logspace(np.log10(10.),np.log10(3000.),100)
LAMBDA=0.003719667767540722

def load_data(path):
    with np.load(path,allow_pickle=False) as z:
        d={k:z[k][0] for k in z.files}
    d['observed']=d['echo_magnitudes'].mean(axis=-1)
    d['mask']=d['estimation_mask'].astype(bool)
    d['y']=d['observed'][d['mask']]
    d['labels']=d['cluster_labels'][d['mask']].astype(int)-1
    return d
