"""Camera counts and float64 temporal averaging."""
import numpy as np

def load_data(path):
    """Load uint16 (batch,block,frame,y,x) detector ADU."""
    with np.load(path,allow_pickle=False) as d:return d['intensities']

def temporal_mean(stack):
    """Float64 mean of (frame,y,x), preserving upstream summation order."""
    mean=np.zeros(stack.shape[1:],dtype=np.float64)
    for frame in stack:mean+=frame
    return mean/len(stack)
