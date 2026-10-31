"""Bound observations and original numeric column conventions."""
import numpy as np

def load_data(path):
    with np.load(path,allow_pickle=False) as f:
        times=f['times'][0];values=f['measurements'][0];errors=f['reported_uncertainty'][0];bounds=f['group_bounds'][0];flags=f['response_flags'][0]
    groups=[]
    for gi,(a,b) in enumerate(bounds):
        # Preserve original source's explicit single-precision table columns.
        y=values[a:b].astype(np.float32);s=errors[a:b].astype(np.float32)
        if gi==0:s=s*np.float32(.84)
        if gi==5:s=np.sqrt(s*s+np.float32(.02**2))
        t=times[a:b];flag=flags[a:b];wt=np.float32(1.)/(s*s)
        groups.append((t,y,s,wt,flag))
    return groups

