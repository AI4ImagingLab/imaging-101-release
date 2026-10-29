"""Public calibrated acceleration-to-response input interface."""
import numpy as np
SHAPES={'accelerations':(1,20001,5),'observation_time':(1,20001),'reconstruction_time':(1,100001),'mass':(1,10,10),'stiffness':(1,10,10),'damping':(1,10,10),'sensor_matrix':(1,5,10)}

def load_observations(path):
    """Return unbatched public arrays, all float64; no hidden input reads."""
    with np.load(path,allow_pickle=False) as z:
        data={}
        for key,shape in SHAPES.items():
            a=z[key]
            if a.shape!=shape or a.dtype!=np.float64 or not np.isfinite(a).all():raise ValueError('Invalid public '+key)
            data[key]=a[0].copy()
    for name in ['observation_time','reconstruction_time']:
        if np.any(np.diff(data[name])<=0):raise ValueError('Non-increasing time grid')
    return data

def observation_indices(observation_time,reconstruction_time):
    """Native matching after four-decimal rounding; retain every observation."""
    t=np.round(reconstruction_time,4);q=np.round(observation_time,4)
    index=np.searchsorted(t,q)
    if np.any(index>=len(t)) or not np.array_equal(t[index],q):raise ValueError('Observation time is outside prediction grid')
    return index
