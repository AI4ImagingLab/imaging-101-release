"""Read only measured delays, emission intervals and processed odometry."""
from pathlib import Path
import numpy as np

def load_data(path=Path('data/raw_data.npz')):
    shapes={'tdoa_successive':(1,13,10),'tdoa_between':(1,9,14),'emission_intervals':(1,14),'displacement':(1,13,3)}
    with np.load(path,allow_pickle=False) as z:
        if set(z.files)!=set(shapes):raise ValueError('Unexpected observation keys')
        result={k:z[k].copy() for k in shapes}
    for k,a in result.items():
        if a.shape!=shapes[k] or a.dtype!=np.float64 or not np.isfinite(a).all():raise ValueError('Invalid public array '+k)
    if result['emission_intervals'][0,0]!=0 or np.any(result['emission_intervals'][0,1:]<=0):raise ValueError('Invalid emission schedule')
    return {k:v[0] for k,v in result.items()}
