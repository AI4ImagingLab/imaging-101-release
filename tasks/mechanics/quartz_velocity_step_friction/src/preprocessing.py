from pathlib import Path
import numpy as np
JUMP_TIME_S=24.8
def prepare(path=Path('data/raw_data.npz')):
    with np.load(path,allow_pickle=False) as f:t=f['time'][0];x=f['loadpoint_displacement'][0];mu=f['friction'][0]
    jump=int(np.argmin(abs(t-JUMP_TIME_S)))
    b_minus_a=float((np.mean(mu[:jump+1])-np.mean(mu[-51:]))/np.log(10))
    mu0=float(np.mean(mu[:jump+1])+b_minus_a*np.log(.1))
    return {'time':t,'displacement':x,'friction':mu,'b_minus_a':b_minus_a,'mu0':mu0,'jump_index':jump}
