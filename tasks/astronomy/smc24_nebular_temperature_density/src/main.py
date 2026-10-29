from pathlib import Path
import numpy as np
from .physics_model import load_ions
from .preprocessing import load_observations,observation_ratios
from .solvers import crossed_solution

def run(output='output/reconstruction.npz'):
    ions=load_ions();raw,corrected=load_observations();ratios=observation_ratios(corrected)
    result=np.array([crossed_solution(ions,t,n,ratios[t],ratios[n]) for t in ['NII','OIII'] for n in ['OII','SII']])
    if not np.all(np.isfinite(result)):raise ValueError('Crossed diagnostic convergence failure')
    Path(output).parent.mkdir(exist_ok=True,parents=True)
    np.savez(output,reconstruction=result[None])
    return result,ions,raw,corrected,ratios

if __name__=='__main__':print(run()[0])
