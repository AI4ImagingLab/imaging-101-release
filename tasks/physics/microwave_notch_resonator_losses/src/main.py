"""Run full sweep calibration and published circle/phase fitting."""
from pathlib import Path
import numpy as np
from .preprocessing import load_data
from .solvers import reconstruct

def run(output='output/reconstruction.npz'):
    frequency,scattering=load_data()
    fit,response,normalized,calibration=reconstruct(frequency,scattering)
    values=dict(resonance_frequency=np.array([[fit['fr']]]),quality_factors=np.array([[fit['Ql'],fit['Qc_dia_corr'],fit['Qi_dia_corr']]]),reconstruction=response[None])
    Path(output).parent.mkdir(parents=True,exist_ok=True)
    np.savez(output,**values)
    return values,fit,normalized,calibration

if __name__=='__main__':
    values,fit,_,_=run()
    print(fit)
