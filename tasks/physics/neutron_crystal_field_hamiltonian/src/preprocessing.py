"""Public observations and active same-data resolution estimation."""
from pathlib import Path
import numpy as np
from scipy.optimize import curve_fit
from scipy.special import wofz
from .physics_model import voigt

def load_data(path=None):
    path=Path(path) if path else Path(__file__).resolve().parents[1]/'data/raw_data.npz'
    with np.load(path,allow_pickle=False) as z:return {k:z[k][0].copy() for k in z.files}

def preliminary_profile(x,*p):
    def peak(x,a,center,alpha,gamma):
        sigma=(.5*alpha)/np.sqrt(2*np.log(2))
        return a*np.real(wofz(((x-center)+1j*(.5*gamma))/sigma/np.sqrt(2)))/sigma/np.sqrt(2*np.pi)
    result=peak(x-p[1],p[0],0,p[2],p[15])
    for i in range(1,5):result+=peak(x-p[1],p[3*i],p[3*i+1],p[3*i+2],p[16])
    return result+.00015

def estimate_resolution(data):
    initial=[.04,.15,.25,.009,.87,.25,.005,3.47,.25,.005,5.11,.18,.004,5.58,.18,.1,.1]
    p,cov=curve_fit(preliminary_profile,data['energy'][0],data['intensity'][0],sigma=data['reported_error'][0],p0=initial)
    energies=np.array([p[3*i+1] for i in range(5)]);widths=np.array([p[3*i+2] for i in range(5)])
    def line(x,a,b):return a*x+b
    width,wcov=curve_fit(line,energies[1:],widths[1:],p0=[-.008,.16])
    return width,dict(voigt_parameters=p,voigt_covariance=cov,width_covariance=wcov)
