"""Author's fifteen-state HMF oxidation network, concentrations in micromol/L.

Extracted from Boudoire chemical-kinetics, MIT, commit 70db1b3f.
All rates are in inverse hours. Electron weights correct a confirmed upstream
index-count error. Unknown-pool extra oxidation remains phenomenological.
"""
import numpy as np
from scipy.integrate import odeint

RATE_NAMES = ('k11','k12','k21','k22','k3','kD1','kD21','kD22','kD3','kD4','kDx')
SPECIES = ('HMF','DFF','HMFCA','FFCA','FDCA','D_HMF','D_DFF','D_HMFCA','D_FFCA','D_FDCA',
           'Dx_HMF','Dx_DFF','Dx_HMFCA','Dx_FFCA','Dx_FDCA')
FARADAY = 96485.33212331001
VOLUME_L = .1
ELECTRON_WEIGHTS=np.array([0,2,2,4,6,2,4,4,6,8,4,6,6,8,10],dtype=np.float64)

def derivatives(c, time, rates):
    """Return all fifteen mass-action derivatives, micromol/L/hour."""
    a,b,d,e,f,g,h,i,j,k,l = rates
    out = np.empty(15, dtype=np.float64)
    out[0] = -(a+b+g)*c[0]
    out[1] = a*c[0]-(d+h)*c[1]
    out[2] = b*c[0]-(e+i)*c[2]
    out[3] = d*c[1]+e*c[2]-(f+j)*c[3]
    out[4] = f*c[3]-k*c[4]
    out[5:10] = np.array([g,h,i,j,k])*c[:5]-l*c[5:10]
    out[10:15] = l*c[5:10]
    return out

def concentrations(time, initial, rates):
    """Integrate at supplied ascending hours using original odeint defaults."""
    return odeint(derivatives, initial, time, args=(np.asarray(rates),))

def charge(c):
    """Return cumulative charge in C without modifying concentration input."""
    converted=c*1e-6
    terms=[ELECTRON_WEIGHTS[i]*converted[:,i] for i in range(15)]
    result=FARADAY*VOLUME_L*np.sum(terms,axis=0)
    return result-result[0]
