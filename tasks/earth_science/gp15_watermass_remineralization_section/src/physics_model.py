import numpy as np
NAMES=('AAIW','PSUW','ESSW','NPIW','PDW1','UCDW','PDW23','LCDW','AABW')
def usage_penalties(coordinates):
 lat,lon,depth,density=np.asarray(coordinates).T
 p=np.zeros((len(lat),9));densitypen=np.minimum(10000,1000*np.expm1(.5*np.maximum(27.72-density,0)))
 p[:,7]=densitypen;p[:,8]=densitypen
 p[:,1]=np.minimum(10000,100*np.expm1(.05*np.maximum(10-lat,0)))+np.minimum(10000,50*np.expm1(.001*np.maximum(depth-3000,0)))
 return p

def predict(fractions,reactions,endmembers,ratios):return fractions@endmembers+reactions@ratios

def objective_gradient(q,observation,endmembers,ratios,weights,penalty):
 n=len(endmembers);a=np.vstack([endmembers,ratios]);res=(q@a-observation)*weights
 value=np.dot(res,res)+np.sum((q[:n]*penalty)**2)
 grad=2*a@(res*weights);grad[:n]+=2*q[:n]*penalty**2
 return float(value),grad
