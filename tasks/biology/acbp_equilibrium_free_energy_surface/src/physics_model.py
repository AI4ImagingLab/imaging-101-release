"""Explicit two-state equilibrium thermodynamics and fluorescence baselines."""
import numpy as np
R=8.314e-3
ZERO_CELSIUS=-273.15
REFERENCE_CELSIUS=25.
def free_energy(temperature,denaturant,thermodynamic_parameters):
 tm,dh,dcp,m=thermodynamic_parameters;K=np.asarray(temperature)-ZERO_CELSIUS;M=tm-ZERO_CELSIUS
 return dh*(1-K/M)+dcp*(K-M-K*np.log(K/M))-np.asarray(denaturant)*m
def unfolded_fraction(temperature,denaturant,thermodynamic_parameters):
 dg=free_energy(temperature,denaturant,thermodynamic_parameters)
 # Preserve native negative-energy overflow guard, inactive on the bound fit.
 odds=np.exp(-np.maximum(dg,-250.)/(R*(np.asarray(temperature)-ZERO_CELSIUS)))
 return odds/(1+odds)
def baseline_planes(temperature,denaturant,coefficients):
 dt=np.asarray(temperature)-REFERENCE_CELSIUS;aF,bFD,bFT,aU,bUD,bUT,bUT2=np.asarray(coefficients)
 return aF+bFD*denaturant+bFT*dt,aU+bUD*denaturant+bUT*dt+bUT2*dt**2
def fluorescence(parameters,temperature,denaturant,channel):
 p=np.asarray(parameters);u=unfolded_fraction(temperature,denaturant,p[14:18]);result=np.zeros_like(temperature,dtype=float)
 for c in range(2):
  F,U=baseline_planes(temperature,denaturant,p[c*7:(c+1)*7]);result+=np.where(channel==c,(1-u)*F+u*U,0.)
 return result
def parameter_jacobian(parameters,temperature,denaturant,channel,relative_step=1e-5):
 p=np.asarray(parameters);columns=[]
 for i in range(18):
  h=max(abs(p[i]),1)*relative_step;plus=p.copy();minus=p.copy();plus[i]+=h;minus[i]-=h
  columns.append((fluorescence(plus,temperature,denaturant,channel)-fluorescence(minus,temperature,denaturant,channel))/(2*h))
 return np.array(columns).T
