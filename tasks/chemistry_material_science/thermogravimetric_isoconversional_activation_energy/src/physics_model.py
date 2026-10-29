"""Arrhenius integrals, isoconversional objectives and residuals in kJ/mol."""
import numpy as np
from scipy.special import expi
R=.0083144626

def senum_yang(energy,temperature):
    x=energy/(R*temperature)
    num=x**3+18*x**2+88*x+96
    den=x**4+20*x**3+120*x**2+240*x+120
    return (energy/R)*(np.exp(-x)/x)*(num/den)

def temperature_integral(energy,lower,upper):
    a=energy/R
    return a*(expi(-a/upper)-expi(-a/lower))+upper*np.exp(-a/upper)-lower*np.exp(-a/lower)

def interval_integrals(energy,row,data,mode='time'):
    beta=data['beta']
    if mode=='time':
        t0,t1=data['advanced_time'][row:row+2]
        start=data['initial_temperature']
        y0=np.exp(-energy/(R*(start+beta*t0)))
        y1=np.exp(-energy/(R*(start+beta*t1)))
        return (t1-t0)*(y0+y1)/2
    if mode=='temperature':
        lo,hi=data['advanced_temperature'][row:row+2]
        return temperature_integral(energy,lo,hi)/beta
    raise ValueError('mode must be time or temperature')

def ratio_objective(integrals):
    p=np.asarray(integrals)
    return np.sum(np.array([p[j]*(np.sum(1/p)-1/p[j]) for j in range(len(p))]))

def ratio_variance(integrals):
    p=np.asarray(integrals)
    return np.sum((np.array([v/p for v in p])-1)**2)/(len(p)*(len(p)-1))

def centered_log_residual(energy,data):
    logs=np.array([np.log(interval_integrals(e,j,data)) for j,e in enumerate(energy)])
    return logs-logs.mean(axis=1,keepdims=True)
