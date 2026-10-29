"""Explicit local periodic-grid blur model and coefficient-to-moment inversion.

Correlation model extracted from Croughan's BSD-3-Clause DiDaFi reference;
source copyright and license retained in task LICENSE.
"""
import numpy as np
from .preprocessing import PERIOD,AVERAGE
def correlation_model(period):
    PERIOD=period
    def correlation_expression(x_y_coordinates,c_0,c_1,c_2,c_3,c_4,phase_1,phase_2):
            # Define function to fit to auto-correlation of I_g and cross-correlation of I_g and I_sg. Allow phase of trig functions to be a free parameter.
            x2d, y2d = x_y_coordinates
            return c_0 + c_1*np.cos((2. *np.pi *x2d)/PERIOD +phase_1)+ \
                c_2*np.cos((2.*np.pi *y2d)/PERIOD +phase_2)+ \
                c_3*np.cos((2. *np.pi *(x2d-y2d))/PERIOD +phase_1-phase_2)+ \
                c_4*np.cos((2. *np.pi *(x2d+y2d))/PERIOD + phase_1+phase_2)
    return correlation_expression

def average_coefficients(coefficients,average=AVERAGE):
    ny=coefficients.shape[0]//average;nx=coefficients.shape[1]//average
    return np.stack([np.nanmean(coefficients[:ny*average,:nx*average,i].copy().reshape(ny,average,nx,average),axis=(3,1)) for i in range(5)])
def moments_from_coefficients(grid_coefficients,sample_coefficients,period=PERIOD,average=AVERAGE):
    ratios=average_coefficients(sample_coefficients,average)/average_coefficients(grid_coefficients,average)
    transmission=ratios[0];a1=period**2/np.pi**2*np.log(ratios[1]/transmission);a2=period**2/np.pi**2*np.log(ratios[2]/transmission);a3=period**2/(2*np.pi**2)*np.log(ratios[3]/transmission);a4=period**2/(2*np.pi**2)*np.log(ratios[4]/transmission)
    covariance=np.array([(-3*a1+a2-a3-a4)/8,(a1-3*a2-a3-a4)/8,(a3-a4)/4])
    return covariance,transmission,ratios

def mode_ratios(covariance,transmission,period=PERIOD):
    xx,yy,xy=covariance;factor=2*np.pi**2/period**2
    projected=np.array([xx,yy,xx+yy-2*xy,xx+yy+2*xy])
    return np.concatenate([transmission[None],transmission[None]*np.exp(-factor*projected)],axis=0)

def covariance_eigenvalues(covariance):
    xx,yy,xy=covariance;trace=xx+yy;difference=np.hypot(xx-yy,2*xy)
    return np.array([(trace-difference)/2,(trace+difference)/2])
