"""Original sliding correlations and seven-parameter local cosine fits."""
import numpy as np
from scipy.signal import correlate
from scipy.optimize import curve_fit
from .preprocessing import PERIOD,KERNEL,AVERAGE
from .physics_model import correlation_model,moments_from_coefficients

def extract_coefficients(grid,sample,period=PERIOD,kernel=KERNEL,capture_covariance=False):
    ny=grid.shape[0]-2*kernel+1;nx=grid.shape[1]-2*kernel+1;half=round(kernel/2)
    coefficients=np.empty((2,ny,nx,5));covariance=np.empty((2,ny,nx,5,5)) if capture_covariance else None
    x=np.linspace(0,kernel,kernel+1);y=np.linspace(kernel,0,kernel+1);x2d,y2d=np.meshgrid(x,y);coords=np.vstack((x2d.ravel(),y2d.ravel()));model=correlation_model(period)
    for row in range(ny):
        reference_row=grid[row:row+2*kernel,:]
        for channel,image in enumerate([grid,sample]):
            window_row=image[row:row+2*kernel,:]
            for col in range(nx):
                reference=reference_row[half:half+kernel,col+half:col+half+kernel];window=window_row[:,col:col+2*kernel];observed=correlate(window,reference,mode='valid').ravel()
                try:
                    p,c=curve_fit(model,coords,observed);coefficients[channel,row,col]=abs(p[:5])
                    if capture_covariance:
                        sign=np.sign(p[:5]);covariance[channel,row,col]=c[:5,:5]*sign[:,None]*sign[None,:]
                except Exception:
                    coefficients[channel,row,col]=np.nan
                    if capture_covariance:covariance[channel,row,col]=np.nan
    return coefficients,covariance

def reconstruct(grid,sample,period=PERIOD,kernel=KERNEL,average=AVERAGE,capture_covariance=False):
    coeff,fit_cov=extract_coefficients(grid,sample,period,kernel,capture_covariance)
    covariance,transmission,ratios=moments_from_coefficients(*coeff,period,average)
    return covariance,transmission,dict(coefficients=coeff,fit_covariance=fit_cov,ratios=ratios)
