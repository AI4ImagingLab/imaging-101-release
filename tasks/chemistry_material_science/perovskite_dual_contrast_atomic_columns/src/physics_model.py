"""Explicit rotated Gaussian image formation; GPL-3.0-or-later.

Extracted from Atomap Gaussian2D (HyperSpy developers, 2007-2016).
Parameters here are [integrated amplitude, sx, sy, cx, cy, theta].
"""
import math
import numpy as np
def gaussian(p,x,y):
    amplitude,sx,sy,cx,cy,theta=p
    sx2,sy2=sx**2,sy**2
    ct,st=math.cos(theta)**2,math.sin(theta)**2
    a=ct/(2*sx2)+st/(2*sy2)
    b=-math.sin(2*theta)/(4*sx2)+math.sin(2*theta)/(4*sy2)
    c=st/(2*sx2)+ct/(2*sy2)
    return amplitude*(1/(sx*sy*(2*math.pi)))*np.exp(-(a*(x-cx)**2+2*b*(x-cx)*(y-cy)+c*(y-cy)**2))
def physical_columns(parameters,pixel_nm):
    # Internal arrays use [cx,cy,sx,sy,theta,integrated_amplitude].
    p=np.asarray(parameters);c,s=np.cos(p[:,4]),np.sin(p[:,4]);a,b=p[:,2]**2,p[:,3]**2
    return np.column_stack((p[:,:2]*pixel_nm,(c*c*a+s*s*b)*pixel_nm**2,c*s*(b-a)*pixel_nm**2,(s*s*a+c*c*b)*pixel_nm**2,p[:,5]/(2*np.pi*p[:,2]*p[:,3])))
def model_image(parameters,shape):
    out=np.zeros(shape,dtype=np.float64)
    for x,y,sx,sy,theta,amplitude in parameters:
        radius=5*max(sx,sy)
        x0,y0=max(0,math.floor(x-radius)),max(0,math.floor(y-radius))
        x1,y1=min(shape[1],math.ceil(x+radius)),min(shape[0],math.ceil(y+radius))
        yy,xx=np.mgrid[y0:y1,x0:x1]
        out[y0:y1,x0:x1]+=gaussian([amplitude,sx,sy,x,y,theta],xx,yy)
    return out
