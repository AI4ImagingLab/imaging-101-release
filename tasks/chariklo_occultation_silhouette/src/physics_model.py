"""Explicit projected elliptical limb, radial contacts and negative-chord test.

MIT adaptation of SORA extra.utils.get_ellipse_points and occultation fitting.
"""
import numpy as np

def ellipse_contour(parameters,theta):
    f,g,a,oblat,phi=np.asarray(parameters)
    b=a-a*oblat;angle=theta+phi*(np.pi/180.)
    radius=a*b/np.sqrt((a*np.sin(angle))**2+(b*np.cos(angle))**2)
    return np.column_stack([f+radius*np.cos(theta),g+radius*np.sin(theta)])

def contact_chi2(parameters,points,sigma):
    """Vectorized trial objective, retaining the original contact residual order."""
    trials=np.atleast_2d(parameters);f,g,a,o,phi=trials.T
    total=np.zeros(len(trials));b=a-a*o
    for (fi,gi),si in zip(points,sigma):
        theta=np.arctan2(gi-g,fi-f);angle=theta+phi*(np.pi/180.)
        radius=a*b/np.sqrt((a*np.sin(angle))**2+(b*np.cos(angle))**2)
        fm=f+radius*np.cos(theta);gm=g+radius*np.sin(theta)
        total+=((fi-fm)**2+(gi-gm)**2)/si**2
    return total

def negative_margin(parameters,path):
    """Smallest radial separation; positive means every negative sample is clear."""
    f,g,a,o,phi=parameters
    delta=path-np.array([f,g]);radius=np.linalg.norm(delta,axis=1)
    cut=radius<1.5*a
    if not np.any(cut):return float(np.min(radius)-a)
    theta=np.arctan2(delta[cut,1],delta[cut,0]);b=a-a*o
    re=a*b/np.sqrt((a*np.sin(theta+phi*np.pi/180))**2+(b*np.cos(theta+phi*np.pi/180))**2)
    return float(np.min(radius[cut]-re))

def contact_time_residuals(parameters,points,velocity):
    """Independent geometric ray/ellipse quadratic; nearest contact time in seconds."""
    f,g,a,o,phi=parameters;angle=phi*np.pi/180
    rotate=np.array([[np.cos(angle),-np.sin(angle)],[np.sin(angle),np.cos(angle)]])
    p=(points-np.array([f,g]))@rotate.T;v=velocity@rotate.T
    weights=np.array([1/a**2,1/(a*(1-o))**2])
    aa=np.sum(v*v*weights,axis=1);bb=2*np.sum(p*v*weights,axis=1);cc=np.sum(p*p*weights,axis=1)-1
    disc=bb*bb-4*aa*cc
    if np.any(disc<0):return np.full(len(points),np.inf)
    roots=np.column_stack([(-bb-np.sqrt(disc))/(2*aa),(-bb+np.sqrt(disc))/(2*aa)])
    return roots[np.arange(len(points)),np.argmin(abs(roots),axis=1)]
