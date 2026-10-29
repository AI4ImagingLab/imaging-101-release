"""Consistent geometric review metrics and plots."""
import numpy as np
import matplotlib.pyplot as plt

def metrics(a,b):
    """Cosine NCC and dynamic-range NRMSE; inputs already in the same gauge."""
    a=np.asarray(a).ravel();b=np.asarray(b).ravel()
    return {'ncc':float(a@b/(np.linalg.norm(a)*np.linalg.norm(b))),'nrmse':float(np.sqrt(np.mean((a-b)**2))/np.ptp(b))}

def plot_shape(ax,shape,title):
    """Scatter every3Dpoint, pixel scale, no extra centring or alignment."""
    ax.scatter(shape[:,0],shape[:,1],shape[:,2],c=shape[:,2],s=4,cmap='viridis')
    ax.set(xlabel='x (pixel)',ylabel='y (pixel)',zlabel='z (pixel)',title=title)
