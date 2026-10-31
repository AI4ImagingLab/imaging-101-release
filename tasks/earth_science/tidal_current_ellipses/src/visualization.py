import numpy as np
import matplotlib.pyplot as plt
def metrics(a,b):
    a=np.asarray(a).ravel();b=np.asarray(b).ravel()
    return dict(ncc=float(a@b/np.linalg.norm(a)/np.linalg.norm(b)),nrmse=float(np.sqrt(np.mean((a-b)**2))/np.ptp(b)))
def ellipse_plot(cosine,sine,names):
    fig,axes=plt.subplots(2,4,figsize=(12,6.5),layout='constrained');phase=np.linspace(0,2*np.pi,361)
    for k,(a,name)in enumerate(zip(axes.flat,names)):
        curve=cosine[:,k,None]*np.cos(phase)+sine[:,k,None]*np.sin(phase)
        a.plot(*curve,color='#0072B2');a.scatter(*curve[:,0],s=18,color='#D55E00',label='Reference phase')
        a.annotate('',xy=curve[:,30],xytext=curve[:,20],arrowprops=dict(arrowstyle='->',color='#009E73'))
        limit=max(np.max(abs(curve))*1.15,.1);a.set(xlim=(-limit,limit),ylim=(-limit,limit),aspect='equal',title=name,xlabel='East current (cm/s)',ylabel='North current (cm/s)');a.axhline(0,color='.7',lw=.5);a.axvline(0,color='.7',lw=.5)
    fig.suptitle('NOAA CO-OPS measured currents: fresh inferred constituent ellipses\nEach panel has its labeled physical scale; orange is reference phase')
    return fig
