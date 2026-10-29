"""Scientific acquisition, parity and acquisition-sensitivity figures."""
import numpy as np
import matplotlib.pyplot as plt
from .physics_model import reconstruction_grid

def agreement(x,r):
    a=x.ravel();b=r.ravel()
    return float(np.dot(a,b)/(np.linalg.norm(a)*np.linalg.norm(b))) if np.linalg.norm(a) else 0.,float(np.sqrt(np.mean((a-b)**2))/np.ptp(b))

def image_panel(ax,image,title,reference_max=None):
    x,z=reconstruction_grid(); vmax=float(image.max()) if reference_max is None else reference_max
    db=20*np.log10(np.maximum(image/vmax,1e-5))
    h=ax.imshow(db,extent=[x[0]*1e3-.1,x[-1]*1e3+.1,z[-1]*1e3+.1,z[0]*1e3-.1],vmin=-45,vmax=0,cmap='magma',aspect='equal')
    ax.set(xlabel='Lateral x (mm)',ylabel='Depth z (mm)',title=title,xlim=(-20,20),ylim=(50,0))
    return h

def acquisition_figure(data,filtered):
    fig,axs=plt.subplots(1,2,figsize=(10,3.8),layout='constrained')
    sel=data['tx']==data['rx']; a=data['traces'][sel]
    h=axs[0].imshow(a.T,aspect='auto',extent=[.5,64.5,(data['time'][-1]+2e-8)*1e6,-.02],cmap='RdBu_r',vmin=-.08,vmax=.08)
    axs[0].set(xlabel='Pulse-echo element',ylabel='Time (microsecond)',title='a  Measured pulse-echo B-scan',xlim=(1,64),ylim=(data['time'][-1]*1e6,0))
    fig.colorbar(h,ax=axs[0],label='Digitizer amplitude (1)')
    i=np.flatnonzero(sel)[32]
    axs[1].plot(data['time']*1e6,data['traces'][i],color='.5',lw=.8,label='Measured')
    axs[1].plot(data['time']*1e6,np.abs(filtered[i]),color='#0072B2',lw=1,label='Filtered envelope')
    axs[1].set(xlabel='Time (microsecond)',ylabel='Amplitude (1)',title='b  Centre-element trace');axs[1].legend()
    return fig

def comparison_figure(fresh,upstream):
    fig,axs=plt.subplots(1,3,figsize=(10,4.2),layout='constrained')
    h=image_panel(axs[0],upstream,'a  Independently run upstream',upstream.max());image_panel(axs[1],fresh,'b  Fresh local reconstruction',upstream.max())
    fig.colorbar(h,ax=list(axs[:2]),label='Echo amplitude (dB, upstream peak)')
    d=fresh-upstream; limit=max(float(np.max(np.abs(d))),1e-16)
    h=axs[2].imshow(d,extent=[-20.1,20.1,50.1,-.1],aspect='equal',cmap='RdBu_r',vmin=-limit,vmax=limit)
    axs[2].set(xlabel='Lateral x (mm)',ylabel='Depth z (mm)',title='c  Local minus upstream',xlim=(-20,20),ylim=(50,0));fig.colorbar(h,ax=axs[2],label='Amplitude difference (1)')
    return fig

def sensitivity_figure(results,reference):
    fig,axs=plt.subplots(1,len(results),figsize=(11,4),layout='constrained')
    for ax,(name,img) in zip(axs,results.items()): h=image_panel(ax,img,name,reference.max())
    fig.colorbar(h,ax=list(axs),label='Echo amplitude (dB, full-array peak)')
    return fig
