"""Maintainer diagnostic plots; importing this module never changes backend."""
import numpy as np
import matplotlib.pyplot as plt
from .preprocessing import R,Z,prepare
from .physics_model import integrate_power

def metrics(estimate,reference):
    span=np.ptp(reference);rn=np.linalg.norm(reference);xn=np.linalg.norm(estimate)
    if span<=0 or rn<=0:raise ValueError('degenerate reference')
    return {'ncc':float(np.sum(estimate*reference)/(xn*rn)) if xn else 0.,'nrmse':float(np.sqrt(np.mean((estimate-reference)**2))/span)}

def observation_plot(data):
    fig,axs=plt.subplots(1,3,figsize=(15,4),layout='constrained')
    im=axs[0].imshow(data['bolometer'].T,aspect='auto',origin='lower',extent=[data['time'][0],data['time'][-1],.5,120.5])
    axs[0].set(title='Recorded RADCAM channels',xlabel='Time (s)',ylabel='One-based channel')
    fig.colorbar(im,ax=axs[0],label='Native radiance-like channel value')
    valid=data['bolometer'][:,data['good_channels'].astype(int)-1]
    im=axs[1].imshow(valid.T,aspect='auto',origin='lower',extent=[data['time'][0],data['time'][-1],0,valid.shape[1]])
    axs[1].set(title='Independently marked good channels',xlabel='Time (s)',ylabel='Good-channel index')
    fig.colorbar(im,ax=axs[1],label='Native radiance-like channel value')
    sensitivity=data['geometry'].sum(axis=0)
    im=axs[2].imshow(sensitivity,extent=[.624,1.1376,-.75,.75],aspect='auto')
    axs[2].set(title='Diagnostic spatial sensitivity',xlabel='R (m)',ylabel='Z (m)')
    fig.colorbar(im,ax=axs[2],label='Sum of effective collection volumes (m³)')
    return fig

def reconstruction_plot(fresh,reference,times,indices=(20,90,170)):
    fig,axs=plt.subplots(3,3,figsize=(10,10),layout='constrained')
    scale=max(float(fresh[list(indices)].max()),float(reference[list(indices)].max()))/1e6
    lower=min(float(fresh[list(indices)].min()),float(reference[list(indices)].min()))/1e6
    for col,i in enumerate(indices):
        for row,x in enumerate((reference[i]/1e6,fresh[i]/1e6,(fresh[i]-reference[i])/1e6)):
            opts={'vmin':lower,'vmax':scale,'cmap':'inferno'} if row<2 else {'cmap':'coolwarm'}
            im=axs[row,col].imshow(x,extent=[.624,1.1376,-.75,.75],aspect='auto',**opts)
            axs[row,col].set(xlabel='R (m)',ylabel='Z (m)',title=f'{["Upstream offline MAP","Fresh local MAP","Fresh − upstream"][row]}\nt={times[i]:.3f} s')
            fig.colorbar(im,ax=axs[row,col],label='MW/m³')
    return fig

def residual_plot(data,fresh):
    a,y,p,m=prepare(data)
    predicted=fresh.reshape(len(fresh),-1)@a.T
    residual=(predicted-y)/y.max(axis=1)[:,None]
    fig,axs=plt.subplots(1,3,figsize=(14,4),layout='constrained')
    im=axs[0].imshow(residual.T,aspect='auto',cmap='coolwarm',extent=[data['time'][0],data['time'][-1],0,a.shape[0]])
    axs[0].set(title='All good-channel residuals',xlabel='Time (s)',ylabel='Good-channel array index')
    fig.colorbar(im,ax=axs[0],label='Residual / frame maximum')
    axs[1].plot(data['time'],np.sqrt(np.mean(residual**2,axis=1)))
    axs[1].axhline(.035,color='r',ls='--',label='Frozen bound')
    axs[1].set(title='Forward consistency',xlabel='Time (s)',ylabel='Normalized RMS');axs[1].legend()
    axs[2].hist(residual.ravel(),bins=60,color='#246782')
    axs[2].set(title='Residual distribution',xlabel='Normalized residual',ylabel='Count')
    return fig

def power_plot(times,fresh,reference):
    fig,axs=plt.subplots(1,2,figsize=(12,4),layout='constrained')
    for values,label,style in [(reference,'Reproduced upstream','--'),(fresh,'Fresh local','-')]:
        axs[0].plot(times,integrate_power(values)/1e3,style,label=label)
    axs[0].set(title='Toroidally integrated power',xlabel='Time (s)',ylabel='kW');axs[0].legend()
    axs[1].plot(times,np.min(fresh,axis=(1,2))/1e6,label='Minimum')
    axs[1].plot(times,np.max(fresh,axis=(1,2))/1e6,label='Maximum')
    axs[1].set(title='Gaussian estimate is not positivity constrained',xlabel='Time (s)',ylabel='MW/m³');axs[1].legend()
    return fig

def ablation_plot(data,indices,methods):
    fig,axs=plt.subplots(2,len(methods),figsize=(4*len(methods),7),layout='constrained',squeeze=False)
    a,y,_,_=prepare(data);center=len(indices)//2
    maximum=max(x[center].max() for x in methods.values())/1e6
    minimum=min(x[center].min() for x in methods.values())/1e6
    fig.suptitle(f'Fresh comparisons; displayed map at t={data["time"][indices[center]]:.3f} s')
    rms_values=[]
    for col,(name,x) in enumerate(methods.items()):
        im=axs[0,col].imshow(x[center]/1e6,extent=[.624,1.1376,-.75,.75],vmin=minimum,vmax=maximum,cmap='inferno',aspect='auto')
        axs[0,col].set(title=name,xlabel='R (m)',ylabel='Z (m)');fig.colorbar(im,ax=axs[0,col],label='MW/m³')
        residual=(x.reshape(len(indices),-1)@a.T-y[indices])/y[indices].max(axis=1)[:,None]
        rms=np.sqrt(np.mean(residual**2,axis=1));rms_values.append(rms)
        axs[1,col].plot(data['time'][indices],rms,'o-')
        axs[1,col].set(title='Fresh selected-frame comparison',xlabel='Time (s)',ylabel='Normalized residual RMS')
    for ax in axs[1]:ax.set_ylim(0,1.05*max(v.max() for v in rms_values))
    return fig
