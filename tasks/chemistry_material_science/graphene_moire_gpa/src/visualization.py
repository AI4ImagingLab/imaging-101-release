import numpy as np
import matplotlib.pyplot as plt
from .preprocessing import NM_PER_PIXEL

def agreement(x,r):
    a=x.ravel();b=r.ravel();n=np.linalg.norm(a)
    return float(np.dot(a,b)/(n*np.linalg.norm(b))) if n else 0.,float(np.sqrt(np.mean((a-b)**2))/np.ptp(b))

def panel(ax,a,title,cmap='RdBu_r',vmin=None,vmax=None):
    h=ax.imshow(a,extent=[-.5*NM_PER_PIXEL,(a.shape[1]-.5)*NM_PER_PIXEL,(a.shape[0]-.5)*NM_PER_PIXEL,-.5*NM_PER_PIXEL],cmap=cmap,vmin=vmin,vmax=vmax)
    ax.set(xlabel='Column coordinate (nm)',ylabel='Row coordinate (nm)',title=title,xlim=(0,(a.shape[1]-1)*NM_PER_PIXEL),ylim=((a.shape[0]-1)*NM_PER_PIXEL,0));return h

def acquisition_figure(data):
    im=data['image'];f=np.abs(np.fft.fftshift(np.fft.fft2(im-im.mean())))
    fig,axs=plt.subplots(1,2,figsize=(9,4),layout='constrained')
    h=panel(axs[0],im,'a  Measured LEEM contrast','gray');fig.colorbar(h,ax=axs[0],label='Detector counts')
    h=axs[1].imshow(np.log10(np.maximum(f,1)),extent=[-.5-.5/im.shape[1],.5-.5/im.shape[1],.5-.5/im.shape[0],-.5-.5/im.shape[0]],cmap='magma')
    k=data['reference_vectors'];axs[1].scatter(k[:,1],k[:,0],facecolors='none',edgecolors='#56B4E9',s=65)
    axs[1].set(xlim=(-.09,.09),ylim=(.09,-.09),xlabel='Column frequency (cycle/pixel)',ylabel='Row frequency (cycle/pixel)',title='b  Image-defined reference carriers')
    fig.colorbar(h,ax=axs[1],label='log10 Fourier amplitude (count)');return fig

def comparison_figure(fresh,reference):
    fig,axs=plt.subplots(2,3,figsize=(10,6.3),layout='constrained');v=float(np.max(abs(reference)))
    for i in range(2):
        h=panel(axs[i,0],reference[i],f'Upstream component {i}','RdBu_r',-v,v);panel(axs[i,1],fresh[i],f'Fresh component {i}','RdBu_r',-v,v)
        d=fresh[i]-reference[i];lim=max(float(abs(d).max()),1e-14);hd=panel(axs[i,2],d,f'Fresh minus upstream {i}','RdBu_r',-lim,lim);fig.colorbar(hd,ax=axs[i,2],label='Difference (nm)')
    fig.colorbar(h,ax=list(axs[:,:2].flat),label='Inverse moire displacement (nm)');return fig

def sensitivity_figure(fields,reference):
    fig,axs=plt.subplots(1,len(fields),figsize=(11,3.7),layout='constrained');v=np.max(abs(reference[0]))
    for ax,(name,u) in zip(axs,fields.items()):h=panel(ax,u[0],name,'RdBu_r',-v,v)
    fig.colorbar(h,ax=list(axs),label='Row displacement (nm)');return fig
