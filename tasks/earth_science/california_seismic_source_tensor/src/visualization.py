"""Review figures and independently testable physical matrix diagnostics."""
import numpy as np
import matplotlib.pyplot as plt

def tensor_relative_error(estimate,reference):
    return float(np.linalg.norm(np.asarray(estimate)-reference)/np.linalg.norm(reference))

def radiation_pattern(tensor,n=161):
    """Lower-hemisphere equal-area P-wave radiation n^T M n, normalized."""
    x,y=np.meshgrid(np.linspace(-1,1,n),np.linspace(-1,1,n))
    radius=np.sqrt(x*x+y*y);theta=2*np.arcsin(np.clip(radius/np.sqrt(2),0,1))
    az=np.arctan2(x,y)
    ray=np.stack([np.sin(theta)*np.cos(az),np.sin(theta)*np.sin(az),np.cos(theta)],axis=-1)
    rad=np.einsum('...i,ij,...j->...',ray,tensor,ray)/np.linalg.norm(tensor)
    return np.ma.array(rad,mask=radius>1)

def plot_tensor_and_radiation(tensor,title='Fresh full tensor'):
    fig,axs=plt.subplots(1,2,figsize=(9,4),constrained_layout=True)
    im=axs[0].imshow(tensor/1e15,cmap='RdBu_r');fig.colorbar(im,ax=axs[0],label='Moment[1e15 Nm]')
    axs[0].set_xticks(range(3),['N','E','D']);axs[0].set_yticks(range(3),['N','E','D']);axs[0].set_title(title)
    im=axs[1].imshow(radiation_pattern(tensor),origin='lower',extent=(-1,1,-1,1),cmap='RdBu_r',vmin=-1,vmax=1)
    axs[1].set_title('Lower hemisphere P-wave radiation');axs[1].set_axis_off()
    return fig

def plot_waveforms(observed,predicted):
    fig,axs=plt.subplots(8,3,figsize=(12,14),constrained_layout=True)
    time=np.arange(observed.shape[-1])
    for i in range(8):
        for j in range(3):
            ax=axs[i,j];ax.plot(time,observed[i,j]*1e6,color='k',lw=1,label='Measured')
            ax.plot(time,predicted[i,j]*1e6,color='C3',lw=1,label='Fresh prediction')
            if i==0:ax.set_title('ZRT'[j]);ax.legend(fontsize=7)
            if j==0:ax.set_ylabel(f'Station{i+1} [1e-6 cm]')
            if i==7:ax.set_xlabel('Window time[s]')
    return fig
