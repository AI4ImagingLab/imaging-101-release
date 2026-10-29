"""Maintainer review figures; metrics use the public mathematical conventions."""
import numpy as np
import matplotlib.pyplot as plt

METHOD_LABELS={'radial':'Radial symmetry','wa':'Weighted centroid','integer':'Integer peak'}

def metrics(estimate,reference):
    """Cosine NCC and RMS error / reference dynamic range, no alignment."""
    a=np.asarray(estimate,dtype=float);b=np.asarray(reference,dtype=float)
    if a.shape!=b.shape or not np.isfinite(a).all() or not np.isfinite(b).all():raise ValueError('Invalid metric arrays')
    norm=np.linalg.norm(a)*np.linalg.norm(b);dynamic=np.ptp(b)
    if norm==0 or dynamic==0:raise ValueError('Undefined NCC/NRMSE')
    return {'ncc':float(np.dot(a.ravel(),b.ravel())/norm),'nrmse':float(np.sqrt(np.mean((a-b)**2))/dynamic)}

def plot_acquisition(iq):
    """Magnitude snapshots and temporal signal from actual bound observations."""
    fig,ax=plt.subplots(1,3,figsize=(15,4),constrained_layout=True)
    for col,ib in enumerate((0,14,29)):
        value=np.abs(iq[ib,:,:,99]);im=ax[col].imshow(20*np.log10(np.maximum(value/value.max(),1e-4)),vmin=-50,vmax=0,cmap='gray')
        ax[col].set(title=f'Actual IQ magnitude: block {ib+1}, frame 100',xlabel='Lateral acquisition pixel',ylabel='Axial acquisition pixel')
    fig.colorbar(im,ax=ax,label='Relative magnitude (dB)');return fig

def plot_comparisons(fresh,upstream):
    """Preserve all three localization methods and both required quantities."""
    figures=[]
    for key,label in [('density','Track crossings per superpixel'),('speed','Mean track speed (acquisition pixel/s)')]:
        vmax=max(upstream[key][:,:,i].max() for i in range(3))
        fig,axes=plt.subplots(3,3,figsize=(15,11),constrained_layout=True)
        for row,method in enumerate(('radial','wa','integer')):
            ref=upstream[key][:,:,row];value=fresh[method][key]
            values=[ref,value,value-ref]
            for col,a in enumerate(values):
                if col<2:
                    display=np.sqrt(a) if key=='density' else a
                    im=axes[row,col].imshow(display,cmap='inferno' if key=='density' else 'viridis',vmin=0,vmax=np.sqrt(vmax) if key=='density' else vmax)
                    fig.colorbar(im,ax=axes[row,col],shrink=.7,label='sqrt(crossing count)' if key=='density' else label)
                else:
                    bound=max(float(np.max(np.abs(a))),1e-10);im=axes[row,col].imshow(a,cmap='coolwarm',vmin=-bound,vmax=bound)
                    fig.colorbar(im,ax=axes[row,col],shrink=.7,label='Fresh minus upstream')
                axes[row,col].set(title=f'{METHOD_LABELS[method]} | '+['Upstream double','Fresh local double','Difference'][col],xlabel='Lateral superpixel',ylabel='Axial superpixel')
        figures.append(fig)
    return figures

def plot_sampling(fresh,upstream):
    """Localization/track totals and a density profile expose method differences."""
    fig,axes=plt.subplots(2,2,figsize=(13,9),constrained_layout=True)
    ax=axes.ravel()
    x=np.arange(3);methods=('radial','wa','integer')
    for j,(index,label) in enumerate([(0,'Accepted localizations'),(1,'Retained tracks')]):
        old=upstream['counts'][:,:,index].sum(axis=0);new=np.array([fresh[m]['counts'][:,index].sum() for m in methods])
        ax[j].bar(x-.18,old,.36,label='Upstream double');ax[j].bar(x+.18,new,.36,label='Fresh local double')
        ax[j].set(xticks=x,xticklabels=['Radial','Centroid','Integer'],ylabel=label);ax[j].legend()
    for m in methods:ax[2].plot(fresh[m]['density'].sum(axis=0),label=METHOD_LABELS[m])
    ax[2].set(xlabel='Lateral superpixel',ylabel='Axially summed track crossings');ax[2].legend()
    for index,m in enumerate(methods):
        ax[3].plot(np.arange(1,31),upstream['saturation'][:,index],linestyle='--',alpha=.6)
        ax[3].plot(np.arange(1,31),fresh[m]['saturation'],label=METHOD_LABELS[m])
    ax[3].set(xlabel='Complete acquisition blocks accumulated',ylabel='Occupied superpixels',title='Vascular sampling: dashed upstream, solid fresh')
    ax[3].legend()
    return fig
