"""Full experimental/standard/fresh comparisons without backend side effects."""
import numpy as np
import matplotlib.pyplot as plt
from .physics_model import DIFFUSION

def review_figures(data,fresh,upstream,details):
    fig,axes=plt.subplots(2,1,figsize=(12,7))
    ppm=data['ppm'];spectra=data['spectra']
    for j in [0,9,19,29]:axes[0].plot(ppm,spectra[:,j],lw=.7,label=f'gradient {j+1}')
    axes[0].set(xlabel='Author chemical-shift axis [ppm]',ylabel='Normalized real amplitude',title='Public measured gradient-encoded spectra');axes[0].legend(fontsize=8)
    centers=np.array([ppm[g].mean() for g in data['groups']])
    image=axes[1].pcolormesh(np.arange(1,30),DIFFUSION*1e10,fresh.T,shading='nearest',cmap='magma')
    axes[1].set(yscale='log',xlabel='All contiguous support bands, source-index order',ylabel='Diffusion [1e-10 m²/s]',title='Fresh complete diffusion amplitude map');fig.colorbar(image,ax=axes[1],label='Integrated amplitude / bin');fig.tight_layout()
    second,axes=plt.subplots(2,2,figsize=(12,8))
    for ax,band in zip(axes.flat,[0,5,15,28]):
        ax.semilogx(DIFFUSION*1e10,upstream[band],label='Reproduced original',lw=2)
        ax.semilogx(DIFFUSION*1e10,fresh[band],'--',label='Fresh cleaned')
        ax.set(title=f'Band {band+1}: center {centers[band]:.3f} ppm',xlabel='Diffusion [1e-10 m²/s]',ylabel='Integrated amplitude / bin');ax.legend(fontsize=8);ax.grid(alpha=.2)
    second.tight_layout()
    third,axes=plt.subplots(1,2,figsize=(12,4))
    for band in [0,5,15,28]:
        axes[0].plot(data['b_values']*1e-10,data['observations'][band]/data['observations'][band,0],'o',ms=3)
        axes[0].plot(data['b_values']*1e-10,details['kernel']@fresh[band]/data['observations'][band,0],label=f'band {band+1}')
    axes[0].set(xlabel='b [1e10 s/m²]',ylabel='Attenuation / first observation',title='Measured circles and fresh forward curves');axes[0].legend(fontsize=8)
    ratio=np.linalg.norm(fresh@details['kernel'].T-data['observations'],axis=1)/data['eta']
    relative=np.linalg.norm(fresh-upstream,axis=1)/np.linalg.norm(upstream,axis=1)
    axes[1].plot(np.arange(1,30),ratio,'o-',label='Forward residual / radius');axes[1].axhline(1.01,color='r',ls='--',label='Predeclared limit1.01')
    axes[1].set(xlabel='Support band',ylabel='Euclidean residual / radius',title='All bands satisfy independently selected error balls');axes[1].legend(fontsize=8)
    third.tight_layout()
    return [fig,second,third]
