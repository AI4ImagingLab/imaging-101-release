"""Report figures from freshly computed optical operators and acquisitions."""
import numpy as np
import matplotlib.pyplot as plt
LABELS=['I','Q','U','V']

def matrix_comparison(truth,original,fresh):
    fig,ax=plt.subplots(1,4,figsize=(16,3.8))
    for a,m,title in zip(ax,[truth,original,fresh,fresh-truth],['Actual author-synthetic sample','Reproduced original 24-angle','Fresh local 24-angle','Fresh - synthetic truth']):
        difference=title.startswith('Fresh -');limit=max(np.max(abs(m)),1e-18) if difference else max(np.max(abs(truth)),1e-18)
        image=a.imshow(m,vmin=-limit,vmax=limit,cmap='coolwarm')
        for i in range(4):
            for j in range(4):a.text(j,i,f'{m[i,j]:.1e}' if difference else f'{m[i,j]:.3f}',ha='center',va='center',fontsize=8,color='white' if abs(m[i,j])>.55*limit else 'black')
        a.set_xticks(range(4),LABELS);a.set_yticks(range(4),LABELS);a.set(xlabel='Input Stokes component',ylabel='Output component',title=title);fig.colorbar(image,ax=a,fraction=.046,label='Dimensionless')
    fig.tight_layout();return fig

def scan_comparison(angles16,power16,angles24,power24,dense,truth_curve,recovered16,recovered24):
    fig,ax=plt.subplots(1,2,figsize=(13,4))
    ax[0].plot(np.rad2deg(dense),truth_curve,'k-',label='Synthetic truth continuous forward')
    ax[0].plot(np.rad2deg(angles16),power16,'x',label='Original 16-angle observations')
    ax[0].plot(np.rad2deg(angles24),power24,'o',ms=4,fillstyle='none',label='Bound 24-angle observations')
    ax[0].set(xlabel='Generator angle (deg)',ylabel='Power / incident intensity',title='Both complete native acquisition conditions');ax[0].legend(fontsize=8)
    ax[1].plot(np.rad2deg(dense),recovered16-truth_curve,label='Fresh 16-angle minimum-norm inverse')
    ax[1].plot(np.rad2deg(dense),recovered24-truth_curve,label='Fresh 24-angle inverse')
    ax[1].set(xlabel='Generator angle (deg)',ylabel='Forward curve - true curve',title='Aliasing between acquired angles');ax[1].legend(fontsize=8)
    fig.tight_layout();return fig

def rank_comparison(design16,design24,s16,s24):
    fig,ax=plt.subplots(1,2,figsize=(13,4))
    ax[0].semilogy(np.arange(1,len(s16)+1),s16/s16[0],'o-',label='Native 16-angle design')
    ax[0].semilogy(np.arange(1,len(s24)+1),s24/s24[0],'s-',label='Bound 24-angle design');ax[0].axhline(1e-15,color='k',ls='--',label='Native inverse cutoff')
    ax[0].set(xlabel='Singular-value index',ylabel='Normalized singular value',title='12 versus 16 identifiable directions');ax[0].legend(fontsize=8)
    m=ax[1].imshow(design24,aspect='auto',cmap='coolwarm');ax[1].set(xlabel='Row-major Mueller element',ylabel='Acquired angle index',title='Calibrated 24×16 physical design');fig.colorbar(m,ax=ax[1],label='Relative intensity per unit matrix entry')
    fig.tight_layout();return fig
