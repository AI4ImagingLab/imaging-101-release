"""Scientific comparison figures; no import-time backend changes."""
import numpy as np
import matplotlib.pyplot as plt
from .physics_model import predict
LABELS=['H','V','D','A','R','L']
def comparison_figure(counts,projectors,solutions):
    """Return figure comparing measured heatmap, density, eigenvalues and residuals."""
    fig,axes=plt.subplots(2,3,figsize=(14,8),layout='constrained')
    im=axes[0,0].imshow(counts.reshape(6,6),cmap='magma');fig.colorbar(im,ax=axes[0,0],label='Corrected counts')
    axes[0,0].set(title='Measured polarization outcomes',xticks=range(6),xticklabels=LABELS,yticks=range(6),yticklabels=LABELS,xlabel='Photon 2 analyzer',ylabel='Photon 1 analyzer')
    r,s=solutions['Full MLE']
    for ax,array,title in [(axes[0,1],r.real,'Fresh density: real'),(axes[0,2],r.imag,'Fresh density: imaginary')]:
        limit=.51 if 'real' in title else .025
        im=ax.imshow(array,cmap='RdBu_r',vmin=-limit,vmax=limit);fig.colorbar(im,ax=ax);ax.set(title=title,xticks=range(4),xticklabels=['HH','HV','VH','VV'],yticks=range(4),yticklabels=['HH','HV','VH','VV'])
    for label,(rho,scale) in solutions.items():
        pred=predict(rho,projectors,scale)
        axes[1,0].plot(pred,label=label)
        axes[1,1].plot((pred-counts)/s,'.-',label=label)
        axes[1,2].plot(np.linalg.eigvalsh(rho),'o-',label=label)
    axes[1,0].plot(counts,'k.',label='Measured');axes[1,0].set(title='Born count prediction',xlabel='Analyzer pair (row-major)',ylabel='Corrected counts')
    axes[1,1].axhline(0,color='k',lw=.5);axes[1,1].set(title='Normalized observation residual',xlabel='Analyzer pair',ylabel='(Predicted − observed) / full scale')
    axes[1,2].axhline(0,color='k',lw=.5);axes[1,2].set(title='Physicality comparison',xlabel='Ordered eigenvalue index',ylabel='Density eigenvalue')
    for ax in axes[1]: ax.legend(fontsize=8)
    return fig

def state_metrics(rho):
    """Paper characterization: tangle, base2 entropy, 4/3 linear entropy, Bell overlap.
    These characterize states; the ideal Bell vector is not measured truth.
    """
    y=np.array([[0,-1j],[1j,0]])
    spin=np.kron(y,y)
    roots=np.sort(np.sqrt(np.maximum(np.linalg.eigvals(rho@spin@rho.conj()@spin).real,0)))
    concurrence=max(0.,roots[-1]-sum(roots[:-1]))
    eigen=np.maximum(np.linalg.eigvalsh(rho),0);positive=eigen[eigen>0]
    bell=np.array([1,0,0,-1])/np.sqrt(2)
    return {'tangle':float(concurrence**2),'entropy_bits':float(-sum(positive*np.log2(positive))),'linear_entropy':float(4/3*(1-np.trace(rho@rho).real)),'ideal_Bell_overlap':float(np.real(bell@rho@bell))}
