import numpy as np
import matplotlib.pyplot as plt
def metrics(a,b):
    x=np.asarray(a).ravel();y=np.asarray(b).ravel()
    return dict(ncc=float(x@y/np.linalg.norm(x)/np.linalg.norm(y)),nrmse=float(np.sqrt(np.mean((x-y)**2))/np.ptp(y)))
def plot_maps(raw,fields,reference,radii):
    fig,ax=plt.subplots(2,3,figsize=(13,7),layout='constrained')
    items=[raw,fields['radial'][0],fields['radial'][17],fields['radial'][-1],reference['radial'][-1],fields['radial'][-1]-reference['radial'][-1]]
    titles=['Measured photosphere','Fresh flux-balanced boundary',f'Fresh radial field at r={radii[17]:.3f} R☉','Fresh source-surface radial field','Independent original source-surface field','Fresh minus original at source surface']
    extent=[22,382,-1,1]
    for i,(a,title)in enumerate(zip(items,titles)):
        limit=np.max(np.abs(a)) if i==5 else np.percentile(np.abs(a),99)
        limit=max(limit,1e-16)
        im=ax.flat[i].imshow(a,origin='lower',aspect='auto',extent=extent,cmap='RdBu_r',vmin=-limit,vmax=limit)
        ax.flat[i].set_title(title+('\n99% display range' if i!=5 else '\nFull numerical residual range'),fontsize=10)
        ax.flat[i].set(xlabel='Carrington longitude (degree)',ylabel='sin(latitude)');fig.colorbar(im,ax=ax.flat[i],label='Magnetic field (G)')
    fig.suptitle('NSO/AURA/NSF — native GONG magnetic observations and coronal continuation')
    return fig
