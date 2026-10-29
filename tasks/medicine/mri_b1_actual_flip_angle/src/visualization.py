"""Maintainer comparison plots; imports no domain reconstruction tools."""
import numpy as np
from matplotlib import pyplot as plt
UNITS={'reconstruction': 'actual / nominal angle (dimensionless)'}
def metrics(estimate,reference):
    """Uncentered cosine NCC and reference dynamic-range NRMSE."""
    a=np.asarray(estimate).ravel();b=np.asarray(reference).ravel()
    return float(a@b/(np.linalg.norm(a)*np.linalg.norm(b))),float(np.sqrt(np.mean((a-b)**2))/np.ptp(b))
def comparison(fresh,reference,mask=None):
    """Return a figure with reference, fresh and signed-difference panels per key."""
    keys=list(fresh);fig,axes=plt.subplots(len(keys),3,figsize=(12,4*len(keys)),squeeze=False,layout='constrained')
    for row,k in enumerate(keys):
        ref=reference[k][0,:,:,16];new=fresh[k][0,:,:,16];limit=np.max(np.abs(new-ref))
        for col,(a,title) in enumerate([(ref,'Reproduced upstream'),(new,'Fresh local'),(new-ref,'Fresh minus upstream')]):
            im=axes[row,col].imshow(a,cmap='coolwarm' if col==2 else 'viridis',vmin=-limit if col==2 and limit>0 else None,vmax=limit if col==2 and limit>0 else None)
            axes[row,col].set_title(k+'\n'+title);fig.colorbar(im,ax=axes[row,col],shrink=.8,label=UNITS[k])
    return fig
