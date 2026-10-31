"""Review plots and transparent metric calculations; no backend mutation."""
import numpy as np
import matplotlib.pyplot as plt
from .preprocessing import CONDITIONS

def rmse(a,b):
    return float(np.sqrt(np.mean((np.asarray(a)-np.asarray(b))**2)))

def field_grid(fields,titles,limits,cmap='viridis',slice_index=4):
    """fields list of (4,64,64,10), one row per field, columns conditions."""
    fig,axes=plt.subplots(len(fields),4,figsize=(13,3*len(fields)),squeeze=False,layout='constrained')
    for row,(values,title,lim) in enumerate(zip(fields,titles,limits)):
        for c in range(4):
            ax=axes[row,c];im=ax.imshow(values[c,:,:,slice_index].T,origin='lower',vmin=lim[0],vmax=lim[1],cmap=cmap)
            ax.set_title(CONDITIONS[c]+'\n'+title);ax.set_xlabel('x voxel');ax.set_ylabel('y voxel')
            fig.colorbar(im,ax=ax,shrink=.75)
    return fig
