"""Maintainer comparisons; importing this module never changes plotting backend."""
import numpy as np
import matplotlib.pyplot as plt

def compare(images, upstream, fresh):
    """Return a 4-by-4 observation/original/local/residual scientific comparison."""
    fig,axes=plt.subplots(4,4,figsize=(11,10),constrained_layout=True)
    for i in range(4):
        for j,stack in enumerate((images,upstream,fresh,images-fresh)):
            v=stack[i]
            artist=axes[i,j].imshow(v,origin='lower',cmap='viridis' if j<3 else 'coolwarm',
                                     vmin=images[i].min() if j<3 else -np.max(np.abs(v)),
                                     vmax=images[i].max() if j<3 else np.max(np.abs(v)))
            axes[i,j].set_title(f'Source {i+1}: '+('observed','upstream fit','fresh fit','data - fresh')[j])
            fig.colorbar(artist,ax=axes[i,j],shrink=.65,label='MJy/sr')
    return fig
