"""Maintainer comparisons, with no backend changes on import."""
import matplotlib.pyplot as plt
from .solvers import OUTPUT_KEYS

def compare_maps(fresh, upstream):
    """Return figure: upstream, fresh and signed difference for all four maps."""
    fig, axes = plt.subplots(4,3,figsize=(11,12),constrained_layout=True)
    for i,(key,unit) in enumerate(zip(OUTPUT_KEYS,('K','dimensionless','km/s','km/s'))):
        a=upstream[key][0]; b=fresh[key][0]
        for j,value in enumerate((a,b,b-a)):
            image=axes[i,j].imshow(value,origin='lower',cmap='coolwarm' if j==2 else 'viridis',
                                  **({} if j==2 else {'vmin':min(a.min(),b.min()),'vmax':max(a.max(),b.max())}))
            axes[i,j].set_title(key+' ['+unit+'] / '+('upstream','fresh','difference')[j])
            fig.colorbar(image,ax=axes[i,j],shrink=.75)
    return fig
