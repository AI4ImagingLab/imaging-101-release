"""Spatial comparison figures and independently tested numerical summaries."""
import numpy as np
import matplotlib.pyplot as plt
def comparison_metrics(a,b,mask):
    a=a[mask];b=b[mask]
    n=np.linalg.norm(a)*np.linalg.norm(b)
    return {'rmse':float(np.sqrt(np.mean((a-b)**2))),'ncc':float(a@b/n) if n else 0.}
def image_grid(images,titles,limits=None,cmap='viridis',z=48):
    fig,axes=plt.subplots(1,len(images),figsize=(5*len(images),4.6),layout='constrained',squeeze=False)
    for i,(a,title) in enumerate(zip(images,titles)):
        lim=limits[i] if limits else (None,None)
        im=axes[0,i].imshow(a[:,:,z].T,origin='lower',cmap=cmap,vmin=lim[0],vmax=lim[1])
        axes[0,i].set(title=title,xlabel='x voxel (1.5 mm)',ylabel='y voxel (1.5 mm)');fig.colorbar(im,ax=axes[0,i],shrink=.8)
    return fig
