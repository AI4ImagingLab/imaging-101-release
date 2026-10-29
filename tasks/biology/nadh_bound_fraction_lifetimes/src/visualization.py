"""Review metrics and image rendering, using only fresh numerical results."""
import numpy as np
import matplotlib.pyplot as plt

def forward_residual(output,phasors,support):
    t=output['reconstruction'][0];f=output['bound_fraction'][0];w=2*np.pi*.08*np.arange(1,3)[:,None,None]
    prediction=(1-f)[None]/(1-1j*w*t[0])+f[None]/(1-1j*w*t[1])
    error=np.sqrt(np.mean(abs(prediction-phasors)**2,axis=0))
    return np.where(support,error,np.nan)

def image_grid(images,titles,limits=None,cmaps=None):
    fig,axes=plt.subplots(1,len(images),figsize=(5*len(images),4.6),layout='constrained')
    for i,(ax,image,title) in enumerate(zip(np.atleast_1d(axes),images,titles)):
        limit=(None,None) if limits is None else limits[i]
        m=ax.imshow(image,cmap='viridis' if cmaps is None else cmaps[i],vmin=limit[0],vmax=limit[1]);ax.set_title(title);ax.set_axis_off();fig.colorbar(m,ax=ax,shrink=.75)
    return fig
