"""Maintainer comparisons."""
import numpy as np
import matplotlib.pyplot as plt
from .physics_model import field_coordinates

def compare(a,b,reconstruction,reference,confidence):
    x,y=field_coordinates()
    fig,axs=plt.subplots(2,3,figsize=(13,8),constrained_layout=True)
    for ax,img,title in zip(axs[0,:2],(a,b),('Frame A','Frame B')):
        ax.imshow(img,cmap='gray');ax.set_title(title)
    ax=axs[0,2];ax.imshow(a,cmap='gray');ax.quiver(x,y,reconstruction[0,...,0],reconstruction[0,...,1],color='red',angles='xy',scale_units='xy',scale=0.3);ax.set_title('Fresh velocity: pixel / interval')
    for ax,img,title in zip(axs[1],(reference[0,...,1],reconstruction[0,...,1],confidence[0]),('Upstream vertical velocity','Fresh vertical velocity','Peak confidence')):
        im=ax.imshow(img);fig.colorbar(im,ax=ax);ax.set_title(title)
    return fig
