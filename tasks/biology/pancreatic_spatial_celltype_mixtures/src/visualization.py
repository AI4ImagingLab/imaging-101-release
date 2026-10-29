"""Pure plotting helpers; no backend selection or reconstruction."""
import numpy as np
import textwrap
import matplotlib.pyplot as plt

def spatial_panel(ax,coordinates,values,title,vmin=0,vmax=None,cmap="viridis",size=28):
    image=ax.scatter(coordinates[:,0],coordinates[:,1],c=values,s=size,cmap=cmap,vmin=vmin,vmax=vmax,marker="s",linewidths=0,rasterized=True)
    ax.set_aspect("equal");ax.set_title(title,fontsize=9);ax.set_xlabel("native spot x [grid units]");ax.set_ylabel("native spot y [grid units]")
    return image

def field_atlas(coordinates,fields,names,title):
    fig,axes=plt.subplots(4,5,figsize=(18,13),layout="constrained")
    for k,ax in enumerate(axes.flat):
        im=spatial_panel(ax,coordinates,fields[:,k],"\n".join(textwrap.wrap(names[k].replace("_"," "),width=24)),vmax=float(fields[:,k].max()))
        fig.colorbar(im,ax=ax,shrink=.72,label="conditional fraction")
    fig.suptitle(title,fontsize=15)
    return fig
