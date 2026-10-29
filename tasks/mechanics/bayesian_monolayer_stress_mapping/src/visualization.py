"""Scientific stress/traction views; no backend mutation or reconstruction library."""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
def image(ax,value,title,unit="kPa um",limit=None,cmap="RdBu_r"):
    if limit is None:limit=np.max(np.abs(value))
    artist=ax.imshow(value,extent=[0,100,0,100],origin="lower",cmap=cmap,vmin=-limit if cmap=="RdBu_r" else 0,vmax=limit)
    ax.set(title=title,xlabel="x (um)",ylabel="y (um)")
    plt.colorbar(artist,ax=ax,shrink=.8,label=unit);return artist
def principal(ax,stress,title,stride=3,scale=2.5):
    tensor=np.zeros((*stress.shape[1:],2,2));tensor[...,0,0]=stress[0];tensor[...,1,1]=stress[1];tensor[...,0,1]=tensor[...,1,0]=stress[2]
    values,vectors=np.linalg.eigh(tensor)
    x,y=np.meshgrid(np.arange(1,100,2),np.arange(1,100,2));segments=[];colors=[]
    for i in range(0,50,stride):
        for j in range(0,50,stride):
            point=np.array([x[i,j],y[i,j]])
            for k in range(2):
                displacement=scale*values[i,j,k]*vectors[i,j,:,k]
                segments.append([point-displacement,point+displacement]);colors.append("#2166ac" if values[i,j,k]>=0 else "#b2182b")
    ax.add_collection(LineCollection(segments,colors=colors,linewidths=.8));ax.set(xlim=(0,100),ylim=(0,100),aspect="equal",title=title,xlabel="x (um)",ylabel="y (um)")
    ax.plot([],[],color="#2166ac",label="Tensile");ax.plot([],[],color="#b2182b",label="Compressive");ax.legend(fontsize=8)
def export(fig,directory,name,**source):
    from pathlib import Path
    folder=Path(directory);folder.mkdir(parents=True,exist_ok=True)
    fig.tight_layout()
    for suffix in ["png","svg","pdf"]:fig.savefig(folder/(name+"."+suffix),dpi=180,bbox_inches="tight")
    np.savez_compressed(folder/(name+"_source.npz"),**source)
