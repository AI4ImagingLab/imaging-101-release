import numpy as np
import matplotlib.pyplot as plt

def local_coordinates(q,reference):
 phi,lam=reference;return np.column_stack([(q[:,1]-lam)*111.1949266*np.cos(phi/(180/3.1415927)),(q[:,0]-phi)*111.1949266])

def catalog_maps(data,catalogs,labels,title):
 ids=np.flatnonzero(labels==np.argmax(np.bincount(labels)));reference=np.mean(data['initial_catalog'][ids,:2],axis=0);fig,axes=plt.subplots(1,len(catalogs),figsize=(5*len(catalogs),4),layout='constrained');axes=np.atleast_1d(axes);allxy=[]
 for ax,(name,q) in zip(axes,catalogs.items()):
  xy=local_coordinates(q[ids],reference);allxy.append(xy);a=ax.scatter(xy[:,0],xy[:,1],c=q[ids,2],s=7,cmap='viridis_r',vmin=0,vmax=20);ax.set(xlabel='east [km]',ylabel='north [km]',title=name);ax.set_aspect('equal');fig.colorbar(a,ax=ax,label='depth [km]')
 combined=np.concatenate(allxy);lo=combined.min(0)-.2;hi=combined.max(0)+.2
 for ax in axes:ax.set_xlim(lo[0],hi[0]);ax.set_ylim(lo[1],hi[1])
 fig.suptitle(title);return fig
