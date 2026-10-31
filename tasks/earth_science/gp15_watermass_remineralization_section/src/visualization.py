import numpy as np
import matplotlib.pyplot as plt
def section_panels(coordinates,values,titles,label,ncols=3):
 n=len(titles);nr=(n+ncols-1)//ncols;fig,axes=plt.subplots(nr,ncols,figsize=(4*ncols,3*nr),layout='constrained',squeeze=False)
 for j,ax in enumerate(axes.ravel()):
  if j>=n:ax.set_visible(False);continue
  im=ax.scatter(coordinates[:,0],coordinates[:,2],c=values[:,j],s=10,cmap='viridis');ax.invert_yaxis();ax.set_title(titles[j]);ax.set_xlabel('latitude [degrees north]');ax.set_ylabel('depth [m]');fig.colorbar(im,ax=ax,label=label,shrink=.7)
 return fig
