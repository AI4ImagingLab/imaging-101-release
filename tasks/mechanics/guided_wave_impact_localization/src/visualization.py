import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
CASES=('HEI','ANG','TEM')
def localization_comparison(truth,methods,sensors,title):
 fig,axes=plt.subplots(1,3,figsize=(14,4),layout='constrained')
 for c,ax in enumerate(axes):
  ax.scatter(sensors[:,0],sensors[:,1],marker='^',c='black',label='PZT');ax.scatter(truth[c,:,0],truth[c,:,1],facecolors='none',edgecolors='black',label='controlled truth')
  for label,p in methods.items():
   ax.scatter(p[c,:,0],p[c,:,1],s=14,label=label);ax.quiver(truth[c,:,0],truth[c,:,1],p[c,:,0]-truth[c,:,0],p[c,:,1]-truth[c,:,1],angles='xy',scale_units='xy',scale=1,alpha=.35)
  ax.set(xlim=(-5,165),ylim=(-5,125),xlabel='x [mm]',ylabel='y [mm]',title=CASES[c]);ax.set_aspect('equal');ax.legend(fontsize=7)
 fig.suptitle(title);return fig

def uncertainty_ellipses(mean,cov,sensors):
 fig,axes=plt.subplots(1,3,figsize=(14,4),layout='constrained')
 for c,ax in enumerate(axes):
  for i,p in enumerate(mean[c]):
   j=35*c+i;block=cov[2*j:2*j+2,2*j:2*j+2];val,vec=np.linalg.eigh(block);angle=np.degrees(np.arctan2(vec[1,1],vec[0,1]));ax.add_patch(Ellipse(p,2*np.sqrt(max(val[1],0)),2*np.sqrt(max(val[0],0)),angle=angle,facecolor='none',edgecolor='tab:blue',alpha=.5))
  ax.scatter(mean[c,:,0],mean[c,:,1],s=8);ax.scatter(sensors[:,0],sensors[:,1],marker='^',c='black');ax.set(xlim=(-5,165),ylim=(-5,125),xlabel='x [mm]',ylabel='y [mm]',title=CASES[c]);ax.set_aspect('equal')
 fig.suptitle('Fresh predictive covariance: one-SD ellipses (no coverage claim)');return fig
