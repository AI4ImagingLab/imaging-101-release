import numpy as np
import matplotlib.pyplot as plt

def display_waterfall(array,good,factor=64):
 a=np.array(array,copy=True);a[~good]=np.nan
 with np.errstate(invalid='ignore'):
  count=np.sum(np.isfinite(a.reshape(-1,factor,a.shape[1])),axis=1);total=np.nansum(a.reshape(-1,factor,a.shape[1]),axis=1)
  return np.divide(total,count,out=np.full_like(total,np.nan),where=count>0)

def model_comparison(data,model):
 good=data['good_channels'];arrays=[data['measurements'],model,data['measurements']-model];fig,axes=plt.subplots(1,3,figsize=(15,4),layout='constrained')
 for ax,a,label in zip(axes,arrays,['measured','Gaussian model','measured minus model']):
  b=display_waterfall(a,good);im=ax.imshow(b,aspect='auto',origin='lower',extent=[data['times'][0]*1000,data['times'][-1]*1000,data['frequencies'][0],data['frequencies'][-1]],cmap='RdBu_r');ax.set_xlabel('relative time [ms]');ax.set_ylabel('frequency [MHz]');ax.set_title(label+'; 64-channel mean for display');fig.colorbar(im,ax=ax,label='intensity [U]')
 return fig
