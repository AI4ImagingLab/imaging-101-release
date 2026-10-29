"""Measured-arc and fresh inversion review figures."""
import numpy as np
import matplotlib.pyplot as plt
from .physics_model import spectrum,forward
def examples(data,result):
    fig,axs=plt.subplots(2,3,figsize=(13,7),layout='constrained');offsets=np.r_[0,np.cumsum(data['counts'])]
    for j,i in enumerate([0,110,220]):
        sl=slice(offsets[i],offsets[i+1]);e=data['elevation'][sl];y=data['signal'][sl];x=np.sin(np.deg2rad(e))/data['half_wavelength'][i];height=result[0,i]
        mat=np.column_stack((np.cos(2*np.pi*height*x),np.sin(2*np.pi*height*x)));coeff=np.linalg.lstsq(mat,y,rcond=None)[0]
        axs[0,j].plot(e,y,'.',label='Measured detrended SNR');axs[0,j].plot(e,mat@coeff,label='Fresh harmonic fit');axs[0,j].set(title=f'Arc{i}, height {height:.3f}m',xlabel='Corrected elevation (degree)',ylabel='Relative voltage');axs[0,j].legend(fontsize=7)
        h,p=spectrum(x,y);axs[1,j].plot(h,p);axs[1,j].axvline(height,color='r',ls='--');axs[1,j].set(xlabel='Height (m)',ylabel='Harmonic amplitude',title='Fresh explicit least-squares spectrum')
    return fig
def summary(data,result,original,fast):
    fig,axs=plt.subplots(1,3,figsize=(13,4),layout='constrained');meta=data['arc_metadata']
    for freq in [1,20,5]:
        mask=meta[:,2]==freq;axs[0].scatter(meta[mask,0]+meta[mask,4]/24,result[0,mask],s=10,label=str(freq))
    axs[0].set(xlabel='Day of year + UTC fraction',ylabel='Height (m)',title='All324measured arcs');axs[0].legend(title='Frequency code')
    axs[1].plot(result[0]-original[0]);axs[1].set(xlabel='Fixed arc index',ylabel='Difference (m)',title='Fresh minus original direct LS')
    axs[2].hist((fast-original).ravel(),bins=25);axs[2].set(xlabel='Fast NFFT minus direct LS (m)',ylabel='Arc count',title='Independent original branch comparison')
    return fig
