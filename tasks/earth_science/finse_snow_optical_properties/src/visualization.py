import numpy as np
import matplotlib.pyplot as plt
LABELS=['grain radius [um]','algae [cells/mL]','liquid water [volume fraction]','dark equivalent [ppb]','dust [ppb]']
def property_panels(parameters):
 fig,axes=plt.subplots(2,3,figsize=(13,7),layout='constrained')
 for j,ax in enumerate(axes.flat[:5]):ax.plot(parameters[:,j],'.',ms=3);ax.set_xlabel('field spectrum index');ax.set_ylabel(LABELS[j])
 axes.flat[5].axis('off');return fig

def selected_spectra(data,modeled,indices):
 w=295+10*np.arange(212);fig,axes=plt.subplots(2,2,figsize=(12,7),layout='constrained')
 for ax,i in zip(axes.flat,indices):
  ax.plot(data['wavelengths'],data['spectra'][i],color='gray',lw=.8,label='field HCRF');ax.plot(w,modeled[i],'k--',lw=1,label='fresh calibrated response');ax.axvspan(355,1345,color='orange',alpha=.1,label='fit window');ax.set_title(f'field spectrum {i}');ax.set_xlabel('wavelength [nm]');ax.set_ylabel('reflectance [1]');ax.legend(fontsize=8)
 return fig
