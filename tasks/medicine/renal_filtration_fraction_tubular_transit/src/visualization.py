"""Numerical review metrics and plots; no backend mutation."""
import numpy as np
import matplotlib.pyplot as plt
def clinical_statistics(mri,isotope):
    """Inputs mL/min; independent proxy comparison, no bias correction."""
    difference=np.asarray(mri)-np.asarray(isotope)
    return {'bias':float(np.mean(difference)),'sd':float(np.std(difference)),
            'rmse':float(np.sqrt(np.mean(difference**2))),
            'p30':float(np.mean(np.abs(difference)<=.3*np.abs(isotope)))}
def bland_altman(mri,isotope,groups):
    fig,axes=plt.subplots(1,2,figsize=(11,4),layout='constrained')
    for field in [1.,3.]:
        m=np.asarray(groups)==field;x=np.asarray(isotope)[m];y=np.asarray(mri)[m]
        axes[0].scatter(x,y,s=12,alpha=.7,label=f'{field:g} T')
        axes[1].scatter((x+y)/2,y-x,s=12,alpha=.7,label=f'{field:g} T')
    limit=max(np.max(mri),np.max(isotope));axes[0].plot([0,limit],[0,limit],'k--')
    stats=clinical_statistics(mri,isotope)
    for value in [stats['bias'],stats['bias']-1.96*stats['sd'],stats['bias']+1.96*stats['sd']]:axes[1].axhline(value,color='black',ls='--')
    axes[0].set(xlabel='Isotope GFR (mL/min)',ylabel='MRI-derived GFR (mL/min)')
    axes[1].set(xlabel='Mean of methods (mL/min)',ylabel='MRI minus isotope (mL/min)')
    axes[0].legend();return fig
