"""Scientific comparison figures; importing this module preserves the backend."""
import numpy as np
import matplotlib.pyplot as plt

def review_figures(observed,weights,fresh,upstream,indices):
    spectrum=fresh['resonances'][0];reference=upstream['resonances'][0]
    fig,axes=plt.subplots(2,1,figsize=(11,7),gridspec_kw={'height_ratios':[2,1]})
    modes=np.arange(1,77)
    axes[0].plot(modes,spectrum*1e3,'o',mfc='none',ms=5,label='Fresh calculated all modes')
    axes[0].plot(indices+1,observed*1e3,'x',label='Measured entries, inferred correspondence')
    missing=np.setdiff1d(np.arange(76),indices)
    axes[0].plot(missing+1,spectrum[missing]*1e3,'s',color='tab:red',label='Unobserved model modes')
    axes[0].set(ylabel='Frequency [kHz]',title='Full cylinder spectrum and incomplete measured peak list');axes[0].legend(fontsize=9)
    residual=100*(observed/spectrum[indices]-1)
    axes[1].scatter(indices+1,residual,c=np.where(weights>0,'tab:blue','tab:red'),s=24)
    axes[1].axhline(0,color='k',lw=.7);axes[1].set(xlabel='Inferred ordered mode index (one based)',ylabel='Observed / model - 1 [%]',title='All source entries; red points have author zero fit weight')
    for ax in axes:ax.grid(alpha=.2)
    fig.tight_layout()
    second,axes=plt.subplots(1,2,figsize=(11,4))
    positions=np.arange(2)
    axes[0].bar(positions-.17,upstream['reconstruction'][0],.34,label='Reproduced original (printed)')
    axes[0].bar(positions+.17,fresh['reconstruction'][0],.34,label='Fresh cleaned')
    axes[0].set(xticks=positions,xticklabels=['c11','c44'],ylabel='Elastic modulus [GPa]',title='Method reference versus fresh estimate');axes[0].legend(fontsize=8)
    axes[1].plot(modes,(spectrum-reference)*1e6,'o-',ms=3)
    axes[1].axhline(0,color='k',lw=.7);axes[1].set(xlabel='Ordered nonrigid mode',ylabel='Fresh minus original [Hz]',title='Native output rounding limits this comparison')
    second.tight_layout()
    return [fig,second]
