import numpy as np
import matplotlib.pyplot as plt
def show_observations(low,core,stages):
    fig,ax=plt.subplots(1,3,figsize=(14,3.6),layout='constrained')
    ax[0].plot(stages['low_axis'],low,color='#326c91');ax[0].set(xlim=(-8,22),xlabel='Energy loss (eV)',ylabel='Counts/channel',title='Measured zero-loss peak')
    ax[0].axvline(0,color='k',ls=':');ax[0].axvline(float(stages['threshold']),color='#bb7030',ls='--',label='Elastic integration limit');ax[0].legend(fontsize=8)
    ax[1].plot(stages['core_axis'],core,label='Measured plural spectrum');ax[1].plot(stages['core_axis'],stages['background'],label='Continuum calibration');ax[1].set(xlabel='Energy loss (eV)',ylabel='Counts/channel',title='Core-loss continuum');ax[1].legend(fontsize=8)
    n=stages['nfft'];gx=stages['low_axis'][0]+.2*np.arange(n)
    ax[2].plot(gx,stages['gaussian'],color='#bd5c42');ax[2].set(xlim=(-8,8),xlabel='Energy loss (eV)',ylabel='Gaussian density (1/eV)',title=f"Resolution FWHM={stages['fwhm']:.3f} eV")
    return fig
def show_results(stages,reference):
    e=600+.2*np.arange(844);fresh=stages['reconstruction'];ref=reference['reconstruction']
    fig,ax=plt.subplots(1,3,figsize=(14,3.7),layout='constrained')
    ax[0].plot(e,stages['edge']/.2,label='Plural counts / 0.2 eV',color='#a6a8aa');ax[0].plot(e,ref,label='Released source, corrected API',color='#bd5c42',lw=2);ax[0].plot(e,fresh,label='Fresh local extraction',color='#326c91',ls='--');ax[0].set(xlabel='Energy loss (eV)',ylabel='Counts/eV',title='Gaussian-resolved single scattering');ax[0].legend(fontsize=7)
    ax[1].plot(e,fresh-ref,color='#326c91');ax[1].set(xlabel='Energy loss (eV)',ylabel='Local minus upstream (counts/eV)',title='Independent source parity')
    ax[2].plot(e,ref,color='#bd5c42',label='Upstream');ax[2].plot(e,fresh,color='#326c91',ls='--',label='Fresh local');ax[2].set(xlim=(635,665),xlabel='Energy loss (eV)',ylabel='Counts/eV',title='Mn white-line detail');ax[2].legend(fontsize=8)
    return fig
