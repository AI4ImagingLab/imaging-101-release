import numpy as np
import matplotlib.pyplot as plt
def summary_figure(data,details,upstream):
    """Return actual measured,fit,residual and original/fresh SO2 comparison panels."""
    t=data['elapsed_seconds']/60;x=details['wavelength'];y=details['corrected'];f=details['fit'];p=details['parameters']
    fig,axes=plt.subplots(2,3,figsize=(15,8),layout='constrained')
    im=axes[0,0].pcolormesh(x,t,y,shading='auto',cmap='viridis');fig.colorbar(im,ax=axes[0,0],label='Corrected counts');axes[0,0].set(title='Complete measured traverse spectra',xlabel='Wavelength(nm)',ylabel='Elapsed time(min)')
    axes[0,1].plot(t,upstream/1e18,label='Independent original run');axes[0,1].plot(t,p[:,0]/1e18,'--',label='Fresh local reconstruction');axes[0,1].set(title='Two volcanic-plume traverses',xlabel='Elapsed time(min)',ylabel='SO2 column(1e18 molecules/cm²)');axes[0,1].legend(fontsize=8)
    im=axes[0,2].pcolormesh(x,t,100*(y-f)/y,shading='auto',cmap='RdBu_r',vmin=-3,vmax=3);fig.colorbar(im,ax=axes[0,2],label='Residual(%)');axes[0,2].set(title='Fresh fit residuals',xlabel='Wavelength(nm)',ylabel='Elapsed time(min)')
    idx=int(np.argmax(p[:,0]));axes[1,0].plot(x,y[idx],label='Measured');axes[1,0].plot(x,f[idx],'--',label='Fresh fit');axes[1,0].set(title='Largest-column spectrum',xlabel='Wavelength(nm)',ylabel='Corrected counts');axes[1,0].legend()
    axes[1,1].plot(x,100*(y[idx]-f[idx])/y[idx]);axes[1,1].set(title='Largest-column residual',xlabel='Wavelength(nm)',ylabel='Residual(%)')
    axes[1,2].plot(t,100*np.std((y-f)/y,axis=1));axes[1,2].set(title='Fit precision over complete acquisition',xlabel='Elapsed time(min)',ylabel='Residual standard deviation(%)')
    return fig
