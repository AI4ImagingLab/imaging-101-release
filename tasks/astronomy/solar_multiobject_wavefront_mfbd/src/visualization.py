"""Maintainer review figures; importing this module does not select a backend."""
import numpy as np
import matplotlib.pyplot as plt


def contrast(image):
    """Return population RMS contrast in percent, with no normalization fitted to a reference."""
    image=np.asarray(image,dtype='float64')
    mean=np.mean(image)
    if mean <= 0:
        raise ValueError('Contrast requires positive mean intensity')
    return float(100*np.std(image)/mean)


def radial_power(image):
    """Return annularly averaged fluctuation power at cycles per arcsec."""
    image=np.asarray(image,dtype='float64')
    side=image.shape[0]
    fluctuation=image-np.mean(image)
    power=np.abs(np.fft.fft2(fluctuation))**2/side**4
    frequency=np.fft.fftfreq(side,d=0.059)
    x,y=np.meshgrid(frequency,frequency)
    radius=np.sqrt(x*x+y*y)
    spacing=1/(side*0.059)
    labels=np.floor(radius/spacing).astype(int)
    bins=np.bincount(labels.ravel(),weights=power.ravel())
    count=np.bincount(labels.ravel())
    return (np.arange(len(bins))+0.5)*spacing,bins/np.maximum(count,1)


def plot_data(observations):
    """Plot first, maximum-broadband-contrast and burst-mean observed frames."""
    normalized=np.array(observations,copy=True)
    normalized/=np.mean(normalized,axis=(-1,-2),keepdims=True)
    best=int(np.argmax(np.std(normalized[0],axis=(-1,-2))))
    figure,axes=plt.subplots(2,3,figsize=(12,8),layout='constrained')
    for channel,name in enumerate(['Broadband','Narrowband']):
        images=[normalized[channel,0],normalized[channel,best],normalized[channel].mean(0)]
        limits=np.percentile(normalized[channel],[1,99])
        for column,(image,label) in enumerate(zip(images,['First exposure',f'Best WB exposure {best}','Burst mean'])):
            artist=axes[channel,column].imshow(image,cmap='gray',origin='lower',vmin=limits[0],vmax=limits[1],extent=[0,512*0.059,0,512*0.059])
            axes[channel,column].set_title(f'{name}: {label}')
            axes[channel,column].set_xlabel('Detector column [arcsec]')
        axes[channel,0].set_ylabel('Detector row [arcsec]')
        figure.colorbar(artist,ax=axes[channel,:],label='Relative intensity',shrink=0.8)
    return figure


def plot_comparison(best,upstream,fresh,traditional):
    """Show measured best frame, reproduced upstream, fresh local and historical comparison."""
    figure,axes=plt.subplots(2,4,figsize=(16,8),layout='constrained')
    labels=['Best measured frame','Reproduced upstream','Fresh local MOMFBD','Traditional MOMFBD archive']
    for channel,name in enumerate(['Broadband','Narrowband']):
        limits=np.percentile(fresh[channel,1:-1,1:-1],[1,99])
        for column,(array,label) in enumerate(zip([best,upstream,fresh,traditional],labels)):
            image=array[channel]
            axes[channel,column].imshow(image,cmap='gray',origin='lower',vmin=limits[0],vmax=limits[1],extent=[0,476*0.059,0,476*0.059])
            axes[channel,column].set_title(label,fontsize=10)
            axes[channel,column].text(0.03,0.96,f'{contrast(image[1:-1,1:-1]):.2f}% RMS',transform=axes[channel,column].transAxes,va='top',color='yellow',fontsize=10)
            axes[channel,column].set_xlabel('Local column [arcsec]')
        axes[channel,0].set_ylabel(name+'\nLocal row [arcsec]')
    return figure


def plot_wavefront(coefficients,pupil,basis):
    """Compare spatially separated recovered pupil phases and corresponding PSFs."""
    figure,axes=plt.subplots(2,3,figsize=(12,7),layout='constrained')
    for column,index in enumerate([0,60,120]):
        phase=np.einsum('m,mrc->rc',coefficients[index,5],basis)
        field=np.fft.fft2(pupil*np.exp(1j*phase))
        psf=np.abs(field)**2
        psf/=psf.sum()
        value=np.where(pupil>0,phase,np.nan)
        artist=axes[0,column].imshow(value,cmap='RdBu_r',origin='lower')
        axes[0,column].set_title(f'Region {index}, exposure 5')
        figure.colorbar(artist,ax=axes[0,column],label='Pupil phase [rad]',shrink=0.8)
        artist=axes[1,column].imshow(np.log10(np.maximum(np.fft.fftshift(psf),1e-9)),origin='lower',cmap='magma',vmin=-7,vmax=-1)
        axes[1,column].set_title('Unit-flux PSF, centered for display')
        figure.colorbar(artist,ax=axes[1,column],label='log10 PSF',shrink=0.8)
    return figure


def plot_diagnostics(losses,upstream_losses,best,upstream,fresh,traditional):
    """Retain convergence, paired RMS contrasts and both channels' spectral comparisons."""
    figure,axes=plt.subplots(2,2,figsize=(12,9),layout='constrained')
    axes[0,0].plot(np.arange(1,len(upstream_losses)+1),upstream_losses,label='Reproduced upstream')
    axes[0,0].plot(np.arange(1,len(losses)+1),losses,'--',label='Fresh local')
    axes[0,0].set(xlabel='Adam update',ylabel='Reduced objective',title='Joint wavefront optimization')
    axes[0,0].legend()
    arrays=[best,upstream,fresh,traditional]
    names=['Best frame','Upstream','Fresh local','Traditional']
    x=np.arange(4)
    for channel,name in enumerate(['Broadband','Narrowband']):
        axes[0,1].bar(x+(channel-0.5)*0.36,[contrast(array[channel,1:-1,1:-1]) for array in arrays],width=0.36,label=name)
        for array,label in zip(arrays,names):
            frequency,power=radial_power(array[channel,1:-1,1:-1])
            admitted=(np.arange(len(frequency))>0)&(frequency<=0.5/0.059)&(power>0)
            axes[1,channel].semilogy(frequency[admitted],power[admitted],label=label)
        axes[1,channel].axvline(1/0.1761915534734726,color='black',linestyle=':',label='Diffraction cutoff')
        axes[1,channel].set(xlabel='Spatial frequency [cycles/arcsec]',ylabel='Mean annular fluctuation power',title=name+'; mosaicking can affect high frequencies')
        axes[1,channel].legend(fontsize=8)
    axes[0,1].set_xticks(x,names)
    axes[0,1].set(ylabel='RMS contrast [%]',title='Paired passband contrasts; no physical truth implied')
    axes[0,1].legend()
    return figure


def plot_differences(upstream,fresh):
    """Display signed fresh-minus-upstream differences without fitting intensity scales."""
    figure,axes=plt.subplots(1,2,figsize=(10,4),layout='constrained')
    for channel,name in enumerate(['Broadband','Narrowband']):
        difference=fresh[channel]-upstream[channel]
        limit=max(float(np.max(np.abs(difference))),1e-8)
        artist=axes[channel].imshow(difference,cmap='RdBu_r',vmin=-limit,vmax=limit,origin='lower')
        axes[channel].set_title(name+': fresh - upstream')
        figure.colorbar(artist,ax=axes[channel],label='Relative-intensity difference')
    return figure
