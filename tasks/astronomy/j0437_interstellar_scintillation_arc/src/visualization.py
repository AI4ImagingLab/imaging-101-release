"""Review plots using physical coordinates, without changing any reconstruction."""
import numpy as np

def secondary_image(axis,spectrum,wavelength=True):
    power=spectrum.lamsspec if wavelength else spectrum.sspec
    y=spectrum.beta if wavelength else spectrum.tdel
    valid=power[np.isfinite(power)]
    artist=axis.imshow(power,origin='lower',aspect='auto',extent=[spectrum.fdop[0],spectrum.fdop[-1],y[0],y[-1]],cmap='magma',vmin=np.median(valid)-3,vmax=np.max(valid)-3)
    axis.set_xlabel('Doppler frequency (mHz)')
    axis.set_ylabel('Wavelength conjugate β (m⁻¹)' if wavelength else 'Differential delay (μs)')
    if wavelength:
        doppler=np.linspace(spectrum.fdop[0],spectrum.fdop[-1],1000)
        axis.plot(doppler,spectrum.betaeta*doppler**2,'c--',lw=1,label='Fresh fitted thin-screen arc')
        axis.set_ylim(y[0],y[-1]);axis.legend(fontsize=7,loc='upper left')
    return artist
