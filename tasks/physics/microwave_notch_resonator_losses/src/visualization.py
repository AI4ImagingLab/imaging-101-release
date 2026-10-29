"""Scientific comparisons for active notebook review."""
import numpy as np
import matplotlib.pyplot as plt

def comparison(frequency,measured,fresh,reference,normalized):
    x=(frequency-frequency.mean())/1e3
    fig,axs=plt.subplots(2,3,figsize=(14,8),constrained_layout=True)
    axs[0,0].plot(x,np.abs(measured),label='Measured 275 mK');axs[0,0].plot(x,np.abs(fresh),label='Fresh circle/phase fit');axs[0,0].plot(x,np.abs(reference),'--',label='Original method')
    axs[0,0].set(xlabel='Frequency offset (kHz)',ylabel='Raw |S21|');axs[0,0].legend(fontsize=8)
    axs[0,1].plot(measured.real,measured.imag,'.',ms=2,label='Measured');axs[0,1].plot(fresh.real,fresh.imag,label='Fresh');axs[0,1].set(xlabel='Raw real S21',ylabel='Raw imaginary S21');axs[0,1].axis('equal');axs[0,1].legend()
    axs[0,2].plot(normalized.real,normalized.imag,'.',ms=2);axs[0,2].set(xlabel='Normalized real S21',ylabel='Normalized imaginary S21',title='Observation-estimated calibration');axs[0,2].axis('equal')
    axs[1,0].plot(x,(measured-fresh).real,label='Real');axs[1,0].plot(x,(measured-fresh).imag,label='Imaginary');axs[1,0].set(xlabel='Frequency offset (kHz)',ylabel='Measured minus fit');axs[1,0].legend()
    axs[1,1].plot(x,np.angle(measured),label='Measured');axs[1,1].plot(x,np.angle(fresh),label='Fresh');axs[1,1].set(xlabel='Frequency offset (kHz)',ylabel='Raw phase (rad)');axs[1,1].legend()
    axs[1,2].plot(x,np.abs(fresh-reference));axs[1,2].set(xlabel='Frequency offset (kHz)',ylabel='|Fresh minus original|',title='Reproduction error, not physical truth')
    return fig
