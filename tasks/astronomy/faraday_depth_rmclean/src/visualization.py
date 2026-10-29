"""Scientific comparisons; plotting imports do not select a backend."""
import numpy as np
import matplotlib.pyplot as plt
from .solvers import PHI

def relative_l2(estimate,reference):
    """Complex Euclidean relative error, without a phase or scale fit."""
    return float(np.linalg.norm(estimate-reference)/np.linalg.norm(reference))

def plot_spectra(submission,details,reference):
    """Dirty/original/fresh magnitudes, actual complex components, signed differences."""
    fig,axes=plt.subplots(2,2,figsize=(12,7),layout='constrained')
    axes[0,0].plot(PHI,abs(details['dirty']),label='Fresh dirty synthesis')
    axes[0,0].plot(PHI,abs(reference['reconstruction'][0]),label='Independent original restored')
    axes[0,0].plot(PHI,abs(submission['reconstruction'][0]),'--',label='Fresh local restored')
    axes[0,0].set(ylabel='Polarized flux (Jy/beam)',title='Full Faraday-depth interval');axes[0,0].legend(fontsize=8)
    axes[0,1].plot(PHI,submission['reconstruction'][0].real,label='Real: Q')
    axes[0,1].plot(PHI,submission['reconstruction'][0].imag,label='Imaginary: U')
    axes[0,1].set(ylabel='Polarized flux (Jy/beam)',title='Fixed wavelength phase convention');axes[0,1].legend()
    axes[1,0].stem(PHI,abs(submission['components'][0]),basefmt=' ')
    axes[1,0].set(ylabel='Component flux (Jy/beam)',title='Fresh sparse CLEAN components')
    delta=submission['reconstruction'][0]-reference['reconstruction'][0]
    axes[1,1].plot(PHI,delta.real,label='Real error');axes[1,1].plot(PHI,delta.imag,label='Imaginary error')
    axes[1,1].set(ylabel='Fresh minus original (Jy/beam)',title='No fitted phase, shift or scale');axes[1,1].legend()
    for ax in axes.flat:ax.set_xlabel('Faraday depth (rad/m²)')
    return fig

def plot_observation(details):
    """All288 observations, selected physical model, channel residuals and RMSF."""
    fig,axes=plt.subplots(2,2,figsize=(12,7),layout='constrained')
    frequency=details['frequency']/1e9;p=details['polarization'];model=details['model'];sigma=details['noise']
    for ax,measured,predicted,label in [(axes[0,0],p.real,model.real,'Stokes Q'),(axes[0,1],p.imag,model.imag,'Stokes U')]:
        ax.errorbar(frequency,measured,yerr=sigma,fmt='.',ms=2,alpha=.4,label='Observed')
        ax.plot(frequency,predicted,label='Forward from fresh components')
        ax.set(xlabel='Frequency (GHz)',ylabel='Flux (Jy/beam)',title=label);ax.legend(fontsize=8)
    residual=(p-model)/sigma
    axes[1,0].plot(frequency,residual.real,'.',ms=3,label='Q residual / sigma')
    axes[1,0].plot(frequency,residual.imag,'.',ms=3,label='U residual / sigma')
    axes[1,0].set(xlabel='Frequency (GHz)',ylabel='Residual (noise std)',title='Model does not replace data');axes[1,0].legend(fontsize=8)
    axes[1,1].plot(details['phi2'],abs(details['response']),label='|RMSF|')
    axes[1,1].axhline(.5,color='gray',ls=':');axes[1,1].set(xlabel='Offset depth (rad/m²)',ylabel='Response (unitless)',title='Finite positive wavelength coverage')
    return fig

def plot_metrics(report):
    """Measured numerical errors and their frozen boundaries in separate panels."""
    fig,axes=plt.subplots(1,2,figsize=(11,3.5),layout='constrained')
    names=['component_l1_flux','restoration_identity_relative_error']
    axes[0].bar(['Absolute flux budget','Restoration identity'],[max(report['metrics'][k]/report['thresholds'][k]['boundary'],1e-16) for k in names])
    axes[0].axhline(1.,color='red',ls='--',label='Acceptance limit')
    axes[0].set(yscale='log',ylabel='Measured value / v2 boundary');axes[0].legend(fontsize=8)
    key='forward_reduced_chi2';axes[1].bar(['Fresh component forward model'],[report['metrics'][key]])
    axes[1].axhline(report['thresholds'][key]['boundary'],color='red',ls='--',label='Frozen boundary')
    axes[1].set(ylabel='Noise-weighted residual / (2 × channels)');axes[1].legend(fontsize=8)
    return fig
