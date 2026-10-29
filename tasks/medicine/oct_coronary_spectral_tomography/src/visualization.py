"""Computed maintainer review experiments; no backend changes on import."""
from pathlib import Path
import json
import numpy as np
import matplotlib.pyplot as plt
from .solvers import reconstruct
from .physics_model import forward
def relative_error(x,y):
    return float(np.linalg.norm(x.astype(np.complex128)-y)/np.linalg.norm(y))
def run_experiments(data,upstream):
    result,stage=reconstruct(data)
    nodisp,_=reconstruct(data,dispersion_correction=False)
    nowin,_=reconstruct(data,apodize=False)
    profiles=np.stack([np.mean(np.abs(x[0])**2,axis=(0,2)) for x in [result,nodisp,nowin]])
    metrics={'nominal_upstream_relative_l2':relative_error(result,upstream),'no_dispersion_change':relative_error(nodisp,result),'rectangular_window_change':relative_error(nowin,result)}
    return {'nominal':result,'no_dispersion':nodisp,'rectangular':nowin,'depth_power':profiles,'spectral_power':np.mean(abs(stage['spectrum'])**2,axis=(0,2)),'metrics':metrics,'dispersion':stage['dispersion'],'indices':stage['indices']}
def make_figures(data,e):
    figs=[]
    fig,ax=plt.subplots(2,2,figsize=(12,8),constrained_layout=True)
    im=ax[0,0].imshow(data['fringes'][0],aspect='auto',cmap='gray');ax[0,0].set(title='Native coronary detector 1 fringes',xlabel='A-line',ylabel='Sweep sample');fig.colorbar(im,ax=ax[0,0],label='ADU')
    ax[0,1].plot(data['background'].T);ax[0,1].set(title='Independent measured background',xlabel='Sweep sample',ylabel='Mean ADU')
    ax[1,0].plot(e['indices']);ax[1,0].set(title='Calibrated nonlinear spectral lookup',xlabel='Uniform spectral index',ylabel='Oversampled fringe index')
    ax[1,1].plot(np.unwrap(np.angle(e['dispersion'])));ax[1,1].set(title='Independent dispersion correction',xlabel='Uniform spectral index',ylabel='Unwrapped correction phase (rad)');figs.append(fig)
    fig,ax=plt.subplots(2,2,figsize=(12,9),constrained_layout=True)
    amp=np.abs(e['nominal'][0]); floor=max(amp.max()*1e-5,1e-12)
    for c in range(2):
        image=20*np.log10(np.maximum(amp[c],floor));im=ax[0,c].imshow(image,aspect='auto',cmap='gray',vmin=np.max(image)-60,vmax=np.max(image));ax[0,c].set(title=f'Fresh detector {c+1} backscatter',xlabel='A-line',ylabel='Digital depth bin');fig.colorbar(im,ax=ax[0,c],label='20 log10 digital amplitude')
    for c in range(2):
        im=ax[1,c].imshow(np.angle(e['nominal'][0,c]),aspect='auto',cmap='twilight',vmin=-np.pi,vmax=np.pi);ax[1,c].set(title=f'Computed complex phase, detector {c+1}',xlabel='A-line',ylabel='Digital depth bin');fig.colorbar(im,ax=ax[1,c],label='Phase (rad)')
    figs.append(fig)
    fig,ax=plt.subplots(2,2,figsize=(12,8),constrained_layout=True)
    for key,a in zip(['no_dispersion','rectangular'],ax[0]):
        delta=abs(e[key][0,0]-e['nominal'][0,0]);im=a.imshow(np.log10(np.maximum(delta,1e-8)),aspect='auto',cmap='magma');a.set(title='Fresh ablation difference: '+key,xlabel='A-line',ylabel='Depth bin');fig.colorbar(im,ax=a,label='log10 absolute complex difference')
    for row,name in zip(e['depth_power'],['Nominal','No dispersion','Rectangular aperture']):ax[1,0].semilogy(row,label=name)
    ax[1,0].set(title='Actually reconstructed depth-power profiles',xlabel='Depth bin',ylabel='Mean squared digital amplitude');ax[1,0].legend(fontsize=8)
    ax[1,1].semilogy(e['spectral_power']);ax[1,1].set(title='Calibrated measured spectral power',xlabel='Uniform spectral index',ylabel='Mean squared spectral amplitude');figs.append(fig)
    return figs
def export(figures,experiments,directory):
    directory=Path(directory);directory.mkdir(parents=True,exist_ok=True)
    for i,fig in enumerate(figures,1):
        for ext in ['png','svg','pdf']:fig.savefig(directory/f'figure_{i}.{ext}',dpi=140)
    np.savez_compressed(directory/'numeric_sources.npz',depth_power=experiments['depth_power'],spectral_power=experiments['spectral_power'],dispersion=experiments['dispersion'],indices=experiments['indices'])
    (directory/'metrics.json').write_text(json.dumps(experiments['metrics'],indent=2),encoding='utf8')
