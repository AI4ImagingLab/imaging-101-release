"""Scientific comparisons of freshly computed atmospheres and independent runs."""
from pathlib import Path
import json
import hashlib
import numpy as np
import matplotlib.pyplot as plt

LABELS=('B [G]','inclination [rad]','azimuth [rad]','v calibrated [km/s]','Doppler width [angstrom]','opacity ratio','damping','S0 / Ic','S1 / Ic')

def make_figures(data,parameters,diagnostics,references,export_dir=None):
    """Return six actual scientific figure groups and their numerical sources."""
    plt.rcParams.update({'font.size':10,'figure.dpi':110,'axes.spines.top':False,'axes.spines.right':False})
    figs=[];numbers={'fresh_parameters':parameters,'fresh_spectra':diagnostics['spectra'],'observed_stokes':data['raw'],'cavity':data['cavity'],'cycle1':diagnostics['cycle1'],'cycle2':diagnostics['cycle2'],'chi2_cycle1':diagnostics['chi2_cycle1'],'chi2_cycle2':diagnostics['chi2_cycle2']}
    extent=[0,96*.059,64*.059,0]
    def image(ax,a,title,cmap='viridis'):
        im=ax.imshow(a,extent=extent,cmap=cmap);ax.set_title(title);ax.set_xlabel('x [arcsec]');ax.set_ylabel('y [arcsec]');plt.colorbar(im,ax=ax,shrink=.83)
    fig,axes=plt.subplots(3,3,figsize=(15,11),constrained_layout=True)
    for k,ax in enumerate(axes.flat):image(ax,parameters[...,k],LABELS[k], 'twilight' if k==2 else 'viridis')
    fig.suptitle('Fresh full-observation reconstruction: all nine atmospheric maps',fontsize=16);figs.append(fig)

    pixels=[(10,20),(32,48),(50,85)];fig,axes=plt.subplots(4,3,figsize=(15,12),constrained_layout=True)
    wave=np.concatenate(data['regions'])[data['indices']]-6302.4931
    for col,(y,x) in enumerate(pixels):
        for k in range(4):
            ax=axes[k,col];obs=data['raw'][y,x,k];fresh=diagnostics['spectra'][y,x,k,data['indices']];source=references['corrected']['spectra'][y,x,k,data['indices']]
            # Spectral windows have a genuine gap; draw them independently.
            for seg in (slice(0,17),slice(17,30)):
                ax.plot(wave[seg],obs[seg],'o',ms=4,color='#222222',label='Observed' if seg.start==0 else None)
                ax.plot(wave[seg],source[seg],color='#e69f00',lw=3,alpha=.7,label='Independent corrected source' if seg.start==0 else None)
                ax.plot(wave[seg],fresh[seg],color='#0072b2',lw=1,label='Fresh local' if seg.start==0 else None)
            ax.set_title(f'Stokes {"IQUV"[k]}, pixel ({y},{x})');ax.set_xlabel('Wavelength - 6302.4931 [angstrom]');ax.set_ylabel('Normalized intensity')
            if k==0 and col==0:ax.legend(fontsize=8)
    fig.suptitle('Line-shape fits at three observed image positions',fontsize=16);figs.append(fig)

    fig,axes=plt.subplots(2,3,figsize=(15,9),constrained_layout=True)
    original=references['original']['parameters'];fft=references['fft_only']['parameters'];corrected=references['corrected']['parameters']
    image(axes[0,0],original[...,0],'Unmodified source B [G]');image(axes[0,1],fft[...,0],'FFT-copy fix only B [G]');image(axes[0,2],parameters[...,0],'Two fixes / fresh local B [G]')
    image(axes[1,0],fft[...,0]-original[...,0],'FFT fix effect on B [G]','coolwarm');image(axes[1,1],corrected[...,0]-fft[...,0],'Residual rollback effect on B [G]','coolwarm');image(axes[1,2],parameters[...,0]-corrected[...,0],'Fresh - independent corrected B [G]','coolwarm')
    numbers.update(original_parameters=original,fft_only_parameters=fft,corrected_source_parameters=corrected)
    fig.suptitle('Original source, two explicit correctness fixes, and extracted method',fontsize=16);figs.append(fig)

    fig,axes=plt.subplots(2,3,figsize=(15,9),constrained_layout=True)
    image(axes[0,0],diagnostics['chi2_cycle1'],'Fresh cycle 1 weighted chi-square');image(axes[0,1],diagnostics['chi2_cycle2'],'Fresh cycle 2 weighted chi-square');image(axes[0,2],diagnostics['chi2_cycle2']-diagnostics['chi2_cycle1'],'Cycle 2 - cycle 1','coolwarm')
    for label,key,color in [('Cycle 1','chi2_cycle1','#e69f00'),('Cycle 2','chi2_cycle2','#0072b2')]:
        q=np.sort(diagnostics[key].ravel());axes[1,0].plot(q,np.arange(1,len(q)+1)/len(q),label=label,color=color)
    axes[1,0].set(xlabel='Weighted chi-square',ylabel='Empirical cumulative probability',xscale='log');axes[1,0].legend()
    for label,ref in references.items():
        q=np.sort(ref['chi2'].ravel());axes[1,1].plot(q,np.arange(1,len(q)+1)/len(q),label=label)
    axes[1,1].set(xlabel='Weighted chi-square',ylabel='Empirical cumulative probability',xscale='log');axes[1,1].legend()
    for k,key in enumerate(('accepted_cycle1','accepted_cycle2')):axes[1,2].hist(diagnostics[key].ravel(),bins=np.arange(27)-.5,histtype='step',lw=2,label=f'Cycle {k+1}')
    axes[1,2].set(xlabel='Accepted LM steps',ylabel='Pixel count');axes[1,2].legend();fig.suptitle('Actual two-cycle fitting behavior and corrected-source comparison',fontsize=16);figs.append(fig)

    fig,axes=plt.subplots(3,3,figsize=(15,10),constrained_layout=True)
    for k,ax in enumerate(axes.flat):
        lo,hi=np.percentile(parameters[...,k],[.5,99.5]);bins=np.linspace(lo,hi,45)
        ax.hist(diagnostics['cycle1'][...,k].ravel()+(data['cavity'].ravel()+.45 if k==3 else 0),bins=bins,histtype='step',color='#e69f00',lw=2,label='Fresh cycle 1')
        ax.hist(parameters[...,k].ravel(),bins=bins,histtype='step',color='#0072b2',lw=2,label='Fresh cycle 2');ax.set_xlabel(LABELS[k]);ax.set_ylabel('Pixel count')
    axes[0,0].legend();fig.suptitle('Atmospheric distributions: actual first and second fit passes',fontsize=16);figs.append(fig)

    fig,axes=plt.subplots(2,3,figsize=(15,9),constrained_layout=True)
    image(axes[0,0],data['raw'][...,0,0],'Measured continuum-wing intensity / Ic');image(axes[0,1],np.sqrt(np.sum(data['raw'][...,1:,:]**2,axis=(2,3))),'Observed integrated polarization / Ic');image(axes[0,2],data['cavity'],'Independent instrument cavity [km/s]','coolwarm')
    residual=diagnostics['spectra'][...,data['indices']]-data['raw'];sigma=np.array([.003,.0003,.0003,.003/3.5]);rms=np.sqrt(np.mean((residual/sigma[None,None,:,None])**2,axis=(0,1,3)));numbers['fresh_stokes_whitened_rms']=rms
    axes[1,0].bar(np.arange(4),rms,color=['#444444','#009e73','#56b4e9','#cc79a7']);axes[1,0].set(xticks=np.arange(4),xticklabels=list('IQUV'),ylabel='RMS / fitting scale',title='Fresh forward observation residuals')
    axes[1,1].scatter(parameters[...,4].ravel(),parameters[...,5].ravel(),s=2,alpha=.3);axes[1,1].set(xlabel='Doppler width [angstrom]',ylabel='Opacity ratio',title='Thermodynamic trade-off in actual fit')
    error=np.max(np.abs(parameters-corrected)/np.array([1000,1,1,1,.05,10,.1,1,1]),axis=2);image(axes[1,2],error,'Maximum scaled source-parity error')
    fig.suptitle('Acquisition coverage, forward fit, and numerical agreement',fontsize=16);figs.append(fig)
    if export_dir is not None:
        path=Path(export_dir);path.mkdir(parents=True,exist_ok=True);manifest={}
        for i,fig in enumerate(figs,1):
            for suffix in ('png','svg','pdf'):
                p=path/f'figure_{i:02d}.{suffix}';fig.savefig(p,dpi=145);manifest[p.name]=hashlib.sha256(p.read_bytes()).hexdigest()
        np.savez_compressed(path/'numeric_sources.npz',**numbers)
        manifest['numeric_sources.npz']=hashlib.sha256((path/'numeric_sources.npz').read_bytes()).hexdigest()
        (path/'manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf8')
    return figs,numbers
