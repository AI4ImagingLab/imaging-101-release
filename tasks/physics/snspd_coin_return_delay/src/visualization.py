"""Actual real-data reconstructions and honest full-window diagnostics."""
from pathlib import Path
import time,json,hashlib,importlib.util
import numpy as np
import matplotlib.pyplot as plt
from .preprocessing import load_data
from .physics_model import BIN_DEPTH_MM,fit_nuisance
from .solvers import reconstruct,matched_peaks,zero_norm,circular_correlation

def review(task_dir,figure_dir):
    """Actively run full inverse, comparisons and aggregation; export five groups."""
    task=Path(task_dir);dest=Path(figure_dir);dest.mkdir(parents=True,exist_ok=True)
    counts,h=load_data(task/'data/raw_data.npz');start=time.perf_counter();depth,peaks=reconstruct(counts,h);seconds=time.perf_counter()-start
    with np.load(task/'notebooks/reference_outputs/upstream_methods.npz') as z:original=z['original_delay_mm']
    np.savez(task/'output/reconstruction.npz',reconstruction=depth[None])
    spec=importlib.util.spec_from_file_location('snspd_review_judge',task/'evaluation/judge.py');judge=importlib.util.module_from_spec(spec);spec.loader.exec_module(judge);report,code=judge.evaluate(task/'output/reconstruction.npz')
    if code:raise RuntimeError(report)
    totals=counts.sum(-1);summary={'fresh_seconds':seconds,'judge':report,'reference_pixel_photons':int(totals[0,0]),'count_below33000':int((totals<33000).sum()),'number_pixels':2500,'depth_mm_range':[float(depth.min()),float(depth.max())],'not_ground_truth':'All differences below compare methods/resolution, not measured physical range accuracy.'};nums={'fresh_depth_mm':depth,'original_depth_mm':original,'fresh_peak_bins':peaks,'photon_totals':totals,'irf_peak_zero_normalized':h};figs=[]
    def image(ax,a,title,unit,limits=None,cmap='viridis'):
        im=ax.imshow(a,origin='upper',cmap=cmap,**({} if limits is None else {'vmin':limits[0],'vmax':limits[1]}));ax.set(title=title,xlabel='Scan column (pixel)',ylabel='Scan row (pixel)');plt.colorbar(im,ax=ax,label=unit,shrink=.85)
    fig,ax=plt.subplots(2,2,figsize=(12,8),layout='constrained');image(ax[0,0],np.log10(np.maximum(1,totals)),'Actual integrated photons','log10 photons')
    t=np.arange(10000);ax[0,1].plot(t,h);ax[0,1].set(xlabel='Peak-zero temporal bin (1ps/bin)',ylabel='Probability per bin',title='Independent instrument response, full window')
    offset=np.arange(-150,151);ax[1,0].plot(offset,h[offset%10000]);ax[1,0].set(xlabel='Delay from instrument maximum (ps)',ylabel='Probability per bin',title='Measured asymmetric IRF, no Gaussian replacement')
    ax[1,1].hist(np.log10(np.maximum(1,totals.ravel())),bins=40);ax[1,1].axvline(np.log10(33000),color='k',ls='--',label='Paper33000count context, no mask');ax[1,1].set(xlabel='log10 detected photons/pixel',ylabel='Number of pixels',title='All2500 points; highly unequal return strength');ax[1,1].legend(fontsize=8);fig.suptitle('1. Real SNSPD acquisition and independently measured temporal calibration');figs.append(fig)
    fig,ax=plt.subplots(2,3,figsize=(16,8),layout='constrained');limits=(float(depth.min()),float(depth.max()))
    image(ax[0,0],original,'Actual original FullHist result','Relative apparent distance (mm)',limits)
    image(ax[0,1],depth,'Fresh local FullHist result','Relative apparent distance (mm)',limits)
    image(ax[0,2],abs(depth-original),'Measured absolute method discrepancy','mm',(0,1e-7))
    image(ax[1,0],depth,'Display saturation only:3.5..6mm','mm',(3.5,6.));ax[1,0].text(.5,-.25,'All pixels remain in output and evaluation',ha='center',transform=ax[1,0].transAxes,fontsize=9)
    image(ax[1,1],peaks,'All-window strongest matched time','Bin index (1ps/bin)',(0,9999))
    ax[1,2].hist(peaks.ravel(),bins=np.arange(0,10001,100));ax[1,2].set(xlabel='Matched peak bin (1ps/bin)',ylabel='Number of pixels',title='Competing/late returns are retained');fig.suptitle('2. Native50×50 apparent delay; neither absolute range nor physical truth');figs.append(fig)
    fig,ax=plt.subplots(2,2,figsize=(13,8),layout='constrained');examples=[(0,0),(19,19),np.unravel_index(np.argmin(totals),totals.shape),np.unravel_index(np.argmax(totals),totals.shape)];fits=[]
    for a,(r,c) in zip(ax.ravel(),examples):
        y=counts[r,c].astype(float);A,b,model=fit_nuisance(y,h,int(peaks[r,c]));nums[f'hist_{r}_{c}']=y;nums[f'fit_{r}_{c}']=model
        fits.append({'row':int(r),'column':int(c),'photons':int(totals[r,c]),'matched_bin':int(peaks[r,c]),'fitted_signal_photons':A,'fitted_background_per_bin':b,'relative_single_pulse_l2':float(np.linalg.norm(y-model)/max(np.linalg.norm(y),1e-15))})
        a.plot(t,y,lw=.7,label='Measured counts');a.plot(t,model,lw=1,label='Single-pulse diagnostic fit');a.axvline(peaks[r,c],color='k',ls=':',lw=.7);a.set_yscale('symlog',linthresh=1);a.set_ylim(bottom=0);a.set(xlabel='Acquired temporal bin (1ps/bin)',ylabel='Counts/bin, symlog',title=f'Pixel({r},{c}), {totals[r,c]:g}photons, peak{peaks[r,c]}');a.legend(fontsize=8)
    fig.suptitle('3. Actual histories reveal background, low counts and multiple returns');figs.append(fig);summary['actual_histogram_examples']=fits
    fig,ax=plt.subplots(2,2,figsize=(13,8),layout='constrained');y=counts[19,19].astype(float);corr=circular_correlation(zero_norm(y),zero_norm(h));nums['correlation_19_19']=corr
    ax[0,0].plot(t,corr);ax[0,0].set(xlabel='Circular lag (1ps/bin)',ylabel='Normalized correlation',title='Actual double-return pixel(19,19), all lags')
    direct_lags=np.array([0,1,853,856,2800,2812,5000,9999]);direct=np.array([np.dot(np.roll(zero_norm(h),int(s)),zero_norm(y)) for s in direct_lags]);nums['direct_lags']=direct_lags;nums['direct_lag_correlations']=direct
    ax[0,1].plot(direct,corr[direct_lags],'o');lo=min(direct.min(),corr[direct_lags].min());hi=max(direct.max(),corr[direct_lags].max());ax[0,1].plot([lo,hi],[lo,hi],'k--');ax[0,1].set(xlabel='Independent direct lag dot product',ylabel='FFT correlation',title='Independent operator/sign/boundary check')
    confidence=np.empty((50,50));raw_peak=counts.argmax(-1)
    for row in range(50):confidence[row]=matched_peaks(counts[row],h)[1].max(-1)
    nums['max_correlation']=confidence;nums['raw_argmax_bins']=raw_peak
    ax[1,0].scatter(np.maximum(1,totals.ravel()),confidence.ravel(),s=5,alpha=.5);ax[1,0].set_xscale('log');ax[1,0].set(xlabel='Actual detected photons',ylabel='Maximum normalized correlation',title='Confidence/counts are not surface truth')
    ax[1,1].hist((raw_peak-peaks).ravel(),bins=60);ax[1,1].set(xlabel='Raw histogram argmax minus matched bin',ylabel='Number of pixels',title='IRF matching differs from brightest single bin');fig.suptitle('4. Explicit matched-filter evidence on public observations');figs.append(fig)
    fig,ax=plt.subplots(2,3,figsize=(16,8),layout='constrained');aggregation=[]
    for j,factor in enumerate([2,4,10]):
        hcoarse=h.reshape(-1,factor).sum(-1);s=np.empty((50,50),int)
        for row in range(50):s[row]=matched_peaks(counts[row].reshape(50,-1,factor).sum(-1),hcoarse)[0]
        d=(s-s[0,0])*BIN_DEPTH_MM*factor;nums['aggregation_factor_'+str(factor)+'_depth_mm']=d
        error=abs(d-depth)/BIN_DEPTH_MM;aggregation.append({'aggregation_factor':factor,'time_bin_ps':factor,'mean_abs_difference_native_bins':float(error.mean()),'median_abs_difference_native_bins':float(np.median(error)),'max_abs_difference_native_bins':float(error.max())})
        image(ax[0,j],d,f'Post-acquisition aggregation:{factor}ps/bin','Relative apparent mm',limits)
        ax[1,j].hist(np.minimum(error.ravel(),100),bins=np.linspace(0,100,31));ax[1,j].set(xlabel='Absolute difference (native1ps bins), >100 saturated',ylabel='Pixels',title=f'Median {np.median(error):.2f}; mean {error.mean():.2f}bins')
    fig.suptitle('5. Actual histogram aggregation: resolution sensitivity, not new hardware');figs.append(fig);summary['post_acquisition_aggregation']=aggregation
    np.savez_compressed(dest/'numeric_sources.npz',**nums);(dest/'metrics.json').write_text(json.dumps(summary,indent=2,allow_nan=False)+'\n',encoding='utf-8');manifest={}
    for i,fig in enumerate(figs,1):
        for ext in ['png','svg','pdf']:
            p=dest/f'figure_{i}.{ext}';fig.savefig(p,dpi=150);manifest[p.name]=hashlib.sha256(p.read_bytes()).hexdigest()
        plt.close(fig)
    for name in ['numeric_sources.npz','metrics.json']:manifest[name]=hashlib.sha256((dest/name).read_bytes()).hexdigest()
    (dest/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n',encoding='utf-8');return figs,summary
