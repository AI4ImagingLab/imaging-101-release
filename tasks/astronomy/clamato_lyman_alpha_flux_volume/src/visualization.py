"""Scientific review plots from explicitly supplied measured/computed arrays."""
from pathlib import Path
import json
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
from matplotlib.ticker import FixedLocator,FixedFormatter,NullFormatter
from scipy.ndimage import gaussian_filter

def export(fig,directory,name,arrays,metadata):
    """Save paired scientific formats and exact numeric inputs; no computation cache lookup."""
    directory=Path(directory);directory.mkdir(parents=True,exist_ok=True)
    for suffix in ['png','svg','pdf']:
        fig.savefig(directory/f'{name}.{suffix}',dpi=170,bbox_inches='tight')
    np.savez_compressed(directory/f'{name}_numeric.npz',**arrays)
    (directory/f'{name}_numeric.json').write_text(json.dumps(metadata,indent=2,allow_nan=False),encoding='utf8')
    return fig

def contrast(ax,field,title,extent=(0,438,0,24),limit=.6,difference=False):
    if difference:
        limit=max(float(np.max(np.abs(field))),1e-14)
    im=ax.imshow(field,origin='lower',extent=extent,aspect='auto',cmap='RdBu',
                 norm=TwoSlopeNorm(vmin=-limit,vcenter=0,vmax=limit),interpolation='nearest',rasterized=True)
    ax.set(title=title,xlabel='Line-of-sight distance [Mpc/h]',ylabel='Declination distance [Mpc/h]')
    return im

def figure01(observations,directory):
    p,s,d=observations.positions,observations.noise,observations.flux
    xy,group=np.unique(p[:,:2],axis=0,return_inverse=True)
    precision=np.array([np.median(1/s[group==i]) for i in range(len(xy))])
    bins=np.linspace(0,438,51);centers=(bins[:-1]+bins[1:])/2
    coverage=np.array([len(np.unique(group[(p[:,2]>=lo)&(p[:,2]<hi)])) for lo,hi in zip(bins[:-1],bins[1:])])
    fig,axs=plt.subplots(2,2,figsize=(12,8),layout='constrained')
    im=axs[0,0].scatter(xy[:,0],xy[:,1],c=precision,cmap='viridis',s=24,edgecolors='white',linewidths=.25)
    axs[0,0].plot([0,30,30,0,0],[0,0,24,24,0],color='#333333',lw=1)
    axs[0,0].set(xlabel='RA distance [Mpc/h]',ylabel='Dec distance [Mpc/h]',title=f'A  {len(xy)} measured sightlines',aspect='equal')
    fig.colorbar(im,ax=axs[0,0],label='Median inverse contrast-noise (not continuum S/N)')
    axs[0,1].plot(centers,coverage,color='#176d9c',lw=2)
    axs[0,1].fill_between(centers,coverage,color='#176d9c',alpha=.12)
    axs[0,1].set(xlabel='Line-of-sight distance [Mpc/h]',ylabel='Unique sightlines in bin',title='B  Sampling changes through the volume')
    axs[1,0].hist(s,bins=np.geomspace(s.min(),s.max(),45),color='#558c72',alpha=.9)
    axs[1,0].set(xscale='log',xlabel='Released contrast-noise standard deviation',ylabel='Spectral pixels',title='C  Author noise floor already applied')
    axs[1,0].xaxis.set_major_locator(FixedLocator([.2,.5,1.,2.,4.]))
    axs[1,0].xaxis.set_major_formatter(FixedFormatter(['0.2','0.5','1','2','4']))
    axs[1,0].xaxis.set_minor_formatter(NullFormatter())
    axs[1,0].axvline(.2,color='#333333',ls='--',label='0.2 floor');axs[1,0].legend()
    axs[1,1].hist(d,bins=140,color='#8d698c',alpha=.9)
    axs[1,1].set(yscale='log',xlabel='Observed flux contrast',ylabel='Spectral pixels',title='D  Noisy measurements are not clipped')
    fig.suptitle('Real CLAMATO DR1 input: sparse spatial coverage and heterogeneous noise',fontsize=15)
    return export(fig,directory,'figure01_observations',{'positions':p,'noise':s,'contrast':d,'sightline_xy':xy,'precision':precision,'coverage_centers':centers,'coverage':coverage},
                  {'consumer':'Notebook chapter2/figure01; all values from released measurements','units':'coordinates Mpc/h; contrast/noise dimensionless','coverage_bin_width':float(bins[1]-bins[0])})

def figure02(field,directory):
    indices=np.array([0,175,350,525,700,875])
    fig,axs=plt.subplots(2,3,figsize=(12,8),layout='constrained')
    for ax,k in zip(axs.flat,indices):
        im=contrast(ax,field[:,:,k].T,f'z center = {(k+.5)*.5:.2f} Mpc/h',extent=(0,30,0,24))
        ax.set(xlabel='RA distance [Mpc/h]',ylabel='Dec distance [Mpc/h]',aspect='equal')
    fig.colorbar(im,ax=axs.ravel().tolist(),label='Reconstructed contrast: red = stronger absorption',extend='both',shrink=.85)
    fig.suptitle('Fresh complete-volume reconstruction: six native transverse sections',fontsize=15)
    return export(fig,directory,'figure02_spatial_volume',{'slice_indices':indices,'native_sections':field[:,:,indices]},
                  {'consumer':'Notebook chapter5/figure02','axis_order':'x,y,selected z','display_limits':[-.6,.6],'saturation':'Values outside limits saturate; all full values remain in numerical evaluation. White is contrast near zero, not missing data.'})

def figure03(field,original,metrics,directory):
    x=30;z=438;diff=field-original
    fig,axs=plt.subplots(2,3,figsize=(15,8),layout='constrained')
    for ax,a,title in zip(axs[0],[original[x],field[x],diff[x]],['A  Independently executed original','B  Fresh local mathematics','C  Fresh − original']):
        im=contrast(ax,a,title,difference=title.startswith('C'))
        fig.colorbar(im,ax=ax,label='Contrast',extend='both' if not title.startswith('C') else 'neither')
    for ax,a,title in zip(axs[1,:2],[original[:,:,z].T,field[:,:,z].T],['D  Original transverse section','E  Fresh transverse section']):
        im=contrast(ax,a,title,extent=(0,30,0,24));ax.set(xlabel='RA distance [Mpc/h]',ylabel='Dec distance [Mpc/h]',aspect='equal')
        fig.colorbar(im,ax=ax,label='Contrast',extend='both')
    xx=original.ravel()[::35];yy=field.ravel()[::35]
    axs[1,2].scatter(xx,yy,s=2,alpha=.12,rasterized=True,color='#236d8a')
    lim=[min(xx.min(),yy.min()),max(xx.max(),yy.max())];axs[1,2].plot(lim,lim,color='#333333',ls='--')
    axs[1,2].set(xlabel='Original contrast',ylabel='Fresh contrast',title=f"F  All-voxel NCC {metrics['ncc_vs_ref']:.10f}\nNRMSE {metrics['nrmse_vs_ref']:.3g}")
    fig.suptitle('Full real-data method parity; reference is a reconstruction, not truth',fontsize=15)
    return export(fig,directory,'figure03_original_parity',{'original_yz':original[x],'fresh_yz':field[x],'difference_yz':diff[x],'original_xy':original[:,:,z],'fresh_xy':field[:,:,z],'scatter_original':xx,'scatter_fresh':yy},
                  {'consumer':'Notebook chapter5/figure03','metrics_all_voxels':metrics,'display_slices':{'x':x,'z':z},'signed_difference_limits':[-max(float(abs(diff[x]).max()),1e-14),max(float(abs(diff[x]).max()),1e-14)],'scatter_stride':35,'note':'Longitudinal sections use unequal visual axis scales; distances remain physical. No metric subsampling.'})

def figure04(observations,field,info,uniform_slice,uniform_info,original_history,directory):
    s,d=observations.noise,observations.flux
    baseline_residual=(info['predicted_pixels']-d)/s
    uniform_residual=(uniform_info['predicted_pixels']-d)/s
    bins=np.quantile(s,[0,.25,.5,.75,1])
    rows=[]
    for indices in np.array_split(np.argsort(s,kind='stable'),4):
        rows.append([np.sqrt(np.mean(baseline_residual[indices]**2)),np.sqrt(np.mean(uniform_residual[indices]**2))])
    rows=np.asarray(rows)
    fig,axs=plt.subplots(2,3,figsize=(15,8),layout='constrained')
    for ax,a,title in zip(axs[0],[field[30],uniform_slice,uniform_slice-field[30]],['A  Heterogeneous released noise','B  Equal-noise ablation','C  Equal noise − selected model']):
        im=contrast(ax,a,title,difference=title.startswith('C'));fig.colorbar(im,ax=ax,label='Contrast',extend='both' if not title.startswith('C') else 'neither')
    loc=np.arange(4);width=.36
    axs[1,0].bar(loc-width/2,rows[:,0],width,label='Selected noise',color='#246f92')
    axs[1,0].bar(loc+width/2,rows[:,1],width,label='Equal noise',color='#c88249')
    axs[1,0].set(xticks=loc,xticklabels=['Q1','Q2','Q3','Q4'],xlabel='Released noise quartile',ylabel='Weighted data RMS',title='D  Heterogeneous-noise fit');axs[1,0].legend()
    for a,label,color in [(baseline_residual,'Selected noise','#246f92'),(uniform_residual,'Equal noise','#c88249')]:
        axs[1,1].hist(a,bins=np.linspace(-5,5,101),histtype='step',density=True,label=label,color=color)
    axs[1,1].set(xlabel='(Predicted − observed) / released sigma',ylabel='Probability density',title='E  Residuals; displayed interval [−5,5]');axs[1,1].legend()
    for history,label,color in [(original_history,'Original','#333333'),(info['residual_norms'],'Fresh','#246f92'),(uniform_info['residual_norms'],'Equal noise','#c88249')]:
        axs[1,2].semilogy(np.arange(len(history)),history/history[0],label=label,color=color)
    axs[1,2].axhline(.001,color='#777777',ls=':');axs[1,2].set(xlabel='PCG updates',ylabel='Relative linear-system residual',title='F  Actively computed convergence');axs[1,2].legend()
    fig.suptitle('Fresh noise ablation: full-observation solves, native middle-x sections',fontsize=15)
    return export(fig,directory,'figure04_noise_ablation',{'selected_slice':field[30],'uniform_slice':uniform_slice,'selected_weighted_residual':baseline_residual,'uniform_weighted_residual':uniform_residual,'noise_quartiles':bins,'quartile_rms':rows,'original_history':original_history,'fresh_history':info['residual_norms'],'uniform_history':uniform_info['residual_norms']},
                  {'consumer':'Notebook chapter5/figure04','equal_noise_std':float(np.median(s)),'residual_display':'Histogram zoom only; RMS and numeric residuals use every observation. Four disjoint rank quartiles split stable noise ordering, including ties at the author floor.','interpretation':'Fit is to noisy measurements, not independent truth.'})

def figure05(field,directory):
    smooth1=gaussian_filter(field,2,mode='reflect');smooth2=gaussian_filter(field,4,mode='reflect')
    maps=[field,smooth1,smooth2];labels=['No extra smoothing','Extra R = 1 Mpc/h','Extra R = 2 Mpc/h']
    z=(np.arange(field.shape[2])+.5)*.5
    k=2*np.pi*np.fft.rfftfreq(field.shape[2],d=.5)
    power=np.array([.5/field.shape[2]*np.mean(np.abs(np.fft.rfft(a,axis=2))**2,axis=(0,1)) for a in maps])
    profiles=np.array([a[28:32,22:26].mean(axis=(0,1)) for a in maps])
    changes=np.array([np.sqrt(np.mean((a-field)**2))/np.ptp(field) for a in maps])
    fig,axs=plt.subplots(2,3,figsize=(15,8),layout='constrained')
    for ax,a,label in zip(axs[0],maps,labels):
        im=contrast(ax,a[30],label);fig.colorbar(im,ax=ax,label='Contrast',extend='both')
    for profile,p,label in zip(profiles,power,labels):
        axs[1,0].plot(z,profile,label=label)
        axs[1,1].loglog(k[1:],p[1:],label=label)
    axs[1,0].set(xlabel='Line-of-sight distance [Mpc/h]',ylabel='Mean contrast in fixed central 2×2 Mpc/h column',title='D  Structure profiles');axs[1,0].legend(fontsize=8)
    axs[1,1].set(xlabel='Line-of-sight wavenumber [h/Mpc]',ylabel='Mean 1D power [Mpc/h]',title='E  Small scales are suppressed');axs[1,1].legend(fontsize=8)
    axs[1,2].bar(['Native','R=1','R=2'],changes,color=['#246f92','#569a8e','#c88249'])
    axs[1,2].set(ylabel='All-voxel NRMSE versus native fresh map',title='F  Display smoothing changes the target')
    fig.suptitle('Paper-style smoothing is an additional display operation, excluded from submission',fontsize=15)
    return export(fig,directory,'figure05_smoothing',{'native_slice':field[30],'R1_slice':smooth1[30],'R2_slice':smooth2[30],'z':z,'profiles':profiles,'wavenumber':k,'power':power,'nrmse_vs_native':changes},
                  {'consumer':'Notebook chapter5/figure05','kernel_sigma_voxels':[0,2,4],'boundary':'reflect for this explicitly labeled display diagnostic','power_definition':'spacing/Nz * mean over x,y of abs(rfft(field,z))^2, non-DC bins shown; no one-sided factor2 applied','historical_comparison':'R2 is the paper visualization scale; this new diagnostic does not claim identical unpublished boundary handling.'})

def figure06(field,sensitivity_slices,sensitivity_metrics,directory):
    labels=list(sensitivity_slices)
    fig,axs=plt.subplots(2,3,figsize=(15,8),layout='constrained')
    im=contrast(axs[0,0],field[30],'A  Selected correlation lengths 2.5 / 2.0');fig.colorbar(im,ax=axs[0,0],label='Contrast',extend='both')
    deltas=np.array([sensitivity_slices[k]-field[30] for k in labels])
    shared=max(float(np.max(np.abs(deltas))),1e-12)
    for ax,label,delta in zip([axs[0,1],axs[0,2],axs[1,0],axs[1,1]],labels,deltas):
        im=contrast(ax,delta,label+' − selected',limit=shared)
        fig.colorbar(im,ax=ax,label='Signed contrast change')
    values=np.array([[sensitivity_metrics[k]['slice_nrmse_vs_selected'],sensitivity_metrics[k]['weighted_data_rms']] for k in labels])
    ax=axs[1,2];loc=np.arange(len(labels))
    ax.bar(loc,values[:,0],color='#246f92',alpha=.8)
    ax.set(xticks=loc,xticklabels=labels,ylabel='Middle-x section NRMSE versus selected',title='F  Conditional-map sensitivity')
    ax.tick_params(axis='x',labelrotation=35)
    twin=ax.twinx();twin.plot(loc,values[:,1],color='#b76d3f',marker='o');twin.set_ylabel('All-observation weighted data RMS',color='#b76d3f')
    fig.suptitle('Correlation-prior sensitivity: independent fresh solves on unchanged real observations',fontsize=15)
    return export(fig,directory,'figure06_prior_sensitivity',{'selected_slice':field[30],'differences':deltas,'summary':values},
                  {'consumer':'Notebook chapter6/figure06','labels':labels,'measurements':sensitivity_metrics,'difference_limits':[-shared,shared],'scope':'Every variant solves all64332 observations; only the complete native middle-x plane is mapped for these sensitivity diagnostics. Baseline reconstruction and grading cover the full native volume.','uncertainty':'Model sensitivity, not calibrated physical uncertainty or independent truth.'})
