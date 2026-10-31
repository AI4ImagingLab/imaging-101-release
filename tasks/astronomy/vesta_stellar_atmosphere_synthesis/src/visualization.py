"""Scientific comparisons from computed arrays; no cached result becomes fresh."""
import numpy as np
import matplotlib.pyplot as plt
from .solvers import estimate_error_map

LABELS=('Temperature (K)','log gravity (dex)','Metallicity (dex)','Effective width (km/s)')

def review_figures(model,prediction,diagnostics,upstream,reference_scale):
    """Return five review figures and actively computed sensitivity measurements."""
    raw=model.raw;wave=raw['wavelength'];observed=raw['flux'];fresh=prediction['spectrum'][0];original=upstream['spectrum'][0];likelihood=diagnostics['likelihood'];mask=model.mask_indices
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False,'figure.dpi':110})
    figures=[];relative_original=[];relative_fresh=[]
    fig,axes=plt.subplots(2,2,figsize=(13,8),layout='constrained')
    for number,(lo,hi) in enumerate(raw['ranges']):
        selected=(wave>=lo)&(wave<=hi);x=wave[selected];denominator=np.median(observed[selected])
        axes[0,0].plot(x,observed[selected],color='#64748b',lw=.5,alpha=.6,label='Real observations' if number==0 else None)
        axes[0,0].plot(x,original[selected],color='#d97706',lw=.6,alpha=.7,label='Reproduced original' if number==0 else None)
        axes[0,0].plot(x,fresh[selected],color='#2563eb',lw=.6,alpha=.7,label='Fresh local' if number==0 else None)
        relative_original.extend(100*(observed[selected]-original[selected])/denominator)
        relative_fresh.extend(100*(observed[selected]-fresh[selected])/denominator)
    axes[0,0].set(xlabel='Rest wavelength (Å)',ylabel='Author flux units',title='All52570 measured channels / all218 windows');axes[0,0].legend(fontsize=8)
    standardized=(prediction['reconstruction'][0]-upstream['reconstruction'][0])/reference_scale
    axes[0,1].barh(np.arange(4),standardized,color='#2563eb');axes[0,1].axvline(0,color='black',lw=.7);axes[0,1].axvspan(-3,3,color='#dbeafe',alpha=.4,zorder=0);axes[0,1].set_yticks(range(4),LABELS);axes[0,1].set(xlabel='Difference / original-posterior standard deviation',title='Every recovered parameter, method reference only')
    both=np.r_[relative_original,relative_fresh];bins=np.linspace(both.min(),both.max(),100)
    axes[1,0].hist(relative_original,bins=bins,histtype='step',density=True,color='#d97706',label='Original full-window residual')
    axes[1,0].hist(relative_fresh,bins=bins,histtype='step',density=True,color='#2563eb',label='Fresh full-window residual')
    # All tails are included; presentation normalization is explicitly labeled.
    axes[1,0].set_yscale('log');axes[1,0].set(xlabel='Residual / window median observed flux (%)',ylabel='Density (log scale)',title='Nonmask blends and systematic errors remain');axes[1,0].legend(fontsize=8)
    best=diagnostics['best_internal'];internal_scale=np.array(reference_scale,copy=True);internal_scale[0]/=1000;responses=[]
    for j in range(4):
        pair=[]
        for sign in [-1,1]:
            point=best.copy();point[j]+=sign*internal_scale[j];pair.append(model.forward(point))
        responses.append(pair)
    response=np.asarray(responses);response_rms=np.sqrt(np.mean(((response[:,:,mask]-fresh[mask])/likelihood['errors'])**2,axis=2))
    k=np.arange(4);axes[1,1].bar(k-.17,response_rms[:,0],.34,label='Minus1 reference σ',color='#0f766e');axes[1,1].bar(k+.17,response_rms[:,1],.34,label='Plus1 reference σ',color='#fb923c');axes[1,1].set_xticks(k,['Temperature','Gravity','Metallicity','Width'],rotation=15);axes[1,1].set(ylabel='RMS Fe response / inferred discrepancy scale',title='Fresh physical-parameter response, same real data');axes[1,1].legend(fontsize=8)
    fig.suptitle('Real spectral inverse: recovery and complete observation-space evidence',fontsize=16);figures.append(fig)

    # Fixed spread across the full source mask list, plus its paper's6335/6336 lines.
    index=np.unique(np.r_[np.linspace(0,len(raw['masks'])-1,10,dtype=int),np.argmin(abs(raw['masks'][:,0]-6335.33)),np.argmin(abs(raw['masks'][:,0]-6336.82))])
    fig,axes=plt.subplots(3,4,figsize=(14,9),layout='constrained')
    for axis,number in zip(axes.flat,index):
        center,width=raw['masks'][number];selected=abs(wave-center)<=3*width;x=wave[selected];scale=np.median(observed[selected]);noise=likelihood['full_error_map'][selected]*likelihood['second_factor']
        axis.errorbar(x,observed[selected]/scale,yerr=raw['uncertainty'][selected]/scale,fmt='.',color='#334155',ms=3,lw=.5,label='Real flux / author initial σ')
        axis.plot(x,original[selected]/scale,color='#d97706',ls='--',lw=1,label='Reproduced original')
        axis.plot(x,fresh[selected]/scale,color='#2563eb',lw=1.3,label='Fresh local')
        axis.fill_between(x,(fresh[selected]-noise)/scale,(fresh[selected]+noise)/scale,color='#2563eb',alpha=.10,label='Inferred model discrepancy')
        axis.axvspan(center-width,center+width,color='#0f766e',alpha=.07)
        axis.set(title=f'Author mask {center:.2f} Å',xlabel='Wavelength (Å)',ylabel='Flux / observed panel median')
        axis.ticklabel_format(useOffset=False,axis='x')
    for axis in list(axes.flat)[len(index):]:axis.axis('off')
    handles,labels=axes.flat[0].get_legend_handles_labels();fig.legend(handles,labels,ncol=4,loc='outside lower center',fontsize=8)
    fig.suptitle('Distributed Fe-line details: masks for display only; full inverse kept all218 windows',fontsize=15);figures.append(fig)

    original_chain=upstream['chain'][0].reshape(-1,4).copy();original_chain[:,0]*=1000
    fresh_chain=diagnostics['chain'].reshape(-1,4).copy();fresh_chain[:,0]*=1000
    fig,axes=plt.subplots(4,4,figsize=(12,10),layout='constrained')
    for i in range(4):
        for j in range(4):
            axis=axes[i,j]
            if i<j:axis.axis('off');continue
            if i==j:
                values=np.r_[original_chain[:,i],fresh_chain[:,i]];bins=np.linspace(values.min(),values.max(),45)
                axis.hist(original_chain[:,i],bins=bins,density=True,histtype='step',color='#d97706',label='Completed original')
                axis.hist(fresh_chain[:,i],bins=bins,density=True,histtype='step',color='#2563eb',label='Fresh complete sampler')
            else:
                axis.scatter(original_chain[::8,j],original_chain[::8,i],s=2,color='#d97706',alpha=.10)
                axis.scatter(fresh_chain[::8,j],fresh_chain[::8,i],s=2,color='#2563eb',alpha=.12)
            if i==3:axis.set_xlabel(LABELS[j])
            if j==0:axis.set_ylabel(LABELS[i])
            axis.tick_params(labelsize=8)
    axes[0,0].legend(fontsize=8);fig.suptitle('Two identified16000-sample posteriors: correlations and numerical inference uncertainty',fontsize=15);figures.append(fig)

    aligned,aligned_masked,aligned_details=estimate_error_map(model,likelihood['first_parameters'],variant='aligned_diagnostic')
    source=likelihood['full_error_map'];ratio=aligned/source;initial=raw['uncertainty'];old_noise=upstream['noise_reference'][0]
    fig,axes=plt.subplots(2,2,figsize=(13,8),layout='constrained')
    source_ratios=[];aligned_ratios=[];locations=[]
    for lo,hi in raw['ranges']:
        selected=(wave>=lo)&(wave<=hi);locations.append((lo+hi)/2);source_ratios.append(np.median(source[selected]/initial[selected]));aligned_ratios.append(np.median(aligned[selected]/initial[selected]))
    axes[0,0].plot(locations,source_ratios,'.',color='#2563eb',label='Selected source2026 map');axes[0,0].plot(locations,aligned_ratios,'.',color='#0f766e',alpha=.7,label='Aligned-ranking diagnostic');axes[0,0].axhline(1,color='black',ls=':');axes[0,0].set(xlabel='Window central wavelength (Å)',ylabel='Window median map / author initial σ',title='Runtime estimates, never public independent calibration');axes[0,0].legend(fontsize=8)
    values=source/initial;bins=np.geomspace(min(values.min(),(aligned/initial).min()),max(values.max(),(aligned/initial).max()),70)
    axes[0,1].hist(values,bins=bins,histtype='step',color='#2563eb',label='Selected source map');axes[0,1].hist(aligned/initial,bins=bins,histtype='step',color='#0f766e',label='Aligned diagnostic');axes[0,1].axvline(1,color='black',ls=':',label='Calculated source floor is unused');axes[0,1].set_xscale('log');axes[0,1].set(xlabel='Estimated map / initial supplied uncertainty',ylabel='Measured-channel count',title='The original returned map is not error-floored');axes[0,1].legend(fontsize=8)
    axes[1,0].hist(ratio,bins=np.geomspace(ratio.min(),ratio.max(),70),histtype='step',lw=1.5,color='#9333ea');axes[1,0].set_xscale('log');axes[1,0].set(xlabel='Aligned diagnostic / selected source map',ylabel='Channel count',title='Filtering/ranking convention changes estimated weights')
    residual=observed[mask]-fresh[mask]
    stats=dict(original_weighted_rmse=float(np.sqrt(np.mean((residual/old_noise)**2))),fresh_source_weighted_rmse=float(np.sqrt(np.mean((residual/likelihood['errors'])**2))),aligned_diagnostic_weighted_rmse=float(np.sqrt(np.mean((residual/(aligned_masked*likelihood['second_factor']))**2))),map_relative_l2=float(np.linalg.norm(aligned-source)/np.linalg.norm(source)),fraction_changed_above_one_percent=float(np.mean(abs(ratio-1)>.01)),fraction_unfloored_below_initial=float(np.mean(source<initial)),source_peak_count=len(likelihood['error_map_details']['centers']),aligned_peak_count=len(aligned_details['centers']))
    axes[1,1].axis('off');axes[1,1].text(0,.98,'SOURCE VARIANT AND LIMITATIONS\n\n2026 code ranks filtered wavelengths against\nunfiltered residual amplitudes. The aligned\nversion shown here is a diagnostic only.\n\nThe original computes an error floor but\nreturns the unfloored map; this is retained.\n\nChanged weights cannot be interpreted as\nindependent evidence of a better fit.\n\nMap relative L2 change: '+f'{stats["map_relative_l2"]:.3f}'+'\nChannels changing >1%: '+f'{100*stats["fraction_changed_above_one_percent"]:.1f}%',va='top',linespacing=1.5)
    fig.suptitle('Fresh noise-heuristic experiment: preserve, quantify and disclose source limitations',fontsize=15);figures.append(fig)

    fig,axes=plt.subplots(2,2,figsize=(12,7),layout='constrained')
    for j,axis in enumerate(axes.flat):
        oc=upstream['chain'][0,:,:,j].copy();fc=diagnostics['chain'][:,:,j].copy()
        if j==0:oc*=1000;fc*=1000
        axis.plot(oc.mean(0),color='#d97706',lw=.8,alpha=.9,label='Original walker mean');axis.plot(fc.mean(0),color='#2563eb',lw=.8,label='Fresh walker mean');axis.fill_between(np.arange(1000),fc.mean(0)-fc.std(0),fc.mean(0)+fc.std(0),color='#2563eb',alpha=.12,label='Fresh walker spread')
        axis.set(xlabel='Production sweep after100burn',ylabel=LABELS[j],title=f'{LABELS[j]} · all16walkers /1000sweeps')
    axes.flat[0].legend(fontsize=8);fig.suptitle('Complete posterior execution; no shortened chain or cached reconstruction',fontsize=15);figures.append(fig)
    return figures,dict(response_spectra=response,response_masked_rms=response_rms,aligned_noise_map=aligned,aligned_masked_noise_map=aligned_masked,diagnostic_statistics=stats)
