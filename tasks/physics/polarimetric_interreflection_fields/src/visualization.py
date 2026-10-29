"""Measured full-image controls, original/fresh distinction and numeric exports."""
from pathlib import Path
import hashlib,json
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import PowerNorm
from .physics_model import components

NAMES=('U','Re C_sum','Im C_sum','Re C_diff','Im C_diff')
COLORS=('#167d89','#bc5c23','#758291','#a24956','#7764a5')

def _image(fig,ax,a,title,signed=False,limit=None,label='Instrumental radiance',phase=False):
    if limit is None:limit=max(float(np.quantile(abs(a),.995)),1e-15)
    lo=-limit if signed or phase else 0
    cmap='twilight' if phase else 'RdBu_r' if signed else 'magma'
    im=ax.imshow(a,cmap=cmap,vmin=lo,vmax=limit,rasterized=True)
    ax.set(title=title,xlabel='Column (pixel)',ylabel='Row (pixel)')
    fig.colorbar(im,ax=ax,shrink=.83,label=label,extend='both' if signed else 'max')

def make_figures(data,original,fresh,residual,comparisons,upstream_summary,directory=None):
    """Return six figures and computed metrics; inputs never trigger a cached inverse."""
    with plt.rc_context({'font.size':9,'axes.titlesize':10,'figure.titlesize':15,'axes.spines.top':False,'axes.spines.right':False}):
        return _make_figures(data,original,fresh,residual,comparisons,upstream_summary,directory)

def _make_figures(data,original,fresh,residual,comparisons,upstream_summary,directory):
    figs=[];y=data['intensity'];f=fresh['reconstruction'];source=original
    comp=components(f);original_comp=components(source)
    errors=[float(np.linalg.norm(a-b)/np.linalg.norm(b)) for a,b in zip(f,source)]
    stats={'full_original_fresh_channel_relative_l2':errors,'fresh_observation_relative_l2':residual['relative_l2'],
      'independent_original_observation_relative_l2':float(upstream_summary['source_forward_relative_residual']),
      'fresh_unrestricted_nine_basis_relative_l2':comparisons['relative9'],
      'negative_unpolarized_fraction':float(np.mean(f[0]<0)),'minimum_unpolarized':float(f[0].min()),
      'measurement_budget':comparisons['support_metrics'],'added_noise':comparisons['noise_metrics'],
      'interpretation':'Five harmonic fields preserve all source degrees of freedom; not independently certified unique bounce paths. No true path-separated experimental target exists.'}
    numeric={'original_five_fields':source,'fresh_five_fields':f,'polarizer_psg':data['polarizer_psg'],'polarizer_psa':data['polarizer_psa'],
      'fresh_five_basis_rms':residual['rms_map'],'fresh_nine_basis_rms':comparisons['rms9'],'frame_five_relative':residual['frame_relative_l2'],'frame_nine_relative':comparisons['frame_relative9']}
    fig,axs=plt.subplots(2,3,figsize=(17,10),layout='constrained');fig.suptitle('1  Real16-state HDR observations and recorded angular geometry; full native resolution')
    limit=float(np.quantile(y,.995))
    for ax,i in zip(axs.flat,[0,5,10,15]):
        im=ax.imshow(y[i],cmap='magma',norm=PowerNorm(.5,vmin=0,vmax=limit),rasterized=True)
        ax.set(title=f'Frame{i+1}: g={np.rad2deg(data["polarizer_psg"][i]):.0f}, c={np.rad2deg(data["polarizer_psa"][i]):.0f} deg',xlabel='Column (pixel)',ylabel='Row (pixel)')
        fig.colorbar(im,ax=ax,shrink=.83,extend='max',label='HDR radiance; sqrt display only');numeric[f'observed_frame_{i}']=y[i]
    ax=axs[1,1];ax.scatter(np.rad2deg(data['polarizer_psg']),np.rad2deg(data['polarizer_psa']),s=100,color=COLORS[0])
    for i,(g,c) in enumerate(zip(data['polarizer_psg'],data['polarizer_psa'])):ax.annotate(str(i+1),(np.rad2deg(g),np.rad2deg(c)),xytext=(5,3),textcoords='offset points')
    ax.set(title='All16 states; full-rank5 and9 designs, condition2',xlabel='Recorded generator angle (deg)',ylabel='Recorded analyzer angle (deg)',xlim=(-10,155),ylim=(-10,155))
    counts,edges=np.histogram(y,bins=np.geomspace(float(y.min()),float(y.max())*1.00001,150));axs[1,2].stairs(counts,edges,color=COLORS[0]);axs[1,2].set(xscale='log',yscale='log',title='All80,216,064 measured pixel intensities',xlabel='Instrumental radiance; no upper clipping',ylabel='Count / logarithmic bin')
    numeric['intensity_hist_count']=counts;numeric['intensity_hist_edges']=edges;figs.append(fig)

    fig,axs=plt.subplots(3,5,figsize=(23,12),layout='constrained');fig.suptitle('2  All five original and fresh signed fields; identical coordinates and radiometric units')
    for i,name in enumerate(NAMES):
        limit=max(float(np.quantile(abs(source[i]),.995)),1e-15)
        _image(fig,axs[0,i],source[i],'Original '+name,True,limit)
        _image(fig,axs[1,i],f[i],'Fresh '+name,True,limit)
        d=f[i]-source[i];_image(fig,axs[2,i],d,f'Difference; rel L2={errors[i]:.2g}',True,max(float(abs(d).max()),1e-15))
    figs.append(fig)

    fig,axs=plt.subplots(2,3,figsize=(17,10),layout='constrained');fig.suptitle('3  Preserved native amplitudes/phases and distribution view; harmonic labels remain explicit')
    common=max(float(np.quantile(comp['sum_amplitude'],.995)),float(np.quantile(comp['difference_amplitude'],.995)))
    _image(fig,axs[0,0],comp['sum_amplitude'],'Sum-angle amplitude; source F label',False,common)
    _image(fig,axs[0,1],comp['difference_amplitude'],'Difference-angle amplitude; source R label',False,common)
    _image(fig,axs[0,2],comp['unpolarized'],f'U: negative fraction={stats["negative_unpolarized_fraction"]:.4%}',True)
    _image(fig,axs[1,0],np.rad2deg(comp['sum_phase']),'Sum half-angle phase; weak amplitudes uncertain',True,90,'Angle (deg), modulo180',True)
    _image(fig,axs[1,1],np.rad2deg(comp['difference_phase']),'Difference half-angle phase',True,90,'Angle (deg), modulo180',True)
    rgb=np.stack([comp['difference_amplitude'],comp['sum_amplitude'],comp['unpolarized']],axis=-1)/2
    rgb=np.clip(rgb,0,1);rgb=np.where(rgb<=.0031308,12.92*rgb,1.055*rgb**(1/2.4)-.055)
    axs[1,2].imshow(rgb,rasterized=True);axs[1,2].set(title='Author-style R=difference, G=sum, B=U\nDisplay clips radiance/2 only; inverse never does',xlabel='Column (pixel)',ylabel='Row (pixel)')
    figs.append(fig)

    fig,axs=plt.subplots(2,3,figsize=(17,10),layout='constrained');fig.suptitle('4  Actual restricted-model discrepancy and unrestricted linear Mueller control')
    mean=y.mean(axis=0);rms=np.sqrt(np.mean(y.astype(np.float64)**2,axis=0));limit=float(np.quantile(residual['rms_map'],.995))
    _image(fig,axs[0,0],mean,'Mean real HDR observation')
    _image(fig,axs[0,1],residual['rms_map'],f'Fresh5basis RMS; relative={residual["relative_l2"]:.4%}',False,limit)
    _image(fig,axs[0,2],comparisons['rms9'],f'Fresh9basis RMS; relative={comparisons["relative9"]:.4%}',False,limit)
    ratio5=residual['rms_map']/rms;ratio9=comparisons['rms9']/rms;limit=float(np.quantile(ratio5,.99))
    _image(fig,axs[1,0],ratio5,'5basis pixel-relative residual',False,limit,'Relative RMS')
    _image(fig,axs[1,1],ratio9,'9basis pixel-relative residual',False,limit,'Relative RMS')
    axs[1,2].plot(np.arange(1,17),residual['frame_relative_l2'],'o-',color=COLORS[0],label='Selected5basis');axs[1,2].plot(np.arange(1,17),comparisons['frame_relative9'],'o-',color=COLORS[1],label='9basis model diagnostic')
    axs[1,2].set(title='Extra four single-polarizer terms explain structure',xlabel='Measured state',ylabel='Full-spatial relative L2 residual');axs[1,2].legend(fontsize=8)
    numeric.update(mean_observed=mean,observed_rms=rms,pixel_relative_residual5=ratio5,pixel_relative_residual9=ratio9);figs.append(fig)

    fig,axs=plt.subplots(2,3,figsize=(17,10),layout='constrained');fig.suptitle('5  Fresh full-spatial measurement-budget study; angle-only subsets frozen before local extraction')
    counts=[5,8,12,16]
    for ax,count in zip(axs.flat,counts):
        amplitude=components(comparisons['support_fields'][count])['sum_amplitude'];numeric[f'budget_{count}_sum_amplitude']=amplitude
        _image(fig,ax,amplitude,f'{count} states; condition={comparisons["support_metrics"][count]["condition"]:.3g}',False,common)
    for i,name in enumerate(NAMES):axs[1,1].plot(counts,[comparisons['support_metrics'][n]['channel_relative_l2'][i] for n in counts],'o-',label=name,color=COLORS[i])
    axs[1,1].set(title='Native16-state method reference, not path truth',xlabel='Measured states used',ylabel='Whole-channel relative L2 difference');axs[1,1].legend(fontsize=7)
    values=[comparisons['support_metrics'][n]['all_state_prediction_relative_l2'] for n in counts]
    axs[1,2].bar([str(n) for n in counts],values,color=COLORS[:4]);axs[1,2].bar_label(axs[1,2].containers[0],fmt='%.4f',padding=3);axs[1,2].set(title='Predictions evaluated on all16 real states',xlabel='States used in inverse',ylabel='All-state relative L2 residual');figs.append(fig)

    fig,axs=plt.subplots(2,3,figsize=(17,10),layout='constrained');fig.suptitle('6  Actually computed full-grid noise perturbations; diagnostic noise is not measured instrument noise')
    _image(fig,axs[0,0],comp['sum_amplitude'],'Native real observations, no added noise',False,common)
    levels=[.005,.01,.02]
    for ax,fraction in zip([axs[0,1],axs[0,2],axs[1,0]],levels):
        a=comparisons['noise_sum_amplitudes'][fraction];numeric[f'noise_{fraction}_sum_amplitude']=a
        _image(fig,ax,a,f'Added Gaussian sigma={fraction:.1%} of frame RMS',False,common)
    for i,name in enumerate(NAMES):axs[1,1].plot([100*x for x in levels],[comparisons['noise_metrics'][n]['channel_relative_l2'][i] for n in levels],'o-',label=name,color=COLORS[i])
    axs[1,1].set(title='All five fields, same full native observation grid',xlabel='Added sigma (% of frame RMS)',ylabel='Relative L2 difference from native field');axs[1,1].legend(fontsize=7)
    ax=axs[1,2];ax.plot([100*x for x in levels],[comparisons['noise_metrics'][n]['prediction_error_against_unperturbed_real_observations'] for n in levels],'o-',color=COLORS[0]);ax.set(title='Prediction error and retained lower-clipping effect',xlabel='Added sigma (% of frame RMS)',ylabel='Prediction residual versus original observations')
    secondary=ax.twinx();secondary.plot([100*x for x in levels],[100*comparisons['noise_metrics'][n]['newly_lower_clipped_fraction'] for n in levels],'s--',color=COLORS[1]);secondary.set_ylabel('Newly lower-clipped input values (%)',color=COLORS[1]);figs.append(fig)
    if directory is not None:
        directory=Path(directory);directory.mkdir(parents=True,exist_ok=True)
        for i,fig in enumerate(figs,1):
            for ext in ('png','svg','pdf'):fig.savefig(directory/f'figure{i}.{ext}',dpi=160,bbox_inches='tight')
        np.savez_compressed(directory/'figure_numeric_source.npz',**numeric)
        (directory/'figure_metrics.json').write_text(json.dumps(stats,indent=2,allow_nan=False),encoding='utf-8')
        provenance={'numeric_file':'figure_numeric_source.npz','numeric_sha256':hashlib.sha256((directory/'figure_numeric_source.npz').read_bytes()).hexdigest(),
            'fresh':'Native full-data inverse plus four measurement budgets, complete9basis control and three full-grid measured-RMS-scaled Gaussian perturbations actually computed by local mathematics.',
            'original':'Unmodified source twice run on exact full measured frames, full five-field method reference; no actual path truth.',
            'display':'HDR sqrt display and quantile saturation are labeled with colorbar extensions. Source-style RGB clips only for visualization. Every metric and inverse uses full unclipped-upper arrays.',
            'figure_sources':{'1':['observed_frame_*','intensity_hist_*','polarizer_*'],'2':['original_five_fields','fresh_five_fields'],'3':['fresh_five_fields'],'4':['*_basis_rms','pixel_relative_residual*','frame_*_relative'],'5':['budget_*_sum_amplitude','metrics:measurement_budget'],'6':['noise_*_sum_amplitude','metrics:added_noise']}}
        (directory/'figure_provenance.json').write_text(json.dumps(provenance,indent=2),encoding='utf-8')
    return figs,stats
