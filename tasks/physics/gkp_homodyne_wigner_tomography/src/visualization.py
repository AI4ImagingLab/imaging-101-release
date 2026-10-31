"""Actually computed quantum-state comparisons; numerical figure sources exported."""
from pathlib import Path
import json,hashlib
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from .physics_model import Q_GRID,P_GRID,quadrature_probability,displacement,wigner_function
from .preprocessing import variance_model

COLORS=('#167d89','#bc5c23','#66727e','#ad4444')

def state_statistics(density,W):
    spacing=Q_GRID[1]-Q_GRID[0]
    Dp=displacement(density.shape,1.86/np.sqrt(2))
    Dq=displacement(density.shape,1j*np.sqrt(2)*np.pi/1.86)
    return {'purity':float(np.trace(density@density).real),
            'negative_volume_finite_window':float((np.abs(W).sum()-W.sum())*spacing**2/2),
            'wigner_integral_finite_window':float(W.sum()*spacing**2),
            'stabilizer_p':float(abs(np.trace(Dp@density))),
            'stabilizer_q':float(abs(np.trace(Dq@density))),
            'minimum_eigenvalue':float(np.linalg.eigvalsh(density).min())}

def comparison_metrics(result,reference):
    rho=np.zeros((31,31),complex);n=len(result['density_matrix']);rho[:n,:n]=result['density_matrix']
    ref=reference['density_matrix'];W=result['wigner'];refW=reference['wigner']
    return {'density_relative_l2_vs_native_reference':float(np.linalg.norm(rho-ref)/np.linalg.norm(ref)),
            'wigner_dynamic_range_nrmse_vs_native_reference':float(np.sqrt(np.mean((W-refW)**2))/np.ptp(refW)),
            'last_recorded_negative_loglikelihood_per_shot':float(-result['likelihood'][-1]),
            **state_statistics(result['density_matrix'],W)}

def _wplot(fig,ax,W,title,limit):
    im=ax.imshow(W,origin='lower',extent=[-7,7,-7,7],cmap='RdBu_r',vmin=-limit,vmax=limit,rasterized=True)
    ax.set(title=title,xlabel='Q (canonical units)',ylabel='P (canonical units)')
    fig.colorbar(im,ax=ax,label='Wigner density',shrink=.85,extend='both')

def make_figures(data,calibration,upstream,fresh,comparisons,historical,directory=None):
    """Return six figures and full-precision metrics; every input is computed/labeled."""
    with plt.rc_context({'font.size':10,'axes.titlesize':11,'figure.titlesize':16,'axes.spines.top':False,'axes.spines.right':False}):
        return _make_figures(data,calibration,upstream,fresh,comparisons,historical,directory)

def _make_figures(data,calibration,upstream,fresh,comparisons,historical,directory):
    figures=[];numeric={};metrics={}
    ref={'density_matrix':upstream['density_matrix'][0],'wigner':upstream['wigner'][0]}
    all_results={'Fresh native':fresh,**comparisons}
    count=len(calibration['q_accepted'])
    for label,result in all_results.items():
        metrics[label]=comparison_metrics(result,ref)
        metrics[label]['last_recorded_negative_loglikelihood_per_shot']/=count
        for key in ('density_matrix','wigner','likelihood','step_fidelity'):numeric[label.replace(' ','_')+'_'+key]=result[key]
    limit=max(abs(ref['wigner']).max(),abs(fresh['wigner']).max())
    fig,axs=plt.subplots(2,3,figsize=(17,10),layout='constrained');fig.suptitle('1  Real observations, independent calibration and confidence selection')
    rawphase=data['variance_phase'];variance=data['signal_variance']/data['vacuum_variance']*.5
    for i in range(100):
        axs[0,0].plot(rawphase[i]/np.pi,variance[i],color=COLORS[0],alpha=.15,lw=.7)
    axs[0,0].axhline(.5,color='black',ls='--',label='Canonical vacuum variance')
    axs[0,0].set(title='100 unconditional signal calibration scans',xlabel='Raw phase / pi',ylabel='Canonical quadrature variance');axs[0,0].legend(fontsize=8)
    for c,i in zip(COLORS,(0,49,99)):
        axs[0,1].plot(rawphase[i],variance[i],'.',ms=3,color=c)
        axs[0,1].plot(rawphase[i],variance_model(rawphase[i],*calibration['phase_fit_param'][i]),color=c,label=f'Block {i}')
    axs[0,1].set(title='Actual variance fits',xlabel='Raw phase (rad)',ylabel='Canonical variance');axs[0,1].legend(fontsize=8)
    ax=axs[0,2];p=calibration['phase_fit_param'];secondary=ax.twinx()
    ax.plot(np.rad2deg(p[:,0]),color=COLORS[0],label='Offset b')
    ax.plot(np.rad2deg(p[:,1]),color=COLORS[1],label='Distortion amplitude a')
    secondary.plot(np.rad2deg(p[:,2]),color=COLORS[2],ls='--',label='Distortion phase c')
    ax.set(title='Source phase parameters, both axis scales shown',xlabel='Acquisition block',ylabel='Offset / amplitude (deg)');secondary.set_ylabel('Distortion phase (deg)')
    handles,labels=ax.get_legend_handles_labels();h,l=secondary.get_legend_handles_labels();ax.legend(handles+h,labels+l,fontsize=8,loc='lower left')
    for k,c in enumerate(COLORS[:3]):
        hist,edges=np.histogram(calibration['pnr_distance'][:,k],bins=201);centres=(edges[1:]+edges[:-1])/2
        axs[1,0].semilogy(centres,hist,color=c,label=f'Detector {k+1}');axs[1,0].axvline(calibration['cutoff'][k],color=c,ls='--',lw=1)
        numeric[f'confidence_hist_{k}']=hist;numeric[f'confidence_centres_{k}']=centres
    axs[1,0].set(title='Native 80% per-detector confidence cutoffs',xlabel='Classifier distance (stored units)',ylabel='Shots / histogram bin');axs[1,0].legend(fontsize=8)
    qe=np.linspace(calibration['q'].min(),calibration['q'].max(),161);pe=np.linspace(0,2*np.pi,65)
    histograms=[]
    for ax,q,phase,label in [(axs[1,1],calibration['q'],calibration['phi'],'All 3,756,232 raw heralded shots'),(axs[1,2],calibration['q_accepted'],calibration['phi_accepted'],'1,881,028 confidence-accepted shots')]:
        H,_,_=np.histogram2d(np.mod(phase,2*np.pi),q,bins=(pe,qe));H=H/len(q)/np.diff(pe)[:,None]/np.diff(qe)[None,:]
        histograms.append(H)
        im=ax.pcolormesh(pe/np.pi,qe,H.T,norm=LogNorm(vmin=1e-4,vmax=max(H.max(),1e-3)),cmap='viridis',rasterized=True)
        fig.colorbar(im,ax=ax,label='Histogram density; visual only',shrink=.8)
        ax.set(title=label,xlabel='Calibrated phase / pi, folded for display',ylabel='q (canonical units)')
    numeric.update(variance_raw_phase=rawphase,normalized_signal_variance=variance,phase_fit_param=p,confidence_cutoff=calibration['cutoff'],histogram_phase_edges=pe,histogram_q_edges=qe,raw_histogram_density=histograms[0],accepted_histogram_density=histograms[1])
    figures.append(fig)

    fig,axs=plt.subplots(2,4,figsize=(21,10),layout='constrained');fig.suptitle('2  Independent original and fresh full-data reconstruction; all complex coherence retained')
    _wplot(fig,axs[0,0],ref['wigner'],'Independent original, all 100 blocks',limit)
    _wplot(fig,axs[0,1],fresh['wigner'],'Fresh local, all accepted shots',limit)
    difference=fresh['wigner']-ref['wigner'];dl=max(float(abs(difference).max()),1e-15)
    _wplot(fig,axs[0,2],difference,f'Fresh − original; max |difference|={abs(difference).max():.2g}',dl)
    rho=fresh['density_matrix'];magnitude=max(abs(rho).max(),1e-15)
    for ax,a,title,cmap,vmin,vmax in [(axs[0,3],abs(rho),'Density magnitude |rho|','magma',0,magnitude),(axs[1,0],rho.real,'Real density matrix','RdBu_r',-magnitude,magnitude),(axs[1,1],rho.imag,'Imaginary density matrix','RdBu_r',-magnitude,magnitude)]:
        im=ax.imshow(a,origin='lower',cmap=cmap,vmin=vmin,vmax=vmax,rasterized=True);fig.colorbar(im,ax=ax,shrink=.85,label='Dimensionless matrix element');ax.set(title=title,xlabel='Ket photon index',ylabel='Bra photon index')
    for ax,theta,label in [(axs[1,2],0,'Q marginal'),(axs[1,3],np.pi/2,'P marginal')]:
        a=quadrature_probability(ref['density_matrix'],Q_GRID,theta)/(Q_GRID[1]-Q_GRID[0]);b=quadrature_probability(rho,Q_GRID,theta)/(Q_GRID[1]-Q_GRID[0])
        ax.plot(Q_GRID,a,color=COLORS[2],lw=2,label='Independent original');ax.plot(Q_GRID,b,'--',color=COLORS[0],label='Fresh local')
        ax.set(title=label+'; finite-grid normalized',xlabel='Canonical quadrature',ylabel='Density');ax.legend(fontsize=8)
        numeric[label.replace(' ','_')+'_original']=a;numeric[label.replace(' ','_')+'_fresh']=b
    figures.append(fig)

    fig,axs=plt.subplots(2,3,figsize=(17,10),layout='constrained');fig.suptitle('3  Fresh photon-support sensitivity; same accepted real observations and 100 updates')
    labels=['Cutoff 10','Cutoff 20','Fresh native'];results=[comparisons['Cutoff 10'],comparisons['Cutoff 20'],fresh]
    common=max(float(abs(r['wigner']).max()) for r in results)
    for ax,label,r in zip(axs[0],labels,results):_wplot(fig,ax,r['wigner'],label,common)
    for label,r,c in zip(labels,results,COLORS):
        population=np.diag(r['density_matrix']).real
        axs[1,0].semilogy(np.arange(len(population)),population,'o-',ms=3,color=c,label=label)
    axs[1,0].set(title='All retained photon populations',xlabel='Photon number',ylabel='Population');axs[1,0].legend(fontsize=8)
    vals=[metrics[k]['last_recorded_negative_loglikelihood_per_shot'] for k in labels]
    axs[1,1].bar(labels,vals,color=COLORS[:3]);axs[1,1].bar_label(axs[1,1].containers[0],fmt='%.6f',padding=3);axs[1,1].set(title='Last recorded list-mode likelihood',ylabel='Negative log likelihood / accepted shot')
    vals=[metrics[k]['wigner_dynamic_range_nrmse_vs_native_reference'] for k in labels]
    axs[1,2].bar(labels,vals,color=COLORS[:3]);axs[1,2].bar_label(axs[1,2].containers[0],fmt='%.5g',padding=3);axs[1,2].set(title='Difference from native method reference',ylabel='Wigner dynamic-range NRMSE; not truth error')
    figures.append(fig)

    uncal=comparisons['Uncorrected phase']
    fig,axs=plt.subplots(2,3,figsize=(17,10),layout='constrained');fig.suptitle('4  Fresh phase-calibration ablation; common measured coordinate frame, no fitted alignment')
    common=max(abs(fresh['wigner']).max(),abs(uncal['wigner']).max())
    _wplot(fig,axs[0,0],fresh['wigner'],'Fresh calibrated phases',common)
    _wplot(fig,axs[0,1],uncal['wigner'],'Fresh raw phases; calibration omitted',common)
    delta=uncal['wigner']-fresh['wigner'];_wplot(fig,axs[0,2],delta,'Raw-phase field − calibrated field',max(abs(delta).max(),1e-15))
    rmsfit=[];rmsraw=[]
    for i in range(100):
        pf=calibration['phase_fit_param'][i]
        rmsfit.append(np.sqrt(np.mean((variance[i]-variance_model(rawphase[i],*pf))**2)))
        rawmodel=pf[3]*np.cos(rawphase[i])**2+pf[4]*np.sin(rawphase[i])**2
        rmsraw.append(np.sqrt(np.mean((variance[i]-rawmodel)**2)))
    axs[1,0].plot(rmsfit,label='Fitted phase model',color=COLORS[0]);axs[1,0].plot(rmsraw,label='Same extrema, raw phase',color=COLORS[1]);axs[1,0].set(title='Independent variance-channel discrepancy',xlabel='Acquisition block',ylabel='RMS canonical-variance residual');axs[1,0].legend(fontsize=8)
    for label,c in [('Fresh native',COLORS[0]),('Uncorrected phase',COLORS[1])]:
        r=all_results[label];axs[1,1].plot(-r['likelihood']/count,color=c,label=label)
    axs[1,1].set(title='Each model uses its stated phase calibration',xlabel='ML iteration (zero-based recorded index)',ylabel='Negative log likelihood / shot');axs[1,1].legend(fontsize=8)
    names=['Purity','Negative volume','|S_p|','|S_q|'];x=np.arange(4)
    for shift,label,c in [(-.18,'Fresh native',COLORS[0]),(.18,'Uncorrected phase',COLORS[1])]:
        s=metrics[label];values=[s['purity'],s['negative_volume_finite_window'],s['stabilizer_p'],s['stabilizer_q']]
        axs[1,2].bar(x+shift,values,.36,label=label,color=c)
    axs[1,2].set_xticks(x,names);axs[1,2].set(title='Same canonical definitions and displacement scales',ylabel='Dimensionless observable');axs[1,2].legend(fontsize=8)
    numeric['phase_fit_RMS_residual']=np.array(rmsfit);numeric['raw_phase_RMS_residual']=np.array(rmsraw)
    figures.append(fig)

    fig,axs=plt.subplots(2,3,figsize=(17,10),layout='constrained');fig.suptitle('5  Actual native ML trajectory, finite budget and nonclassical observables')
    steps=[1,7,25,100];Wsteps={}
    for ax,n in zip([axs[0,0],axs[0,1],axs[0,2],axs[1,0]],steps):
        W=wigner_function(fresh['snapshots'][n],Q_GRID,P_GRID);Wsteps[n]=W
        _wplot(fig,ax,W,f'Fresh snapshot after {n} updates',limit)
        numeric[f'snapshot_W_{n}']=W
    ax=axs[1,1];ax.plot(np.arange(1,101),fresh['likelihood']/count,label='Fresh local',color=COLORS[0]);ax.plot(np.arange(1,101),upstream['likelihood']/count,'--',label='Independent original',color=COLORS[2]);ax.set(title='Old-state likelihood; new-state step fidelity',xlabel='Completed update count',ylabel='Recorded log likelihood / shot');ax.legend(fontsize=8,loc='center right')
    other=ax.twinx();other.semilogy(np.arange(1,101),1-fresh['step_fidelity'],color=COLORS[1],alpha=.6);other.set_ylabel('1 − step fidelity',color=COLORS[1])
    stats=[];snapsteps=sorted(fresh['snapshots'])
    for n in snapsteps:
        W=Wsteps.get(n)
        if W is None:W=wigner_function(fresh['snapshots'][n],Q_GRID,P_GRID)
        stats.append(state_statistics(fresh['snapshots'][n],W))
    for key,label,c in [('purity','Purity',COLORS[0]),('negative_volume_finite_window','Negative volume',COLORS[1]),('stabilizer_p','|S_p|',COLORS[2]),('stabilizer_q','|S_q|',COLORS[3])]:axs[1,2].plot(snapsteps,[s[key] for s in stats],'o-',label=label,color=c)
    axs[1,2].set(title='Computed trajectory; no uncertainty intervals claimed',xlabel='Completed updates',ylabel='Dimensionless observable');axs[1,2].legend(fontsize=8)
    metrics['trajectory']={str(n):s for n,s in zip(snapsteps,stats)};figures.append(fig)

    fig,axs=plt.subplots(2,3,figsize=(17,10),layout='constrained');fig.suptitle('6  Historical supplied context: first five cached author analyses; final panel fresh real-data inverse')
    hs=['111','133','335','444','333'];common=max([float(abs(historical['W_'+s]).max()) for s in hs]+[float(abs(fresh['wigner']).max())])
    for ax,label in zip(axs.flat,hs):
        _wplot(fig,ax,historical['W_'+label],'Historical release, heralding '+label,common)
        numeric['historical_W_'+label]=historical['W_'+label];numeric['historical_rho_'+label]=historical['rho_'+label]
    _wplot(fig,axs[1,2],fresh['wigner'],'Fresh 333: all 100 raw acquisition blocks',common)
    metrics['historical_context']={'classification':'Author cached analyses, never observations/truth/fresh methods; cached parameters not independently documented',
        'cached_333_stabilizer_p':float(historical['stabilizer_p_expectation_value']),
        'cached_333_stabilizer_q':float(historical['stabilizer_q_expectation_value']),
        'cached_333_effective_squeezing_dB':[float(historical['p_dB']),float(historical['q_dB']),float(historical['symm_dB'])]}
    figures.append(fig)
    numeric.update(upstream_density=ref['density_matrix'],upstream_W=ref['wigner'],Q=Q_GRID,P=P_GRID)
    if directory is not None:
        directory=Path(directory);directory.mkdir(parents=True,exist_ok=True)
        for i,fig in enumerate(figures,1):
            for ext in ('png','svg','pdf'):fig.savefig(directory/f'figure{i}.{ext}',dpi=180,bbox_inches='tight')
        np.savez_compressed(directory/'figure_numeric_source.npz',**numeric)
        (directory/'figure_metrics.json').write_text(json.dumps(metrics,indent=2,allow_nan=False),encoding='utf-8')
        provenance={'arrays':'figure_numeric_source.npz','metrics':'figure_metrics.json','fresh_methods':['Complete calibrated native31state inverse','Full-data lower photon supports10/20','Full-data omitted-phase-calibration inverse','Actual native iteration snapshots'],
            'historical_context':'Only figure6first5panels and explicit historical report fields; no fresh/source parity claims for cached state data',
            'figure_array_mapping':{'1':['variance_raw_phase','normalized_signal_variance','phase_fit_param','confidence_hist_*','*_histogram_density'],
            '2':['upstream_W','Fresh_native_wigner','Fresh_native_density_matrix','*_marginal_*'],'3':['Cutoff_10_*','Cutoff_20_*','Fresh_native_*'],
            '4':['Uncorrected_phase_*','Fresh_native_*','phase_fit_RMS_residual','raw_phase_RMS_residual'],'5':['snapshot_W_*','Fresh_native_likelihood','Fresh_native_step_fidelity','metrics:trajectory'],'6':['historical_W_*','Fresh_native_wigner']},
            'numeric_sha256':hashlib.sha256((directory/'figure_numeric_source.npz').read_bytes()).hexdigest()}
        (directory/'figure_provenance.json').write_text(json.dumps(provenance,indent=2),encoding='utf-8')
    return figures,metrics
