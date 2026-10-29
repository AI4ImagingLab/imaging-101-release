"""Scientific comparisons; importing this module leaves Matplotlib backend untouched."""
import hashlib,json
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from .physics_model import EPOCHS,LATITUDES,LONGITUDES,field_maps,temporal_basis
from .solvers import reconstruct,damping_matrices

def metric_pair(estimate,reference):
    a=np.asarray(estimate).ravel(); b=np.asarray(reference).ravel()
    return {'ncc':float(np.dot(a,b)/(np.linalg.norm(a)*np.linalg.norm(b))) if np.linalg.norm(a)>0 else 0.,'nrmse':float(np.sqrt(np.mean((a-b)**2))/np.ptp(b))}

def scientific_review(data,coefficients,diagnostics,upstream_original,upstream_corrected,export_dir=None):
    """Build five actually computed figure groups; return figures,numeric arrays,metrics."""
    maps=field_maps(coefficients); original_maps=field_maps(upstream_original)
    corrected_maps=field_maps(upstream_corrected)
    numeric={'fresh_coefficients':coefficients,'fresh_maps':maps,'upstream_original_coefficients':upstream_original,'upstream_corrected_coefficients':upstream_corrected,'upstream_original_coefficients_on_correct_physics_maps':original_maps,'upstream_corrected_maps':corrected_maps,'epochs':EPOCHS,'latitudes':LATITUDES,'longitudes':LONGITUDES,'observed_values':data[2],'sigma':data[3],'kind':data[4],'predicted_values':diagnostics['predicted'],'weighted_mae':diagnostics['weighted_mae']}
    results={'fresh_vs_corrected_coefficients':metric_pair(coefficients,upstream_corrected),'original_vs_corrected_coefficients':metric_pair(upstream_original,upstream_corrected)}
    figures={}
    fig,ax=plt.subplots(2,3,figsize=(13,7),constrained_layout=True)
    for row,idx in enumerate([0,4]):
        for comp,name in enumerate(['North','East','Down']):
            limit=np.max(np.abs(maps[[0,4],comp]))/1000
            im=ax[row,comp].pcolormesh(LONGITUDES,LATITUDES,maps[idx,comp]/1000,cmap='RdBu_r',vmin=-limit,vmax=limit,shading='nearest')
            ax[row,comp].set(title=f'{name} | year {EPOCHS[idx]:.0f} CE',xlabel='Longitude [deg]',ylabel='Geodetic latitude [deg]')
            fig.colorbar(im,ax=ax[row,comp],label='Magnetic field [µT]')
    figures['01_global_vector_fields']=fig
    fig,ax=plt.subplots(1,3,figsize=(13,3.8),constrained_layout=True)
    for axis,kind,title in zip(ax,[4,5,6],['Intensity','Inclination','Declination']):
        mask=data[4]==kind; obs=data[2][mask]; pred=diagnostics['predicted'][mask]
        scale=.001 if kind==4 else 180/np.pi; unit='µT' if kind==4 else 'deg'
        axis.scatter(obs*scale,pred*scale,s=3,alpha=.35,rasterized=True)
        lim=[min(obs.min(),pred.min())*scale,max(obs.max(),pred.max())*scale]
        axis.plot(lim,lim,color='black',lw=1); axis.set(title=title,xlabel=f'Observed [{unit}]',ylabel=f'Fresh modeled [{unit}]')
        residual=obs-pred
        if kind==6: residual=(residual+np.pi)%(2*np.pi)-np.pi
        results[f'{title.lower()}_weighted_rms']=float(np.sqrt(np.mean((residual/data[3][mask])**2)))
    figures['02_observation_consistency']=fig
    time=np.linspace(-1800,1800,73); temp=temporal_basis(time)
    numeric['coefficient_times']=time; numeric['fresh_gauss_timeseries']=temp@coefficients
    fig,ax=plt.subplots(1,3,figsize=(13,3.8),constrained_layout=True)
    for i,(axis,label) in enumerate(zip(ax,['g10','g11','h11'])):
        axis.plot(time,(temp@upstream_original)[:,i]/1000,'--',label='Unmodified original run')
        axis.plot(time,(temp@upstream_corrected)[:,i]/1000,label='Corrected oracle run')
        axis.plot(time,(temp@coefficients)[:,i]/1000,':',label='Fresh cleaned run')
        axis.set(title=label,xlabel='Year CE',ylabel='Gauss coefficient [µT]')
    ax[0].legend(fontsize=7); figures['03_independent_runs_and_evolution']=fig
    fig,ax=plt.subplots(1,2,figsize=(11,4),constrained_layout=True)
    ax[0].plot(np.arange(1,len(diagnostics['weighted_mae'])+1),diagnostics['weighted_mae'],'o-')
    ax[0].set(xlabel='Gauss–Newton linearization',ylabel='Mean |residual / sigma|',title='Fresh convergence (before each update)')
    damping=[1e-4,1e-3,1e-2]; sweep=[]; s,t=damping_matrices(); energies=[]; residuals=[]
    for d in damping:
        c,diag=reconstruct(data,temporal_damping=d); sweep.append(c)
        res=data[2]-diag['predicted']; mask=data[4]==6; res[mask]=(res[mask]+np.pi)%(2*np.pi)-np.pi
        residuals.append(np.sqrt(np.mean((res/data[3])**2))); energies.append(c.ravel()@t@c.ravel()/4000)
    numeric.update(damping_sweep=np.array(damping),damping_sweep_coefficients=np.array(sweep),damping_sweep_weighted_rms=np.array(residuals),damping_sweep_acceleration=np.array(energies))
    ax[1].plot(energies,residuals,'o-')
    for x,y,d in zip(energies,residuals,damping): ax[1].annotate(f' τ={d:g}',(x,y))
    ax[1].margins(x=.18)
    ax[1].set(xlabel='Acceleration quadratic form / duration [model units]',ylabel='Weighted RMS residual',title='Actually recomputed temporal-prior sensitivity')
    figures['04_convergence_and_prior_sensitivity']=fig
    fig,ax=plt.subplots(1,2,figsize=(11,4),constrained_layout=True)
    coverage=ax[0].scatter(np.rad2deg(data[1][:,1]),90-np.rad2deg(data[1][:,0]),c=data[0],s=5,cmap='viridis',rasterized=True)
    ax[0].set(xlabel='Longitude [deg]',ylabel='Geocentric latitude [deg]',title='Irregular observation coverage')
    fig.colorbar(coverage,ax=ax[0],label='Observation year CE')
    delta=maps[2,2]-original_maps[2,2]
    im=ax[1].pcolormesh(LONGITUDES,LATITUDES,delta,shading='nearest',cmap='RdBu_r'); fig.colorbar(im,ax=ax[1],label='Down-field change [nT]')
    ax[1].set(xlabel='Longitude [deg]',ylabel='Geodetic latitude [deg]',title='Physics correction impact on inferred coefficients\nBoth coefficient sets evaluated with correct field law')
    numeric['correction_impact_down_field']=delta
    figures['05_coverage_and_physics_corrections']=fig
    results['temporal_prior_sweep']=[{'tau':d,'weighted_rms':r,'acceleration':e} for d,r,e in zip(damping,residuals,energies)]
    fig,ax=plt.subplots(1,2,figsize=(11,4),constrained_layout=True)
    from scipy.interpolate import BSpline
    from .physics_model import KNOTS
    gauss=temporal_basis(EPOCHS)@coefficients
    sv=BSpline(KNOTS,coefficients,3).derivative()(EPOCHS)
    powers=[]; velocities=[]
    for epoch,row,dr in zip(EPOCHS,gauss,sv):
        power=[]; vel=[]; start=0
        for degree in range(1,4):
            stop=start+2*degree+1
            power.append((degree+1)*np.sum(row[start:stop]**2))
            vel.append((degree+1)*np.sum(dr[start:stop]**2)); start=stop
        powers.append(power); velocities.append(vel)
        ax[0].semilogy([1,2,3],power,'o-',label=f'{epoch:.0f} CE')
        ax[1].semilogy([1,2,3],vel,'o-',label=f'{epoch:.0f} CE')
    ax[0].set(xlabel='Spherical harmonic degree',ylabel='Surface field power [nT²]',title='Resolved spatial power')
    ax[1].set(xlabel='Spherical harmonic degree',ylabel='Secular-variation power [nT² / year²]',title='Actually inferred time variation')
    for axis in ax: axis.set_xticks([1,2,3])
    ax[0].legend(fontsize=7)
    numeric.update(degree_power=np.array(powers),degree_sv_power=np.array(velocities))
    figures['06_spatial_and_secular_variation_power']=fig
    if export_dir is not None:
        out=Path(export_dir); out.mkdir(parents=True,exist_ok=True)
        for name,figure in figures.items():
            for ext in ['png','svg','pdf']: figure.savefig(out/f'{name}.{ext}',dpi=170)
        np.savez_compressed(out/'numeric_sources.npz',**numeric)
        (out/'metrics.json').write_text(json.dumps(results,indent=2,allow_nan=False),encoding='utf8')
        manifest={f.name:hashlib.sha256(f.read_bytes()).hexdigest() for f in out.iterdir() if f.is_file() and f.name!='manifest.json'}
        (out/'manifest.json').write_text(json.dumps({'figures':list(figures),'sha256':manifest,'provenance':'Fresh computation from src modules; explicitly labeled independent original/corrected upstream captures; no truth is claimed.'},indent=2),encoding='utf8')
    return figures,numeric,results
