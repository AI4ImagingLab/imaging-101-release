"""Active scientific comparisons and reproducible numeric-source figure exports."""
import hashlib,json,time
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from .solvers import reconstruct
from .preprocessing import normalize_clock
from .physics_model import accumulate_events

def metrics(x,reference):
    u=x.ravel();v=reference.ravel();den=np.linalg.norm(u)*np.linalg.norm(v)
    return dict(ncc=float(u@v/den) if den else 0.,nrmse=float(np.sqrt(np.mean((u-v)**2))/np.ptp(v)))

def run_experiments(events,angular_velocity,intrinsics):
    start=time.perf_counter();images={};states={}
    images['TV'],states['TV']=reconstruct(events,angular_velocity,intrinsics)
    images['Laplacian'],states['Laplacian']=reconstruct(events,angular_velocity,intrinsics,method='Laplacian')
    tau,duration=normalize_clock(events)
    images['seconds_api'],states['seconds_api']=reconstruct(events,angular_velocity,intrinsics,warp_fraction=duration)
    weights=np.array([.03,.1,.3,1.]);weight_images=[];weight_diagnostics=[]
    for weight in weights:
        x,z=reconstruct(events,angular_velocity,intrinsics,weight=float(weight));weight_images.append(x)
        weight_diagnostics.append([np.linalg.norm(z['A']@z['raw_log_image'].ravel()-z['niwe'].ravel())/np.linalg.norm(z['niwe']),z['history'][-1,1]])
    factors=np.array([.8,1.,1.2]);motion_images=[];motion_energy=[]
    for factor in factors:
        x,z=reconstruct(events,angular_velocity*factor,intrinsics);motion_images.append(x);motion_energy.append(np.mean(z['iwe']**2))
    raw_iwe=accumulate_events(tau,np.zeros_like(states['TV']['displacement']))
    return dict(images=images,states=states,weights=weights,weight_images=np.array(weight_images),weight_diagnostics=np.array(weight_diagnostics),factors=factors,motion_images=np.array(motion_images),motion_energy=np.array(motion_energy),raw_iwe=raw_iwe,events=events,angular_velocity=angular_velocity,intrinsics=intrinsics,seconds=time.perf_counter()-start)

def make_figures(result,references,export_dir=None):
    plt.rcParams.update({'font.size':10,'axes.titlesize':11,'figure.facecolor':'white','savefig.facecolor':'white'})
    images=result['images'];state=result['states']['TV'];events=result['events'];figures=[];numbers={}
    for name,x in images.items():numbers['fresh_'+name]=x
    for name,x in references.items():numbers['upstream_'+name]=x
    for key in ['events','angular_velocity','intrinsics','weights','weight_images','weight_diagnostics','factors','motion_images','motion_energy','raw_iwe']:numbers[key]=result[key]
    for key in ['iwe','niwe','displacement','unit_flowx','unit_flowy','flow_magnitude','history','raw_log_image']:numbers[key]=state[key]
    table={name:metrics(images[name],references[name]) for name in ['TV','Laplacian']}
    def show(ax,image,title,cmap='gray',vmin=None,vmax=None,unit=None):
        im=ax.imshow(image,cmap=cmap,vmin=vmin,vmax=vmax,origin='upper');ax.set(title=title,xlabel='x (pixel)',ylabel='y (pixel)')
        if unit:ax.figure.colorbar(im,ax=ax,shrink=.75,label=unit)
    # Actual event coordinates, time and independent rotational instrument geometry.
    f,a=plt.subplots(2,3,figsize=(14,8),layout='constrained');f.suptitle('Real rotational event acquisition and independent IMU calibration',fontsize=15)
    pos=events[:,3]>0;bins=[np.arange(181),np.arange(241)]
    densityp=np.histogram2d(events[pos,2],events[pos,1],bins=bins)[0];densitym=np.histogram2d(events[~pos,2],events[~pos,1],bins=bins)[0]
    numbers['positive_density']=densityp;numbers['negative_density']=densitym
    show(a[0,0],densityp,'Positive events',cmap='magma',unit='events / pixel');show(a[0,1],densitym,'Negative events',cmap='magma',unit='events / pixel')
    a[0,2].plot((events[:,0]-events[0,0])*1e3,np.arange(len(events)),color='#0072B2');a[0,2].set(xlabel='Time since first event (ms)',ylabel='Cumulative events',title=f'{len(events):,} measured events')
    yy,xx=np.mgrid[:180:15,:240:15];d=state['displacement'];a[1,0].quiver(xx,yy,d[0,::15,::15],d[1,::15,::15],angles='xy',scale_units='xy',scale=1,color='#0072B2');a[1,0].set(xlim=(0,239),ylim=(179,0),xlabel='x (pixel)',ylabel='y (pixel)',title='Independent gyro-derived window displacement')
    lim=np.percentile(abs(state['iwe']),99)
    show(a[1,1],result['raw_iwe'],'Uncompensated signed accumulation','RdBu_r',-lim,lim,'signed events / pixel');show(a[1,2],state['iwe'],'Correctly motion-compensated IWE','RdBu_r',-lim,lim,'signed events / pixel')
    figures.append(('01_real_events_geometry',f))
    # Exact selected bound-original comparisons; references are method results, not truth.
    f,a=plt.subplots(2,3,figsize=(14,7.7),layout='constrained');f.suptitle('Independent original method and fresh local reconstruction on identical real observations',fontsize=14)
    for row,name in enumerate(['TV','Laplacian']):
        show(a[row,0],references[name],f'Qualified original {name}',vmin=0,vmax=1,unit='relative log radiance')
        show(a[row,1],images[name],f'Fresh local {name}',vmin=0,vmax=1,unit='relative log radiance')
        delta=images[name]-references[name];numbers[name+'_parity_difference']=delta
        error_limit=max(float(np.max(abs(delta))),1e-16)
        show(a[row,2],delta,f'Fresh minus original; max {abs(delta).max():.2e}','RdBu_r',-error_limit,error_limit,'relative log radiance')
    figures.append(('02_original_local_parity',f))
    # Physical observation residual in the unnormalized reconstruction's event units.
    f,a=plt.subplots(2,3,figsize=(14,7.8),layout='constrained');f.suptitle('Directional image-gradient constraints and measured residuals',fontsize=15)
    b=state['niwe'];pred=(state['A']@state['raw_log_image'].ravel()).reshape(180,240);error=pred-b;lim=np.percentile(abs(b),99)
    numbers['directional_prediction']=pred;numbers['directional_residual']=error
    for ax,x,title in zip(a[0],[b,pred,error],['Measured normalized IWE','Predicted one-pixel log difference','Predicted minus measured']):show(ax,x,title,'RdBu_r',-lim,lim,'unit-contrast log change / pixel')
    a[1,0].hist(error.ravel(),bins=100,color='#0072B2',alpha=.8);a[1,0].set(xlabel='Residual (unit-contrast log change / pixel)',ylabel='Pixel count',title=f'Relative L2 residual {np.linalg.norm(error)/np.linalg.norm(b):.4f}')
    a[1,1].plot(b[90],color='#0072B2',label='Measured NIWE');a[1,1].plot(pred[90],color='#D55E00',label='Predicted');a[1,1].set(xlabel='x (pixel)',ylabel='Unit-contrast log change / pixel',title='Center-row directional constraint');a[1,1].legend()
    parity=[table[k]['nrmse'] for k in ['TV','Laplacian']];a[1,2].bar(['TV','Laplacian'],parity,color=['#0072B2','#D55E00']);a[1,2].set(ylabel='Dynamic-range NRMSE vs original',title='Numerical extraction agreement')
    figures.append(('03_observation_consistency',f))
    # Regularization is changed only in explicitly labeled, actively reconstructed ablations.
    f,a=plt.subplots(2,3,figsize=(14,7.8),layout='constrained');f.suptitle('Fresh split-Bregman convergence and regularization sensitivity',fontsize=15)
    h=state['history'];a[0,0].plot(np.arange(1,len(h)+1),h[:,0]+.01*h[:,1],marker='o',color='#0072B2');a[0,0].set(xlabel='Outer Bregman iteration',ylabel='Data term + 0.01 × anisotropic TV',title='Declared effective objective')
    a[0,1].semilogy(np.arange(1,len(h)+1),h[:,2],marker='o',color='#009E73');a[0,1].set(xlabel='Outer Bregman iteration',ylabel='L2 norm of outer update',title='Actual stopping history')
    a[0,2].semilogx(result['weights'],result['weight_diagnostics'][:,0],marker='o',color='#D55E00');a[0,2].axvline(.1,color='black',ls='--',label='Standard');a[0,2].set(xlabel='Original TV parameter λ (dimensionless)',ylabel='Relative NIWE L2 residual',title='Data / prior tradeoff');a[0,2].legend()
    for ax,index in zip(a[1],[0,1,3]):show(ax,result['weight_images'][index],f'Fresh TV λ={result["weights"][index]:g}',vmin=0,vmax=1,unit='relative log radiance')
    figures.append(('04_convergence_regularization',f))
    # The earlier code's seconds-clock mismatch remains visible as an actual comparison.
    f,a=plt.subplots(2,3,figsize=(14,7.8),layout='constrained');f.suptitle('Timing conventions and independent-motion sensitivity: every map freshly computed',fontsize=14)
    show(a[0,0],images['seconds_api'],'Seconds-clock control; corrected pixel lookup',vmin=0,vmax=1,unit='relative log radiance')
    show(a[0,1],images['TV'],'Dimensionally corrected window clock',vmin=0,vmax=1,unit='relative log radiance')
    show(a[0,2],images['TV']-images['seconds_api'],'Clock-adaptation image difference','RdBu_r',-.3,.3,'relative log radiance')
    for ax,index in zip(a[1,:2],[0,2]):show(ax,result['motion_images'][index],f'Independent gyro scale × {result["factors"][index]:g}',vmin=0,vmax=1,unit='relative log radiance')
    a[1,2].plot(result['factors'],result['motion_energy'],marker='o',color='#0072B2');a[1,2].axvline(1,color='black',ls='--',label='Recorded gyro');a[1,2].axhline(np.mean(result['raw_iwe']**2),color='#D55E00',ls=':',label='No compensation');a[1,2].set(xlabel='Gyro calibration multiplier',ylabel='Mean squared signed IWE (events² / pixel²)',title='Measured edge concentration');a[1,2].legend()
    figures.append(('05_timing_motion_sensitivity',f))
    if export_dir:
        dest=Path(export_dir);dest.mkdir(parents=True,exist_ok=True)
        for name,f in figures:
            for extension in ['png','svg','pdf']:f.savefig(dest/(name+'.'+extension),dpi=180)
        np.savez_compressed(dest/'numeric_sources.npz',**numbers)
        (dest/'metrics.json').write_text(json.dumps(dict(parity=table,active_experiment_seconds=result['seconds'],clock_convention='unit tau with pixel/window displacement'),indent=2),encoding='utf8')
        hashes={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in dest.iterdir() if p.is_file() and p.name!='figure_manifest.json'}
        (dest/'figure_manifest.json').write_text(json.dumps(dict(groups=[n for n,_ in figures],source='active canonical notebook with local src modules',reference='independent unit-adapted original method on the same real IJRR boxes window; not physical truth',files=hashes),indent=2),encoding='utf8')
    return figures,table
