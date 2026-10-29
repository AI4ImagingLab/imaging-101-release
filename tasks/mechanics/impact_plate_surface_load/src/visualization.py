"""Numeric scientific review of measured deflection and reconstructed load."""
from pathlib import Path
import hashlib,json
import numpy as np
import matplotlib.pyplot as plt
from .preprocessing import PIXEL_SIZE,SAMPLING_RATE,filter_deflection
from .physics_model import kinematics,hermite_virtual_fields
from .solvers import reconstruct,pressure_from_fields

def ncc_nrmse(a,b):
    """Cosine NCC and dynamic-range-normalized RMS, matching public contract."""
    a=a.ravel();b=b.ravel();return float(np.dot(a,b)/(np.linalg.norm(a)*np.linalg.norm(b))),float(np.sqrt(np.mean((a-b)**2))/np.ptp(b))

def _image(fig,ax,a,title,unit,offset=0.,cmap='coolwarm',limit=None):
    extent=[offset*1e3,(offset+(a.shape[1]-1)*PIXEL_SIZE)*1e3,(offset+(a.shape[0]-1)*PIXEL_SIZE)*1e3,offset*1e3]
    m=ax.imshow(a,extent=extent,origin='upper',cmap=cmap,vmin=-limit if limit else None,vmax=limit,interpolation='nearest');ax.set(title=title,xlabel='column-coordinate y (mm)',ylabel='row-coordinate x (mm)');fig.colorbar(m,ax=ax,shrink=.8,label=unit)

def _force(p,window=20):
    start=20+(20-window)//2
    return np.sum(p[:,start:start+30,start:start+30],axis=(1,2))*PIXEL_SIZE**2

def make_figures(observations,result,upstream,hammer_time,hammer_force):
    """Return five figures, exact plotted numeric sources and fresh metric values.

    All inputs are arrays; no hidden loads or backend selection occurs here.
    Parameter variants are fresh full250-frame reconstructions of the same data.
    """
    obs=observations[0];p=result['reconstruction'][0];ref=upstream['reconstruction'][0];times=np.arange(len(p))/SAMPLING_RATE;tms=times*1e3
    uncorrected=upstream.get('uncorrected_pipeline_reconstruction',upstream['reconstruction'])[0];uncorrected_force=_force(uncorrected)
    core=_force(p);whole=np.sum(p,axis=(1,2))*PIXEL_SIZE**2;hf=np.interp(times,hammer_time,hammer_force);index=int(np.argmax(core));frames=np.array([0,max(0,index-25),index,min(len(p)-1,index+40)])
    stack=kinematics(filter_deflection(observations)[0]);virtual=hermite_virtual_fields();bending=[];inertial=[]
    for i in range(len(p)):
        _,parts=pressure_from_fields({k:v[i] for k,v in stack.items() if k!='time'},virtual);bending.append(parts['bending']);inertial.append(parts['inertial'])
    bending=np.array(bending);inertial=np.array(inertial)
    nc,nr=ncc_nrmse(p,ref);_,ferr=ncc_nrmse(core,hf);_,werr=ncc_nrmse(whole,hf)
    metrics={'pressure_ncc_vs_upstream':nc,'pressure_dynamic_range_nrmse_vs_upstream':nr,'force_core_dynamic_range_nrmse_vs_loadcell':ferr,'force_whole_dynamic_range_nrmse_vs_loadcell':werr,'force_core_peak_N':float(core.max()),'hammer_peak_N':float(hammer_force.max()),'force_core_peak_time_ms':float(tms[index]),'hammer_peak_time_ms':float(hammer_time[np.argmax(hammer_force)]*1e3),'force_core_impulse_Ns':float(np.trapezoid(core,times)),'force_hammer_impulse_Ns':float(np.trapezoid(hf,times)),'whole_force_tail_limitation':'All-field integral is retained and displayed; boundary noise and repeated differentiation create a large late-time tail.'}
    numeric={'times_s':times,'selected_frame_indices':frames,'observed_deflection_selected_m':obs[frames],'pressure_selected_Pa':p[frames],'original_pressure_selected_Pa':ref[frames],'core_force_N':core,'uncorrected_original_pipeline_core_force_N':uncorrected_force,'whole_force_N':whole,'hammer_time_s':hammer_time,'hammer_force_N':hammer_force,'hammer_interpolated_N':hf,'bending_pressure_peak_Pa':bending[index],'inertial_pressure_peak_Pa':inertial[index],'core_bending_force_N':_force(bending),'core_inertial_force_N':_force(inertial),'plane_pressure_rmse_vs_upstream_Pa':np.sqrt(np.mean((p-ref)**2,axis=(1,2)))}
    for k,v in virtual.items():numeric['virtual_'+k]=v
    for k in ['slope_x','slope_y','curv_xx','curv_yy','curv_xy','acceleration']:numeric['peak_'+k]=stack[k][index]
    figures=[];fig,axes=plt.subplots(2,3,figsize=(13,8),layout='constrained');fig.suptitle('Real impact experiment | optically estimated plate deflection and local kinematics',fontsize=14)
    for j,i in enumerate(frames[:3]):_image(fig,axes[0,j],obs[i]*1e6,f'Estimated deflection at {tms[i]:.3f} ms','micrometer')
    _image(fig,axes[1,0],stack['slope_x'][index]*1e3,'Peak-force slope in row direction','mrad')
    _image(fig,axes[1,1],stack['curv_xx'][index],'Peak-force row curvature','1/m')
    _image(fig,axes[1,2],stack['acceleration'][index],'Peak-force transverse acceleration','m/s^2');figures.append(fig)
    fig,axes=plt.subplots(2,3,figsize=(13,8),layout='constrained');fig.suptitle('Fresh full-field pressure | moving Hermite patch, bending and inertia',fontsize=14);lim=float(np.max(np.abs(p[frames])))/1e3
    for j,i in enumerate(frames):_image(fig,axes.flat[j],p[i]/1e3,f'Pressure at {tms[i]:.3f} ms','kPa',9.5*PIXEL_SIZE,limit=lim)
    _image(fig,axes[1,1],bending[index]/1e3,'Bending contribution at peak force','kPa',9.5*PIXEL_SIZE)
    _image(fig,axes[1,2],inertial[index]/1e3,'Inertial contribution at peak force','kPa',9.5*PIXEL_SIZE);figures.append(fig)
    fig,axes=plt.subplots(2,3,figsize=(13,8),layout='constrained');fig.suptitle('Independent load-cell comparison | fixed clock, scale and source-defined region',fontsize=14)
    ax=axes[0,0];ax.plot(tms,whole,label='whole valid field');ax.plot(tms,core,label='fixed source subregion');ax.plot(hammer_time*1e3,hammer_force,label='independent impact hammer');ax.set(title='All three original experimental comparisons',xlabel='source local time (ms)',ylabel='force (N)');ax.legend(fontsize=8)
    ax=axes[0,1];ax.plot(tms,core,label='fresh local corrected-observation VFM');ax.plot(tms,uncorrected_force,':',label='unmodified full original pipeline');ax.plot(hammer_time*1e3,hammer_force,label='independent load cell');ax.set(xlim=(.8,3.),title='Load-region comparison at original scale',xlabel='time (ms)',ylabel='force (N)');ax.legend(fontsize=8)
    ax=axes[0,2];ax.plot(tms,core-hf,label='subregion error');ax.plot(tms,whole-hf,label='whole-field error');ax.set(title='Errors over all 250 frames',xlabel='time (ms)',ylabel='force error (N)');ax.legend(fontsize=8)
    ax=axes[1,0];ax.plot(tms,_force(bending),label='bending');ax.plot(tms,_force(inertial),label='inertial');ax.plot(tms,core,label='sum');ax.set(title='Elastic and inertial force decomposition',xlabel='time (ms)',ylabel='force (N)');ax.legend(fontsize=8)
    ax=axes[1,1];ax.bar(['local region','whole field'],[ferr,werr],color=['tab:blue','tab:orange']);ax.set(title='Dynamic-range NRMSE vs load cell',ylabel='dimensionless error');ax.axhline(.1,color='gray',ls=':',label='0.10 reference level');ax.legend(fontsize=8)
    ax=axes[1,2];ax.plot(tms,numeric['plane_pressure_rmse_vs_upstream_Pa']);ax.ticklabel_format(axis='y',style='sci',scilimits=(0,0));ax.set(title='Fresh minus independently run source',xlabel='time (ms)',ylabel='pressure RMSE (Pa)');figures.append(fig)
    fig,axes=plt.subplots(2,3,figsize=(13,8),layout='constrained');fig.suptitle('Physics and resolution | explicit Hermite virtual displacement and curvatures',fontsize=14)
    for j,key in enumerate(['deflection','curv_xx','curv_xy']):_image(fig,axes[0,j],virtual[key],f'Virtual {key}','dimensionless' if key=='deflection' else '1/m^2')
    center=virtual['deflection'].shape[0]//2;coord=(np.arange(20)+.5)*PIXEL_SIZE*1e3;ax=axes[1,0];ax.plot(coord,virtual['deflection'][center]);ax.set(title='Virtual-displacement line profile',xlabel='patch coordinate (mm)',ylabel='dimensionless')
    ax=axes[1,1];ax.plot(tms,np.max(np.abs(obs),axis=(1,2))*1e6);ax.set(title='Observed peak absolute deflection',xlabel='time (ms)',ylabel='micrometer')
    ax=axes[1,2];ax.plot(tms,np.max(np.abs(stack['acceleration']),axis=(1,2)));ax.set(title='Filtered peak absolute acceleration',xlabel='time (ms)',ylabel='m/s^2');numeric['peak_abs_deflection_m']=np.max(np.abs(obs),axis=(1,2));numeric['peak_abs_acceleration_m_per_s2']=np.max(np.abs(stack['acceleration']),axis=(1,2));figures.append(fig)
    fig,axes=plt.subplots(2,3,figsize=(13,8),layout='constrained');fig.suptitle('Fresh parameter sensitivity | same optical observations, fixed physical load region',fontsize=14)
    sigma_errors=[];window_errors=[]
    for sigma in [3.,6.,9.]:
        variant=p if sigma==6 else reconstruct(observations,time_sigma=sigma)['reconstruction'][0]
        force=_force(variant);numeric[f'force_sigma_{int(sigma)}_N']=force;_,e=ncc_nrmse(force,hf);sigma_errors.append(e);axes[0,0].plot(tms,force,label=f'sigma {int(sigma)} frames')
    axes[0,0].plot(tms,hf,'k--',label='loadcell');axes[0,0].set(title='Temporal bandwidth and force recovery',xlabel='time (ms)',ylabel='force (N)');axes[0,0].legend(fontsize=8)
    for win in [12,20,28]:
        variant=p if win==20 else reconstruct(observations,window=win)['reconstruction'][0]
        force=_force(variant,win);numeric[f'force_window_{win}_N']=force;_,e=ncc_nrmse(force,hf);window_errors.append(e);axes[0,1].plot(tms,force,label=f'{win} x {win} patch')
    axes[0,1].plot(tms,hf,'k--',label='loadcell');axes[0,1].set(title='Spatial virtual patch and force recovery',xlabel='time (ms)',ylabel='force (N)');axes[0,1].legend(fontsize=8)
    axes[0,2].bar(['sigma3','sigma6','sigma9'],sigma_errors);axes[0,2].set(title='Temporal-filter sensitivity vs loadcell',ylabel='dynamic-range NRMSE')
    axes[1,0].bar(['12x12','20x20','28x28'],window_errors);axes[1,0].set(title='Virtual-patch sensitivity vs loadcell',ylabel='dynamic-range NRMSE')
    _image(fig,axes[1,1],p[index]/1e3,'Canonical pressure and fixed load region','kPa',9.5*PIXEL_SIZE)
    from matplotlib.patches import Rectangle
    axes[1,1].add_patch(Rectangle(((20+9.5)*PIXEL_SIZE*1e3,(20+9.5)*PIXEL_SIZE*1e3),30*PIXEL_SIZE*1e3,30*PIXEL_SIZE*1e3,fill=False,edgecolor='black',lw=1.5))
    axes[1,2].plot(tms,numeric['force_sigma_3_N']-numeric['force_sigma_6_N'],label='sigma3minus6');axes[1,2].plot(tms,numeric['force_window_28_N']-numeric['force_window_20_N'],label='patch28minus20');axes[1,2].set(title='Computed sensitivity differences',xlabel='time (ms)',ylabel='force change (N)');axes[1,2].legend(fontsize=8)
    numeric['sigma_values']=np.array([3.,6.,9.]);numeric['sigma_force_errors']=np.array(sigma_errors);numeric['window_values']=np.array([12,20,28]);numeric['window_force_errors']=np.array(window_errors);metrics['sigma_force_errors']=sigma_errors;metrics['window_force_errors']=window_errors;figures.append(fig)
    for fig in figures:
        for ax in fig.axes:ax.title.set_fontsize(10)
    return figures,numeric,metrics

def export_figures(figures,numeric,metrics,directory):
    directory=Path(directory);directory.mkdir(parents=True,exist_ok=True)
    for i,fig in enumerate(figures,1):
        for ext in ['png','svg','pdf']:fig.savefig(directory/f'figure_{i}.{ext}',dpi=150)
    np.savez_compressed(directory/'numeric_sources.npz',**numeric);(directory/'metrics.json').write_text(json.dumps(metrics,indent=2,allow_nan=False)+'\n',encoding='utf8')
    manifest={'source':'Fresh local canonical inverse, full250frame freshly recomputed sensitivities, separately captured original pressure baseline and independent measured hammer trace; no fitted clock or amplitude.','sha256':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(directory.iterdir()) if p.is_file() and p.name!='hash_manifest.json'}};(directory/'hash_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n',encoding='utf8')
