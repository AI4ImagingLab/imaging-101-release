"""Fresh bound-data scientific comparisons; no plotting backend side effects."""
from pathlib import Path
import json,time
import numpy as np
import matplotlib.pyplot as plt
from .solvers import reconstruct
from .physics_model import diagnostics,forward_spectrum

def metrics(x,y):
    result={}
    for i,name in enumerate(['jx','jy']):
        if np.linalg.norm(y[i])==0:raise ValueError('Undefined zero reference norm')
        result['ncc_'+name]=float(np.vdot(x[i].ravel(),y[i].ravel()).real/(np.linalg.norm(x[i])*np.linalg.norm(y[i]))) if np.linalg.norm(x[i]) else 0.
        result['relative_l2_'+name]=float(np.linalg.norm(x[i]-y[i])/np.linalg.norm(y[i]))
    return result

def run_experiments(data,corrected_references,unmodified):
    start=time.perf_counter();currents=[];stages=[];rows=[]
    for cutoff in [.5,1.,1.5]:
        t=time.perf_counter();out,stage=reconstruct(data,cutoff=cutoff);elapsed=time.perf_counter()-t
        current=out['reconstruction'][0];m=metrics(current,corrected_references[len(currents)])
        currents.append(current);stages.append(stage);rows.append([cutoff,m['ncc_jx'],m['ncc_jy'],m['relative_l2_jx'],m['relative_l2_jy'],elapsed])
    nominal=currents[1];stage=stages[1];diagnostic=diagnostics(nominal,data)
    heights=[]
    for height in [.5,1.,1.5]:heights.append(reconstruct(data,standoff=height)[0]['reconstruction'][0])
    no_psf=reconstruct(data,use_psf=False)[0]['reconstruction'][0]
    constant=reconstruct(data,pad_mode='constant')[0]['reconstruction'][0]
    legacy=unmodified['reconstruction'][0]
    legacy_diag=diagnostics(legacy.real,data)
    legacy_imag=[float(np.linalg.norm(z.imag)/np.linalg.norm(z.real)) for z in legacy]
    return {'currents':np.asarray(currents),'stages':stages,'table':np.asarray(rows),'heights':np.asarray(heights),'no_psf':no_psf,'constant_boundary':constant,'legacy':legacy,'diagnostic':diagnostic,'metrics':{**metrics(nominal,corrected_references[1]),**{k:v for k,v in diagnostic.items() if np.isscalar(v)},'imaginary_ratio':stage['imaginary_ratio'],'unmodified_imaginary_ratio_jx':legacy_imag[0],'unmodified_imaginary_ratio_jy':legacy_imag[1],'unmodified_real_filtered_forward_residual':legacy_diag['filtered_forward_residual'],'unmodified_real_divergence_ratio':legacy_diag['divergence_ratio'],'constant_boundary_relative_difference':float(np.linalg.norm(constant-nominal)/np.linalg.norm(nominal)),'total_fresh_experiment_seconds':time.perf_counter()-start}}

def make_figures(data,e):
    figures=[];s=e['stages'][1];x=s['padded_xs'];y=s['padded_ys'];full_extent=[x[0],x[-1],y[0],y[-1]];extent=[data['xs'][0],data['xs'][-1],data['ys'][0],data['ys'][-1]];crop=(slice(75,-75),slice(100,-100));nom=e['currents'][1]
    def imshow(fig,ax,array,title,unit,extent=None,signed=False,limit=None):
        kw={'origin':'lower','aspect':'auto','cmap':'RdBu_r' if signed else 'viridis'}
        if extent is not None:kw['extent']=extent
        if signed:
            bound=float(np.max(abs(array))) if limit is None else limit;kw.update(vmin=-bound,vmax=bound)
        image=ax.imshow(array,**kw);ax.set(title=title,xlabel='x (um)' if extent else 'Column index',ylabel='y (um)' if extent else 'Row index');fig.colorbar(image,ax=ax,label=unit)
    fig,ax=plt.subplots(2,3,figsize=(16,9),constrained_layout=True)
    imshow(fig,ax[0,0],data['flux'],'Real measured SQUID lock-in response','Flux response (Phi0/A)',extent,True)
    imshow(fig,ax[0,1],data['psf_native'],'Independent native dipole calibration','Calibration (mPhi0)',[data['psf_xs'][0],data['psf_xs'][-1],data['psf_ys'][0],data['psf_ys'][-1]],True)
    imshow(fig,ax[0,2],data['psf_coarse'][140:161,190:211],'Actual coarse index kernel (center zoom)','Calibration (mPhi0)',signed=True)
    imshow(fig,ax[1,0],s['padded_mag'],'Fixed linear-ramp extension (not measured)','Flux response (Phi0/A)',full_extent,True)
    ax[1,0].add_patch(plt.Rectangle((extent[0],extent[2]),extent[1]-extent[0],extent[3]-extent[2],fill=False,ec='black',ls='--'))
    imshow(fig,ax[1,1],np.log10(1+abs(s['mag_k'])),'Padded observation spatial spectrum','log10(1+FFT amplitude)')
    imshow(fig,ax[1,2],s['H'],'Nominal radial Hanning window','Dimensionless');figures.append(fig)
    fig,ax=plt.subplots(2,3,figsize=(16,9),constrained_layout=True);limit=float(max(abs(nom).max(),abs(e['legacy'].real).max()))
    imshow(fig,ax[0,0],nom[0],'Fresh corrected Jx: complete periodic support','Relative current (a.u.)',full_extent,True,limit)
    imshow(fig,ax[0,1],nom[1],'Fresh corrected Jy: complete periodic support','Relative current (a.u.)',full_extent,True,limit)
    imshow(fig,ax[0,2],e['legacy'][0].real,'Unmodified oracle Jx, real part only\nReproduced this batch; not truth','Relative current (a.u.)',full_extent,True,limit)
    imshow(fig,ax[1,0],nom[0]-e['legacy'][0].real,'Fresh corrected minus untouched Jx','Relative current difference (a.u.)',full_extent,True)
    imshow(fig,ax[1,1],e['legacy'][0].imag,'Untouched source discarded imaginary Jx','Imaginary relative current (a.u.)',full_extent,True)
    speed=np.hypot(nom[0][crop],nom[1][crop]);imshow(fig,ax[1,2],speed,'Fresh vector field in measured window','Relative speed (a.u.)',extent)
    gridx,gridy=np.meshgrid(data['xs'],data['ys']);skip=(slice(None,None,10),slice(None,None,10));ax[1,2].quiver(gridx[skip],gridy[skip],nom[0][crop][skip],nom[1][crop][skip],color='white',pivot='mid');figures.append(fig)
    fig,ax=plt.subplots(2,3,figsize=(16,9),constrained_layout=True);limit=float(np.max(abs(e['currents'])))
    for col,cutoff in enumerate([.5,1.,1.5]):
        for row,name in enumerate(['Jx','Jy']):imshow(fig,ax[row,col],e['currents'][col,row][crop],f'Fresh {name}, kx_max=ky_max={cutoff}\nCommon signed scale; no actual current truth','Relative current (a.u.)',extent,True,limit)
    figures.append(fig)
    fig,ax=plt.subplots(2,3,figsize=(16,9),constrained_layout=True)
    target=np.fft.ifft2(np.fft.ifftshift(e['diagnostic']['target_k'])).real;prediction=np.fft.ifft2(np.fft.ifftshift(e['diagnostic']['prediction_k'])).real;limit=float(max(abs(target).max(),abs(prediction).max()))
    imshow(fig,ax[0,0],target[crop],'Filtered observation, DC removed','Flux response (Phi0/A)',extent,True,limit)
    imshow(fig,ax[0,1],prediction[crop],'Independent local forward from fresh real J','Flux response (Phi0/A)',extent,True,limit)
    imshow(fig,ax[0,2],(prediction-target)[crop],'Forward minus filtered observation','Flux response error (Phi0/A)',extent,True)
    ax[1,0].plot(data['xs'],data['flux'][75],label='Measured raw');ax[1,0].plot(data['xs'],target[150,100:-100],label='Filtered/DC0');ax[1,0].plot(data['xs'],prediction[150,100:-100],'--',label='Forward');ax[1,0].set(title='Central observed-row profile',xlabel='x (um)',ylabel='Flux response (Phi0/A)');ax[1,0].legend()
    jk=np.stack([s['jx_k'],s['jy_k']]);div=s['Kx']*jk[0]+s['Ky']*jk[1]
    imshow(fig,ax[1,1],abs(div)/max(np.max(abs(s['K']*jk[0])),np.max(abs(s['K']*jk[1]))),'Spectral divergence / maximum scale','Dimensionless')
    labels=['Jx imaginary','Jy imaginary','Real-field forward','Real-field divergence'];old=[e['metrics']['unmodified_imaginary_ratio_jx'],e['metrics']['unmodified_imaginary_ratio_jy'],e['metrics']['unmodified_real_filtered_forward_residual'],e['metrics']['unmodified_real_divergence_ratio']];new=[float(np.linalg.norm(z.imag)/np.linalg.norm(z.real)) for z in s['full_complex']]+[e['metrics']['filtered_forward_residual'],e['metrics']['divergence_ratio']]
    loc=np.arange(4);ax[1,2].bar(loc-.18,old,.36,label='Untouched source');ax[1,2].bar(loc+.18,new,.36,label='Fresh corrected');ax[1,2].set(yscale='log',xticks=loc,xticklabels=labels,title='Why explicit discrete corrections matter',ylabel='Relative diagnostic');ax[1,2].tick_params(axis='x',labelrotation=20);ax[1,2].legend();figures.append(fig)
    fig,ax=plt.subplots(2,3,figsize=(16,9),constrained_layout=True);maximum=float(np.max(np.hypot(e['heights'][:,0],e['heights'][:,1])))
    for i,height in enumerate([.5,1.,1.5]):
        image=ax[0,i].imshow(np.hypot(e['heights'][i,0][crop],e['heights'][i,1][crop]),origin='lower',extent=extent,vmin=0,vmax=maximum);ax[0,i].set(title=f'Fresh assumed height {height}um\nModel sensitivity, not new acquisition',xlabel='x (um)',ylabel='y (um)');fig.colorbar(image,ax=ax[0,i],label='Relative current magnitude (a.u.)')
    middle=150;ax[1,0].plot(s['Kx'][middle],s['H'][middle],label='Hanning');ax[1,0].plot(s['Kx'][middle],abs(s['psf_k'][middle])/np.max(abs(s['psf_k'])),label='|P| / max|P|');ax[1,0].set(title='Regularization and calibrated response',xlabel='kx (rad/um), ky=0',ylabel='Dimensionless');ax[1,0].legend()
    # Separate normalized profiles are sensitivity diagnostics, never evaluation alignment.
    for array,label in [(nom,'Given calibrated PSF'),(e['no_psf'],'Identity PSF assumption'),(e['constant_boundary'],'Zero instead of ramp boundary')]:
        profile=np.hypot(array[0][150,100:-100],array[1][150,100:-100]);ax[1,1].plot(data['xs'],profile/profile.max(),label=label)
    ax[1,1].set(title='Assumption ablations: shape profiles\nIndividually normalized for display only',xlabel='x (um)',ylabel='Magnitude / own profile maximum');ax[1,1].legend(fontsize=8)
    ax[1,2].bar(['0.5','1.0','1.5'],[np.linalg.norm(v) for v in e['heights']]);ax[1,2].set(title='Height sensitivity of full relative field',xlabel='Assumed standoff (um)',ylabel='Full current L2 norm (a.u.)');figures.append(fig)
    return figures

def export(figures,data,e,directory):
    directory=Path(directory);directory.mkdir(parents=True,exist_ok=True)
    for i,fig in enumerate(figures,1):
        for extension in ['png','svg','pdf']:fig.savefig(directory/f'figure_{i}.{extension}',dpi=140)
    values={'fresh_currents':e['currents'],'parameter_table':e['table'],'height_currents':e['heights'],'no_psf':e['no_psf'],'constant_boundary':e['constant_boundary'],'unmodified_complex':e['legacy'],'forward_k':e['diagnostic']['prediction_k'],'filtered_observation_k':e['diagnostic']['target_k']}
    for i,stage in enumerate(e['stages']):
        for key,value in stage.items():values[f'cutoff{i}_{key}']=np.asarray(value)
    np.savez_compressed(directory/'numeric_sources.npz',**values)
    (directory/'metrics.json').write_text(json.dumps(e['metrics'],indent=2,allow_nan=False),encoding='utf8')
