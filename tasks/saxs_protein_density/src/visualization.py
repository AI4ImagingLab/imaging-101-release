"""Scientific measured-data, convergence and nonuniqueness review."""
from pathlib import Path
import json,hashlib,itertools
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter,affine_transform
from scipy.spatial.transform import Rotation
from scipy.optimize import minimize
from scipy.interpolate import interp1d
from .physics_model import forward_scattering,shell_grid
from .preprocessing import VOXEL

def cosine(a,b):
    return float(np.dot(a.ravel(),b.ravel())/(np.linalg.norm(a)*np.linalg.norm(b)))

def _core(rho):
    blurred=gaussian_filter(np.clip(rho,0,None),1.5)
    return np.where(blurred>=.2*blurred.max(),blurred,0)

def _moments(a):
    xyz=np.indices(a.shape).reshape(3,-1);w=a.ravel();center=xyz@w/w.sum();offset=xyz-center[:,None]
    covariance=(offset*w)@offset.T/w.sum();_,basis=np.linalg.eigh(covariance)
    return center,basis

def align_for_display(reference,moving):
    """Rigid/handedness gauge only; inertia starts + Powell on a coarse grid.

    Returns a linearly resampled display array and diagnostic cosine. No
    alignment is passed to the Judge, reconstruction or spatial-truth metric.
    """
    ref=_core(reference)[::2,::2,::2];mov=_core(moving)[::2,::2,::2]
    rc,rb=_moments(ref);mc,mb=_moments(mov)
    def score(matrix):
        warped=affine_transform(mov,matrix,mc-matrix@rc,output_shape=ref.shape,order=1,mode='constant',prefilter=False)
        return cosine(ref,warped)
    starts=[]
    for perm in itertools.permutations(range(3)):
        for signs in itertools.product([-1.,1.],repeat=3):
            matrix=mb[:,perm]@np.diag(signs)@rb.T;starts.append((score(matrix),matrix))
    best=max(starts,key=lambda p:p[0])
    for _,start in sorted(starts,key=lambda p:p[0],reverse=True)[:2]:
        fit=minimize(lambda angles:-score(Rotation.from_rotvec(angles).as_matrix()@start),np.zeros(3),method='Powell',options={'maxiter':12,'xtol':.002,'ftol':1e-5})
        matrix=Rotation.from_rotvec(fit.x).as_matrix()@start;candidate=score(matrix)
        if candidate>best[0]:best=(candidate,matrix)
    matrix=best[1];warped=affine_transform(moving,matrix,2*mc-matrix@(2*rc),output_shape=reference.shape,order=1,mode='constant',prefilter=False)
    return warped,{'core_registration_cosine':best[0],'aligned_raw_density_cosine':cosine(reference,warped),'handedness_determinant':float(np.linalg.det(matrix)),'matrix_output_to_input':matrix.tolist(),'offset_output_to_input_voxel':(2*mc-matrix@(2*rc)).tolist()}

def _plane(fig,ax,a,title,unit='1e-5 / A^3',limit=None):
    extent=[-45,45,45,-45];im=ax.imshow(a,extent=extent,cmap='coolwarm',vmin=-limit if limit else None,vmax=limit,interpolation='nearest');ax.set(title=title,xlabel='coordinate (A)',ylabel='coordinate (A)');fig.colorbar(im,ax=ax,shrink=.8,label=unit)

def _profile(rho,data):
    q,intensity=forward_scattering(rho);mask=q<=data['q'][0,-1]
    m=data['measured_mask'][0];pred=interp1d(q[mask],intensity[mask],kind='cubic',fill_value='extrapolate')(data['q'][0,m])
    return q,intensity,pred

def make_figures(data,canonical,upstream,comparisons,scores):
    """Return six genuine comparisons, exact numerical sources and diagnostics.

    comparisons are actively reconstructed by the caller, never filled from
    upstream arrays; upstream values remain explicitly labeled separately.
    """
    rho=canonical['reconstruction'][0];history=canonical['history'];m=data['measured_mask'][0];q=data['q'][0,m];i0=data['smoothed_intensity'][0,0];obs=data['intensity'][0,m]/i0;err=data['sigma'][0,m]/i0
    qc,curve,pred=_profile(rho,data);residual=(obs-pred)/err
    numeric={'q_measured_per_A':q,'observed_normalized':obs,'sigma_normalized':err,'final_shell_q_per_A':qc,'final_shell_intensity':curve,'fresh_prediction_all_measured_q':pred,'standardized_residual':residual,'fresh_density_per_A3':rho,'canonical_chi_monitor':history['chi'],'canonical_rg_monitor_A':history['rg'].real,'canonical_support_volume_A3':history['support_volume']}
    figures=[];crop=slice(20,45);sub=rho[crop,crop,crop];limit=float(np.max(np.abs(sub))*1e5)
    fig,axes=plt.subplots(2,3,figsize=(13,8),layout='constrained');fig.suptitle('Real lysozyme SAXS | all measured frequencies and observation-derived fit')
    ax=axes[0,0];ax.errorbar(q,obs,yerr=err,fmt='.',ms=2,lw=.3,label='real measured');ax.plot(q,pred,label='fresh density');ax.plot(data['q'][0],data['smoothed_intensity'][0]/i0,'--',label='GNOM fit');ax.set(yscale='log',xlabel='q (1/A)',ylabel='I(q)/I(0)',title='Complete acquired curve');ax.legend(fontsize=8)
    ax=axes[0,1];ax.plot(q,residual);ax.axhline(0,color='k',ls=':');ax.set(xlabel='q (1/A)',ylabel='(observed-predicted)/sigma',title='Every measured-point residual')
    ax=axes[0,2];ax.plot(q,q*q*obs,label='measured');ax.plot(q,q*q*pred,label='fresh');ax.set(xlabel='q (1/A)',ylabel='q^2 I(q)/I(0) (1/A^2)',title='Kratky representation');ax.legend(fontsize=8)
    low=q<=.08;ax=axes[1,0];ax.plot(q[low]**2,np.log(obs[low]),'.',label='measured');ax.plot(q[low]**2,np.log(pred[low]),label='fresh');ax.set(xlabel='q^2 (1/A^2)',ylabel='log normalized I',title='Guinier region uses actual measurements');ax.legend(fontsize=8)
    ax=axes[1,1];ax.loglog(qc[1:],curve[1:],label='final density all shells');ax.axvline(q.max(),color='k',ls=':',label='last measured q');ax.set(xlabel='q (1/A)',ylabel='normalized intensity',title='Unmeasured tail is a declared prior');ax.legend(fontsize=8)
    ax=axes[1,2];ax.plot(q,err/obs);ax.set(xlabel='q (1/A)',ylabel='sigma / I',title='Supplied relative measurement uncertainty');figures.append(fig)
    fig,axes=plt.subplots(2,3,figsize=(13,8),layout='constrained');fig.suptitle('Fresh 3D normalized density | orthogonal views, signed tails retained')
    for j in range(3):
        a=np.take(sub,12,axis=j)*1e5;numeric[f'central_slice_axis{j}_per_A3']=a/1e5;_plane(fig,axes[0,j],a,f'Central plane normal to axis {j}',limit=limit)
        a=np.sum(sub,axis=j)*VOXEL;numeric[f'core_projection_axis{j}_per_A2']=a;_plane(fig,axes[1,j],a*1e3,f'90 A display region projection, axis {j}','1e-3 / A^2')
    figures.append(fig)
    densities=[upstream['reconstruction'][0]]+[upstream[f'density_seed{s}'][0] for s in [42,43,44,45]]
    aligned=[densities[0]];alignment=[{'core_registration_cosine':1.,'aligned_raw_density_cosine':1.,'handedness_determinant':1.}]
    for moving in densities[1:]:
        a,info=align_for_display(densities[0],moving);aligned.append(a);alignment.append(info)
    aligned=np.array(aligned);mean=aligned.mean(axis=0);std=aligned.std(axis=0);numeric['aligned_original_densities_per_A3']=aligned;numeric['aligned_ensemble_mean_per_A3']=mean;numeric['aligned_ensemble_std_per_A3']=std
    fig,axes=plt.subplots(2,3,figsize=(13,8),layout='constrained');fig.suptitle('Five independent original starts | rigid orientation and handedness aligned for display')
    for j,a in enumerate(list(aligned)+[mean]):
        image=np.sum(a[crop,crop,crop],axis=2)*VOXEL*1e3;_plane(fig,axes.flat[j],image,f'Original seed {41+j}' if j<5 else 'Aligned ensemble mean (display only)','1e-3 / A^2')
    figures.append(fig)
    fig,axes=plt.subplots(2,3,figsize=(13,8),layout='constrained');fig.suptitle('Actual inverse diagnostics | original monitors and final-output limitations')
    for j,key in enumerate(['chi','rg','support_volume']):
        axes[0,j].plot(np.arange(len(history[key])),np.real(history[key]),label='fresh local')
        upstream_key={'chi':'chi_seed41','rg':'rg_seed41','support_volume':'supportV_seed41'}[key];source=upstream[upstream_key];length=len(history[key]);axes[0,j].plot(np.arange(length),source[:length].real,'--',label='original qualified run');axes[0,j].set(xlabel='iteration',ylabel=['original monitor chi2','monitor Rg (A)','support volume (A^3)'][j],title=['Convergence monitor (pre-final stage)','Scattering radius monitor','Adaptive compact support'][j]);axes[0,j].legend(fontsize=8)
    legacy_chi=upstream['unmodified_halfspace_chi'];axes[0,0].plot(legacy_chi[legacy_chi>0],':',label='unmodified half-space original');axes[0,0].legend(fontsize=7);axes[0,0].set_yscale('log');axes[0,2].set_yscale('log')
    ax=axes[1,0];diff=rho-upstream['reconstruction'][0];_plane(fig,ax,diff[crop,crop,crop][:,:,12]*1e5,'Fresh minus qualified original','1e-5 / A^3')
    coordinates=(np.arange(64)-32)*VOXEL;radius=np.sqrt(sum(x*x for x in np.meshgrid(coordinates,coordinates,coordinates,indexing='ij')));edges=np.arange(0,211,5);positive,_=np.histogram(radius,edges,weights=np.clip(rho,0,None)*VOXEL**3);negative,_=np.histogram(radius,edges,weights=-np.clip(rho,None,0)*VOXEL**3);centers=(edges[1:]+edges[:-1])/2;ax=axes[1,1];ax.plot(centers,positive,label='positive mass');ax.plot(centers,negative,label='absolute negative mass');ax.axvline(40,color='k',ls=':',label='Dmax soft radius');ax.set(xlabel='radius from grid center (A)',ylabel='normalized mass / 5 A bin',title='Final Fourier projection leaves signed tails');ax.legend(fontsize=8);numeric['mass_radius_A']=centers;numeric['radial_positive_mass']=positive;numeric['radial_negative_mass']=negative
    ax=axes[1,2];hq=upstream['halfspace_q_full'];raw=upstream['halfspace_unweighted_intensity'];rot=upstream['halfspace_unweighted_rotated_intensity'];full=upstream['halfspace_weighted_intensity'];fullrot=upstream['halfspace_weighted_rotated_intensity'];keep=(hq>0)&(hq<=q.max());ax.plot(hq[keep],(rot[keep]/raw[keep]-1)*100,'o-',label='unmodified half-space');ax.plot(hq[keep],(fullrot[keep]/full[keep]-1)*100,'s--',label='correct Hermitian weights');ax.set(xlabel='q (1/A)',ylabel='intensity change after axis swap (%)',title='Grid-exact 90 degree rotation exposes source bias');ax.legend(fontsize=7)
    for key in ['q_full','q_half','unweighted_intensity','weighted_intensity','unweighted_rotated_intensity','weighted_rotated_intensity']:numeric['halfspace_'+key]=upstream['halfspace_'+key]
    numeric['unmodified_original_chi_monitor']=legacy_chi[legacy_chi>0];figures.append(fig)
    fig,axes=plt.subplots(2,3,figsize=(13,8),layout='constrained');fig.suptitle('Fresh reconstruction sensitivity | independent starts and stopping tolerance')
    labels=['canonical']+list(comparisons);results={'canonical':canonical,**comparisons}
    for name,result in results.items():
        d=result['reconstruction'][0];_,_,predicted=_profile(d,data);axes[0,0].plot(q,(predicted-obs)/err,label=name);h=result['history'];axes[0,1].plot(h['chi'],label=name);numeric[f'fresh_{name}_density_per_A3']=d;numeric[f'fresh_{name}_chi']=h['chi'];numeric[f'fresh_{name}_predicted']=predicted
    axes[0,0].set(xlabel='q (1/A)',ylabel='standardized residual',title='Actual final densities fit the same data');axes[0,0].legend(fontsize=7);axes[0,1].set(xlabel='iteration',ylabel='source monitor chi2',yscale='log',title='Fresh monitor histories');axes[0,1].legend(fontsize=7)
    for ax,key,title in [(axes[0,2],'chi2','All-measurement final-density chi2'),(axes[1,0],'negative_mass_fraction','Negative/positive mass after final projection'),(axes[1,1],'outside_mass_fraction','Soft-support residual')]:
        ax.bar(np.arange(len(labels)),[scores[name][key] for name in labels]);ax.set_xticks(np.arange(len(labels)),labels,rotation=30,ha='right');ax.set(title=title,ylabel='dimensionless')
    ax=axes[1,2];ax.bar(np.arange(len(labels)),[results[name]['history']['iterations'] for name in labels]);ax.set_xticks(np.arange(len(labels)),labels,rotation=30,ha='right');ax.set(title='Actual executed iterations',ylabel='iteration');figures.append(fig)
    # Agreement between random-start groups is uncertainty context, not truth or
    # gold-standard independent-acquisition resolution.
    even=aligned[[0,2,4]].mean(axis=0);odd=aligned[[1,3]].mean(axis=0);full_q=np.fft.fftfreq(64)*2*np.pi/VOXEL;qr=np.sqrt(sum(x*x for x in np.meshgrid(full_q,full_q,full_q,indexing='ij')));shell=np.floor(qr/(2*np.pi/(64*VOXEL))+1e-10).astype(int);shell=np.maximum(shell,0);ea=np.fft.fftn(even);ob=np.fft.fftn(odd);count=np.bincount(shell.ravel());numerator=np.bincount(shell.ravel(),weights=np.real(ea*np.conj(ob)).ravel());den1=np.bincount(shell.ravel(),weights=np.abs(ea).ravel()**2);den2=np.bincount(shell.ravel(),weights=np.abs(ob).ravel()**2);fsc=numerator/np.sqrt(den1*den2);fsc_q=np.bincount(shell.ravel(),weights=qr.ravel())/count
    fig,axes=plt.subplots(2,3,figsize=(13,8),layout='constrained');fig.suptitle('Identifiability review | aligned ensemble variability is not molecular truth')
    _plane(fig,axes[0,0],mean[crop,crop,crop][:,:,12]*1e5,'Aligned original ensemble mean')
    _plane(fig,axes[0,1],std[crop,crop,crop][:,:,12]*1e5,'Across-start density standard deviation')
    _plane(fig,axes[0,2],(even-odd)[crop,crop,crop][:,:,12]*1e5,'Difference of random-start group means')
    ax=axes[1,0];ax.plot(fsc_q[1:],fsc[1:]);ax.axhline(.5,color='gray',ls=':',label='illustrative 0.5 level');ax.set(xlabel='spatial frequency (1/A)',ylabel='Fourier shell correlation',ylim=(-.1,1.05),title='Same-data start-group agreement, not truth');ax.legend(fontsize=7)
    ax=axes[1,1];values=[x['aligned_raw_density_cosine'] for x in alignment];ax.bar(np.arange(5),values);ax.set_xticks(np.arange(5),[str(s) for s in [41,42,43,44,45]]);ax.set(xlabel='original seed',ylabel='cosine to seed41 after display alignment',ylim=(0,1.05),title='Diagnostic spatial agreement; no Judge boundary')
    ax=axes[1,2];ax.axis('off');ax.text(0,.95,'All maps use the same real SAXS curve.\nRotations and handedness are display gauges.\nResidual shape variations remain.\nNo hidden atomic model or voxel truth is used.\nFive starts illustrate variability; they do not\ncertify publication-level structural resolution.\nFinal signed tails remain in evaluated outputs.',va='top',fontsize=10)
    numeric['group_fsc_q_per_A']=fsc_q;numeric['group_fsc']=fsc;numeric['display_aligned_cosines']=np.array(values);numeric['start_group_difference_per_A3']=even-odd;figures.append(fig)
    for fig in figures:
        for ax in fig.axes:ax.title.set_fontsize(10)
    measurements={'scores':scores,'display_alignment':alignment,'interpretation':'No molecular shape truth. Raw signed final-density maps are scored before display alignment. All fresh comparisons actually reconstructed. Same-data start-group FSC is ensemble consistency only.'}
    return figures,numeric,measurements

def export_figures(figures,numeric,metrics,directory):
    path=Path(directory);path.mkdir(parents=True,exist_ok=True)
    for index,figure in enumerate(figures,1):
        for suffix in ['png','svg','pdf']:figure.savefig(path/f'figure_{index}.{suffix}',dpi=150)
    np.savez_compressed(path/'numeric_sources.npz',**numeric);(path/'metrics.json').write_text(json.dumps(metrics,indent=2,allow_nan=False)+'\n',encoding='utf8')
    manifest={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(path.iterdir()) if p.is_file() and p.name!='hash_manifest.json'};(path/'hash_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n',encoding='utf8')
