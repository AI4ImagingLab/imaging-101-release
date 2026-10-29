"""Scientific figures from measured and freshly reconstructed numeric arrays."""
from pathlib import Path
import hashlib,json
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import hsv_to_rgb
from .preprocessing import split_channels
from .physics_model import forward,polarization
from .solvers import reconstruct

def ncc_nrmse(estimate,reference):
    """Cosine NCC, then RMS error/reference dynamic range (real arrays)."""
    e=np.asarray(estimate).ravel();r=np.asarray(reference).ravel()
    return float(np.dot(e,r)/(np.linalg.norm(e)*np.linalg.norm(r))),float(np.sqrt(np.mean((e-r)**2))/np.ptp(r))

def _image(fig,ax,a,title,unit,cmap='viridis',vmin=None,vmax=None):
    m=ax.imshow(a,cmap=cmap,vmin=vmin,vmax=vmax,origin='upper',interpolation='nearest')
    ax.set_title(title,fontsize=10);ax.set_xlabel('column (superpixel)');ax.set_ylabel('row (superpixel)')
    fig.colorbar(m,ax=ax,shrink=.79,label=unit)

def _smooth(a,size):
    p=size//2;x=np.pad(a,[(0,0)]*(a.ndim-2)+[(p,p),(p,p)],mode='reflect')
    return sum(x[...,i:i+a.shape[-2],j:j+a.shape[-1]] for i in range(size) for j in range(size))/(size*size)

def make_figures(raw,result,upstream):
    """Return five figures, exact plotted numeric arrays and measured metrics.

    raw is (1,Z,2H,2W); result/upstream are declared output dictionaries.
    No file writes and no plotting-backend selection occurs here.
    """
    channels=split_channels(raw)[0];s=result['reconstruction'][0];a=result['orientation'][0];d=result['degree'][0]
    z=int(np.argmax(np.sum(s[:,0],axis=(1,2))));h,w=d.shape[-2:];row=h//2;col=w//2
    metrics={'display_plane_index':z,'canonical_offset_ADU':0.0,'nominal_lateral_superpixel_nm':138.0}
    for k in ['reconstruction','degree']:
        nc,nr=ncc_nrmse(result[k],upstream[k]);metrics[k+'_ncc']=nc;metrics[k+'_dynamic_range_nrmse']=nr
    delta=result['orientation']-upstream['orientation'];weight=np.sum(upstream['reconstruction'][:,:,1:]**2,axis=2)
    metrics['orientation_weighted_periodic_rmse_rad']=float(np.sqrt(np.sum(weight*(np.arctan2(np.sin(2*delta),np.cos(2*delta))/2)**2)/np.sum(weight)))
    predicted=forward(s);res=predicted-channels
    metrics['observation_relative_residual']=float(np.linalg.norm(res.ravel())/np.linalg.norm(channels.ravel()))
    numeric={'raw_display_plane':raw[0,z],'analyzer_display_plane':channels[z],'stokes_display_plane':s[z],'degree_display_plane':d[z],'angle_display_plane_rad':a[z], 'display_plane_index':np.array(z),'stokes_axial_row_section':s[:,:,row,:], 'degree_axial_row_section':d[:,row,:],'angle_axial_row_section_rad':a[:,row,:],'stokes_axial_col_section':s[:,:,:,col], 'residual_display_plane':res[z]}
    figures=[]
    fig,axes=plt.subplots(2,3,figsize=(13,8),layout='constrained');fig.suptitle(f'Real NR4A T-cell observations | acquired plane {z}',fontsize=15)
    lo,hi=np.percentile(channels[z],[1,99]);
    for i,label in enumerate(['0 deg','45 deg','90 deg','135 deg']):_image(fig,axes.flat[i],channels[z,i],f'Analyzer {label}','ADU','gray',lo,hi)
    m=axes[1,1].imshow(raw[0,z],cmap='gray',origin='upper');axes[1,1].set(title='Original interlaced detector image',xlabel='column (raw pixel)',ylabel='row (raw pixel)');fig.colorbar(m,ax=axes[1,1],shrink=.79,label='ADU')
    axes[1,2].hist(channels[z].reshape(4,-1).T,bins=60,label=['0','45','90','135'],histtype='step');axes[1,2].set(title='Measured channel distributions',xlabel='intensity (ADU)',ylabel='number of superpixels');axes[1,2].legend(title='analyzer (deg)')
    figures.append(fig)
    fig,axes=plt.subplots(2,3,figsize=(13,8),layout='constrained');fig.suptitle('Fresh local Stokes inversion | effective detector polarization',fontsize=15)
    _image(fig,axes[0,0],s[z,0],'Total intensity S0','ADU','gray')
    vmax=float(np.percentile(np.abs(s[z,1:]),99));_image(fig,axes[0,1],s[z,1],'Signed linear component S1','ADU','coolwarm',-vmax,vmax);_image(fig,axes[0,2],s[z,2],'Signed linear component S2','ADU','coolwarm',-vmax,vmax)
    _image(fig,axes[1,0],d[z],'Effective linear polarization degree','dimensionless','magma',0,float(np.percentile(d[z],99)))
    _image(fig,axes[1,1],np.rad2deg(a[z]),'Effective angle (pi periodic)','degree','twilight',-90,90)
    hsv=np.stack(((a[z]/np.pi+.5)%1,np.clip(d[z]/np.percentile(d[z],99),0,1),np.clip(s[z,0]/np.percentile(s[z,0],99),0,1)),axis=-1)
    numeric['angle_degree_intensity_rgb']=hsv_to_rgb(hsv);axes[1,2].imshow(numeric['angle_degree_intensity_rgb']);axes[1,2].set(title='Hue=angle, saturation=degree, value=S0',xlabel='column (superpixel)',ylabel='row (superpixel)')
    figures.append(fig)
    fig,axes=plt.subplots(2,3,figsize=(13,8),layout='constrained');fig.suptitle('Full acquired stack | axial coordinate remains plane index',fontsize=15)
    _image(fig,axes[0,0],np.max(s[:,0],axis=0),'Maximum intensity over all planes','ADU','gray')
    _image(fig,axes[0,1],s[:,0,row,:],f'S0 section at row {row}','ADU','gray');axes[0,1].set_ylabel('acquired plane index')
    _image(fig,axes[0,2],s[:,0,:,col],f'S0 section at column {col}','ADU','gray');axes[0,2].set_ylabel('acquired plane index');axes[0,2].set_xlabel('row (superpixel)')
    _image(fig,axes[1,0],d[:,row,:],'Degree in the same section','dimensionless','magma');axes[1,0].set_ylabel('acquired plane index')
    _image(fig,axes[1,1],np.rad2deg(a[:,row,:]),'Angle in the same section','degree','twilight',-90,90);axes[1,1].set_ylabel('acquired plane index')
    axes[1,2].plot(np.sum(s[:,0],axis=(1,2)),label='sum S0');axes[1,2].set(title='Intensity over all observed planes',xlabel='acquired plane index',ylabel='sum intensity (ADU)');axes[1,2].grid(alpha=.2)
    numeric['max_intensity_projection']=np.max(s[:,0],axis=0);numeric['plane_sum_S0']=np.sum(s[:,0],axis=(1,2));figures.append(fig)
    fig,axes=plt.subplots(2,3,figsize=(13,8),layout='constrained');fig.suptitle('Analyzer-model residuals and independently reproduced source agreement',fontsize=15)
    vmax=float(np.percentile(np.abs(res[z]),99));
    for i,label in enumerate(['0 deg','45 deg','90 deg','135 deg']):_image(fig,axes.flat[i],res[z,i],f'Predicted minus measured: {label}','ADU','coolwarm',-vmax,vmax)
    plane_res=np.sqrt(np.mean(res**2,axis=(1,2,3)));plane_error=np.sqrt(np.mean((s-upstream['reconstruction'][0])**2,axis=(1,2,3)));numeric['plane_analyzer_rmse_ADU']=plane_res;numeric['plane_local_upstream_rmse_ADU']=plane_error
    axes[1,1].plot(plane_res,label='analyzer RMSE');axes[1,1].set(title='All-channel residual across the full stack',xlabel='acquired plane index',ylabel='RMSE (ADU)');axes[1,1].grid(alpha=.2)
    axes[1,2].plot(plane_error,color='tab:green');axes[1,2].set(title='Fresh local minus upstream baseline',xlabel='acquired plane index',ylabel='Stokes RMSE (ADU)');axes[1,2].ticklabel_format(axis='y',style='sci',scilimits=(0,0));axes[1,2].grid(alpha=.2)
    figures.append(fig)
    fig,axes=plt.subplots(2,3,figsize=(13,8),layout='constrained');fig.suptitle('Measured-data sensitivity | nominal background and spatial averaging',fontsize=15)
    offsets=np.array([0.,50.,100.,150.]);pmean=[];adiff=[]
    # Fresh computations; display-plane sensitivity is explicitly labeled, not a replacement for the canonical inverse.
    for off in offsets:
        r=reconstruct(raw[:,z:z+1],offset=float(off));pmean.append(float(np.mean(r['degree'])));dd=r['orientation'][0,0]-a[z];adiff.append(float(np.max(np.abs(np.arctan2(np.sin(2*dd),np.cos(2*dd))/2))))
        numeric['degree_offset_'+str(int(off))]=r['degree'][0,0]
    numeric['offsets_ADU']=offsets;numeric['mean_degree_by_offset']=np.array(pmean);numeric['max_periodic_angle_change_rad']=np.array(adiff)
    _image(fig,axes[0,0],d[z],'Canonical zero-offset degree','dimensionless','magma')
    _image(fig,axes[0,1],numeric['degree_offset_100'],'Sensitivity: subtract 100 ADU (assumed)','dimensionless','magma')
    axes[0,2].plot(offsets,pmean,'o-');axes[0,2].set(title='Degree depends on detector pedestal',xlabel='assumed subtracted offset (ADU)',ylabel='mean degree (dimensionless)');axes[0,2].grid(alpha=.2)
    averaged=_smooth(s[z],3);sa,sd=polarization(averaged);numeric['stokes_mean3']=averaged;numeric['degree_mean3']=sd;numeric['angle_mean3_rad']=sa
    _image(fig,axes[1,0],sd,'Degree after 3x3 Stokes averaging','dimensionless','magma')
    axes[1,1].hist(d[z].ravel(),bins=80,histtype='step',label='canonical');axes[1,1].hist(sd.ravel(),bins=80,histtype='step',label='3x3 Stokes average');axes[1,1].set(title='Averaging mixes orientations and noise',xlabel='degree (dimensionless)',ylabel='superpixel count');axes[1,1].legend()
    axes[1,2].plot(offsets,adiff,'o-',color='tab:green');axes[1,2].set(title='Common-pedestal angle cancellation',xlabel='assumed subtracted offset (ADU)',ylabel='maximum periodic angle change (rad)');axes[1,2].grid(alpha=.2)
    metrics['max_offset_angle_change_rad']=float(max(adiff));metrics['display_mean_degree_by_offset']={str(int(k)):v for k,v in zip(offsets,pmean)};figures.append(fig)
    for figure in figures:
        for axis in figure.axes:
            axis.title.set_fontsize(10)
    return figures,numeric,metrics

def export_figures(figures,numeric,metrics,directory):
    """Export five scientific groups in PNG/SVG/PDF plus exact source arrays and hashes."""
    directory=Path(directory);directory.mkdir(parents=True,exist_ok=True)
    for i,fig in enumerate(figures,1):
        for suffix in ['png','svg','pdf']:fig.savefig(directory/f'figure_{i}.{suffix}',dpi=150)
    np.savez_compressed(directory/'numeric_sources.npz',**numeric)
    (directory/'metrics.json').write_text(json.dumps(metrics,indent=2,allow_nan=False)+'\n',encoding='utf8')
    manifest={'provenance':'Every plotted reconstruction/sensitivity array was freshly computed by local src modules from full real public observations. Upstream difference uses separately identified hidden unmodified source baseline.','sha256':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(directory.iterdir()) if p.is_file() and p.name!='hash_manifest.json'}}
    (directory/'hash_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n',encoding='utf8')
