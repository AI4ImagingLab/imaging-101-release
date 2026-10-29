"""Scientific review figures from actually computed fields and native data."""
from pathlib import Path
import hashlib,json,os,textwrap
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
from scipy.ndimage import map_coordinates
from .physics_model import zero_moment
from .preprocessing import PIXEL_ARCSEC,VELOCITY_START_KM_S,VELOCITY_STEP_KM_S

BLUE='#2368a0';ORANGE='#d86b26';PURPLE='#8548a3';GRAY='#656b72'


def rms(array):
    """Unnormalised RMS; plotted physical units remain those of the array."""
    return float(np.sqrt(np.mean(np.asarray(array,dtype=float)**2)))


def signed_norm(*arrays):
    limit=max(float(np.nanmax(abs(np.asarray(a)))) for a in arrays)
    return TwoSlopeNorm(vmin=-(limit or 1e-12),vcenter=0,vmax=(limit or 1e-12))


def _image(fig,axis,array,title,unit,cmap='viridis',norm=None,vmin=None,vmax=None):
    im=axis.imshow(array,origin='lower',cmap=cmap,norm=norm,vmin=vmin,vmax=vmax,interpolation='nearest')
    axis.set_title(title,fontsize=10)
    axis.set_xlabel('image x pixel');axis.set_ylabel('image y pixel')
    fig.colorbar(im,ax=axis,shrink=.78,pad=.025,label=unit)
    return im


def save_figure(fig,slug,numeric,caption,directory=None):
    """Export actual figure PNG/SVG/PDF and numeric sources to a private folder."""
    target=directory or os.environ.get('IMAGING_FIGURE_DIR')
    if not target:return fig
    path=Path(target);path.mkdir(parents=True,exist_ok=True)
    for suffix in ['png','svg','pdf']:
        fig.savefig(path/(slug+'.'+suffix),dpi=160,bbox_inches='tight',facecolor='white')
    source=path/(slug+'_numeric.npz')
    np.savez_compressed(source,**{key:np.asarray(value) for key,value in numeric.items()})
    record=dict(group=slug,caption=caption,consumer='Canonical active notebook,ngc2403_tilted_ring_kinematics',numeric_file=source.name,numeric_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),arrays={key:dict(shape=list(np.asarray(value).shape),dtype=str(np.asarray(value).dtype)) for key,value in numeric.items()})
    (path/(slug+'_source.json')).write_text(json.dumps(record,indent=2),encoding='utf-8')
    return fig


def plot_data(cube,mask,moment,statistics,directory=None):
    fig,axes=plt.subplots(2,3,figsize=(14,8.5),layout='constrained')
    fig.suptitle('1  |  Complete observed H I cube and observation-only support',fontsize=15)
    integrated=zero_moment(cube,mask)
    _image(fig,axes[0,0],integrated,'Masked integrated intensity','Jy/beam km/s','magma')
    _image(fig,axes[0,1],moment,'Intensity-weighted observed velocity','km/s','viridis')
    _image(fig,axes[0,2],mask.sum(axis=0),'Number of detected channels','channels','cividis',vmin=0,vmax=62)
    for ax,k in zip(axes[1,:2],[20,40]):
        _image(fig,ax,cube[k],f'Native channel{k}: {VELOCITY_START_KM_S+k*VELOCITY_STEP_KM_S:.2f} km/s','Jy/beam','magma')
    ax=axes[1,2]
    ax.hist(cube.ravel(),bins=120,color=BLUE,alpha=.8)
    for label,key,color in [('4-sigma detection','threshold',ORANGE),('2.5-sigma growth','growth_threshold',PURPLE)]:
        ax.axvline(statistics[key],color=color,label=label,lw=1.5)
    ax.set_yscale('log');ax.set_xscale('symlog',linthresh=.003)
    ax.set_xticks([-.01,0,.01,.1],['-0.01','0','0.01','0.1'])
    ax.minorticks_off()
    ax.set(xlabel='observed intensity / Jy/beam',ylabel='voxel count',title='All 1,395,000 observed samples')
    ax.legend(fontsize=8)
    caption='All62x150x150 actual observed samples retained. White pixels in the first two summaries mean no detected intensity denominator; the source is not assumed absent there. Kinematic output still contains all41 rings. Signed negative native noise samples remain in the cube.'
    fig.supxlabel(textwrap.fill(caption,155),fontsize=8)
    return save_figure(fig,'01_native_data',dict(cube=cube,mask=mask,moment=moment,integrated_intensity=integrated),caption,directory)


def plot_operator(original_raw,fresh_raw,original_kernel,fresh_kernel,original_smoothed,fresh_smoothed,directory=None):
    fig,axes=plt.subplots(2,3,figsize=(13.5,8),layout='constrained')
    fig.suptitle('2  |  Independently captured acquisition stages and local mathematics',fontsize=15)
    k=int(np.argmax(original_raw.sum(axis=(1,2))))
    limit=max(float(original_raw[k].max()),float(fresh_raw[k].max()))
    _image(fig,axes[0,0],original_raw[k],f'Original cloud integral: channel{k}','pre-normalization model flux','magma',vmin=0,vmax=limit)
    _image(fig,axes[0,1],fresh_raw[k],f'Fresh conditional operator: channel{k}','pre-normalization model flux','magma',vmin=0,vmax=limit)
    delta=fresh_raw[k].astype(float)-original_raw[k]
    _image(fig,axes[0,2],delta,'Fresh minus original conditional integral','model-flux difference','RdBu_r',norm=signed_norm(delta))
    _image(fig,axes[1,0],fresh_kernel,'Explicit normalized telescope beam','dimensionless','viridis')
    ax=axes[1,1];x=np.arange(original_kernel.shape[1])-original_kernel.shape[1]//2
    ax.plot(x,original_kernel[original_kernel.shape[0]//2],color=GRAY,lw=2.5,label='original captured')
    ax.plot(x,fresh_kernel[fresh_kernel.shape[0]//2],color=BLUE,marker='o',ms=3,lw=1,label='fresh explicit kernel')
    ax.set(xlabel='kernel x offset / pixel',ylabel='normalized weight',title='Beam central row');ax.legend(fontsize=8)
    ax=axes[1,2];channel=np.arange(original_raw.shape[0])
    ax.plot(channel,original_smoothed.sum(axis=(1,2)),color=GRAY,lw=2.5,label='original convolved')
    ax.plot(channel,fresh_smoothed.sum(axis=(1,2)),color=BLUE,lw=1,label='fresh convolved')
    ax.plot(channel,original_raw.sum(axis=(1,2)),color=ORANGE,lw=1,alpha=.8,label='raw cloud integral')
    ax.set(xlabel='native velocity-channel index',ylabel='summed model flux',title='Complete62-channel diagnostic spectra');ax.legend(fontsize=8)
    caption='The9x9 inner-annulus diagnostic is explicitly labeled and does not replace the complete benchmark. Its forward test is conditioned on original captured cloud coordinates/deviates; random sampling quality is tested statistically and whole fields against independent original seeds. Zero signed differences are neutral.'
    fig.supxlabel(textwrap.fill(caption,155),fontsize=8)
    return save_figure(fig,'02_operator_parity',dict(original_raw=original_raw,fresh_raw=fresh_raw,original_kernel=original_kernel,fresh_kernel=fresh_kernel,original_smoothed=original_smoothed,fresh_smoothed=fresh_smoothed),caption,directory)


def plot_fields(original,fresh,source_seed2,source_seed3,prepass,directory=None):
    radius=np.arange(15,1216,30)
    fig,axes=plt.subplots(2,3,figsize=(14,8),layout='constrained')
    fig.suptitle('3  |  All41 physical profiles: qualified original and fresh reconstruction',fontsize=15)
    names=['Circular rotation','Intrinsic dispersion','Receding-side position angle'];units=['km/s','km/s','degree']
    ensemble=np.stack([original,source_seed2,source_seed3])
    for j,(name,unit) in enumerate(zip(names,units)):
        ax=axes[0,j]
        ax.fill_between(radius,ensemble[:,:,j].min(axis=0),ensemble[:,:,j].max(axis=0),color=ORANGE,alpha=.25,label='range of3 original seeds, not CI')
        ax.plot(radius,original[:,j],color=GRAY,lw=3,label='qualified original')
        ax.plot(radius,fresh[:,j],color=BLUE,lw=1,marker='o',ms=2.5,label='fresh local')
        if j==0:ax.plot(radius[:-1],prepass[:,1],color=PURPLE,ls='--',lw=1,label='fresh2D prepass,40 rings')
        ax.set(xlabel='radius / arcsec',ylabel=unit,title=name)
        ax.legend(fontsize=7,loc='best')
        ax=axes[1,j]
        for arr,label,color in [(source_seed2,'original ISEED -2',ORANGE),(source_seed3,'original ISEED -3',PURPLE),(fresh,'fresh local',BLUE)]:
            difference=arr[:,j]-original[:,j]
            if j==2:difference=(difference+180)%360-180
            ax.plot(radius,difference,color=color,lw=1,marker='o' if label=='fresh local' else None,ms=2,label=label)
        ax.axhline(0,color=GRAY,lw=.8)
        ax.set(xlabel='radius / arcsec',ylabel='difference / '+unit,title='Differences from qualified original')
        if j==0:ax.legend(fontsize=7)
    caption='Reference agreement is not galaxy truth. The independently rerun source-seed range shows stochastic integration sensitivity, not posterior uncertainty. PA is constant only after the selected median regularization; all41 values are returned and evaluated. The2D prepass has40 fitted rings and is shown as such.'
    fig.supxlabel(textwrap.fill(caption,155),fontsize=8)
    return save_figure(fig,'03_full_profiles',dict(radius_arcsec=radius,original=original,fresh=fresh,source_seed2=source_seed2,source_seed3=source_seed3,prepass=prepass),caption,directory)


def spectral_slice(cube,angle_deg,radius_arcsec=None):
    """Bilinear actual-cube cut along the receding major axis; no velocity crop."""
    radius=np.linspace(-1100,1100,181) if radius_arcsec is None else np.asarray(radius_arcsec,dtype=float)
    angle=np.deg2rad(angle_deg)
    x=77-radius*np.sin(angle)/PIXEL_ARCSEC
    y=77+radius*np.cos(angle)/PIXEL_ARCSEC
    channel=np.arange(cube.shape[0])[:,None]
    z=np.broadcast_to(channel,(cube.shape[0],radius.size))
    xx=np.broadcast_to(x,z.shape);yy=np.broadcast_to(y,z.shape)
    values=map_coordinates(np.asarray(cube,dtype=float),[z,yy,xx],order=1,mode='constant',cval=0)
    return radius,values


def _pv(fig,ax,array,radius,title,unit,norm=None,vmin=None,vmax=None,cmap='magma'):
    end=VELOCITY_START_KM_S+VELOCITY_STEP_KM_S*(array.shape[0]-1)
    image=ax.imshow(array,origin='upper',extent=[radius[0],radius[-1],end,VELOCITY_START_KM_S],aspect='auto',interpolation='nearest',cmap=cmap,norm=norm,vmin=vmin,vmax=vmax)
    ax.set(xlabel='signed major-axis radius / arcsec',ylabel='LOS velocity / km/s',title=title)
    fig.colorbar(image,ax=ax,shrink=.8,pad=.025,label=unit)


def plot_residuals(cube,original_model,fresh_model,position_angle,directory=None):
    radius,data_pv=spectral_slice(cube,position_angle)
    _,original_pv=spectral_slice(original_model,position_angle)
    _,fresh_pv=spectral_slice(fresh_model,position_angle)
    fig,axes=plt.subplots(3,3,figsize=(15,12),layout='constrained')
    fig.suptitle('4  |  Native spectral structure and full-cube residuals',fontsize=15)
    lo=min(data_pv.min(),0);hi=max(data_pv.max(),original_pv.max(),fresh_pv.max())
    for ax,value,title in zip(axes[0],[data_pv,original_pv,fresh_pv],['Observed cube','Qualified original model','Fresh local model']):
        _pv(fig,ax,value,radius,title,'Jy/beam',vmin=lo,vmax=hi)
    d1=data_pv-original_pv;d2=data_pv-fresh_pv;implementation=fresh_pv-original_pv
    common=signed_norm(d1,d2)
    _pv(fig,axes[1,0],d1,radius,'Observed minus original','Jy/beam',norm=common,cmap='RdBu_r')
    _pv(fig,axes[1,1],d2,radius,'Observed minus fresh','Jy/beam',norm=common,cmap='RdBu_r')
    _pv(fig,axes[1,2],implementation,radius,'Fresh minus original model','Jy/beam',norm=signed_norm(implementation),cmap='RdBu_r')
    maps=[np.sqrt(np.mean((cube.astype(float)-model)**2,axis=0)) for model in [original_model,fresh_model]]
    hi=max(float(m.max()) for m in maps)
    _image(fig,axes[2,0],maps[0],'Original residual RMS, all62 channels','Jy/beam','magma',vmin=0,vmax=hi)
    _image(fig,axes[2,1],maps[1],'Fresh residual RMS, all62 channels','Jy/beam','magma',vmin=0,vmax=hi)
    delta=np.sqrt(np.mean((fresh_model.astype(float)-original_model)**2,axis=0))
    _image(fig,axes[2,2],delta,'Implementation difference RMS','Jy/beam','viridis',vmin=0)
    caption='Major-axis cuts are display summaries only; all1395000 native voxels enter forward RMS. All62 velocity channels are shown. Signed residuals use symmetric limits and neutral zero, without clipping. Original/fresh model differences quantify implementation/integration effects, not independent physical error.'
    fig.supxlabel(textwrap.fill(caption,155),fontsize=8)
    return save_figure(fig,'04_full_cube_residuals',dict(radius_arcsec=radius,data_pv=data_pv,original_pv=original_pv,fresh_pv=fresh_pv,original_residual_RMS_map=maps[0],fresh_residual_RMS_map=maps[1],implementation_RMS_map=delta),caption,directory)


def plot_initialization(cube,initial,prepass,fresh,initial_model,fresh_model,directory=None):
    radius=np.arange(15,1216,30)
    fig,axes=plt.subplots(2,3,figsize=(14,8.5),layout='constrained')
    fig.suptitle('5  |  Velocity-field initialization versus complete3D fitting',fontsize=15)
    ax=axes[0,0]
    ax.plot(radius[:-1],prepass[:,1],color=PURPLE,ls='--',label='fitted2D prepass,40 rings')
    ax.plot(radius,initial[:,0],color=ORANGE,alpha=.8,label='actual selected initialization')
    ax.plot(radius,fresh[:,0],color=BLUE,label='fresh final3D fit')
    ax.set(xlabel='radius / arcsec',ylabel='rotation / km/s',title='Rotation and boundary-start conventions');ax.legend(fontsize=7)
    ax=axes[0,1]
    ax.plot(radius,initial[:,2],color=ORANGE,label='initial geometry')
    ax.plot(radius,fresh[:,2],color=BLUE,label='selected regularized final PA')
    ax.set(xlabel='radius / arcsec',ylabel='position angle / degree',title='Geometric regularization');ax.legend(fontsize=8)
    ax=axes[0,2]
    ax.plot(radius,initial[:,1],color=ORANGE,label='supplied dispersion start')
    ax.plot(radius,fresh[:,1],color=BLUE,label='fresh intrinsic dispersion')
    ax.set(xlabel='radius / arcsec',ylabel='dispersion / km/s',title='Beam mixing is included in the3D fit');ax.legend(fontsize=8)
    before=np.sqrt(np.mean((initial_model.astype(float)-cube)**2,axis=0))
    after=np.sqrt(np.mean((fresh_model.astype(float)-cube)**2,axis=0))
    maximum=max(float(before.max()),float(after.max()))
    _image(fig,axes[1,0],before,'Initialized model residual RMS','Jy/beam','magma',vmin=0,vmax=maximum)
    _image(fig,axes[1,1],after,'Final3D model residual RMS','Jy/beam','magma',vmin=0,vmax=maximum)
    measures=np.array([rms(cube),rms(initial_model.astype(float)-cube),rms(fresh_model.astype(float)-cube)])
    ax=axes[1,2];bars=ax.bar(['zero emission','initialized','final3D'],measures,color=[GRAY,ORANGE,BLUE])
    for bar,value in zip(bars,measures):ax.text(bar.get_x()+bar.get_width()/2,value,f'{value:.6f}',ha='center',va='bottom',fontsize=8)
    ax.set(ylabel='full-cube RMS / Jy/beam',title='All1395000 observed voxels');ax.margins(y=.2)
    caption='These are freshly computed stages of the selected method, not independent truth. Source boundary starts are retained: rotation120 on the first and unmatched last initialization knots. The40-ring2D solution is not mislabeled as a complete41-ring inverse. Every displayed model was actually evaluated on the unchanged observation.'
    fig.supxlabel(textwrap.fill(caption,155),fontsize=8)
    return save_figure(fig,'05_initialization_comparison',dict(radius_arcsec=radius,initial=initial,prepass=prepass,fresh=fresh,initialized_residual_RMS_map=before,final_residual_RMS_map=after,full_cube_RMS=measures),caption,directory)


def plot_sensitivity(cube,nominal_model,variant_models,position_angle,directory=None):
    keys=['rotation_minus','rotation_plus','dispersion_minus','dispersion_plus','angle_minus','angle_plus']
    names=['rotation','dispersion','position angle'];steps=[[-5,0,5],[-20,0,20],[-3,0,3]]
    units=['relative change / %','relative change / %','offset / degree']
    nominal=rms(nominal_model.astype(float)-cube)
    scores=np.array([rms(variant_models[key].astype(float)-cube) for key in keys])
    fig,axes=plt.subplots(2,3,figsize=(14,8.5),layout='constrained')
    fig.suptitle('6  |  Controlled physical forward sensitivity on the same observation',fontsize=15)
    for j,ax in enumerate(axes[0]):
        values=[scores[2*j],nominal,scores[2*j+1]]
        ax.plot(steps[j],values,color=[BLUE,ORANGE,PURPLE][j],marker='o',lw=2)
        ax.set(xlabel=units[j],ylabel='full-cube RMS / Jy/beam',title='Perturb '+names[j]+' only')
        ax.ticklabel_format(axis='y',style='plain',useOffset=False)
        for x,value in zip(steps[j],values):ax.annotate(f'{value:.6f}',(x,value),xytext=(0,7),textcoords='offset points',ha='center',fontsize=8)
        ax.margins(x=.2,y=.25)
    radius,nominal_pv=spectral_slice(nominal_model,position_angle)
    selected=['rotation_plus','dispersion_plus','angle_plus']
    changes=[spectral_slice(variant_models[key],position_angle)[1]-nominal_pv for key in selected]
    norm=signed_norm(*changes)
    for ax,value,title in zip(axes[1],changes,['Vrot+5% minus nominal','dispersion+20% minus nominal','PA+3degrees minus nominal']):
        _pv(fig,ax,value,radius,title,'Jy/beam',norm=norm,cmap='RdBu_r')
    caption='All six alternatives are newly computed forward predictions with the original observations unchanged. Other fields stay fixed; nuisance brightness follows the same observation-only estimator. These are sensitivity diagnostics, not reoptimized inverse solutions, synthetic benchmark observations or confidence intervals. Shared signed limits include every plotted value.'
    fig.supxlabel(textwrap.fill(caption,155),fontsize=8)
    numeric=dict(nominal_full_cube_RMS=nominal,variant_full_cube_RMS=scores,radius_arcsec=radius,nominal_pv=nominal_pv)
    numeric.update({key:value for key,value in zip(selected,changes)})
    return save_figure(fig,'06_forward_sensitivity',numeric,caption,directory)
