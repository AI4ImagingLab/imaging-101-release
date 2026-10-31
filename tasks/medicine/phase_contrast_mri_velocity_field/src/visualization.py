"""Actual scientific review figures; no backend selection or upstream imports."""
from pathlib import Path
import json,hashlib
import numpy as np
import matplotlib.pyplot as plt

def plot_component_means(ax,velocity,original_means):
    """Plot all temporal samples, pairing fresh/original by direction and color."""
    means=velocity.mean(axis=(2,3,4))
    for d,(label,color) in enumerate(zip('xyz',['#1764ab','#d95f02','#387b4d'])):
        ax.plot(means[:,d]*100,color=color,label=label+' fresh')
        ax.plot(original_means[:,d]*100,'--',color=color,linewidth=1,label=label+' original')
    return means

def make_figures(images,result,diagnostics,upstream,comparisons,judge_report,export_dir=None):
    """Return six real-data figure groups; optionally export plotted numeric data.

    Public layout is batch,time,encoding,z,y,x; output is batch,time,component,z,y,x.
    Upstream sections are original-run measurements, never fresh reconstructions.
    comparisons maps mask fractions to independently fresh result dictionaries.
    """
    obs=images[0];v=result['reconstruction'][0];tshow=[0,6,13,20];iz=27;it=6
    mag=diagnostics['reference_magnitude'].transpose(3,2,1,0)
    mask=diagnostics['mask'].transpose(3,2,1,0)
    extentxy=(0,150,0,100);sources={};figures=[]
    style={'font.size':10,'axes.titlesize':10,'axes.labelsize':9,'figure.titlesize':13,'svg.fonttype':'none'}

    def panel(fig,ax,a,title,unit,cmap='viridis',extent=extentxy,vmin=None,vmax=None):
        im=ax.imshow(a,origin='lower',extent=extent,aspect='equal',cmap=cmap,vmin=vmin,vmax=vmax,interpolation='nearest')
        ax.set_title(title);ax.set_xlabel('stored x (mm)');ax.set_ylabel('stored y (mm)')
        fig.colorbar(im,ax=ax,shrink=.83,pad=.025,label=unit)
        return im

    with plt.rc_context(style):
        fig,axes=plt.subplots(2,4,figsize=(15,7),layout='constrained')
        for d,label in enumerate(['Reference','Encoding x','Encoding y','Encoding z']):
            amplitude=np.abs(obs[it,d,iz]);phase=np.angle(obs[it,d,iz])
            panel(fig,axes[0,d],amplitude*1e5,f'{label}: magnitude','signal x1e-5','gray',vmin=0,vmax=float(np.abs(obs[it,:,iz]).max()*1e5))
            panel(fig,axes[1,d],phase,f'{label}: complex phase','rad','twilight',vmin=-np.pi,vmax=np.pi)
            sources[f'observed_magnitude_{d}']=amplitude*1e5;sources[f'observed_phase_{d}']=phase
        fig.suptitle('1 | Real four-encoding MRI observations; native z=27, temporal phase=6')
        figures.append(fig)

        fig,axes=plt.subplots(3,4,figsize=(15,10),layout='constrained')
        for d,label in enumerate('xyz'):
            for j,t in enumerate(tshow):
                a=np.where(mask[t,iz],v[t,d,iz]*100,np.nan)
                panel(fig,axes[d,j],a,f'{label} encoding, phase {t}','cm/s','RdBu_r',vmin=-3,vmax=3)
                sources[f'velocity_{label}_t{t}_xy']=a
        fig.suptitle('2 | Fresh relative velocity: all directions over time; shared +/-3 cm/s display saturation')
        figures.append(fig)

        speed=np.sqrt(np.sum(v.astype(np.float64)**2,axis=1))*100
        sections=[(mag[it,27],speed[it,27],(0,150,0,100),'xy at z=27','stored x (mm)','stored y (mm)'),
                  (mag[it,:,33,:],speed[it,:,33,:],(0,150,0,80),'xz at y=33','stored x (mm)','stored z (mm)'),
                  (mag[it,:,:,50],speed[it,:,:,50],(0,100,0,80),'yz at x=50','stored y (mm)','stored z (mm)')]
        fig,axes=plt.subplots(2,3,figsize=(13,8),layout='constrained')
        for j,(m,s,e,label,xlabel,ylabel) in enumerate(sections):
            for row,a,unit,cmap,upper in [(0,m*1e5,'signal x1e-5','gray',None),(1,s,'cm/s','magma',4.)]:
                panel(fig,axes[row,j],a,('Reference magnitude: ' if row==0 else 'Relative speed: ')+label,unit,cmap,e,vmin=0,vmax=upper)
                axes[row,j].set_xlabel(xlabel);axes[row,j].set_ylabel(ylabel)
            sources['orthogonal_magnitude_'+str(j)]=m*1e5;sources['orthogonal_speed_'+str(j)]=s
        fig.suptitle('3 | Complete3D observation boundary and recovered relative speed; no anatomical-axis inference')
        figures.append(fig)

        coverage=mask.mean(axis=(1,2,3));counts=mask.sum(axis=(1,2,3))
        high_signal=mag.mean(axis=0)>.3*mag.max()
        curves=np.array([np.mean(v[:,d,high_signal],axis=1) for d in range(3)])
        speed_q=np.quantile(speed[:,high_signal],[.5,.95],axis=1)
        fig,axes=plt.subplots(2,3,figsize=(14,8),layout='constrained')
        means=plot_component_means(axes[0,0],v,upstream['mean_components'])
        for d,label in enumerate('xyz'):
            axes[1,0].plot(curves[d]*100,label=label)
        axes[0,0].set(title='All-volume component means',xlabel='temporal phase index',ylabel='relative velocity (cm/s)');axes[0,0].legend(ncol=2,fontsize=8)
        axes[0,1].plot(coverage*100,label='fresh');axes[0,1].plot(upstream['mask_fraction']*100,'--',label='original');axes[0,1].set(title='Time-varying observed support',xlabel='temporal phase index',ylabel='in-support voxels (%)');axes[0,1].legend()
        axes[0,2].plot(counts,label='fresh');axes[0,2].plot(upstream['mask_count'],'--',label='original');axes[0,2].set(title='Every phase uses the full spatial grid',xlabel='temporal phase index',ylabel='observed-support voxel count');axes[0,2].legend()
        axes[1,0].set(title='Observed high-signal region (mean magnitude>0.3global max)',xlabel='temporal phase index',ylabel='relative velocity (cm/s)');axes[1,0].legend()
        axes[1,1].plot(speed_q[0],label='median');axes[1,1].plot(speed_q[1],label='95th percentile');axes[1,1].set(title='High-signal relative speed distribution',xlabel='temporal phase index',ylabel='speed (cm/s)');axes[1,1].legend()
        axes[1,2].hist(v.mean(axis=0).ravel()*100,bins=80,histtype='step');axes[1,2].set(title='Mask after centering: final mean can be nonzero',xlabel='voxel/direction temporal mean (cm/s)',ylabel='voxel-direction count')
        fig.suptitle('4 | Actual full-volume temporal behavior; phase index is not a calibrated time in seconds')
        sources.update(mean_components_cm_s=means*100,source_mean_components_cm_s=upstream['mean_components']*100,mask_fraction_percent=coverage*100,source_mask_fraction_percent=upstream['mask_fraction']*100,mask_count=counts,source_mask_count=upstream['mask_count'],high_signal_component_means_cm_s=curves*100,high_signal_speed_quantiles_cm_s=speed_q)
        h,e=np.histogram(v.mean(axis=0).ravel()*100,bins=80);sources['temporal_mean_histogram']=h;sources['temporal_mean_histogram_edges']=e
        figures.append(fig)

        fig,axes=plt.subplots(3,3,figsize=(12,10),layout='constrained')
        for d,label in enumerate('xyz'):
            old=upstream['xy'][it,d];fresh=v[it,d,iz]
            panel(fig,axes[d,0],old*100,label+' independently reproduced original','cm/s','RdBu_r',vmin=-3,vmax=3)
            panel(fig,axes[d,1],fresh*100,label+' fresh local','cm/s','RdBu_r',vmin=-3,vmax=3)
            diff=(fresh.astype(float)-old.astype(float))*1e9
            panel(fig,axes[d,2],diff,f'Fresh-original; max {np.max(abs(diff)):.3g}','nm/s','RdBu_r',vmin=-30,vmax=30)
            sources[f'source_{label}_xy']=old*100;sources[f'fresh_{label}_xy']=fresh*100;sources[f'parity_{label}_nm_s']=diff
        # Consume and verify both additional independently captured original sections.
        for key,section in [('xz',v[:,:,:,33,:]),('yz',v[:,:,:,:,50])]:
            delta=section.astype(float)-upstream[key].astype(float)
            sources['source_'+key+'_parity_max_m_s']=np.array(float(abs(delta).max()))
        fig.suptitle('5 | Independent original versus fresh extraction; method agreement is not measured velocity truth')
        figures.append(fig)

        fig,axes=plt.subplots(2,3,figsize=(14,8),layout='constrained')
        for j,fraction in enumerate([.05,.1,.2]):
            variant=comparisons[fraction]['reconstruction'][0];sp=np.sqrt(np.sum(variant[it,:,iz].astype(float)**2,axis=0))*100
            panel(fig,axes[0,j],sp,f'Fresh support threshold {fraction:.2f}','cm/s','magma',vmin=0,vmax=4)
            sources['support_speed_'+str(fraction)]=sp
        for d,label in enumerate('xyz'):
            values=v[:,d][mask]*100;h,e=np.histogram(values,bins=120)
            axes[1,0].stairs(h,e,label=label);sources['velocity_histogram_'+label]=h;sources['velocity_histogram_edges_'+label]=e
        axes[1,0].set(title='Full in-support range, including displayed saturations',xlabel='relative velocity (cm/s)',ylabel='voxel-phase count',yscale='log');axes[1,0].legend()
        uncentered=comparisons['uncentered']['reconstruction'][0]
        for d,label in enumerate('xyz'):
            axes[1,1].plot(uncentered[:,d,27,33,50]*100,'--',label=label+' principal')
            axes[1,1].plot(v[:,d,27,33,50]*100,label=label+' centered')
        axes[1,1].set(title='One fixed voxel: temporal gauge changes the estimand',xlabel='temporal phase index',ylabel='velocity (cm/s)');axes[1,1].legend(ncol=2,fontsize=8)
        report=judge_report['metrics'];axes[1,2].axis('off')
        text=('Independent complete-grid Judge\n\n'+f"Phase-phasor RMS: {report['phase_phasor_rms']:.6g}\n"+f"Outside-support max: {report['outside_mask_max']:.6g} m/s\n"+f"Global DR-NRMSE: {report['nrmse']:.6g}\n"+f"Global cosine: {report['ncc']:.12f}\n\n"+'No independently measured velocity truth.\nNo unwrapped absolute/steady velocity.\nNo anatomical gradient-sign calibration.\nLow-signal and acquisition artifacts persist.\nThe public temporal gauge is fixed, not fitted.')
        axes[1,2].text(0,.98,text,va='top',fontsize=10)
        sources['fixed_voxel_principal_cm_s']=uncentered[:,:,27,33,50]*100;sources['fixed_voxel_centered_cm_s']=v[:,:,27,33,50]*100
        fig.suptitle('6 | Fresh support/gauge comparisons, full dynamic range and independently checked phase closure')
        figures.append(fig)

    if export_dir:
        dest=Path(export_dir);dest.mkdir(parents=True,exist_ok=True)
        for k,fig in enumerate(figures,1):
            for ext in ('png','svg','pdf'):fig.savefig(dest/f'figure_{k}.{ext}',dpi=150,bbox_inches='tight')
        np.savez_compressed(dest/'numeric_sources.npz',**sources)
        (dest/'metrics.json').write_text(json.dumps(judge_report,indent=2,allow_nan=False),encoding='utf8')
        files=[p for p in dest.iterdir() if p.name!='hash_manifest.json' and p.is_file()]
        manifest={p.name:dict(bytes=p.stat().st_size,sha256=hashlib.file_digest(open(p,'rb'),'sha256').hexdigest()) for p in files}
        (dest/'hash_manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf8')
    return figures
