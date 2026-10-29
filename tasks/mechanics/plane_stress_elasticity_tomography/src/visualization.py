"""Scientific comparisons from computed arrays; importing selects no backend."""
from pathlib import Path
import hashlib
import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.tri as mtri
from .physics_model import PlaneStressModel

METHODS=('tv','smooth','tikhonov')
LABELS={'tv':'TV','smooth':'Gaussian smoothness','tikhonov':'Quadratic L2'}
BLUE='#0072B2';ORANGE='#D55E00';GREEN='#009E73'

def quality_metrics(estimate: np.ndarray,reference: np.ndarray):
    x=np.asarray(estimate).ravel();r=np.asarray(reference).ravel()
    if x.shape!=r.shape or not np.isfinite(x).all() or not np.isfinite(r).all():
        raise ValueError('Metric arrays must have the same finite shape')
    if np.linalg.norm(x)==0 or np.linalg.norm(r)==0 or np.ptp(r)==0:
        raise ValueError('Undefined correlation or reference range')
    return {'ncc_vs_ref':float(x@r/(np.linalg.norm(x)*np.linalg.norm(r))),
            'nrmse_vs_ref':float(np.sqrt(np.mean((x-r)**2))/np.ptp(r))}

def comparison_metrics(acquisition,field: np.ndarray,truth: np.ndarray):
    measured=acquisition.displacement.ravel();model=PlaneStressModel(acquisition)
    predicted=model.forward(field)
    result=quality_metrics(field,truth)
    result['relative_displacement_error']=float(np.linalg.norm(predicted-measured)/np.linalg.norm(measured))
    result['maximum_abs_principal_strain']=float(np.max(abs(np.linalg.eigvalsh(model.strain(predicted)))))
    return result

def _field(ax,mesh,values,title,vmin=0,vmax=350,cmap='viridis'):
    image=ax.tripcolor(mesh,facecolors=np.asarray(values),shading='flat',
                        cmap=cmap,vmin=vmin,vmax=vmax,rasterized=False)
    ax.set(title=title,xlabel='x (L0)',ylabel='y (L0)',aspect='equal')
    return image

def _nodal(ax,mesh,values,title,scale,vmax):
    image=ax.tricontourf(mesh,scale*np.asarray(values),levels=np.linspace(-vmax,vmax,31),cmap='RdBu_r',extend='both')
    ax.set(title=title,xlabel='x (L0)',ylabel='y (L0)',aspect='equal')
    return image

def _profile(mesh,field):
    x=np.linspace(.01,9.99,400);indices=mesh.get_trifinder()(x,np.full_like(x,3.3))
    assert np.all(indices>=0)
    return x,np.asarray(field)[indices]

def make_figures(acquisition,truth: np.ndarray,original: dict,qualified: dict,
                 fresh: dict,sensitivity: dict,diagnostics: dict):
    """Six complete groups; all three original/qualified/fresh methods required.

    original/qualified values are numeric dictionaries with reconstruction,
    modulus_history and objective_history. fresh/sensitivity values are actual
    ReconstructionResult objects. No missing group is replaced or synthesized.
    """
    for group in (original,qualified,fresh):
        if set(group)!=set(METHODS):raise ValueError('All three computed methods are required')
    if set(sensitivity)!={.5,1.,2.}:raise ValueError('Require three actual TV multipliers')
    nodes=acquisition.nodes;mesh=mtri.Triangulation(nodes[:,0],nodes[:,1],acquisition.triangles)
    truth=np.asarray(truth).ravel();model=PlaneStressModel(acquisition)
    scored={method:comparison_metrics(acquisition,fresh[method].reconstruction,truth) for method in METHODS}
    qscore={method:comparison_metrics(acquisition,qualified[method]['reconstruction'].ravel(),truth) for method in METHODS}
    figures={}
    with plt.rc_context({'font.size':9,'axes.titlesize':10,'axes.labelsize':9,
                         'legend.fontsize':8,'svg.fonttype':'none','pdf.fonttype':42}):
        fig,axes=plt.subplots(1,3,figsize=(13.6,4.1),layout='constrained')
        ax=axes[0];ax.triplot(mesh,color='.72',lw=.35)
        fixed=acquisition.fixed_dofs
        ax.scatter(nodes[fixed[:,0],0],nodes[fixed[:,0],1],s=13,marker='s',color=BLUE,label='x constrained')
        ax.scatter(nodes[fixed[:,1],0],nodes[fixed[:,1],1],s=14,marker='_',color=ORANGE,label='y constrained')
        loaded=np.any(acquisition.nodal_force!=0,axis=1)
        force=acquisition.nodal_force[loaded]
        ax.quiver(nodes[loaded,0],nodes[loaded,1],force[:,0],force[:,1],angles='xy',scale_units='xy',scale=.0035,color=GREEN,width=.006)
        ax.set(xlim=(-.4,11.1),ylim=(-.3,10.3),aspect='equal',title='A  Known mesh, load and constraints',xlabel='x (L0)',ylabel='y (L0)')
        ax.text(.4,9.5,'441 nodes / 800 elements\n0.00275 F0 per loaded node',va='top',fontsize=8)
        ax.legend(loc='lower left',framealpha=.95)
        image=axes[1].tricontourf(mesh,1000*np.linalg.norm(acquisition.displacement,axis=1),levels=25,cmap='viridis')
        axes[1].set(title='B  Observed displacement magnitude',xlabel='x (L0)',ylabel='y (L0)',aspect='equal')
        fig.colorbar(image,ax=axes[1],label='|u observed| (10^-3 L0)')
        strain=diagnostics['true_principal_max_per_element']
        image=_field(axes[2],mesh,1000*strain,'C  Independent generator strain QA',vmax=1000*np.max(strain))
        fig.colorbar(image,ax=axes[2],label='max |principal strain| (10^-3)')
        axes[2].text(.3,9.7,f'max={np.max(strain):.6f} < 0.01',va='top',color='white',bbox=dict(facecolor='black',alpha=.6,pad=3))
        figures['01_acquisition_and_validity']=fig

        fig,axes=plt.subplots(2,4,figsize=(16,7.3),layout='constrained')
        maximum=max(350.,max(float(np.max(original[m]['reconstruction'])) for m in METHODS))
        maximum=float(np.ceil(maximum/50)*50)
        image=_field(axes[0,0],mesh,truth,'A  Actual simulated modulus',vmax=maximum)
        for column,method in enumerate(METHODS,1):
            field=original[method]['reconstruction'].ravel()
            _field(axes[0,column],mesh,field,f'{chr(65+column)}  Unmodified source: {LABELS[method]}',vmax=maximum)
        for column,method in enumerate(METHODS):
            _field(axes[1,column],mesh,fresh[method].reconstruction,f'{chr(69+column)}  Fresh local: {LABELS[method]}',vmax=maximum)
        fig.colorbar(image,ax=list(axes.ravel()[:-1]),label='Young modulus (E0)',shrink=.87)
        error=fresh['tv'].reconstruction-truth;limit=float(np.max(abs(error)))
        image=_field(axes[1,3],mesh,error,'H  Fresh TV minus actual truth',-limit,limit,'RdBu_r')
        fig.colorbar(image,ax=axes[1,3],label='Signed modulus error (E0)')
        figures['02_original_and_fresh_methods']=fig

        fig,axes=plt.subplots(2,3,figsize=(14,7.7),layout='constrained')
        image=_field(axes[0,0],mesh,qualified['tv']['reconstruction'].ravel(),'A  Qualified source TV',vmax=350)
        _field(axes[0,1],mesh,fresh['tv'].reconstruction,'B  Fresh local TV',vmax=350)
        fig.colorbar(image,ax=list(axes[0,:2]),label='Young modulus (E0)')
        difference=1000*(fresh['tv'].reconstruction-qualified['tv']['reconstruction'].ravel())
        limit=float(np.max(abs(difference)))
        image=_field(axes[0,2],mesh,difference,'C  Numerical difference: fresh minus source',-limit,limit,'RdBu_r')
        fig.colorbar(image,ax=axes[0,2],label='Modulus difference (10^-3 E0)')
        for values,label,color,style in ((truth,'Actual truth','black','-'),(qualified['tv']['reconstruction'].ravel(),'Qualified TV',ORANGE,'--'),(fresh['tv'].reconstruction,'Fresh TV',BLUE,':')):
            x,y=_profile(mesh,values);axes[1,0].plot(x,y,label=label,color=color,ls=style,lw=1.6)
        axes[1,0].set(title='D  Piecewise-element profile at y=3.3 L0',xlabel='x (L0)',ylabel='Young modulus (E0)');axes[1,0].legend()
        locations=np.arange(3);width=.34
        for column,(metric,title,ylabel) in enumerate((('nrmse_vs_ref','E  Actual-truth recovery','Dynamic-range NRMSE'),('relative_displacement_error','F  Public-observation consistency','Relative displacement L2 error')),1):
            ax=axes[1,column]
            ax.bar(locations-width/2,[qscore[m][metric] for m in METHODS],width,color=ORANGE,label='Qualified source')
            ax.bar(locations+width/2,[scored[m][metric] for m in METHODS],width,color=BLUE,label='Fresh local')
            ax.set(title=title,ylabel=ylabel,xticks=locations,xticklabels=['TV','Gaussian','L2']);ax.legend();ax.grid(axis='y',alpha=.2)
        figures['03_qualified_parity_and_metrics']=fig

        fig,axes=plt.subplots(2,3,figsize=(14,7),layout='constrained')
        for column,method in enumerate(METHODS):
            q=qualified[method];result=fresh[method]
            axes[0,column].semilogy(np.arange(len(q['objective_history'])),q['objective_history'],color=ORANGE,label='Qualified source')
            axes[0,column].semilogy(np.arange(len(result.objective_history)),result.objective_history,color=BLUE,ls='--',label='Fresh local')
            axes[0,column].set(title=f'{chr(65+column)}  {LABELS[method]}: identical objective',xlabel='Accepted update',ylabel='Objective (dimensionless)');axes[0,column].legend();axes[0,column].grid(alpha=.2)
            histories=((original[method]['modulus_history'],'Unmodified source','.4','-'),(q['modulus_history'],'Qualified source',ORANGE,'-'),(result.modulus_history,'Fresh local',BLUE,'--'))
            for history,label,color,style in histories:
                values=[quality_metrics(row,truth)['nrmse_vs_ref'] for row in history]
                axes[1,column].plot(np.arange(len(values)),values,label=label,color=color,ls=style)
            axes[1,column].set(title=f'{chr(68+column)}  {LABELS[method]}: common truth metric',xlabel='Accepted update',ylabel='Dynamic-range NRMSE');axes[1,column].legend();axes[1,column].grid(alpha=.2)
        figures['04_convergence_and_actual_stops']=fig

        fig,axes=plt.subplots(2,4,figsize=(16,7),layout='constrained')
        predicted=fresh['tv'].predicted_displacement.reshape(-1,2);observed=acquisition.displacement
        residual=predicted-observed
        for row,component in enumerate(('x','y')):
            limit=1000*max(float(np.max(abs(observed[:,row]))),float(np.max(abs(predicted[:,row]))))
            for column,(values,title) in enumerate(((observed[:,row],'Observed'),(predicted[:,row],'Fresh TV predicted'))):
                image=_nodal(axes[row,column],mesh,values,f'{chr(65+row*4+column)}  {title} u({component})',1000,limit)
                fig.colorbar(image,ax=axes[row,column],label=f'u({component}) (10^-3 L0)')
            residual_limit=1e6*float(np.max(abs(residual[:,row])))
            image=_nodal(axes[row,2],mesh,residual[:,row],f'{chr(67+row*4)}  Predicted minus observed',1e6,residual_limit)
            fig.colorbar(image,ax=axes[row,2],label=f'Residual u({component}) (10^-6 L0)')
        for component,label,color in ((0,'x',BLUE),(1,'y',ORANGE)):
            axes[0,3].scatter(1000*observed[:,component],1000*predicted[:,component],s=7,alpha=.6,label=label,color=color)
        limits=np.array([min(observed.min(),predicted.min()),max(observed.max(),predicted.max())])*1000
        axes[0,3].plot(limits,limits,color='black',lw=1)
        axes[0,3].set(title='D  Predicted versus observed',xlabel='Observed (10^-3 L0)',ylabel='Predicted (10^-3 L0)');axes[0,3].legend()
        measured=observed.ravel();sigma=np.sqrt((.001*np.abs(measured))**2+(.001*np.ptp(measured))**2)
        for method,color in zip(METHODS,(BLUE,ORANGE,GREEN)):
            normalized=(fresh[method].predicted_displacement-measured)/sigma
            axes[1,3].hist(normalized,bins=45,density=True,histtype='step',color=color,label=f'{LABELS[method]}: RMS {np.sqrt(np.mean(normalized**2)):.2f}')
        guide=np.linspace(-5,5,300);axes[1,3].plot(guide,np.exp(-guide**2/2)/np.sqrt(2*np.pi),'k--',label='Unit Gaussian guide')
        axes[1,3].set(title='H  Noise-scaled residuals; bias retained',xlabel='Residual / estimated true-noise SD',ylabel='Density');axes[1,3].legend(fontsize=7)
        figures['05_observation_fits_and_noise']=fig

        fig,axes=plt.subplots(2,3,figsize=(14,7.6),layout='constrained')
        for column,multiplier in enumerate((.5,1.,2.)):
            image=_field(axes[0,column],mesh,sensitivity[multiplier].reconstruction,f'{chr(65+column)}  Fresh TV multiplier {multiplier:g}',vmax=350)
        fig.colorbar(image,ax=list(axes[0,:]),label='Young modulus (E0)')
        for multiplier in (.5,1.,2.):
            value=comparison_metrics(acquisition,sensitivity[multiplier].reconstruction,truth)
            axes[1,0].scatter(value['nrmse_vs_ref'],value['relative_displacement_error'],s=65,edgecolors='black' if multiplier==1 else 'none',color=BLUE)
            axes[1,0].annotate(f'  {multiplier:g}x',(value['nrmse_vs_ref'],value['relative_displacement_error']))
        axes[1,0].set(title='D  Actual regularization tradeoff',xlabel='Dynamic-range NRMSE versus truth',ylabel='Relative displacement L2 error');axes[1,0].grid(alpha=.2)
        x=np.arange(3);width=.35
        axes[1,1].bar(x-width/2,diagnostics['wrong_homogeneous_column_errors'],width,color=ORANGE,label='Wrong homogeneous-column state')
        axes[1,1].bar(x+width/2,diagnostics['correct_current_derivative_step_errors'],width,color=BLUE,label='Current-state independent FD check')
        axes[1,1].set(title='E  Independent derivative qualification',ylabel='Relative derivative discrepancy',xticks=x,xticklabels=['Element0','Element399','Element799'],yscale='log');axes[1,1].legend(fontsize=7);axes[1,1].grid(axis='y',alpha=.2)
        axes[1,2].bar([0,1],diagnostics['load_case_maximum_strains'],color=[ORANGE,BLUE])
        axes[1,2].axhline(.01,color='black',ls='--',label='Fixed small-strain ceiling')
        axes[1,2].set(title='F  Actual source acquisition qualification',xticks=[0,1],xticklabels=['Excluded large load','Bound small load'],ylabel='Maximum absolute principal strain',yscale='log');axes[1,2].legend(fontsize=7)
        figures['06_sensitivity_and_independent_physics']=fig
    return figures

def export_figures(figures: dict,directory: str | Path,numeric_sources: dict,metadata: dict):
    """Write actual scientific figures and hashed numeric source/metric manifests."""
    destination=Path(directory);destination.mkdir(parents=True,exist_ok=True)
    for key,value in numeric_sources.items():
        if np.asarray(value).dtype.kind=='O':raise ValueError(f'Object numeric source {key}')
    np.savez_compressed(destination/'numeric_sources.npz',**numeric_sources)
    manifest={'metadata':metadata,'numeric_sources_sha256':hashlib.sha256((destination/'numeric_sources.npz').read_bytes()).hexdigest(),'groups':{}}
    for name,figure in figures.items():
        files={}
        for extension in ('png','svg','pdf'):
            path=destination/f'{name}.{extension}'
            figure.savefig(path,dpi=170,bbox_inches='tight')
            files[extension]={'filename':path.name,'sha256':hashlib.sha256(path.read_bytes()).hexdigest()}
        manifest['groups'][name]={'files':files,'source_keys':metadata['figure_sources'][name]}
    (destination/'figure_manifest.json').write_text(json.dumps(manifest,indent=2,allow_nan=False)+'\n',encoding='utf8')
    return manifest
