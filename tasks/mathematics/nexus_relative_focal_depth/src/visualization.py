"""Computed focus-stack review figures; no reconstruction is synthesized here."""
from pathlib import Path
import json
import numpy as np
import matplotlib.pyplot as plt
from .physics_model import objective


def image_metrics(estimate,reference):
    a=np.asarray(estimate,np.float64).ravel();b=np.asarray(reference,np.float64).ravel()
    if not np.isfinite(a).all() or not np.isfinite(b).all() or len(a)!=len(b):raise ValueError('Invalid metric arrays')
    norm=np.linalg.norm(b);span=np.ptp(b)
    if not(norm>0 and span>0):raise ValueError('Degenerate reference')
    scale=np.linalg.norm(a)
    return {'ncc':float(np.dot(a,b)/(scale*norm)) if scale>0 else 0.,
            'nrmse':float(np.sqrt(np.mean((a-b)**2))/span)}


def relative_surface_figure(images,result,sharpness):
    """Measured relative surface and focus-evidence margin; no metric3D claim."""
    fig=plt.figure(figsize=(16,7),layout='constrained');ax=fig.add_subplot(121,projection='3d')
    fig.suptitle('7. Dense relative focal surface; no metric3D geometry is claimed')
    stride=12;y,x=np.mgrid[0:1080:stride,0:1920:stride];z=result[::stride,::stride]
    colors=images[9,::stride,::stride,::-1].astype(float)/255
    ax.plot_surface(x,y,z,facecolors=colors,linewidth=0,antialiased=False,shade=False)
    ax.set(xlabel='Column(pixel)',ylabel='Row(pixel)',zlim=(-10,10))
    ax.set_box_aspect((1920,1080,850));ax.view_init(elev=40,azim=-100)
    ax.text2D(.02,.98,'Vertical axis: relative focal coordinate',transform=ax.transAxes,va='top')
    ax=fig.add_subplot(122)
    sorted_response=np.sort(sharpness,axis=0)
    margin=(sorted_response[-1]-sorted_response[-2])/(sorted_response[-1]+np.float32(.3))
    im=ax.imshow(margin,cmap='magma',vmin=0,vmax=float(np.percentile(margin,99)))
    ax.set(title='Observed top-two focus-score margin\nSmall margin signals local ambiguity; not posterior variance',xlabel='Column(pixel)',ylabel='Row(pixel)')
    fig.colorbar(im,ax=ax,extend='max',label='Dimensionless normalized score margin')
    return fig


def make_figures(images,result,details,upstream,source_diagnostics,classical,variants,tv_only,directory=None):
    """Return8 actual science groups, numeric metrics and figure provenance.

    Main result/details and all classical/weight/ROF results must already be
    computed by the caller. The separately labeled source variants are existing
    actual oracle measurements. No cached array is substituted for fresh input.
    """
    plt.rcParams.update({'font.size':10,'axes.titlesize':11,'figure.titlesize':16,
                          'axes.spines.top':False,'axes.spines.right':False})
    figures=[];captions=[]
    def field(ax,a,title,limits=(-10,10),cmap='viridis'):
        im=ax.imshow(a,cmap=cmap,vmin=limits[0],vmax=limits[1],interpolation='nearest')
        ax.set(title=title,xlabel='Column(pixel)',ylabel='Row(pixel)')
        fig=ax.figure;fig.colorbar(im,ax=ax,shrink=.82,label='Relative focal coordinate')
        return im
    def add(fig,caption):figures.append(fig);captions.append(caption)
    fig,axes=plt.subplots(4,5,figsize=(18,11),layout='constrained')
    fig.suptitle('1. Complete real19-frame author acquisition; display stride4 only')
    for k,ax in enumerate(axes.flat):
        if k<19:
            ax.imshow(images[k,::4,::4,::-1]);ax.set_title(f'Frame{k:02d}; relative setting{20*(k/18-.5):.2f}')
            ax.set_xticks([]);ax.set_yticks([])
        else:
            ax.axis('off');ax.text(.05,.75,'All19 native1080×1920\nBGR observations enter every\nfull reconstruction.\n\nDisplay decimation changes\nonly this image atlas.\nNo metric lens calibration.',transform=ax.transAxes,va='top')
    add(fig,'All19 original real frames; plot-only stride4, no reconstruction decimation.')

    fig,axes=plt.subplots(2,3,figsize=(17,9),layout='constrained')
    fig.suptitle('2. Original-method agreement and independently executed source diagnostics')
    field(axes[0,0],upstream,'Unmodified original400steps(method reference)')
    field(axes[0,1],result,'Fresh local400steps')
    field(axes[0,2],source_diagnostics['corrected_initializer_result'][0],'Qualified source diagnostic: zero-based initializer')
    for ax,a,label in [(axes[1,0],result,'Fresh−original'),(axes[1,1],source_diagnostics['no_fma_result'][0],'Original compiler noFMA−original'),(axes[1,2],source_diagnostics['corrected_initializer_result'][0],'Qualified initializer−original')]:
        difference=a.astype(float)-upstream
        limit=max(float(np.percentile(np.abs(difference),99.5)),1e-7)
        im=ax.imshow(difference,cmap='RdBu_r',vmin=-limit,vmax=limit)
        ax.set(title=label+'\n99.5% colour extent; all pixels scored',xlabel='Column(pixel)',ylabel='Row(pixel)')
        fig.colorbar(im,ax=ax,shrink=.82,extend='both',label='Focal-coordinate difference')
    add(fig,'Source FMA and qualified initializer variants are independently executed oracle results, not fresh local results or physical truth. Difference colour limits affect display only.')

    fig,axes=plt.subplots(2,2,figsize=(13,9),layout='constrained')
    fig.suptitle('3. Actually computed classical focus estimates')
    for ax,(name,a) in zip(axes.flat,classical.items()):field(ax,a,name)
    add(fig,'All classical methods freshly use all19 native observations. Box size and median size appear in each method name.')

    fig,axes=plt.subplots(2,2,figsize=(13,9),layout='constrained')
    fig.suptitle('4. Joint variational reconstruction: relative data/TV tradeoff')
    for ax,a in zip(axes.flat,[4.,6.,12.,28.]):
        x=result if a==6 else variants[a]
        field(ax,x,f'Fresh400steps: a={a:g}, equivalent TV weight1/a={1/a:.4f}')
    add(fig,'Paper-relevant a4/12/28 plus selected a6, all400steps on this later author acquisition; not a claimed replication of the paper ARRI figures or absolute-depth scale.')

    fig,axes=plt.subplots(2,3,figsize=(18,9),layout='constrained')
    fig.suptitle('5. Joint reconstruction versus estimating then denoising')
    seed=classical['MLAP21×21 + median5×5']
    for ax,a,title in [(axes[0,0],seed,'Fresh classical input'),(axes[0,1],tv_only[6.],'Fresh ROF only: weight6,250steps'),(axes[0,2],tv_only[26.],'Fresh ROF only: weight26,250steps')]:field(ax,a,title)
    field(axes[1,0],result,'Fresh joint method, a6')
    row=750
    for a,label in [(seed,'Classical'),(tv_only[6.],'ROF6'),(tv_only[26.],'ROF26'),(result,'Joint a6')]:axes[1,1].plot(a[row],label=label,lw=1.2)
    axes[1,1].set(title=f'Native row{row}, every column',xlabel='Column(pixel)',ylabel='Relative focal coordinate');axes[1,1].legend()
    groups={'Classical':seed,'ROF6':tv_only[6.],'ROF26':tv_only[26.],'Joint':result}
    vals=[objective(details['coefficients'],x)['tv']/x.size for x in groups.values()]
    axes[1,2].bar(list(groups),vals,color=['#777777','#ba6d2c','#b39b32','#007f7d'])
    axes[1,2].set(title='Spatial TV per pixel; smoothness is not accuracy',ylabel='Focal-coordinate change/pixel')
    add(fig,'ROF comparisons use full native classical21/median5 map, explicit250-step primal-dual solves; weights6/26 are defined in this relative-coordinate scale, not a claimed metric/paper-distance calibration.')

    fig,axes=plt.subplots(2,3,figsize=(18,9),layout='constrained')
    fig.suptitle('6. Fresh actual intermediate fields; fixed400-step budget')
    for ax,n in zip(axes.flat,[1,2,10,100,400]):field(ax,details['steps'][n]['u'],f'Actual fresh iteration{n}')
    iterations=list(details['steps']);errors=[image_metrics(details['steps'][n]['u'],upstream)['nrmse'] for n in iterations]
    axes[1,2].loglog(iterations,errors,'o-',color='#007f7d')
    axes[1,2].set(title='Agreement with original final method result',xlabel='Fresh iteration',ylabel='Dynamic-range NRMSE')
    add(fig,'Every plotted intermediate field is actually computed; sparse markers are recorded iterations, not an invented dense trajectory or proof of global convergence.')

    fig=relative_surface_figure(images,result,details['sharpness'])
    add(fig,'Surface uses plot-only stride12 and original frame9 colours; x/y are pixel axes and z is a differently scaled focal coordinate. Margin is a data ambiguity diagnostic, not calibrated uncertainty.')

    fig,axes=plt.subplots(2,2,figsize=(14,9),layout='constrained')
    fig.suptitle('8. Focus-curve evidence and limits of a conditional reference')
    t=20*(np.arange(19)/18-.5);u=np.linspace(-10,10,160)
    for row,col in [(250,400),(500,600),(625,1240)]:
        s=details['sharpness'][:,row,col];normalized=s/(s.max()+np.float32(.3))
        label=f'({row},{col})';points=axes[0,0].plot(t,normalized,'o',ms=3,label=label)[0]
        axes[0,0].plot(u,np.polyval(details['coefficients'][:,row,col],u),lw=1,color=points.get_color())
    axes[0,0].set(title='Actual normalized focus data and degree8fits',xlabel='Relative focus setting',ylabel='Normalized MLAP');axes[0,0].legend()
    field(axes[0,1],details['initial'],'Original-convention fresh initializer',limits=(-11.12,8.89))
    names=[];contrast=[];tv=[]
    for name,a in [('MLAP9',classical['MLAP9×9']),('MLAP41+median11',classical['MLAP41×41 + median11×11']),('ROF6',tv_only[6.]),('ROF26',tv_only[26.]),('Joint a6',result),('Source zero-base diagnostic',source_diagnostics['corrected_initializer_result'][0])]:
        score=objective(details['coefficients'],a);names.append(name);contrast.append(-score['negative_contrast']/a.size);tv.append(score['tv']/a.size)
    axes[1,0].barh(names,contrast,color='#007f7d');axes[1,0].set(title='Fitted contrast evidence; all pixels',xlabel='Mean fitted contrast')
    axes[1,1].barh(names,tv,color='#ba6d2c');axes[1,1].set(title='Spatial regularity; all pixels',xlabel='TV per pixel')
    add(fig,'Contrast and TV are distinct conditional diagnostics, not optical accuracy. Original initializer extends one frame below the polynomial node range; main iterative fields are clipped to the declared output range.')

    all_methods={'fresh_main_a6':result,**{f'classical_{k}':v for k,v in classical.items()},
       **{f'fresh_variational_a{k:g}':v for k,v in variants.items()},
       **{f'fresh_rof_weight{k:g}':v for k,v in tv_only.items()},
       'oracle_qualified_initializer':source_diagnostics['corrected_initializer_result'][0],
       'oracle_compiler_no_fma':source_diagnostics['no_fma_result'][0]}
    measured={name:{**image_metrics(a,upstream),**objective(details['coefficients'],a),
                     'fraction_at_output_bounds':float(np.mean((a<=-10)|(a>=10)))} for name,a in all_methods.items()}
    if directory is not None:
        directory=Path(directory);directory.mkdir(parents=True,exist_ok=True)
        for k,fig in enumerate(figures,1):
            for extension in ['png','svg','pdf']:fig.savefig(directory/f'figure{k}.{extension}',dpi=150)
        np.savez_compressed(directory/'figure_numeric_sources.npz',upstream=upstream,**all_methods)
        (directory/'metrics.json').write_text(json.dumps(measured,indent=2,allow_nan=False),encoding='utf8')
        (directory/'figure_provenance.json').write_text(json.dumps({str(k):caption for k,caption in enumerate(captions,1)},indent=2),encoding='utf8')
    return figures,measured,captions
