"""Scientific review figures from actually computed forces and diagnostics.

No numerical reconstruction is performed by a plotting function and no
Matplotlib backend is changed at import time. Dense 3D triangle artists are
rasterized within vector exports; axes, text, legends and colorbars stay vector.
"""
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors,cm
from matplotlib.collections import LineCollection
from matplotlib.ticker import MaxNLocator
from mpl_toolkits.mplot3d.art3d import Poly3DCollection,Line3DCollection
from .physics_model import triangle_geometry,cell_volumes

def style():
    plt.rcParams.update({'font.size':10,'axes.titlesize':11,'figure.titlesize':16,'axes.labelsize':10,
       'axes.spines.top':False,'axes.spines.right':False,'savefig.dpi':170,'figure.facecolor':'white','axes.facecolor':'#fafbfe',
       'figure.constrained_layout.hspace':.12,'figure.constrained_layout.wspace':.04,'figure.constrained_layout.h_pad':.1})

def centers(result):
    mesh=result['mesh'];p=mesh.vertices[mesh.triangles].mean(axis=1)
    interfaces=result['interfaces'];ic=np.array([p[np.all(mesh.region_labels==k,axis=1)].mean(axis=0) for k in interfaces])
    cc=[]
    for cell in result['cells']:
        faces=np.any(mesh.region_labels==cell,axis=1);cc.append(mesh.vertices[np.unique(mesh.triangles[faces])].mean(axis=0))
    return ic,np.array(cc)

def _axes3d(ax,mesh,title):
    ax.set_title(title,loc='left',fontweight='bold',pad=12)
    ax.set_xlabel('x [mesh unit]',labelpad=1);ax.set_ylabel('y [mesh unit]',labelpad=1);ax.set_zlabel('z [mesh unit]',labelpad=1)
    ax.set_box_aspect(np.ptp(mesh.vertices,axis=0));ax.view_init(20,60)
    for axis in [ax.xaxis,ax.yaxis,ax.zaxis]:axis.set_major_locator(MaxNLocator(3))
    ax.grid(False)

def surface(ax,result,values,cut=False,norm=None,cmap='viridis',explode=False):
    mesh=result['mesh'];t=mesh.triangles;xyz=mesh.vertices[t]
    _,inv=np.unique(mesh.region_labels,axis=0,return_inverse=True)
    facevalues=values[inv]
    if explode:
        _,cc=centers(result);origin=mesh.vertices.mean(axis=0);polygons=[];fv=[]
        for idx,cell in enumerate(result['cells'][1:],start=1):
            keep=np.any(mesh.region_labels==cell,axis=1)
            polygons.append(xyz[keep]+.75*(cc[idx]-origin));fv.append(facevalues[keep])
        xyz=np.concatenate(polygons);facevalues=np.concatenate(fv)
    elif cut:
        keep=xyz.mean(axis=1)[:,1]<=np.mean(mesh.vertices[:,1]);xyz=xyz[keep];facevalues=facevalues[keep]
    else:
        keep=np.any(mesh.region_labels==0,axis=1);xyz=xyz[keep];facevalues=facevalues[keep]
    norm=norm or colors.Normalize(np.min(values),np.max(values))
    palette=plt.get_cmap(cmap)
    collection=Poly3DCollection(xyz,facecolors=palette(norm(facevalues)),edgecolor='none',linewidth=0,rasterized=True)
    ax.add_collection3d(collection);ax.auto_scale_xyz(xyz[:,:,0].ravel(),xyz[:,:,1].ravel(),xyz[:,:,2].ravel())
    return cm.ScalarMappable(norm=norm,cmap=palette)

def _signed(values):
    limit=max(float(np.max(abs(values))),1e-14)
    return colors.Normalize(-limit,limit)

def plot_observations(result):
    style();mesh=result['mesh'];fig=plt.figure(figsize=(13,10),layout='constrained')
    ax=fig.add_subplot(221,projection='3d')
    category=np.arange(len(result['interfaces']))%20
    surface(ax,result,category,norm=colors.Normalize(0,19),cmap='tab20');_axes3d(ax,mesh,'A  Real segmented embryo: exterior surface')
    ax=fig.add_subplot(222,projection='3d')
    surface(ax,result,category,cut=True,norm=colors.Normalize(0,19),cmap='tab20');_axes3d(ax,mesh,'B  Observed interface labels inside a cutaway')
    ax=fig.add_subplot(223)
    a=np.array(list(result['geometry']['angles'].values()))*180/np.pi
    ax.hist(a,bins=50,color='#236e9b',alpha=.85);ax.axvline(120,color='#d86c44',linestyle='--',label='Equal-tension ideal: 120°')
    ax.set(xlabel='Mean chord wedge angle [degree]',ylabel='Wedge keys',title='C  Measured contact geometry');ax.legend(frameon=False)
    ax=fig.add_subplot(224)
    ax.hist(result['areas'],bins=np.geomspace(result['areas'].min(),result['areas'].max(),25),color='#388b7b')
    ax.set_xscale('log');ax.set(xlabel='Interface area [mesh unit²]',ylabel='Interfaces',title='D  All 357 observed interfaces')
    fig.suptitle('Geometry is the observation; force is the unknown',fontweight='bold')
    return fig,{'wedge_angle_degrees':a,'interface_areas':result['areas'],'cell_volumes':cell_volumes(mesh)[1]}

def plot_atlas(result):
    style();mesh=result['mesh'];fig=plt.figure(figsize=(14,10),layout='constrained')
    gamma=result['tensions'];p=result['pressures'];ic,cc=centers(result)
    norm=colors.LogNorm(.35,8)
    ax=fig.add_subplot(221,projection='3d');m=surface(ax,result,gamma,cut=True,norm=norm)
    _axes3d(ax,mesh,'A  Relative interface tensions: cutaway');cb=fig.colorbar(m,ax=ax,shrink=.65,label='γ / mean(γ)',ticks=[.5,1,2,4,8]);cb.ax.set_yticklabels(['0.5','1','2','4','8']);cb.minorticks_off()
    ax=fig.add_subplot(222,projection='3d');m=surface(ax,result,gamma,explode=True,norm=norm)
    _axes3d(ax,mesh,'B  Exploded cells expose contact tensions');cb=fig.colorbar(m,ax=ax,shrink=.65,label='γ / mean(γ)',ticks=[.5,1,2,4,8]);cb.ax.set_yticklabels(['0.5','1','2','4','8']);cb.minorticks_off()
    ax=fig.add_subplot(223,projection='3d')
    q=ax.scatter(*cc[1:].T,c=p[1:],s=100,cmap='plasma',vmin=0,vmax=p.max(),edgecolors='white',linewidths=.4)
    _axes3d(ax,mesh,'C  All 64 cell pressures at geometry centroids');fig.colorbar(q,ax=ax,shrink=.65,label='p [mean tension / mesh unit]')
    ax=fig.add_subplot(224)
    index={int(k):i for i,k in enumerate(result['cells'])}
    keep=np.all(result['interfaces']!=0,axis=1);pairs=np.array([[index[int(a)],index[int(b)]] for a,b in result['interfaces'][keep]])
    seg=cc[pairs][:,:,[0,2]]
    lines=LineCollection(seg,array=gamma[keep],norm=norm,cmap='viridis',linewidths=.4+1.5*np.sqrt(result['areas'][keep]/result['areas'].max()),alpha=.7)
    ax.add_collection(lines);q=ax.scatter(cc[1:,0],cc[1:,2],c=p[1:],s=45,cmap='plasma',vmin=0,vmax=p.max(),edgecolors='white',linewidths=.4)
    ax.autoscale();ax.set_aspect('equal');ax.set(xlabel='x [mesh unit]',ylabel='z [mesh unit]',title='D  Force graph: edges γ, nodes p')
    fig.colorbar(q,ax=ax,shrink=.8,label='p [mean tension / mesh unit]')
    fig.suptitle('A complete conditional mechanical atlas',fontweight='bold')
    return fig,{'tensions':gamma,'pressures':p,'interface_centers':ic,'cell_centers':cc,'graph_interface_pairs':result['interfaces'],'explosion_fraction':np.array(.75)}

def plot_parity(result,original):
    style();fig=plt.figure(figsize=(13,10),layout='constrained');ic,cc=centers(result)
    arrays={}
    for j,key in enumerate(['tensions','pressures']):
        x=result[key];r=original[key].reshape(-1);difference=x-r
        ax=fig.add_subplot(2,2,j+1);ax.scatter(r,x,s=12,alpha=.7,color=['#246b98','#b65583'][j])
        lo=min(x.min(),r.min());hi=max(x.max(),r.max());ax.plot([lo,hi],[lo,hi],color='.3',linestyle='--',linewidth=1)
        unit='relative tension' if j==0 else 'mean tension / mesh unit'
        ax.set(xlabel=f'Independent author run [{unit}]',ylabel=f'Fresh local result [{unit}]',title=f'{chr(65+j)}  All {len(x)} {key}')
        ax=fig.add_subplot(2,2,j+3,projection='3d');c=ic if key=='tensions' else cc
        q=ax.scatter(*c.T,c=difference,s=18 if j==0 else 70,cmap='coolwarm',norm=_signed(difference))
        _axes3d(ax,result['mesh'],f'{chr(67+j)}  Signed fresh − author {key}')
        fig.colorbar(q,ax=ax,shrink=.65,label=f'Difference [{unit}]')
        arrays['author_'+key]=r;arrays['fresh_'+key]=x;arrays['difference_'+key]=difference
    fig.suptitle('Independent implementation agreement is not biological truth',fontweight='bold')
    return fig,arrays

def _metrics(x,r):
    return np.sqrt(np.mean((x-r)**2))/np.ptp(r),np.dot(x,r)/(np.linalg.norm(x)*np.linalg.norm(r))

def plot_comparisons(result,methods):
    style();fig,axes=plt.subplots(2,3,figsize=(16,10),layout='constrained')
    names=[k for k,v in methods.items() if 'tensions' in v]
    labels={'young_dupre':'Young–Dupré','projection':'Projection','lami':'Lami','inverse_lami':'Inverse Lami','log_lami':'Log Lami','variational_gram':'Variational Gram','variational_operator':'Variational Operator','unweighted_young_dupre':'No length weight'}
    normalized={k:{q:v/values['tensions'].mean() for q,v in values.items()} for k,values in methods.items() if 'tensions' in values}
    values=[normalized[k]['tensions'] for k in names]
    axes[0,0].boxplot(values,tick_labels=[labels[k] for k in names],showfliers=True,flierprops={'markersize':2});axes[0,0].set_yscale('log')
    axes[0,0].tick_params(axis='x',rotation=55);axes[0,0].set(ylabel='γ / arithmetic mean',title='A  Same geometry, different tension estimators')
    axes[0,1].boxplot([normalized[k]['pressures'][1:] for k in names],tick_labels=[labels[k] for k in names],flierprops={'markersize':2})
    axes[0,1].tick_params(axis='x',rotation=55);axes[0,1].set(ylabel='p [unit-mean tension / mesh unit]',title='B  Conditional pressure distributions')
    junction=[];force=[]
    for k in names:
        g=normalized[k]['tensions'];p=normalized[k]['pressures']
        junction.append(np.linalg.norm(result['M'][:-1]@g))
        a=result['D'].T@g;r=a-result['V'].T@p
        force.append(np.linalg.norm(r)/np.linalg.norm(a))
    positions=np.arange(len(names))
    axes[0,2].bar(positions-.18,np.array(junction)/junction[0],.36,label='Junction residual')
    axes[0,2].bar(positions+.18,np.array(force)/force[0],.36,label='Vertex-force residual')
    axes[0,2].set_xticks(positions,[labels[k] for k in names],rotation=55);axes[0,2].legend(frameon=False,fontsize=8)
    axes[0,2].set_yscale('log');axes[0,2].set(ylabel='Residual norm / Young–Dupré norm',title='C  Competing noisy-geometry objectives')
    r=result['pressures'][1:]
    for k,color in [('laplace','#a14467'),('laplace_weighted_curvature','#207d8c')]:
        axes[1,0].scatter(r,methods[k]['pressures'][1:],s=18,alpha=.7,label=k.replace('_',' '),color=color)
    axes[1,0].axline((0,0),slope=1,color='.4',linestyle='--');axes[1,0].legend(frameon=False,fontsize=8)
    axes[1,0].set(xlabel='Variational p [mean tension / mesh unit]',ylabel='Curvature-law p [same unit]',title='D  Curvature and variational pressure disagree')
    delta=methods['variational_gram']['tensions']-result['tensions'];ic,_=centers(result)
    q=axes[1,1].scatter(ic[:,0],ic[:,2],c=delta,cmap='coolwarm',norm=_signed(delta),s=22)
    axes[1,1].set_aspect('equal');axes[1,1].set(xlabel='x [mesh unit]',ylabel='z [mesh unit]',title='E  Gram − Young–Dupré γ: every interface')
    fig.colorbar(q,ax=axes[1,1],label='Δγ [unit-mean tension]')
    axes[1,2].plot(np.sort(result['tensions']),label='Length weighted',color='#246b98')
    axes[1,2].plot(np.sort(methods['unweighted_young_dupre']['tensions']),label='Equal line weights',color='#c47b37')
    axes[1,2].set_yscale('log');axes[1,2].legend(frameon=False)
    axes[1,2].set(xlabel='Sorted interface rank',ylabel='Relative tension',title='F  The paper’s length-weighting ablation')
    fig.suptitle('Actual formulation comparisons; no force truth is supplied',fontweight='bold')
    arrays={k+'_'+q:v for k,values in methods.items() for q,v in values.items()}
    arrays.update(junction_residual=np.array(junction),relative_vertex_force=np.array(force))
    return fig,arrays

def plot_sensitivity(result,perturbed,sigmas):
    style();mesh=result['mesh'];fig=plt.figure(figsize=(15,10),layout='constrained');ic,cc=centers(result)
    ax=fig.add_subplot(231,projection='3d');surface(ax,result,np.ones(len(ic)),cut=True,norm=colors.Normalize(0,2),cmap='Greys')
    edge=result['geometry']['malformed_edges'];segments=mesh.vertices[edge]
    ax.add_collection3d(Line3DCollection(segments,colors='#d33f35',linewidths=1.3));_axes3d(ax,mesh,'A  All 128 malformed three-region edges')
    triples=sorted({tuple(sorted(k)) for k in result['geometry']['angles']})
    angles=result['geometry']['angles'];closure=np.array([2*np.pi-angles[(a,b,c)]-angles[(b,a,c)]-angles[(a,c,b)] for a,b,c in triples])*180/np.pi
    ax=fig.add_subplot(232);ax.hist(closure,bins=40,color='#5d7298');ax.set_yscale('log');ax.set(xlabel='360° − sum of mean chord wedges [degree]',ylabel='Triple lines',title='B  Chord reading retains geometric nonideality')
    gamma_error=np.array([_metrics(v['tensions'],result['tensions'])[0] for v in perturbed]);pressure_error=np.array([_metrics(v['pressures'],result['pressures'])[0] for v in perturbed])
    ax=fig.add_subplot(233);ax.plot(sigmas,gamma_error,'o-',color='#246b98',label='All tensions');ax.plot(sigmas,pressure_error,'s-',color='#b65583',label='All pressures')
    ax.set(xlabel='Coordinate jitter σ [mesh unit]',ylabel='Range-normalized change',title='C  Fresh perturbations of every coordinate');ax.legend(frameon=False)
    ax=fig.add_subplot(234);delta=perturbed[-1]['tensions']-result['tensions'];q=ax.scatter(ic[:,0],ic[:,2],c=delta,norm=_signed(delta),cmap='coolwarm',s=24)
    ax.set_aspect('equal');ax.set(xlabel='x [mesh unit]',ylabel='z [mesh unit]',title=f'D  Tension change at σ={sigmas[-1]:g}');fig.colorbar(q,ax=ax,label='Δγ')
    ax=fig.add_subplot(235)
    for sigma,v in zip(sigmas,perturbed):ax.scatter(result['pressures'][1:],v['pressures'][1:],s=12,alpha=.6,label=f'σ={sigma:g}')
    ax.axline((0,0),slope=1,color='.4',linestyle='--');ax.legend(frameon=False,fontsize=8)
    ax.set(xlabel='Original conditional p',ylabel='Perturbed-geometry p',title='E  Every cell pressure responds')
    ax=fig.add_subplot(236);negative=[np.count_nonzero(v['tensions']<=0) for v in perturbed]
    minimum=[v['tensions'].min() for v in perturbed]
    ax.plot(sigmas,minimum,'o-',color='#bb684c');ax.axhline(0,color='.4',linestyle='--')
    ax.text(.03,.10,'Nonpositive counts: '+', '.join(str(n) for n in negative),transform=ax.transAxes,va='bottom',fontsize=9)
    ax.set(xlabel='Coordinate jitter σ [mesh unit]',ylabel='Minimum relative tension',title='F  Positivity margin; no result is clipped')
    fig.suptitle('Geometry sensitivity is a stress test, not calibrated uncertainty',fontweight='bold')
    arrays={'sigmas':np.array(sigmas),'closure_degrees':closure,'gamma_nrmse':gamma_error,'pressure_nrmse':pressure_error,'nonpositive_tensions':np.array(negative),'minimum_tensions':np.array(minimum),'malformed_edges':edge}
    for i,v in enumerate(perturbed):
        arrays[f'perturbed_{i}_tensions']=v['tensions'];arrays[f'perturbed_{i}_pressures']=v['pressures'];arrays[f'perturbed_{i}_vertices']=v['mesh'].vertices
    return fig,arrays

def plot_scaling(result,scaled,scales,gauged,precision,thresholds):
    style();fig,axes=plt.subplots(2,3,figsize=(15,9),layout='constrained')
    gx=[];px=[]
    for scale,v in zip(scales,scaled):
        gx.append(np.max(abs(v['tensions']-result['tensions'])));px.append(np.max(abs(v['pressures']*scale-result['pressures'])))
    axes[0,0].bar([str(v) for v in scales],np.maximum(gx,1e-18),color='#3386a6');axes[0,0].set_yscale('log')
    axes[0,0].set(xlabel='Coordinate scale factor',ylabel='Maximum |Δγ|',title='A  Tensions are invariant under mesh scaling')
    for scale,v in zip(scales,scaled):axes[0,1].plot(result['cells'][1:],v['pressures'][1:]*scale,'.',label=f's={scale:g}')
    axes[0,1].plot(result['cells'][1:],result['pressures'][1:],color='.3',linewidth=.6,label='Original')
    axes[0,1].legend(frameon=False);axes[0,1].set(xlabel='Observed nonconsecutive cell label',ylabel='s × p(sx) [mean tension / mesh unit]',title='B  Pressures scale as inverse length')
    axes[0,2].scatter(result['tensions'],gauged['tensions']/2,s=12,color='#3a8e7c');axes[0,2].axline((0,0),slope=1,color='.4',linestyle='--')
    axes[0,2].set(xlabel='Mean-one tension',ylabel='Mean-two tension / 2',title='C  Gauge changes rescale the whole atlas')
    labels=[v['label'] for v in precision];x=np.arange(len(labels));width=.36
    for j,key in enumerate(['tensions','pressures']):
        values=np.array([v[key+'_nrmse'] for v in precision]);axes[1,0].bar(x+(j-.5)*width,np.maximum(values,1e-18),width,label=key)
        axes[1,0].axhline(thresholds[key]['nrmse_boundary'],color=['#246b98','#b65583'][j],linestyle='--',linewidth=1)
    axes[1,0].set_yscale('log');axes[1,0].set_xticks(x,labels,rotation=35,ha='right');axes[1,0].legend(frameon=False,fontsize=8)
    axes[1,0].set(ylabel='Range-normalized numerical error',title='D  Fresh arithmetic precision; frozen limits dashed')
    s=np.linalg.svd(result['M'][:-1],compute_uv=False)
    axes[1,1].semilogy(np.arange(1,len(s)+1),s,color='#276b98');axes[1,1].set(xlabel='Singular-value rank',ylabel='Singular value',title='E  Complete junction-system spectrum')
    means=np.array([result['tensions'].mean(),gauged['tensions'].mean()]);medium=np.array([result['pressures'][0],gauged['pressures'][0]])
    axes[1,2].bar(['Mean-one γ','Mean-two γ'],means,color=['#2876a0','#c47c37']);axes[1,2].set(ylabel='Arithmetic mean tension',title='F  Absolute force scale is a convention')
    fig.suptitle('Independent physical invariants and measured numerical reproducibility',fontweight='bold')
    arrays={'scales':np.array(scales),'max_tension_scale_error':np.array(gx),'max_rescaled_pressure_error':np.array(px),'singular_values':s,'gauge_means':means,'medium_pressures':medium,'mean_two_tensions':gauged['tensions'],'mean_two_pressures':gauged['pressures']}
    for i,v in enumerate(scaled):arrays[f'scaled_{i}_tensions']=v['tensions'];arrays[f'scaled_{i}_pressures']=v['pressures']
    return fig,arrays

def export_figure(fig,figure_id,directory,numeric):
    """Export paired PNG/SVG/PDF and full-precision numeric plot sources."""
    directory=Path(directory);directory.mkdir(parents=True,exist_ok=True)
    for suffix in ['png','svg','pdf']:fig.savefig(directory/(figure_id+'.'+suffix),dpi=170,bbox_inches='tight',pad_inches=.15)
    np.savez_compressed(directory/(figure_id+'_numeric.npz'),**numeric)
