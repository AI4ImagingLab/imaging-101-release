"""Actual uniform/bilateral reconstructions and numerical sensitivity plots."""
import time
import numpy as np
import matplotlib.pyplot as plt
from .solvers import integrate
from .physics_model import system


def quality(height,reference):
    """Same defined NCC/NRMSE as Judge, used for explicit method comparisons."""
    x=height.ravel();r=reference.ravel()
    norm=np.linalg.norm(r);span=np.ptp(r);xn=np.linalg.norm(x)
    if norm<=0 or span<=0:raise ValueError('Degenerate reference')
    return float(x@r/(xn*norm)) if xn>0 else 0.,float(np.sqrt(np.mean((x-r)**2))/span)


def experiments(data,fresh):
    variants={};rows=[];sensitivity=[]
    for name,k0 in [('vase',4.),('tent',1.)]:
        normal,mask=data[name];reference=fresh[0]['reconstruction' if name=='vase' else 'tent_reconstruction'][0]
        for k in [0.,1.,2.,4.,8.]:
            start=time.perf_counter()
            if k==k0:z=reference;diag=fresh[1][name]
            else:z,diag=integrate(normal,mask,k=k)
            variants[name+'_'+str(k)]=z
            ncc,nrmse=quality(z,reference);rows.append([0 if name=='vase' else 1,k,ncc,nrmse,diag['weighted_residual_rms'],diag['iterations'],time.perf_counter()-start])
        for tol in [1e-2,1e-3,1e-5]:
            start=time.perf_counter();z,diag=integrate(normal,mask,k=k0,cg_tol=tol)
            ncc,nrmse=quality(z,reference);sensitivity.append([0 if name=='vase' else 1,tol,ncc,nrmse,diag['weighted_residual_rms'],diag['iterations'],time.perf_counter()-start])
    return variants,np.asarray(rows),np.asarray(sensitivity)


def figures(data,fresh,upstream,variants,rows,sensitivity):
    plt.rcParams.update({'font.size':8,'axes.titlesize':9,'axes.labelsize':8})
    out,diagnostics=fresh;result={}
    fig,axes=plt.subplots(2,5,figsize=(17,7),layout='constrained')
    for row,(name,key) in enumerate([('vase','reconstruction'),('tent','tent_reconstruction')]):
        normal,mask=data[name];uniform=variants[name+'_0.0'];z=out[key][0]
        for col,(a,title,cmap) in enumerate([(normal*.5+.5,'Quantized normals',None),(uniform,'Uniform (k=0)','viridis'),(z,'Fresh bilateral','viridis'),(diagnostics[name]['horizontal_weights'],'Horizontal weights','coolwarm'),(diagnostics[name]['vertical_weights'],'Vertical weights','coolwarm')]):
            ax=axes[row,col]
            if col in [1,2]:im=ax.imshow(a,cmap=cmap,vmin=min(uniform.min(),z.min()),vmax=max(uniform.max(),z.max()))
            elif col>=3:im=ax.imshow(a,cmap=cmap,vmin=0,vmax=1)
            else:im=ax.imshow(a)
            ax.set(title=name.title()+'\n'+title,xlabel='Column (pixel)',ylabel='Row (pixel)')
            if col:fig.colorbar(im,ax=ax,shrink=.75,label='Weight' if col>=3 else 'Zero-mean height (pixel units)')
    result['field_comparison']=fig
    fig=plt.figure(figsize=(13,7),layout='constrained')
    for row,(name,key) in enumerate([('vase','reconstruction'),('tent','tent_reconstruction')]):
        normal,mask=data[name];z=out[key][0];rr,cc=np.indices(mask.shape)
        for col,(h,title) in enumerate([(variants[name+'_0.0'],'Uniform'),(z,'Bilateral')]):
            ax=fig.add_subplot(2,3,row*3+col+1,projection='3d');ax.plot_surface(cc,rr,h,cmap='viridis',rstride=2,cstride=2,linewidth=0,antialiased=True)
            ax.set(xlabel='Column (pixel)',ylabel='Row (pixel)',zlabel='Height (pixel)',title=name.title()+' · '+title);ax.view_init(elev=35,azim=-60)
        ax=fig.add_subplot(2,3,row*3+3);r=mask.shape[0]//2
        ax.plot(cc[r],variants[name+'_0.0'][r],label='Uniform');ax.plot(cc[r],z[r],label='Fresh bilateral');ax.plot(cc[r],upstream[name+'_depth'][r],'--',label='Original bilateral')
        ax.set(xlabel='Column (pixel)',ylabel='Zero-mean height (pixel)',title='Center-row discontinuity profile');ax.legend(fontsize=8)
    result['surfaces_profiles']=fig
    fig,axes=plt.subplots(2,3,figsize=(13,7),layout='constrained')
    for row,(name,key) in enumerate([('vase','reconstruction'),('tent','tent_reconstruction')]):
        z=out[key][0];diag=diagnostics[name];energy=diag['energy']
        axes[row,0].plot(np.arange(1,len(energy)+1),energy,label='Fresh');old=upstream[name+'_energy'];axes[row,0].plot(np.arange(1,len(old)+1),old,'--',label='Original');axes[row,0].set(xlabel='IRLS iteration',ylabel='Weighted squared normal residual',title=name.title()+' source energy');axes[row,0].legend()
        delta=z-upstream[name+'_depth'];bound=max(float(np.max(np.abs(delta))),1e-13)
        im=axes[row,1].imshow(delta,cmap='RdBu_r',vmin=-bound,vmax=bound);fig.colorbar(im,ax=axes[row,1],label='Fresh − original height (pixel)');axes[row,1].set(title='Numerical parity',xlabel='Column (pixel)',ylabel='Row (pixel)')
        a,b,_=system(*data[name]);res=(a@z[data[name][1]]-b).reshape(4,-1)
        for i,label in enumerate(['Up','Down','Right','Left']):axes[row,2].hist(res[i],bins=60,histtype='step',label=label,log=True)
        axes[row,2].set(xlabel='Signed normal constraint residual',ylabel='Constraint count',title='All constraints, including boundary rows');axes[row,2].legend(fontsize=8)
    result['numerics_physics']=fig
    fig,axes=plt.subplots(2,2,figsize=(10,7),layout='constrained')
    for i,name in enumerate(['vase','tent']):
        a=rows[rows[:,0]==i];b=sensitivity[sensitivity[:,0]==i]
        axes[0,0].plot(a[:,1],a[:,3],'o-',label=name);axes[0,1].plot(a[:,1],a[:,4],'o-',label=name)
        axes[1,0].semilogx(b[:,1],b[:,3],'o-',label=name);axes[1,1].semilogx(b[:,1],b[:,5],'o-',label=name)
    for ax in axes.ravel():ax.legend()
    axes[0,0].set(xlabel='Bilateral sharpness k',ylabel='NRMSE vs selected method',title='Method variation, not truth error')
    axes[0,1].set(xlabel='Bilateral sharpness k',ylabel='Weighted normal residual RMS')
    axes[1,0].set(xlabel='Inner CG relative tolerance',ylabel='NRMSE vs canonical CG tolerance')
    axes[1,1].set(xlabel='Inner CG relative tolerance',ylabel='Outer IRLS iterations')
    result['sensitivity']=fig
    return result
