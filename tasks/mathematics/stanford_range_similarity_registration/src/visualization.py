"""Measured-scan comparisons; all fitting uses local mathematical operators."""
import time
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import cKDTree
from .solvers import reconstruct


def nearest_distances(target,points):
    """Full-source distance to measured target samples, metres."""
    return cKDTree(target).query(points,workers=1)[0]


def icp(data,iterations=100):
    """Explicit point-to-point rigid ICP comparison, identity initialization.

    Unlike CPD this comparison fixes scale=1 and uses hard nearest neighbors.
    Fit every40th native vertex, return all transformed source observations.
    """
    x=data['target_points'][::40];y=data['source_points'][::40];tree=cKDTree(x)
    r=np.eye(3);t=np.zeros(3);previous=np.inf;history=[]
    for _ in range(iterations):
        z=y@r+t;distance,index=tree.query(z);match=x[index]
        ym=y.mean(0);xm=match.mean(0)
        u,_,vt=np.linalg.svd((y-ym).T@(match-xm))
        correction=np.eye(3);correction[-1,-1]=np.linalg.det(u@vt)
        r=u@correction@vt;t=xm-ym@r
        cost=float(np.mean((y@r+t-match)**2));history.append(cost)
        if abs(previous-cost)<1e-12:break
        previous=cost
    return data['source_points']@r+t,np.asarray(history)


def experiments(data,fresh):
    """Actually refit outlier and resolution sweeps, plus explicit ICP baseline."""
    rows=[]
    for kind,values in [('outlier_fraction',[0.,.05,.1,.2,.3]),('fit_stride',[20,40,80,160])]:
        for value in values:
            start=time.perf_counter()
            if (kind=='outlier_fraction' and value==.1) or (kind=='fit_stride' and value==40):
                out,diag=fresh
            else:
                kw={'outlier_fraction':value} if kind=='outlier_fraction' else {'stride':value}
                out,diag=reconstruct(data,**kw)
            d=nearest_distances(data['target_points'],out['reconstruction'][0])
            rows.append([0 if kind=='outlier_fraction' else 1,value,np.sqrt(np.mean(d*d)),np.median(d),diag['iterations'],out['scale'].item(),time.perf_counter()-start])
    aligned,history=icp(data)
    return np.asarray(rows),aligned,history


def figures(data,fresh,upstream,sweep,icp_points,icp_history):
    """Four report figures; same native samples, scale and metric definitions."""
    out,diag=fresh;x=data['target_points'];y=data['source_points'];z=out['reconstruction'][0]
    distances=[nearest_distances(x,a) for a in [y,icp_points,z]]
    result={}
    fig,axes=plt.subplots(2,3,figsize=(13,8),layout='constrained')
    for col,(points,title) in enumerate([(y,'Before registration'),(icp_points,'Rigid nearest-neighbor ICP'),(z,'Similarity CPD (fresh)')]):
        for row,(a,b,label) in enumerate([(0,1,'x–y'),(0,2,'x–z')]):
            ax=axes[row,col];ax.scatter(x[::8,a]*1000,x[::8,b]*1000,s=.5,color='#B8B8B8',label='Target laser scan')
            ax.scatter(points[::8,a]*1000,points[::8,b]*1000,s=.5,color='#147D92',alpha=.65,label='Source laser scan')
            ax.set(xlabel='x (mm)',ylabel=('y' if b==1 else 'z')+' (mm)',title=title+' · '+label,aspect='equal')
            if row==0 and col==0:ax.legend(markerscale=6,fontsize=8)
    result['registration']=fig
    fig,axes=plt.subplots(1,3,figsize=(14,4),layout='constrained')
    for i,(d,label) in enumerate(zip(distances,['Unregistered','Rigid ICP','Similarity CPD'])):
        q=np.sort(d)*1000;axes[0].plot(q,np.arange(1,len(q)+1)/len(q),label=label)
    axes[0].set(xlabel='Nearest measured target distance (mm)',ylabel='Fraction of all source samples',xlim=(0,12));axes[0].legend(fontsize=8)
    im=axes[1].scatter(z[::4,0]*1000,z[::4,1]*1000,c=distances[2][::4]*1000,s=2,cmap='magma',vmin=0,vmax=8)
    axes[1].set(xlabel='x (mm)',ylabel='y (mm)',title='CPD residual / partial overlap',aspect='equal');fig.colorbar(im,ax=axes[1],label='Nearest-sample distance (mm)')
    axes[2].bar(['Before','ICP','CPD'],[np.sqrt(np.mean(d*d))*1000 for d in distances],color=['#999999','#CA8A30','#147D92'])
    axes[2].set(ylabel='All-source nearest-target RMS (mm)',title='Geometry agreement, not pose truth')
    result['measured_consistency']=fig
    fig,axes=plt.subplots(1,3,figsize=(14,4),layout='constrained');h=diag['history']
    axes[0].plot(h[:,0],h[:,1],label='Fresh local');axes[0].plot(upstream['output_history'][:,0],upstream['output_history'][:,1],'--',label='Independent upstream')
    axes[0].set(xlabel='EM iteration',ylabel='EM surrogate Q (dimensionless)',title='Source convention: previous variance');axes[0].legend(fontsize=8)
    axes[1].semilogy(h[:,0],h[:,2]);axes[1].set(xlabel='EM iteration',ylabel='Normalized isotropic variance')
    axes[2].plot(h[:,0],h[:,3]);axes[2].axhline(1,color='gray',ls=':');axes[2].set(xlabel='EM iteration',ylabel='Isotropic scale',title='Similarity, not fixed-scale registration')
    result['convergence']=fig
    fig,axes=plt.subplots(1,2,figsize=(10,4),layout='constrained')
    for k,ax,title in [(0,axes[0],'Uniform outlier mixture weight'),(1,axes[1],'Deterministic native fit stride')]:
        a=sweep[sweep[:,0]==k];ax.plot(a[:,1],a[:,2]*1000,'o-',label='RMS');ax.plot(a[:,1],a[:,3]*1000,'s--',label='Median');ax.set(xlabel=title,ylabel='All-source distance (mm)');ax.legend()
    result['sensitivity']=fig
    return result
