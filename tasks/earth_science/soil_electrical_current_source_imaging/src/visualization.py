import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import lsq_linear
from .physics_model import weighted_system,regularizer
from .solvers import reconstruct
COLORS=['#246a91','#cc643c','#5d8d58']
def image(ax,a,title,unit='fraction',vmin=None,vmax=None,cmap='viridis'):
    im=ax.imshow(a.T,origin='lower',extent=[.01,.52,.005,.515],cmap=cmap,vmin=vmin,vmax=vmax,aspect='equal');ax.set(xlabel='Source-plane x / m',ylabel='Source-plane y / m',title=title);plt.colorbar(im,ax=ax,label=unit,shrink=.85)
def acquisition(d):
    fig,ax=plt.subplots(1,3,figsize=(15,4.5),layout='constrained');i=np.arange(len(d['observations']));ax[0].plot(i,d['observations'],color=COLORS[0]);ax[0].set(xlabel='Electrode measurement index',ylabel='Transfer resistance / Ω',title='Original author simulated observations')
    g=d['green_matrix'];im=ax[1].imshow(g,aspect='auto',cmap='RdBu_r');plt.colorbar(im,ax=ax[1],label='Ω / source fraction');ax[1].set(xlabel='Spatial source index',ylabel='Measurement index',title='Independent known-medium response')
    coord=d['source_coordinates'];ax[2].scatter(coord[:,0],coord[:,1],c=np.arange(289),s=12,cmap='viridis');ax[2].set(xlabel='Source-plane x / m',ylabel='Source-plane y / m',title='289 candidate current-release nodes',aspect='equal')
    return fig,{'observations':d['observations'],'green_matrix':g,'source_coordinates':coord}
def graph(d):
    R=regularizer(d['source_coordinates']);coord=d['source_coordinates'];fig,ax=plt.subplots(1,3,figsize=(15,4.6),layout='constrained')
    for r in R:
        edge=np.flatnonzero(r);ax[0].plot(coord[edge,0],coord[edge,1],color=COLORS[0],lw=.8)
    for ix in range(16):ax[0].plot([coord[ix*17+16,0],coord[(ix+1)*17+16,0]],[.5,.5],color=COLORS[1],lw=2,ls='--')
    ax[0].set(xlabel='x / m',ylabel='y / m',title='Exact source graph; dashed edges omitted',aspect='equal')
    eig=np.linalg.eigvalsh(R.T@R);ax[1].plot(np.arange(289),eig,color=COLORS[0]);ax[1].set(xlabel='Eigenmode index',ylabel='Graph Laplacian eigenvalue',title='Connected graph: one constant null mode')
    s=np.linalg.svd(d['green_matrix'],compute_uv=False);ax[2].semilogy(np.arange(1,len(s)+1),s,color=COLORS[2]);ax[2].set(xlabel='Singular-value index',ylabel='Response singular value / Ω',title='Broad conductive response is ill-conditioned')
    return fig,{'regularizer':R,'graph_eigenvalues':eig,'response_singular_values':s}
def components(m,a):
    fig,ax=plt.subplots(1,3,figsize=(15,4.7),layout='constrained');hi=max(m.max(),a.max());image(ax[0],a,'Original bounded iCSD',vmin=0,vmax=hi);image(ax[1],m,'Fresh local reconstruction',vmin=0,vmax=hi);image(ax[2],m-a,'Fresh − original',cmap='RdBu_r');fig.suptitle('Spatial current release • method reference, no physical grid truth',fontweight='bold')
    return fig,{'fresh':m,'original':a,'difference':m-a}
def residuals(d,p,m):
    A,b,R,w=weighted_system(d);r=p-d['observations'];fig,ax=plt.subplots(1,3,figsize=(15,4.5),layout='constrained');i=np.arange(len(p));ax[0].plot(i,d['observations'],label='Original observations',color=COLORS[0]);ax[0].plot(i,p,label='Fresh predicted',ls='--',color=COLORS[1]);ax[0].legend();ax[0].set(xlabel='Measurement index',ylabel='Transfer resistance / Ω',title='All204 measurements retained')
    ax[1].plot(i,w*r,color=COLORS[2]);ax[1].axhline(0,color='black',lw=.6);ax[1].set(xlabel='Measurement index',ylabel='Relative weighted residual',title=f'Weighted RMS {np.sqrt(np.mean((w*r)**2)):.3f}')
    ax[2].scatter(d['observations'],p,s=12,alpha=.6,color=COLORS[0]);lo=min(p.min(),d['observations'].min());hi=max(p.max(),d['observations'].max());ax[2].plot([lo,hi],[lo,hi],color='black',ls='--');ax[2].set(xlabel='Original simulated / Ω',ylabel='Predicted / Ω',title=f'Current sum={m.sum():.7f}')
    return fig,{'prediction':p,'residual':r,'weighted_residual':w*r}
def comparisons(d,m):
    default,pdefault,rdefault=reconstruct(d,conservation_weight=1e5);A,b,R,w=weighted_system(d)
    # Explicit altered complete-graph ablation; preserve selected graph above.
    edges=[]
    for i in range(17):
        for j in range(17):
            for di,dj in [(1,0),(0,1)]:
                if i+di<17 and j+dj<17:
                    row=np.zeros(289);row[17*i+j]=-1;row[17*(i+di)+j+dj]=1;edges.append(row)
    full=np.array(edges);Af=np.vstack([w[:,None]*d['green_matrix'],1000*np.ones((1,289)),full]);bf=np.r_[w*d['observations'],1000,np.zeros(len(full))];rr=lsq_linear(Af,bf,bounds=(0,1));altered=rr.x.reshape(17,17)
    fig,ax=plt.subplots(1,3,figsize=(15,4.7),layout='constrained');hi=max(m.max(),default.max(),altered.max());image(ax[0],m,'Selected: conservation weight1000',vmin=0,vmax=hi);image(ax[1],default,'Original default: weight100000',vmin=0,vmax=hi);image(ax[2],altered,'Altered full-edge graph ablation',vmin=0,vmax=hi);fig.suptitle(f'Numerical conditioning • default optimality {rdefault.optimality:.2g}, selected graph preserved',fontweight='bold')
    return fig,{'selected':m,'original_default':default,'complete_graph_ablation':altered,'default_optimality':np.array(rdefault.optimality)}
def sensitivity(d):
    weights=np.geomspace(.1,1000,9);maps=[];res=[];rough=[];sums=[]
    R=regularizer(d['source_coordinates'])
    for wr in weights:
        m,p,r=reconstruct(d,regularization_weight=wr);maps.append(m);res.append(np.linalg.norm(p-d['observations']));rough.append(np.linalg.norm(R@m.ravel()));sums.append(m.sum())
    maps=np.array(maps);fig,ax=plt.subplots(2,3,figsize=(15,9),layout='constrained');hi=maps.max()
    for a,idx in zip(ax[0],[0,4,8]):image(a,maps[idx],f'Fresh wr={weights[idx]:g}',vmin=0,vmax=hi)
    ax[1,0].loglog(weights,res,marker='o',color=COLORS[0]);ax[1,0].set(xlabel='Graph penalty weight',ylabel='Observation residual L2 / Ω',title='Data fit versus smoothing')
    ax[1,1].loglog(res,rough,marker='o',color=COLORS[1]);ax[1,1].set(xlabel='Observation residual L2 / Ω',ylabel='Graph roughness L2',title='Actual regularization tradeoff')
    ax[1,2].semilogx(weights,sums,marker='o',color=COLORS[2]);ax[1,2].axhline(1,ls='--',color='black');ax[1,2].set(xlabel='Graph penalty weight',ylabel='Total source-current fraction',title='Soft conservation retained')
    return fig,{'weights':weights,'maps':maps,'residual_l2':np.array(res),'roughness':np.array(rough),'current_sums':np.array(sums)}
