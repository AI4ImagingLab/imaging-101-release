import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import least_squares
from .preprocessing import prepare
from .physics_model import YoungLaplaceSolution,ylderiv
from .solvers import solve,jacobian_residual
def review(image,contour,result,upstream):
    curve=result['reconstruction'];p=result['parameters'];ref=upstream['profile']
    f1,ax=plt.subplots(1,2,figsize=(12,5),layout='constrained')
    ax[0].imshow(image,cmap='gray',vmin=0,vmax=255);ax[0].plot(contour[:,0],contour[:,1],'.',ms=2,label='observed boundary')
    ax[0].plot(curve[:,0],curve[:,1],lw=1,label='fresh equilibrium fit');ax[0].legend(fontsize=8);ax[0].set(title='Complete acquired camera frame',xlabel='Image column x (px)',ylabel='Image row y (px)')
    ax[1].plot(contour[:,0],contour[:,1],'k.',ms=3,label='all293observed points')
    ax[1].plot(ref[:,0],ref[:,1],'--',label='native method reference');ax[1].plot(curve[:,0],curve[:,1],lw=1,label='fresh local')
    ax[1].invert_yaxis();ax[1].axis('equal');ax[1].set(title='Entire selected free-drop boundary',xlabel='x (px)',ylabel='y (px)');ax[1].legend(fontsize=8)
    internal=p.copy();internal[[1,4]]*=-1;points=prepare(contour)
    J,res,_=jacobian_residual(points,internal,result['profile_size'])
    f2,ax=plt.subplots(1,3,figsize=(13,3.7),layout='constrained')
    ax[0].scatter(res[:,0],res[:,1],s=9);ax[0].axhline(0,color='k',lw=.5);ax[0].set(xlabel='Signed arclength / apex radius',ylabel='Signed closest error (px)',title='Fresh observed-contour residuals')
    ax[1].hist(res[:,1],bins=25);ax[1].set(xlabel='Signed error (px)',ylabel='Point count',title=f'RMS {np.sqrt(np.mean(res[:,1]**2)):.4f}px')
    ax[2].plot(np.linspace(0,1,401),curve-ref);ax[2].legend(['x','y']);ax[2].set(xlabel='Normalized silhouette position',ylabel='Fresh minus native (px)',title='Cross-NumPy numerical agreement')
    f3,ax=plt.subplots(1,2,figsize=(11,4),layout='constrained')
    ax[0].semilogy(np.arange(len(result['objectives'])),result['objectives'],'o-');ax[0].set(xlabel='Native LM step',ylabel='SSR / (N−4), pixel²',title='Actual full-contour convergence')
    names=['R (px)','B','angle (rad)','L (px)'];fresh=np.r_[p[2:],result['capillary_length']];u=upstream['parameters'];native=np.r_[u[2:],u[2]/np.sqrt(u[3])]
    x=np.arange(4);ax[1].bar(x-.18,native/native,.36,label='native / native');ax[1].bar(x+.18,fresh/native,.36,label='fresh / native');ax[1].set_xticks(x,names);ax[1].set_ylim(.9999,1.0001);ax[1].set(ylabel='Ratio to method reference',title='Computed parameter comparison (not truth)');ax[1].legend(fontsize=8)
    s=np.linspace(.001,result['profile_size'],300);values=YoungLaplaceSolution(p[3],1)(s);der=np.array([ylderiv(v,t,p[3]) for v,t in zip(values,s)])
    f4,ax=plt.subplots(1,2,figsize=(11,4),layout='constrained')
    curv=der[:,2]+np.sin(values[:,2])/values[:,0]
    ax[0].plot(values[:,1],curv,label='sum of principal curvatures ×R');ax[0].plot(values[:,1],2-p[3]*values[:,1],'--',label='2 − B z/R');ax[0].legend(fontsize=8);ax[0].set(xlabel='Height / R',ylabel='Dimensionless curvature',title='Hydrostatic-capillary balance')
    ax[1].plot(s,values[:,3],label='dr/dB');ax[1].plot(s,values[:,4],label='dz/dB');ax[1].legend();ax[1].set(xlabel='Arclength / R',ylabel='Dimensionless sensitivity',title='Shape information varies along the drop')
    return f1,f2,f3,f4
def robustness(contour,result):
    records=[]
    for threshold in [350,450]:
        selected=contour[contour[:,1]>=threshold]
        try:r=solve(selected)
        except RuntimeError as error:r={'error':str(error)}
        records.append((threshold,len(selected),r))
    # Gravity-free diagnostic: geometric least-squares circle from all measured points.
    p=result['parameters'];guess=[p[0],p[1]-p[2],p[2]]
    def residual(v):return np.sqrt(np.sum((contour-v[:2])**2,axis=1))-v[2]
    circle=least_squares(residual,guess,xtol=1e-12,ftol=1e-12,gtol=1e-12).x
    f,ax=plt.subplots(1,3,figsize=(14,4),layout='constrained')
    ax[0].plot(contour[:,0],contour[:,1],'k.',ms=3,label='observed')
    for label,r in [('full',result)]+[(f'y≥{t}px',r) for t,_,r in records if 'error' not in r]:ax[0].plot(r['reconstruction'][:,0],r['reconstruction'][:,1],label=label)
    ax[0].invert_yaxis();ax[0].set_aspect('equal');ax[0].legend(fontsize=8);ax[0].set(xlabel='x (px)',ylabel='y (px)',title='Fresh fits to predefined visible ranges')
    vals=[result['parameters'][3]]+[r['parameters'][3] if 'error' not in r else np.nan for _,_,r in records]
    ax[1].bar(['full']+[f'y≥{t}' for t,_,_ in records],vals);ax[1].set(ylabel='Fitted Bond number',title='Reduced height weakens shape inference')
    for i,(_,_,r) in enumerate(records,1):
        if 'error' in r:ax[1].text(i,.5,'Failed:\nstep limit',transform=ax[1].get_xaxis_transform(),ha='center',fontsize=8)
    theta=np.linspace(0,2*np.pi,401);xy=circle[:2]+circle[2]*np.stack((np.cos(theta),np.sin(theta)),axis=1)
    ax[2].plot(contour[:,0],contour[:,1],'k.',ms=3,label='observed');ax[2].plot(xy[:,0],xy[:,1],label='fresh gravity-free circle');ax[2].plot(result['reconstruction'][:,0],result['reconstruction'][:,1],label='fresh Young–Laplace')
    ax[2].invert_yaxis();ax[2].set_aspect('equal');ax[2].legend(fontsize=8);ax[2].set(xlabel='x (px)',ylabel='y (px)',title=f'Circle residual RMS {np.sqrt(np.mean(residual(circle)**2)):.1f}px')
    return f,[(t,n,r.get('error',r.get('parameters',np.array([])).tolist())) for t,n,r in records],circle
