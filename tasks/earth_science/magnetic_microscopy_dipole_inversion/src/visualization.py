"""Review plots; backend remains under caller control."""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from .physics_model import forward

def angles(moments):
    """Return inclination down, declination from +y toward +x, degrees."""
    x,y,z=moments.T
    return np.degrees(np.arctan2(-z,np.hypot(x,y))),np.degrees(np.arctan2(x,y))

def review(field,x,y,result,reference):
    """Create acquisition, fit, upstream comparison and quality-filter plots."""
    p,m,b,s,w,c,prep,det=result
    f,dx,dy,dz,tga,stretch=prep;xx,yy=np.meshgrid(x,y)
    predicted=forward(xx,yy,10.,p,m)
    fig,axes=plt.subplots(2,3,figsize=(14,8),layout='constrained')
    maps=(field,f,tga,stretch,predicted,f-predicted)
    titles=('Measured Bz at 5 um','Continued Bz at 10 um','Total gradient amplitude','Contrast-stretched gradient','All recovered dipoles','Observation minus model')
    for index,(ax,a,title) in enumerate(zip(axes.ravel(),maps,titles)):
        options=dict(vmin=-10000,vmax=10000) if index in (0,1,4,5) else {}
        im=ax.imshow(a,origin='lower',extent=(x[0],x[-1],y[0],y[-1]),cmap='RdBu_r' if a is not tga and a is not stretch else 'viridis',**options)
        ax.set(title=title,xlabel='x [um]',ylabel='y [um]');fig.colorbar(im,ax=ax,shrink=.7)
    for window in det:
        axes[1,0].add_patch(Rectangle((window[0],window[2]),window[1]-window[0],window[3]-window[2],fill=False,edgecolor='white',linewidth=.4))
    axes[1,1].scatter(p[:,0],p[:,1],s=5,facecolors='none',edgecolors='k',linewidths=.5)
    fig2,axes2=plt.subplots(2,3,figsize=(14,8),layout='constrained')
    inc,dec=angles(m);ri,rd=angles(reference['moments'][0]);good=(s[:,0]>=.85)&(s[:,1]>5)
    for ax,mask,title in zip(axes2[0,:2],(np.ones(len(p),bool),good),('All 75 recovered sources','R2 >= 0.85 and SNR > 5 dB')):
        scatter=ax.scatter(dec[mask],inc[mask],c=np.linalg.norm(m[mask],axis=1),cmap='plasma')
        ax.set(xlabel='Declination [degrees]',ylabel='Inclination [degrees]',title=title);fig2.colorbar(scatter,ax=ax,label='Moment [A m2]')
    axes2[0,2].scatter(rd,ri,facecolors='none',edgecolors='k',label='Independent upstream')
    axes2[0,2].scatter(dec,inc,s=8,label='Fresh cleaned');axes2[0,2].legend();axes2[0,2].set(xlabel='Declination [degrees]',ylabel='Inclination [degrees]',title='Direction comparison')
    axes2[1,0].hist(s[:,0],bins=20);axes2[1,0].axvline(.85,color='r');axes2[1,0].set(xlabel='R2',ylabel='Source count')
    axes2[1,1].hist(s[:,1],bins=20);axes2[1,1].axvline(5,color='r');axes2[1,1].set(xlabel='SNR [dB]',ylabel='Source count')
    scatter=axes2[1,2].scatter(p[:,0],p[:,1],c=p[:,2]);axes2[1,2].set(xlabel='x [um]',ylabel='y [um]',title='Recovered source depth');fig2.colorbar(scatter,ax=axes2[1,2],label='z [um]')
    return fig,fig2
