"""Traction review figures; does not select a Matplotlib backend."""
import numpy as np
import matplotlib.pyplot as plt

def review(displacement,fresh,upstream,infinite):
    """Plot measured vector displacement, all signed components, comparisons."""
    fig,axes=plt.subplots(2,3,figsize=(13,7),layout='constrained')
    arrays=(displacement[0],displacement[1],np.linalg.norm(displacement,axis=0),fresh[0],fresh[1],np.linalg.norm(fresh,axis=0))
    titles=('Measured ux [image pixels]','Measured uy [image pixels]','Measured displacement magnitude [pixels]','Fresh traction tx [Pa]','Fresh traction ty [Pa]','Fresh traction magnitude [Pa]')
    for ax,a,title in zip(axes.ravel(),arrays,titles):
        im=ax.imshow(a,origin='lower',cmap='RdBu_r' if 'magnitude' not in title else 'viridis');ax.set(title=title,xlabel='Grid column',ylabel='Grid row');fig.colorbar(im,ax=ax)
    fig2,axes2=plt.subplots(2,3,figsize=(13,7),layout='constrained')
    limit=max(np.linalg.norm(fresh,axis=0).max(),np.linalg.norm(upstream,axis=0).max(),np.linalg.norm(infinite,axis=0).max())
    for ax,a,title in zip(axes2[0],(upstream,fresh,infinite),('Independent upstream finite','Fresh local finite','Fresh local half-space comparison')):
        im=ax.imshow(np.linalg.norm(a,axis=0),origin='lower',vmin=0,vmax=limit,cmap='magma');ax.set(title=title,xlabel='Grid column',ylabel='Grid row');fig2.colorbar(im,ax=ax,label='Pa')
    for ax,a,title in zip(axes2[1,:2],(fresh-upstream,fresh-infinite),('Fresh minus upstream','Finite minus half-space')):
        im=ax.imshow(np.linalg.norm(a,axis=0),origin='lower',cmap='viridis');ax.set(title=title);fig2.colorbar(im,ax=ax,label='Vector difference [Pa]')
    yy,xx=np.indices(fresh.shape[1:]);axes2[1,2].quiver(xx[::2,::2],yy[::2,::2],fresh[0,::2,::2],fresh[1,::2,::2]);axes2[1,2].set(title='Fresh vector traction direction',xlabel='Grid column',ylabel='Grid row',aspect='equal')
    return fig,fig2
