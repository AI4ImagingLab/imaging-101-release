import numpy as np

def wrap(angle):
    """Original g2o principal angle in (-pi,pi]."""
    result=np.fmod(angle+np.pi,2*np.pi)
    return np.where(result<=0,result+np.pi,result-np.pi)

def residual_jacobians(poses,data):
    """Canonical z^-1 Ti^-1 Tj error and world-additive Jacobians, all edges."""
    ids=data['edge_indices'];z=data['relative_poses'];a=poses[ids[:,0]];b=poses[ids[:,1]]
    dx=b[:,0]-a[:,0];dy=b[:,1]-a[:,1];c=np.cos(a[:,2]);s=np.sin(a[:,2]);cz=np.cos(z[:,2]);sz=np.sin(z[:,2])
    qx=c*dx+s*dy;qy=-s*dx+c*dy;ux=qx-z[:,0];uy=qy-z[:,1]
    e=np.column_stack((cz*ux+sz*uy,-sz*ux+cz*uy,wrap(b[:,2]-a[:,2]-z[:,2])))
    ja=np.zeros((len(z),3,3));jb=ja.copy();cc=np.cos(a[:,2]+z[:,2]);ss=np.sin(a[:,2]+z[:,2])
    ja[:,0,0]=-cc;ja[:,0,1]=-ss;ja[:,1,0]=ss;ja[:,1,1]=-cc
    ja[:,0,2]=cz*qy-sz*qx;ja[:,1,2]=-sz*qy-cz*qx;ja[:,2,2]=-1
    jb[:,:2,:2]=-ja[:,:2,:2];jb[:,2,2]=1
    return e,ja,jb

def weighted_error(poses,data):
    e=residual_jacobians(poses,data)[0]
    return float(np.einsum('ni,nij,nj->',e,data['information'],e))
