"""Explicit published differentiable opacity acquisition model.

Adapted from ShapeRecDeepPrior, MIT LICENSE. This preserves the author's
half-voxel coordinate convention and zero weights at exact integer knots.
The model is an approximation to geometric silhouettes, not a calibrated
Beer-Lambert experiment. All coordinates and densities are dimensionless.
"""
import torch

def rotation_matrix(angles):
    """Return author row-coordinate rotation Rz@Ry@Rx, angles[3] radians.

    Rotation geometry is evaluated on the angle tensor's device (CPU in the
    selected native training case) before coordinates transfer to the volume.
    """
    x,y,z=angles
    cx,sx=torch.cos(x),torch.sin(x);cy,sy=torch.cos(y),torch.sin(y);cz,sz=torch.cos(z),torch.sin(z)
    X=torch.eye(3,dtype=angles.dtype,device=angles.device);Y=X.clone();Z=X.clone()
    X[1,1]=cx;X[1,2]=sx;X[2,1]=-sx;X[2,2]=cx
    Y[0,0]=cy;Y[0,2]=-sy;Y[2,0]=sy;Y[2,2]=cy
    Z[0,0]=cz;Z[0,1]=sz;Z[1,0]=-sz;Z[1,1]=cz
    return Z.mm(Y).mm(X)

def normalized_grid(size,dtype=torch.float32,device='cpu'):
    """Return [size**3,3] inclusive [-1,1] grid, C order x,y,z."""
    axis=torch.linspace(-1.,1.,steps=size,dtype=dtype,device=device)
    x,y,z=torch.meshgrid(axis,axis,axis,indexing='ij')
    return torch.stack((x.reshape(-1),y.reshape(-1),z.reshape(-1)),dim=1)

def resample_volume(volume,coordinates):
    """Explicit eight-corner source interpolation; output same cubic shape.

    Clamp to [0,N-1], then independently floor/ceil. If any coordinate is an
    exact integer, floor==ceil and all associated weights vanish. Preserve
    this published-code artifact, including zero clamped boundary samples.
    """
    q=torch.clamp(coordinates,min=0,max=volume.shape[0]-1).to(volume.device)
    x,y,z=q[:,0],q[:,1],q[:,2]
    fx,fy,fz=torch.floor(x),torch.floor(y),torch.floor(z)
    cx,cy,cz=torch.ceil(x),torch.ceil(y),torch.ceil(z)
    ix,iy,iz=fx.long(),fy.long(),fz.long();jx,jy,jz=cx.long(),cy.long(),cz.long()
    values=(torch.abs((x-fx)*(y-fy)*(z-fz))*volume[jx,jy,jz]
           +torch.abs((x-fx)*(y-fy)*(z-cz))*volume[jx,jy,iz]
           +torch.abs((x-fx)*(y-cy)*(z-fz))*volume[jx,iy,jz]
           +torch.abs((x-fx)*(y-cy)*(z-cz))*volume[jx,iy,iz]
           +torch.abs((x-cx)*(y-fy)*(z-fz))*volume[ix,jy,jz]
           +torch.abs((x-cx)*(y-fy)*(z-cz))*volume[ix,jy,iz]
           +torch.abs((x-cx)*(y-cy)*(z-fz))*volume[ix,iy,jz]
           +torch.abs((x-cx)*(y-cy)*(z-cz))*volume[ix,iy,iz])
    return values.reshape(volume.shape)

def rotate_volume(volume,angles):
    """Row-grid rotation and exact source mapping (q+1)*(N-.5)/2."""
    grid=normalized_grid(volume.shape[0],dtype=angles.dtype,device=angles.device)
    q=grid.mm(rotation_matrix(angles));q=(q+1.)*float(volume.shape[0]-.5)/2
    return resample_volume(volume,q)

def binary_projection(volume,tau=.5):
    """Sum along y and map optical thickness to opacity, [x,z]."""
    return 1.-torch.exp(-tau*volume.sum(1))

def project(volume,views):
    """Return opacity[x,z,view] using explicit resampling for each view."""
    return torch.cat([binary_projection(rotate_volume(volume,a)).unsqueeze(2) for a in views],dim=2)

def objective(projections,observations,views,prior,view_weight=.1):
    """Mean absolute opacity error minus mean cosine view-prior agreement."""
    data=torch.nn.functional.l1_loss(projections,observations)
    return data-view_weight*torch.cos(views-prior).mean().to(data.device),data


def inverse_rotate_volume(volume,angles):
    """Apply inverse rotation with the same native interpolation convention."""
    grid=normalized_grid(volume.shape[0],dtype=angles.dtype,device=angles.device)
    q=grid.mm(rotation_matrix(angles).T);q=(q+1.)*float(volume.shape[0]-.5)/2
    return resample_volume(volume,q)


def carve(silhouettes,views,device='cuda'):
    """Original sequential space carving, including resampling each view.

    silhouette[x,z,view]<.01 removes every voxel on its y ray. Unlike a
    simultaneous geometric visual hull, each intermediate volume is rotated
    and interpolated back; preserve this selected source comparison.
    """
    images=torch.as_tensor(silhouettes,dtype=torch.float32,device=device)
    angles=torch.as_tensor(views,dtype=torch.float32,device='cpu')
    with torch.no_grad():
        volume=torch.ones(images.shape[0],images.shape[0],images.shape[0],device=device)
        for i,a in enumerate(angles):
            volume=rotate_volume(volume,a)
            rays=torch.nonzero(images[:,:,i]<.01,as_tuple=False)
            volume[rays[:,0],:,rays[:,1]]=0.
            volume=inverse_rotate_volume(volume,a)
    return volume.cpu().numpy()
