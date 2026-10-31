"""Review-only density/rotation metrics and physical display helpers."""
import numpy as np
import matplotlib.pyplot as plt

def rotation_matrices(angles):
    out=[]
    for x,y,z in angles:
        cx,sx=np.cos(x),np.sin(x);cy,sy=np.cos(y),np.sin(y);cz,sz=np.cos(z),np.sin(z)
        out.append(np.array([[cz,sz,0],[-sz,cz,0],[0,0,1]])@np.array([[cy,0,-sy],[0,1,0],[sy,0,cy]])@np.array([[1,0,0],[0,cx,sx],[0,-sx,cx]]))
    return np.array(out)

def density_metrics(x,y):
    a=x.astype(float).ravel();b=y.astype(float).ravel();n=np.linalg.norm(a)
    return {'ncc':float(np.clip(a@b/(n*np.linalg.norm(b)),-1,1)) if n else 0.,'nrmse':float(np.sqrt(np.mean((a-b)**2))/np.ptp(b)),'iou':float(np.count_nonzero((a>.5)&(b>.5))/np.count_nonzero((a>.5)|(b>.5)))}

def pose_errors(estimate,reference):
    a=rotation_matrices(estimate);b=rotation_matrices(reference)
    return np.arccos(np.clip((np.einsum('vij,vij->v',a,b)-1)/2,-1,1))

def surface_points(volume,threshold=.5,max_points=12000):
    """Boundary-voxel centers, thinned ONLY for 3D display, never metrics."""
    occupied=volume>threshold;interior=occupied.copy()
    for axis in range(3):interior&=np.roll(occupied,1,axis)&np.roll(occupied,-1,axis)
    points=np.argwhere(occupied&~interior)
    return points[::max(1,int(np.ceil(len(points)/max_points)))]/(volume.shape[0]-1)*2-1

def plot_surface(ax,volume,title):
    points=surface_points(volume)
    ax.scatter(*points.T,s=.5,c=points[:,2],cmap='viridis',rasterized=True)
    ax.set(title=title,xlabel='x (normalized)',ylabel='y (normalized)',zlabel='z (normalized)',xlim=(-1,1),ylim=(-1,1),zlim=(-1,1));ax.set_box_aspect((1,1,1));ax.view_init(20,-55)
