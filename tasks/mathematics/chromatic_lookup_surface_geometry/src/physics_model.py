"""Per-ray empirical radiometric and depth acquisition calibration."""
import numpy as np

def calibration_rays(raw,rows,rank=25):
    """Return RGB dictionary (N,749,3), axial depths (N,749), float32.

    Rows are C-order flattened indices of the fixed650x650 sensor ROI. Each
    dictionary is explicitly evaluated as its supplied rank25 matrix product.
    Smaller ranks are maintainer compression controls, not the benchmark model.
    """
    rows=np.asarray(rows,dtype=np.int64)
    if rows.ndim!=1 or np.any(rows<0) or np.any(rows>=650*650):raise ValueError('Invalid calibration pixel indices')
    if not isinstance(rank,int) or not 1<=rank<=25:raise ValueError('Rank must be integer1..25')
    color=np.empty((len(rows),749,3),dtype=np.float32)
    left=raw['left_color'][0].reshape(3,-1,25);right=raw['right_color'][0]
    for c in range(3):color[:,:,c]=left[c,rows,:rank]@right[c,:rank]
    depth=raw['left_depth'][0].reshape(-1,25)[rows,:rank]@raw['right_depth'][0,:rank]
    return color,depth

def project_depth_to_points(depth,intrinsic,distortion,roi=(650,200,1300,850)):
    """Camera XYZ in mm from axial Z; five Brown-Conrady inverse iterations.

    This visualization helper uses calibrated intrinsics, not target geometry.
    Axes are right/down/forward; invalid sentinel points are left in the array
    for the caller to select using the fixed observation support.
    """
    d=np.asarray(depth);K=np.asarray(intrinsic,dtype=float);k1,k2,p1,p2,k3=np.asarray(distortion,dtype=float).ravel()
    yy,xx=np.indices(d.shape);xd=(xx+roi[0]-K[0,2])/K[0,0];yd=(yy+roi[1]-K[1,2])/K[1,1]
    x=xd.copy();y=yd.copy()
    for _ in range(5):
        r=x*x+y*y;radial=1+r*(k1+r*(k2+r*k3))
        dx=2*p1*x*y+p2*(r+2*x*x);dy=p1*(r+2*y*y)+2*p2*x*y
        x=(xd-dx)/radial;y=(yd-dy)/radial
    return np.stack((x*d,y*d,d),axis=-1)
