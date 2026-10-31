"""Quartic straight-line covariance determinant and radial acquisition geometry.

Adapted from AMI Research Group (2010-2014), CC-BY-NC-SA3.0 Spain.
"""
import numpy as np
from scipy.signal import convolve2d
from .preprocessing import CENTER
def determinant_polynomial(points,line_ids):
    """Coefficients P[i,j] of sum(det(unnormalized scatter)) in k2^i k4^j.

    Each transformed coordinate is affine in the two radial coefficients.
    Explicit polynomial convolution expands the covariance determinants.
    """
    total=np.zeros((5,5))
    for group in np.unique(line_ids):
        q=points[line_ids==group];r2=np.sum(q*q,axis=1)
        terms=np.stack((q,q*r2[:,None],q*r2[:,None]**2),axis=1)
        terms-=terms.mean(axis=0)
        xx=np.zeros((3,3));xy=xx.copy();yy=xx.copy()
        for sample in terms:
            px=np.array([[sample[0,0],sample[2,0]],[sample[1,0],0.]])
            py=np.array([[sample[0,1],sample[2,1]],[sample[1,1],0.]])
            xx+=convolve2d(px,px);xy+=convolve2d(px,py);yy+=convolve2d(py,py)
        total+=convolve2d(xx,yy)-convolve2d(xy,xy)
    return total
def transform_points(points,coefficients):
    q=np.asarray(points)-CENTER;r=np.sqrt(np.sum(q*q,axis=-1))
    return CENTER+q*np.polynomial.polynomial.polyval(r,coefficients)[...,None]
def straightness(points,line_ids):
    """Mean squared orthogonal point-to-best-line distance in pixel squared."""
    error=0.
    for k in np.unique(line_ids):
        q=points[line_ids==k];q=q-q.mean(axis=0)
        error+=np.linalg.eigvalsh(q.T@q)[0]
    return float(error/len(points))

def inverse_radius_lookup(coefficients,max_radius):
    """Original sequential Newton radii, tolerance1e-12, maximum10000 updates."""
    size=int(max_radius+1.5);lookup=np.zeros(size);root=1.
    derivative=np.arange(1,6)*coefficients
    for distance in range(1,size):
        for iteration in range(10000):
            value=root*np.polynomial.polynomial.polyval(root,coefficients)-distance
            slope=np.polynomial.polynomial.polyval(root,derivative)
            if slope==0:break
            updated=root-value/slope
            if abs(root-updated)<abs(root)*1e-12:
                root=updated;break
            root=updated
        lookup[distance]=root/distance
    lookup[0]=lookup[1]
    return lookup

def rectify(image,coefficients):
    """Original all-visible framing, inverse radial LUT and bilinear uint8 output.

    Original image corners use width,height rather than width-1,height-1.
    Invalid mapped pixels are black; valid edge weights are renormalized.
    """
    height,width,_=image.shape
    corners=np.array([[0,0],[width,0],[0,height],[width,height]],dtype=float)
    radii=np.linalg.norm(corners-CENTER,axis=1)
    scales=np.polynomial.polynomial.polyval(radii,coefficients)
    scale=np.min(radii*scales/np.linalg.norm(CENTER))
    largest=max(radii.max(),(radii*scales).max())
    lut=inverse_radius_lookup(coefficients,largest)
    yy,xx=np.indices((height,width),dtype=float)
    qx=scale*xx-(scale*CENTER[0]-CENTER[0])-CENTER[0]
    qy=scale*yy-(scale*CENTER[1]-CENTER[1])-CENTER[1]
    r=np.sqrt(qx*qx+qy*qy)
    index=r.astype(int);valid=index<largest
    safe=np.minimum(index,len(lut)-2)
    factor=lut[safe]+(r-safe)*(lut[safe+1]-lut[safe])
    x=CENTER[0]+qx*factor;y=CENTER[1]+qy*factor
    n=np.trunc(x).astype(int);m=np.trunc(y).astype(int)
    valid&=(n>=0)&(n<width)&(m>=0)&(m<height)
    di=y-m;dj=x-n;total=np.zeros_like(image,dtype=float);mass=np.zeros((height,width))
    for dy,dx,w in [(0,0,(1-di)*(1-dj)),(1,0,di*(1-dj)),(0,1,(1-di)*dj),(1,1,di*dj)]:
        use=valid&(m+dy<height)&(n+dx<width)
        if dy or dx:use&=w>0
        total[use]+=w[use,None]*image[(m+dy)[use],(n+dx)[use]]
        mass[use]+=w[use]
    result=np.zeros_like(image);use=mass>0
    result[use]=(total[use]/mass[use,None]).astype(np.uint8)
    # Documented roundoff correction: bilinear interpolation must preserve a
    # constant cell exactly. Native divide-then-truncate can turn 213 into212.
    # This branch uses only the observed four neighbouring intensities.
    nn=np.clip(n,0,width-1);mm=np.clip(m,0,height-1)
    v=image[mm,nn]
    constant=(v==image[np.minimum(mm+1,height-1),nn])&(v==image[mm,np.minimum(nn+1,width-1)])&(v==image[np.minimum(mm+1,height-1),np.minimum(nn+1,width-1)])
    keep=constant&use[...,None]
    result[keep]=v[keep]
    return result
