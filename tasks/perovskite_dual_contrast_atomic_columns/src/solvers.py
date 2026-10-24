"""Local least-squares and lattice recovery mathematics; GPL-3.0-or-later.

Extracted from Atomap contributors, with NumPy arrays replacing framework objects.
"""
import numpy as np
import math
from scipy.spatial import cKDTree
from scipy.optimize import leastsq
from .physics_model import gaussian,model_image,physical_columns
def fit_crop(image,initial,x_offset,y_offset):
    yy,xx=np.indices(image.shape);xx=xx+x_offset;yy=yy+y_offset
    # Original free_parameters are alphabetically ordered, not component order.
    order=np.array([0,3,4,5,1,2]);inverse=np.argsort(order)
    def residual(free):return (gaussian(free[inverse],xx,yy)-image).ravel()
    result=leastsq(residual,np.asarray(initial)[order],col_deriv=1,full_output=True)
    return result[0][inverse]
def nearest(positions,k=9):
    return cKDTree(positions,leafsize=100).query(positions,k=k)[1][:,1:]
def center_of_mass(image,position,radius):
    cx,cy=[int(round(v)) for v in position];r=int(np.ceil(radius))
    padded=np.pad(image,r+1)
    a=padded[cy+1:cy+2*r+2,cx+1:cx+2*r+2].copy()
    yy,xx=np.indices(a.shape);a[(xx-r)**2+(yy-r)**2>radius**2]=0
    total=np.sum(a)
    if total==0:raise ValueError('Zero-mass local observation')
    return np.array([cx-r+np.sum(xx*a)/total,cy-r+np.sum(yy*a)/total])
def fit_one(image,parameters,radius,already_fitted=False):
    p=parameters.copy();x,y=p[:2]
    x0=max(0,int(round(x-radius)));x1=int(round(x+radius))+1
    y0=max(0,int(round(y-radius)));y1=int(round(y+radius))+1
    a=image[y0:y1,x0:x1].astype(np.float64,copy=True)
    yy,xx=np.indices(a.shape);xx=xx+x0;yy=yy+y0
    for attempt in range(10):
        mask=(xx-x)**2+(yy-y)**2<=radius**2
        a=a*mask
        values=np.sort(a[mask]);a-=np.median(values[:max(1,int(.03*len(values)))]);a[a<0]=0
        amp=p[5] if already_fitted else np.median(np.sort(a.ravel())[-max(1,int(.03*a.size)):])*10
        initial=np.array([amp,p[2],p[3],x,y,p[4]])
        if a.size>=6:
            g=fit_crop(a,initial,x0,y0)
            iy,ix=int(round(g[4]-y0)),int(round(g[3]-x0))
            valid=(0<iy<a.shape[0] and 0<ix<a.shape[1] and mask[iy,ix] and g[0]>=0 and min(abs(g[1:3]))>0 and max(abs(g[1:3]))/min(abs(g[1:3]))<=4)
            if valid:return np.array([g[3],g[4],abs(g[1]),abs(g[2]),g[5]%math.pi,g[0]]),True
        if attempt==9:
            p[:2]=center_of_mass(image,p[:2],radius);p[5]=0;return p,already_fitted
        # Preserve the selected source's first-retry radius assignment.
        # It reuses this radius on later retries rather than recropping.
        if attempt==0:radius*=.95
    raise AssertionError('Unreachable fit state')
def gaussian_pass(image,parameters,neighbor_indices,fraction,fitted=None):
    p=parameters.copy();flags=np.zeros(len(p),bool) if fitted is None else fitted.copy()
    for i in range(len(p)):
        radius=np.min(np.linalg.norm(p[neighbor_indices[i],:2]-p[i,:2],axis=1))*fraction
        p[i],flags[i]=fit_one(image,p[i],radius,bool(flags[i]))
    return p,flags
def refine(image,parameters,fraction,gaussian_repetitions=1,com=False):
    p=np.asarray(parameters,dtype=np.float64).copy();nn=nearest(p[:,:2]);flags=np.zeros(len(p),bool)
    if com:
        for i in range(len(p)):
            radius=np.min(np.linalg.norm(p[nn[i],:2]-p[i,:2],axis=1))*fraction
            p[i,:2]=center_of_mass(image,p[i,:2],radius)
    for _ in range(gaussian_repetitions):
        # Source refine_sublattice invokes nominal fixed/free rotation passes;
        # its fit routine actually leaves rotation free in both calls.
        p,flags=gaussian_pass(image,p,nn,fraction,flags)
        p,flags=gaussian_pass(image,p,nn,fraction,flags)
    return p
def translation_vectors(positions,separation=12):
    """Explicit density-connected clustering of ordered displacement vectors."""
    p=np.asarray(positions,dtype=np.float32)
    differences=p[None,:,:]-p[:,None,:]
    mask=~np.eye(len(p),dtype=bool)
    mask&=(differences[:,:,0]**2+differences[:,:,1]**2)<(7*separation)**2
    d=differences[mask]
    if len(d)>150000:
        raise ValueError('Selected full-field binding exceeds validated fingerprint cap')
    scale=d.std();q=d/scale;tree=cKDTree(q)
    labels=np.full(len(q),-1,int);visited=np.zeros(len(q),bool);cluster=0
    for i in range(len(q)):
        if visited[i]:continue
        visited[i]=True;neighbors=tree.query_ball_point(q[i],.1)
        if len(neighbors)<10:continue
        labels[i]=cluster;stack=list(neighbors)
        while stack:
            j=stack.pop()
            if not visited[j]:
                visited[j]=True;near=tree.query_ball_point(q[j],.1)
                if len(near)>=10:stack.extend(k for k in near if not visited[k])
            if labels[j]<0:labels[j]=cluster
        cluster+=1
    vectors=[]
    for i in range(cluster):
        mean=q[labels==i].mean(axis=0)*scale
        vectors.append(tuple(float(format(v,'.2f')) for v in mean))
    vectors.sort(key=lambda v:math.hypot(*v));tol=separation/1.5
    remove=set()
    for i,a in enumerate(vectors):
        for j,b in enumerate(vectors[i+1:],i+1):
            if math.dist(a,b)<tol:remove.add(j)
    vectors=[a for i,a in enumerate(vectors) if i not in remove]
    remove=[]
    for i,a in enumerate(vectors):
        length=math.hypot(*a)
        for n in range(-4,5):
            for b in vectors[i+1:]:
                if math.dist(b,(n*a[0],n*a[1]))<tol:
                    if abs(length-math.hypot(*b))<length/10:
                        axis=0 if abs(a[0])>=abs(a[1]) else 1
                        remove.append(b if a[axis]>b[axis] else a)
                    else:remove.append(b)
    return [v for v in vectors if v not in remove][:6]
def atomic_planes(positions,vector,separation=12):
    p=np.asarray(positions);v=np.asarray(vector);nn=nearest(p,15)
    used=set();planes=[]
    def walk(start,direction):
        path=[start]
        while True:
            i=path[-1];choices=nn[i];distance=np.linalg.norm(p[choices]-(p[i]+direction),axis=1)
            j=int(np.argmin(distance))
            if distance[j]>=.5*separation:break
            nxt=int(choices[j])
            if nxt in path:raise ValueError('Cyclic atomic plane')
            path.append(nxt)
        return path
    for i in range(len(p)):
        if i in used:continue
        forward,reverse=walk(i,v),walk(i,-v)
        plane=forward+reverse[1:]
        if len(plane)==1:continue
        used.update(plane)
        start=p[reverse[-1]]
        plane.sort(key=lambda j:math.dist(p[j],start));planes.append(plane)
    orthogonal=np.array([1e8*v[1],-1e8*v[0]])
    planes.sort(key=lambda ids:np.linalg.norm(p[ids]-orthogonal,axis=1).min())
    return planes
def midpoint_lattice(positions,axis_index):
    vectors=translation_vectors(positions)
    vector=vectors[axis_index];planes=atomic_planes(positions,vector)
    if not planes or sum(len(a)==2 for a in planes)/len(planes)>.6:
        raise ValueError('Required crystallographic direction is fragmented')
    out=[]
    for ids in planes:
        for j,k in zip(ids[:-1],ids[1:]):out.append((positions[j]+positions[k])*.5)
    return np.asarray(out)
def subtract_columns(image,parameters,neighbor_indices,fraction=.35):
    fitted,_=gaussian_pass(image,parameters,neighbor_indices,fraction)
    return image-model_image(fitted,image.shape)
def initial_parameters(positions):
    p=np.zeros((len(positions),6),np.float64);p[:,:2]=positions;p[:,2:4]=1.2;p[:,4]=.01;p[:,5]=1
    return p
def reconstruct(adf,abf,pixel_nm=.012443924564929288,capture=False):
    from .preprocessing import filter_adf,find_peaks
    states={};filtered=filter_adf(adf);peaks=find_peaks(filtered)
    a=refine(adf,initial_parameters(peaks),.35,gaussian_repetitions=2)
    b_initial=initial_parameters(midpoint_lattice(a[:,:2],1))
    b_image=subtract_columns(adf,a,nearest(a[:,:2],15))
    b=refine(b_image,b_initial,.25,com=True)
    o_initial=initial_parameters(midpoint_lattice(b[:,:2],0))
    reciprocal=1/abf
    abf_minus_a=subtract_columns(reciprocal,a,nearest(a[:,:2],15))
    o_image=subtract_columns(abf_minus_a,b,nearest(b_initial[:,:2],9),fraction=.30)
    o=refine(o_image,o_initial,.25,com=True)
    if capture:
        states=dict(filtered=filtered,initial_peaks=peaks,a_parameters=a,b_parameters=b,oxygen_parameters=o,b_image=b_image,abf_minus_a=abf_minus_a,oxygen_image=o_image,b_initial=b_initial,oxygen_initial=o_initial)
    out={key:physical_columns(p,pixel_nm)[None] for key,p in zip(['a_columns','b_columns','oxygen_columns'],[a,b,o])}
    return out,states
