"""Explicit tilted-ring emission, spatial beam and selected AZIM L1 criterion.

Mathematics extracted from BBarolo (GPL-2.0-or-later), Di Teodoro &
Fraternali (2015). The private original and qualification are indexed in
test/validation.md. Arrays are channel,y,x; only general NumPy/SciPy primitives.
"""
import math
from functools import lru_cache
import numpy as np
from scipy.signal import fftconvolve

SPATIAL_STEP_ARCMIN=np.array([-.266666673126,.266666673126],dtype=float)
CHANNEL_STEP_M_S=-5152.860000000001
VELOCITY_FIRST_M_S=289610.96899999998


def beam_kernel(major_arcsec=59.3287870284, minor_arcsec=58.6053617292,
                angle_deg=-79.446372986, cutoff=1e-4):
    """Native Gaussian kernel, elliptical cutoff, source rounded extent.

    FWHM beam angles are sky PA; the array x-axis rotation is PA+90 degrees.
    Spatial pixel signs enter the sky model; kernel spacings are absolute.
    """
    # Convpars stores atan2(sin(2PA),cos(2PA)) in template T=float32
    # before dividing by two. Preserve that discretization, not ideal PA.
    native_twice=np.float32(math.atan2(math.sin(2*np.deg2rad(angle_deg)),math.cos(2*np.deg2rad(angle_deg))))
    phi=float(native_twice)/2+np.pi/2
    cs,sn=math.cos(phi),math.sin(phi)
    extend=math.sqrt(-math.log(cutoff)/math.log(2))
    rx=.5*major_arcsec*extend;ry=.5*minor_arcsec*extend
    px,py=abs(SPATIAL_STEP_ARCMIN)*60
    hx=int(nearest_integer(max(abs(rx*cs),abs(ry*sn))/px))
    hy=int(nearest_integer(max(abs(rx*sn),abs(ry*cs))/py))
    yy,xx=np.indices((2*hy+1,2*hx+1));x=(xx-hx)*px;y=(yy-hy)*py
    xr=x*cs+y*sn;yr=-x*sn+y*cs
    values=np.exp(-4*np.log(2)*((xr/major_arcsec)**2+(yr/minor_arcsec)**2))
    values=np.where(values>=cutoff,values,0)
    total=np.add.accumulate(values.ravel())[-1]
    return values/total


def internal_rings(radius_edges_arcsec, parameters, density=1.0):
    """Float32 radial quadrature, selected fixed geometry and native bounding box.

    parameters rows are [rotation km/s,dispersion km/s,position angle deg].
    A single row defines one fitted annulus; multiple rows interpolate a disk.
    Returns internal rings, metadata and [radius,PA,inc,x0,y0] edge geometry.
    """
    radius=np.asarray(radius_edges_arcsec,dtype=np.float32)
    p=np.asarray(parameters,dtype=np.float32)
    if p.ndim==1:p=np.repeat(p[None],radius.size,axis=0)
    if p.shape!=(radius.size,3):raise ValueError('One parameter row per radial knot is required')
    # Each assignment here is float32 in the original ring container.
    r=(radius.astype(float)/60).astype(np.float32)
    width=.75*min(abs(SPATIAL_STEP_ARCMIN))
    start=r[0]+width/2 if r[0]!=0 else width/2
    sample=np.float32(start)
    values=np.column_stack((
        (p[:,0]*np.float32(1000)).astype(np.float32),
        (p[:,1]*np.float32(1000)).astype(np.float32),
        np.full(radius.size,np.float32(np.deg2rad(60))),
        (p[:,2].astype(float)*np.pi/180).astype(np.float32),
        np.broadcast_to(np.asarray(density,dtype=np.float32),(radius.size,)),
        np.full(radius.size,np.float32(10/60)),np.full(radius.size,77,dtype=np.float32),
        np.full(radius.size,77,dtype=np.float32),np.full(radius.size,np.float32(132.8)*np.float32(1000)))).astype(np.float32)
    output=[]
    for i in range(1,radius.size):
        slope=((values[i]-values[i-1])/np.float32(r[i]-r[i-1])).astype(np.float32)
        while sample<r[i]:
            delta=np.float32(sample-r[i-1])
            interpolated=(values[i-1]+slope*delta).astype(np.float32)
            output.append(np.r_[float(sample),interpolated.astype(float),62.])
            sample=np.float32(float(sample)+width)
    rings=np.asarray(output,dtype=float)
    # A freshly addRings-constructed fitted annulus retains radsep=0.
    # Its outer edge already includes the 15-arcsec half width.
    extent=int(math.ceil((float(radius[-1])+3*10)/(abs(SPATIAL_STEP_ARCMIN[0])*60)))
    bx=max(0,77-extent);by=bx;ex=min(150,77+extent+1);ey=ex
    px=abs(SPATIAL_STEP_ARCMIN[0]);py=abs(SPATIAL_STEP_ARCMIN[1]);instrument=float(np.float32(1/(2*np.sqrt(2*np.log(2)))))*abs(CHANNEL_STEP_M_S)
    meta=np.array([len(rings),ex-bx,ey-by,62,bx,by,ex,ey,*SPATIAL_STEP_ARCMIN,1,VELOCITY_FIRST_M_S,CHANNEL_STEP_M_S,abs(CHANNEL_STEP_M_S),instrument,px*py,width])
    edges=np.column_stack((radius,p[:,2],np.full(radius.size,60),np.full(radius.size,77),np.full(radius.size,77)))
    return rings,meta,edges


def zero_moment(cube,mask):
    """Masked integrated intensity, Jy/beam km/s, original float32 summation."""
    total=np.add.accumulate((np.asarray(cube,dtype=np.float32)*mask).astype(np.float32),axis=0,dtype=np.float32)[-1]
    value=(total.astype(float)*abs(CHANNEL_STEP_M_S)/1000).astype(np.float32)
    return np.where(total!=0,value,np.float32(np.nan))


def azimuthal_profile(moment0,parameters):
    """Author full-ring 2x2 subpixel flux estimate, including blank accounting.

    Columns are [integrated sum,number of valid subpixels,blank subpixels,
    sky-plane surface brightness,face-on surface brightness]. The surface
    brightness units are Jy/beam km/s/arcsec². This estimates nuisance flux
    from observations; it is not a gas-density truth or an added calibration.
    """
    image=np.asarray(moment0,dtype=np.float32)
    p=np.asarray(parameters,dtype=np.float32)
    yy,xx=np.indices(image.shape)
    # Author Dx/Dy and 2x2 subdivision are stored as float32 (16 and 8).
    spacing=np.float32(abs(SPATIAL_STEP_ARCMIN[0])*60)
    xcenter=(spacing*(xx-77)).astype(np.float32)
    ycenter=(spacing*(yy-77)).astype(np.float32)
    # x-major, y, subpixel-x, subpixel-y is the original accumulation order.
    offsets=np.array([[-spacing/4,-spacing/4],[-spacing/4,spacing/4],
                      [spacing/4,-spacing/4],[spacing/4,spacing/4]],dtype=np.float32)
    x=(xcenter.T.ravel()[:,None]+offsets[None,:,0]).ravel()
    y=(ycenter.T.ravel()[:,None]+offsets[None,:,1]).ravel()
    values=np.repeat(image.T.ravel(),4)
    valid=np.isfinite(values)
    radius=np.arange(15,1216,30,dtype=np.float32)
    output=[]
    for r,angle in zip(radius,p[:,2]):
        phi=-(float(angle)+90)*np.pi/180
        cosphi,sinphi=np.cos(phi),np.sin(phi)
        xp=(x.astype(float)*cosphi-y.astype(float)*sinphi).astype(np.float32)
        yp=((x.astype(float)*sinphi+y.astype(float)*cosphi)/np.cos(np.pi/3)).astype(np.float32)
        squared=xp*xp+yp*yp
        inner=np.float32(max(float(r)-15,0));outer=np.float32(float(r)+15)
        inside=(squared>=inner*inner)&(squared<outer*outer)
        n=int(np.count_nonzero(inside&valid));blanks=int(np.count_nonzero(inside&~valid))
        total=float(np.add.accumulate(values[inside&valid].astype(float))[-1])/4 if n else 0.
        # Native Num/4 and area are float expressions before double assignment.
        area=np.float32(np.float32(n)/np.float32(4))*np.float32(spacing*spacing)
        surface=np.float32(total/float(area)) if area else np.float32(0)
        output.append([total,n,blanks,float(surface),float(surface)*np.cos(np.pi/3)])
    return np.asarray(output)


def full_model(cube,mask,parameters,seed=-1):
    """Complete original post-fit disk emission, with observed annular flux.

    The lowest positive face-on profile selects the author's power-of-ten
    scaling to keep Monte-Carlo fluxes away from numerical underflow.
    No missing sky/velocity voxels are excluded from the returned cube.
    """
    profile=azimuthal_profile(zero_moment(cube,mask),parameters)
    positive=profile[:,-1][profile[:,-1]>0]
    if not positive.size:raise ValueError('No positive observation-derived radial flux')
    minimum=float(positive.min());scale=np.float32(1)
    while minimum<.1:minimum*=10;scale=np.float32(scale*10)
    while minimum>10:minimum/=10;scale=np.float32(scale/10)
    density=(profile[:,-1]*float(scale)).astype(np.float32)
    radius=np.arange(15,1216,30,dtype=np.float32);radius[0]=0;radius[-1]+=15
    rings,metadata,_=internal_rings(radius,parameters,density)
    # Output uses the complete native observation box, regardless of geometry.
    metadata=metadata.copy();metadata[1:8]=[150,150,62,0,0,150,150]
    model=beam_convolve(cloud_cube(rings,metadata,seed=seed),beam_kernel())
    return (model/scale).astype(np.float32),profile


def beam_convolve(cube, kernel, scale=1.0):
    """Zero-padded, linear SAME beam convolution; source positive-tail cutoff."""
    result=fftconvolve(np.asarray(cube,dtype=np.float64),np.asarray(kernel)[None],mode='same',axes=(-2,-1))
    return np.where(result<1e-12,0,result*scale).astype(np.float32)


def ring_geometry(shape, ring_edges, box_origin=(0,0), pixel_arcsec=16.00000038756):
    """Annulus and disk azimuth; ring columns radius,PA,inclination,x0,y0."""
    yy,xx=np.indices(shape)
    angle,inclination,x0,y0=ring_edges[-1,1:]
    phi,inc=np.deg2rad([angle,inclination])
    dx=xx-(x0-box_origin[0]);dy=yy-(y0-box_origin[1])
    xr=-dx*np.sin(phi)+dy*np.cos(phi)
    yr=(-dx*np.cos(phi)-dy*np.sin(phi))/np.cos(inc)
    radius=np.hypot(xr,yr)
    theta=np.where(radius<.1,0,np.arctan2(yr,xr))
    inside=(radius>=ring_edges[0,0]/pixel_arcsec)&(radius<=ring_edges[-1,0]/pixel_arcsec)
    return inside,theta


def azimuthal_loss(cube, observation, mask, ring_edges, metadata):
    """Normalize one annulus and return full original L1/blank penalty summary.

    metadata=(nx,ny,nz,bloX,bloY,nedges,data_noise,wpow,bweight,ftype).
    Floating-point descending source accumulation is retained deliberately.
    """
    nx,ny,nz,bx,by,_=np.asarray(metadata[:6],dtype=int)
    noise,wpow,bweight,ftype=metadata[6:]
    if ftype!=2:
        raise ValueError('Selected loss is FTYPE=2, weighted absolute residual')
    inside,theta=ring_geometry((ny,nx),ring_edges,(bx,by))
    weight=np.abs(np.cos(theta) if wpow>=0 else np.sin(theta))**abs(wpow)
    # Descending y,x,z is the actual original float32 addition order.
    positions=np.argwhere(inside)[::-1]
    ys,xs=positions.T
    def source_order(array):
        return array[::-1,ys,xs].T.ravel()
    model_sum=np.add.accumulate(source_order(cube),dtype=np.float32)[-1]
    obs_sum=np.add.accumulate(source_order((observation*mask).astype(np.float32)),dtype=np.float32)[-1]
    factor=np.float32(obs_sum/model_sum) if model_sum!=0 else np.float32(0)
    output=np.array(cube,dtype=np.float32,copy=True)
    output[:,ys,xs]*=factor
    m=source_order(output)
    o=source_order(np.where(observation>0,observation,np.float32(noise)))
    good=source_order(mask)
    use=good|(m!=0)
    blanks=int(np.count_nonzero(~good&use));count=int(use.sum())
    values=np.repeat(weight[ys,xs],nz)*np.abs((m-o).astype(np.float32)).astype(float)
    residual_sum=float(np.add.accumulate(values[use],dtype=float)[-1]) if use.any() else 0.
    if count==blanks:
        value=math.inf
    else:
        penalty=float(np.float32(1)+np.float32(blanks)/np.float32(count))**bweight
        value=penalty*residual_sum/(count-blanks)
    summary=np.array([obs_sum,model_sum,factor,residual_sum,count,blanks,int(inside.sum()),value])
    return output,summary


class CloudRandom:
    """General MT19937 plus explicit libstdc++ float canonical/polar sampling.

    This is an integration convention, not a claim that a seed establishes
    scientific quality. Complete independent-seed runs define the field gates.
    A uint32 pool and vectorized pair rejection leave the mathematical draws
    unchanged. No NumPy normal-distribution implementation is called.
    """
    def __init__(self, seed=-1):
        self.generator=np.random.RandomState(seed&0xffffffff)
        self.pool=np.empty(0,dtype=np.uint32);self.cursor=0
        self.saved=None

    def _ensure(self,n):
        if self.cursor+n>self.pool.size:
            tail=self.pool[self.cursor:]
            self.pool=np.r_[tail,self.generator.randint(0,2**32,size=max(65536,n),dtype=np.uint32)]
            self.cursor=0

    def canonical(self,n=1):
        self._ensure(n)
        raw=self.pool[self.cursor:self.cursor+n].astype(np.float32)
        self.cursor+=n
        canonical=np.minimum(raw*np.float32(2**-32),np.nextafter(np.float32(1),np.float32(0)))
        return canonical

    def uniform(self,n=1):
        return self.canonical(n)*np.float32(2)-np.float32(1)

    def normal(self,n):
        parts=[];left=n
        if self.saved is not None:
            parts.append(np.array([self.saved],dtype=np.float32));left-=1;self.saved=None
        while left:
            pairs=max(8,int(math.ceil(left/.75)))
            values=self.uniform(2*pairs).reshape(-1,2)
            radius=(values*values).sum(axis=1,dtype=np.float32)
            indices=np.flatnonzero((radius<=1)&(radius!=0))
            need=(left+1)//2
            chosen=indices[:need]
            if not chosen.size:
                continue
            # Undo only unused look-ahead words; preserve accepted/rejected pairs.
            # _ensure may have moved the pool, so reconstruct from current cursor.
            self.cursor-=2*(pairs-int(chosen[-1])-1)
            r=radius[chosen]
            mult=np.sqrt((-np.float32(2)*np.log(r))/r,dtype=np.float32)
            generated=(values[chosen][:,::-1]*mult[:,None]).ravel()
            take=min(left,generated.size)
            parts.append(generated[:take]);left-=take
            if generated.size>take:
                self.saved=generated[-1]
        return np.concatenate(parts)


def nearest_integer(value):
    """C++ lround: halfway cases go away from zero."""
    value=np.asarray(value)
    return np.where(value>=0,np.floor(value+.5),np.ceil(value-.5)).astype(np.int64)


def cloud_cube(internal_rings, metadata, seed=-1, cdens=10.0, quadrature=None):
    """Selected optically-thin H I cloud integral on a native velocity grid.

    Internal columns: radius arcmin,Vrot/disp m/s,inc/PA rad,density,z0 arcmin,
    center x/y pixel,Vsys m/s,velocity samples per cloud. Actual off-box clouds
    consume their spatial draws and skip all velocity draws, as in the source.
    Optional quadrature is an explicit deterministic integration input with
    columns ring index,3 spatial uniform deviates,62 standard LOS deviates;
    frozen original quadrature separates operator parity from stochastic tests.
    """
    nr,nx,ny,nz,bx,by,ex,ey=map(int,metadata[:8])
    dx,dy,crpix,crval,dv,width,instrument,pixarea,dr=metadata[8:]
    out=np.zeros((nz,ny,nx),dtype=np.float32)
    # Geometry and random quadrature do not depend on Vrot/dispersion.
    # Reusing exactly those computed draws is the same source common-random
    # integration, and does not cache fitted fields, objective values or cubes.
    geometry=np.asarray(internal_rings)[:,[0,3,4,5,6,7,8,9,10]]
    if quadrature is None:
        samples=_cloud_samples(tuple(geometry.ravel()),tuple(np.asarray(metadata)),seed,float(cdens))
    else:samples=_conditioned_cloud_samples(internal_rings,metadata,quadrature,cdens)
    for row,(profiles,cosaz,deviates,increment) in zip(internal_rings,samples):
        radius,rotation,dispersion,inc,pa,density,z0,x0,y0,systemic,nv=row
        if not profiles.size:continue
        sigma=math.sqrt(dispersion*dispersion+instrument*instrument)
        velocity=systemic+rotation*cosaz[:,None]*math.sin(inc)+deviates.astype(float)*sigma
        channel=nearest_integer((velocity-crval)/dv+crpix-1)
        index=channel*(nx*ny)+profiles[:,None]
        use=(channel>=0)&(channel<nz)
        np.add.at(out.ravel(),index[use],increment)
    return out


def _conditioned_cloud_samples(internal_rings,metadata,quadrature,cdens=10.0):
    """Evaluate all geometric/LOS formulas at supplied integration variates."""
    nr,nx,ny,nz,bx,by,ex,ey=map(int,metadata[:8])
    dx,dy,crpix,crval,dv,width,instrument,pixarea,dr=metadata[8:]
    quadrature=np.asarray(quadrature,dtype=float);samples=[]
    for ir,row in enumerate(internal_rings):
        radius,rotation,dispersion,inc,pa,density,z0,x0,y0,systemic,nv=row
        nv=int(nv);area=2*np.pi*radius*dr/pixarea
        clouds=int(nearest_integer(cdens*density*area))
        if nearest_integer(clouds/area)<10:clouds=int(nearest_integer(10*area))
        increment=np.float32(np.float32(density*area*pixarea/(clouds*nv))*np.float32(3600/width*1000))
        selected=quadrature[quadrature[:,0]==ir]
        if selected.shape[1]!=4+nv:raise ValueError('Quadrature velocity-sample count mismatch')
        u=selected[:,1:4]
        R=np.sqrt((radius-.5*dr)**2+2*dr*radius*abs(u[:,0]))
        az=2*np.pi*abs(u[:,1]);height=np.arctanh(u[:,2])*z0
        x=R*np.cos(az);y=R*np.sin(az)*np.cos(inc)-height*np.sin(inc)
        gx=nearest_integer(x0+(x*np.sin(pa)-y*np.cos(pa))/dx)
        gy=nearest_integer(y0+(x*np.cos(pa)+y*np.sin(pa))/dy)
        inside=(gx>=bx)&(gx<ex)&(gy>=by)&(gy<ey)
        profiles=(gy-by)*nx+gx-bx
        samples.append((profiles[inside],np.cos(az[inside]),selected[inside,4:].astype(np.float32),increment))
    return tuple(samples)


@lru_cache(maxsize=16)
def _cloud_samples(geometry_key,metadata_key,seed,cdens):
    """Memoize only source random quadrature at identical physical geometry."""
    metadata=np.asarray(metadata_key)
    nr,nx,ny,nz,bx,by,ex,ey=map(int,metadata[:8])
    dx,dy,crpix,crval,dv,width,instrument,pixarea,dr=metadata[8:]
    random=CloudRandom(seed);samples=[]
    for row in np.asarray(geometry_key).reshape(-1,9):
        radius,inc,pa,density,z0,x0,y0,systemic,nv=row
        if density==0:
            samples.append((np.empty(0,dtype=np.int64),np.empty(0),np.empty((0,int(nv)),dtype=np.float32),np.float32(0)))
            continue
        nv=int(nv)
        area=2*np.pi*radius*dr/pixarea
        clouds=int(nearest_integer(cdens*density*area))
        if nearest_integer(clouds/area)<10:
            clouds=int(nearest_integer(10*area))
        flux=np.float32(density*area*pixarea/(clouds*nv))
        cd2i=np.float32(3600/width*1000)
        increment=np.float32(np.float32(1)*flux*cd2i)
        sini=math.sin(inc);cosi=math.cos(inc);sinpa=math.sin(pa);cospa=math.cos(pa)
        profiles=[];cosaz=[];deviates=[]
        for _ in range(clouds):
            u=random.uniform(3).astype(float)
            R=math.sqrt((radius-.5*dr)**2+2*dr*radius*abs(u[0]))
            az=2*np.pi*abs(u[1]);height=math.atanh(u[2])*z0
            x=R*math.cos(az);y=R*math.sin(az)*cosi-height*sini
            gx=int(nearest_integer(x0+(x*sinpa-y*cospa)/dx))
            gy=int(nearest_integer(y0+(x*cospa+y*sinpa)/dy))
            if not (bx<=gx<ex and by<=gy<ey):continue
            profiles.append((gy-by)*nx+gx-bx);cosaz.append(math.cos(az));deviates.append(random.normal(nv))
        samples.append((np.asarray(profiles,dtype=np.int64),np.asarray(cosaz),np.asarray(deviates,dtype=np.float32).reshape(-1,nv),increment))
    return tuple(samples)
