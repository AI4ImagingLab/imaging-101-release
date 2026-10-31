"""Explicit Cartesian Fourier acquisition and shell-amplitude projection.

Extracted mathematics from DENSS, GPL-3.0-or-later; SUNY copyright.
The sole qualified correction weights real-FFT Hermitian partners exactly.
"""
import numpy as np
from scipy.fft import rfftn,irfftn
from scipy.interpolate import interp1d
from scipy.ndimage import center_of_mass,gaussian_filter,label,generate_binary_structure
from .preprocessing import SIZE,SIDE,VOXEL

def shell_grid(size: int=SIZE, side: float=SIDE) -> dict:
    spacing=side/size
    qx=np.fft.fftfreq(size)*2*np.pi/spacing;qz=np.fft.rfftfreq(size)*2*np.pi/spacing
    mesh=np.meshgrid(qx,qx,qz,indexing='ij');radius=np.sqrt(sum(a*a for a in mesh))
    step=radius[radius>0].min()-1e-8;bins=int(radius.max()/step)
    edges=np.linspace(0,bins*step,bins+1);labels=np.searchsorted(edges,radius,'right')-1
    multiplicity=np.ones(size//2+1);multiplicity[1:-1]=2
    weights=np.broadcast_to(multiplicity,radius.shape).copy()
    counts=np.bincount(labels.ravel(),weights=weights.ravel())
    grid={'labels':labels,'weights':weights,'counts':counts,'size':size,'side':side,'voxel':spacing}
    grid['q']=shell_mean(radius,grid)
    return grid

def shell_mean(values: np.ndarray, grid: dict) -> np.ndarray:
    return np.bincount(grid['labels'].ravel(),weights=(values*grid['weights']).ravel())/grid['counts']

def forward_scattering(density: np.ndarray, grid: dict | None=None) -> tuple[np.ndarray,np.ndarray]:
    """Return q(1/A), normalized intensity from density(1/A^3)."""
    grid=shell_grid() if grid is None else grid
    fourier=rfftn(density*grid['voxel']**3,workers=1)
    return grid['q'],shell_mean(np.abs(fourier)**2,grid)

def scattering_target(data: dict, grid: dict) -> tuple[np.ndarray,np.ndarray,np.ndarray]:
    """Interpolate GNOM observed-data fit, extend q^-4 and normalize I(0)=1."""
    q=data['q'][0];fit=data['smoothed_intensity'][0];sigma=data['sigma'][0]
    selected=(grid['q']>=q.min())&(grid['q']<=q.max());qdata=grid['q'][selected]
    target=np.interp(qdata,q,fit);extend=grid['q'][grid['q']>=qdata.max()]
    high=extend**-4;high=high/high[0]*target[-1]
    qdata=np.concatenate((qdata,extend[1:]));target=np.concatenate((target,high[1:]))
    selected=np.isin(grid['q'],qdata,assume_unique=True)
    errors=np.interp(qdata,q,sigma);scale=1/target[0]
    return target*scale,errors*scale,selected

def reciprocal_projection(rho: np.ndarray, target: np.ndarray, selected: np.ndarray, grid: dict) -> tuple[np.ndarray,dict]:
    fourier=rfftn(rho,workers=1);fourier[np.abs(fourier)==0]=1e-16
    average=shell_mean(np.abs(fourier)**2,grid)
    factors=np.sqrt(target/average);factors[~selected]=1
    projected=fourier*factors[grid['labels']]
    return irfftn(projected,workers=1).real,{'fourier':projected,'intensity':average,'factors':factors}

def profile_chi2(qcalc: np.ndarray, intensity: np.ndarray, data: dict, original_scale_fit: bool=True) -> float:
    """Original monitor: cubic shell interpolation, weighted amplitude fit, no offset.

    This monitor is intentionally distinct from evaluation of final rho.
    """
    valid=data['measured_mask'][0];q=data['q'][0,valid];observed=data['intensity'][0,valid];sigma=data['sigma'][0,valid]
    keep=qcalc<=q.max();pred=interp1d(qcalc[keep],intensity[keep],kind='cubic',fill_value='extrapolate')(q)
    scale=np.linalg.lstsq((observed/sigma)[:,None],pred/sigma,rcond=None)[0][0] if original_scale_fit else 1.0
    return float(np.mean(((scale*observed-pred)/(scale*sigma))**2))

def center_integer(rho: np.ndarray, maximum_first: bool=True) -> np.ndarray:
    center=np.array(rho.shape)//2
    if maximum_first:
        shift=np.rint(center-np.unravel_index(np.abs(rho).argmax(),rho.shape)).astype(int)
        rho=np.roll(rho,shift,axis=(0,1,2))
    shift=np.rint(center-np.array(center_of_mass(np.abs(rho)))).astype(int)
    return np.roll(rho,shift,axis=(0,1,2))

def update_support(rho: np.ndarray, sigma: float, threshold: float=.2) -> tuple[np.ndarray,np.ndarray]:
    rho=center_integer(rho)
    blurred=gaussian_filter(np.abs(rho),sigma=sigma,mode='wrap')
    return rho,blurred>=threshold*blurred.max()

def keep_connected(rho: np.ndarray,support: np.ndarray) -> tuple[np.ndarray,np.ndarray]:
    labels,count=label(support,structure=generate_binary_structure(3,3))
    sums=np.array([np.sum(rho[labels==i]) for i in range(1,count+1)])
    support=labels==(np.argmax(sums)+1);rho=rho.copy();rho[~support]=0
    return rho,support

def real_space_update(projected: np.ndarray,support: np.ndarray,iteration: int,sigma: float) -> tuple[np.ndarray,np.ndarray,float]:
    rho=np.where(support,projected,0);rho[rho<0]=0
    if iteration%500==0:
        shift=np.rint(np.array(rho.shape)//2-np.array(center_of_mass(np.abs(rho)))).astype(int)
        rho=np.roll(rho,shift,axis=(0,1,2));support=np.roll(support,shift,axis=(0,1,2))
    if iteration>=100 and iteration%20==1:
        rho,support=update_support(rho,sigma)
        if sigma>1.5:sigma*=.99
    if iteration==500:
        rho,support=update_support(rho,sigma);rho,support=keep_connected(rho,support)
    return rho,support,sigma
