"""Local Kirchhoff-Love kinematics, Hermite virtual field and virtual work."""
import numpy as np
from scipy.signal import convolve2d
from .preprocessing import PIXEL_SIZE,SAMPLING_RATE
YOUNG_MODULUS=190e9
POISSON_RATIO=0.3
DENSITY=7934.0
THICKNESS=4.95e-3
WINDOW_SIZE=20

def bending_stiffness(modulus: float=YOUNG_MODULUS, poisson: float=POISSON_RATIO, thickness: float=THICKNESS) -> tuple[float,float]:
    """Return (D11,D12) in N m for an isotropic thin plate."""
    d=thickness**3/12*modulus/(1-poisson**2)
    return d,poisson*d

def kinematics(deflection: np.ndarray, spacing: float=PIXEL_SIZE, sampling_rate: float=SAMPLING_RATE) -> dict[str,np.ndarray]:
    """Return slopes(rad), curvatures(1/m), acceleration(m/s^2), time(s).

    Input (time,row,column) in meters. Original x is row direction, y is column
    direction. Slopes are minus deflection gradients. numpy.gradient uses centered
    interior/one-sided first-order edge differences, preserved from source.
    """
    a=np.asarray(deflection,dtype=np.float64)
    if a.ndim!=3 or min(a.shape)<2:raise ValueError('need time,row,column with at least two samples each')
    sx,sy=np.gradient(-a,spacing,spacing,axis=(1,2),edge_order=1)
    xx,xy=np.gradient(sx,spacing,spacing,axis=(1,2),edge_order=1)
    yx,yy=np.gradient(sy,spacing,spacing,axis=(1,2),edge_order=1)
    dt=1.0/sampling_rate
    acc=np.gradient(np.gradient(a,axis=0,edge_order=1)/dt,axis=0,edge_order=1)/dt
    return {'deflection':a,'slope_x':sx,'slope_y':sy,'curv_xx':xx,'curv_yy':yy,'curv_xy':.5*(xy+yx),'acceleration':acc,'time':np.arange(a.shape[0])/sampling_rate}

def hermite_virtual_fields(window: int=WINDOW_SIZE, spacing: float=PIXEL_SIZE) -> dict[str,np.ndarray]:
    """Sample a unit-center 2x2 Hermite16 virtual patch at pixel midpoints.

    C1 virtual deflection vanishes with first derivatives at the patch boundary.
    Return deflection(dimensionless), curv_xx/yy/xy(1/m^2). The tensor-product
    representation explicitly evaluates cubic Hermite shape functions; it does
    not call a finite-element or inverse package.
    """
    if not isinstance(window,int) or window<4 or window%2:raise ValueError('window must be even and at least4')
    half=window//2;length=window*spacing;local=(np.arange(half)+.5)/half
    up=3*local**2-2*local**3
    h=np.concatenate((up,1-up))
    derivative=(6*local-6*local**2)/(length/2)
    dh=np.concatenate((derivative,-derivative))
    second=(6-12*local)/(length/2)**2
    ddh=np.concatenate((second,-second))
    return {'deflection':np.outer(h,h),'curv_xx':-np.outer(ddh,h),'curv_yy':-np.outer(h,ddh),'curv_xy':-np.outer(dh,dh)}

def virtual_work_terms(fields: dict, virtual: dict, density: float=DENSITY, thickness: float=THICKNESS) -> tuple[np.ndarray,np.ndarray,np.ndarray]:
    """Return unnormalized D11/D12 and inertial terms on valid moving patches.

    convolve2d implements the source's index reversal, valid windows only. The
    area factor spacing^2 cancels between virtual works and the normalization.
    """
    conv=lambda a,b:convolve2d(a,b,mode='valid')
    xx,yy,xy=fields['curv_xx'],fields['curv_yy'],fields['curv_xy']
    vx,vy,vxy=virtual['curv_xx'],virtual['curv_yy'],virtual['curv_xy']
    a11=conv(xx,vx)+conv(yy,vy)+2*conv(xy,vxy)
    a12=conv(xx,vy)+conv(yy,vx)-2*conv(xy,vxy)
    inertia=density*thickness*conv(fields['acceleration'],virtual['deflection'])
    return a11,a12,inertia

def pressure_virtual_work(pressure: np.ndarray, virtual_deflection: np.ndarray) -> np.ndarray:
    """Forward external virtual work per unit sampled area, pressure in Pa.

    Sum(p*w*) has units Pa for dimensionless w*. Actual work for a unit-meter
    virtual displacement multiplies this by pixel area (m^2) and1m.
    """
    return convolve2d(pressure,virtual_deflection,mode='valid')
