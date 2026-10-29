"""Lorentz mapping and nonlinear count conservation, in the original cgs units.

Adapted from PROBLEM, copyright (c)2017 Flash Center for Computational Science,
MIT License; see task LICENSE and private provenance in test/validation.md.
"""
import numpy as np
from scipy.interpolate import RegularGridInterpolator

RI_CM = 1.3
LI_CM = 0.2
RS_CM = 20.0
V_CMS = 5.24e9
M_PROTON_G = 1.67e-24
ESU = 4.803e-10
C_CMS = 3.0e10
MAGNIFICATION = (RS_CM + RI_CM + .5 * LI_CM) / (RI_CM + .5 * LI_CM)

def mapping_derivatives(potential, x, y):
    """Return gradient(cm) and Hessian(dimensionless), axes x then y.

    np.gradient uses centered two-point differences inside and first-order
    one-sided differences on edges. Normal gradient is the identity map on
    each boundary; the mixed Hessian is zero on all boundary pixels, as source.
    """
    dx = x[1, 0] - x[0, 0]
    dy = y[0, 1] - y[0, 0]
    gx, gy = np.gradient(potential, dx, dy)
    gx[0, :] = x[0, :]; gx[-1, :] = x[-1, :]
    gy[:, 0] = y[:, 0]; gy[:, -1] = y[:, -1]
    hxx, hxy = np.gradient(gx, dx, dy)
    _, hyy = np.gradient(gy, dx, dy)
    hxy[0, :] = 0; hxy[-1, :] = 0
    hxy[:, 0] = 0; hxy[:, -1] = 0
    return gx, gy, hxx, hyy, hxy

def fluence_interpolator(x, y, fluence):
    """Bilinear observations on the object-coordinate grid; exterior mean fill."""
    return RegularGridInterpolator((x[:, 0], y[0, :]), fluence, method='linear',
                                   bounds_error=False, fill_value=np.mean(fluence))

def transport_state(potential, x, y, incident, interpolate):
    """Compute log[observed(gradient Phi)*abs(det Hess Phi)/incident]."""
    gx, gy, hxx, hyy, hxy = mapping_derivatives(potential, x, y)
    determinant = hxx * hyy - hxy ** 2
    mapped_fluence = interpolate((gx, gy))
    residual = np.log(mapped_fluence * np.absolute(determinant) / incident)
    return dict(phix=gx, phiy=gy, phixx=hxx, phiyy=hyy, phixy=hxy,
                det=determinant, fluxn=mapped_fluence, Fn=residual)

def velocity_from_mapping(x, y, map_x, map_y):
    """Transverse velocity impulse (2,nx,ny), cm/s, original rounded constants."""
    return np.stack([MAGNIFICATION * (V_CMS / RS_CM) * (map_x - x),
                     MAGNIFICATION * (V_CMS / RS_CM) * (map_y - y)])

def magnetic_from_velocity(velocity):
    """Into-screen beam, x right/y up: Bx=-mp*c/q*wy, By=mp*c/q*wx, G cm.

    The laboratory right-handed Z axis points OUT of the screen. The beam is
    along -Z, so its direction cross B has components (+By,-Bx).
    """
    factor = M_PROTON_G * C_CMS / ESU
    return np.stack([-factor * velocity[1], factor * velocity[0]])

def magnetic_from_mapping(x, y, map_x, map_y):
    """Return all two path-integrated transverse magnetic components, G cm."""
    return magnetic_from_velocity(velocity_from_mapping(x, y, map_x, map_y))

def mapping_from_magnetic(x, y, magnetic):
    """Inverse Lorentz conversion for independent observation-consistency checks."""
    factor = M_PROTON_G * C_CMS / ESU
    wx, wy = magnetic[1] / factor, -magnetic[0] / factor
    return x + RS_CM / (MAGNIFICATION * V_CMS) * wx, y + RS_CM / (MAGNIFICATION * V_CMS) * wy

def ray_forward_fluence(x, y, magnetic, incident_mean, subpixels=8, beam_direction=-1):
    """Independent ray-bin discretization for scientific model checks, counts/pixel.

    Sample a uniform incident sheet with subpixels**2 equal-weight rays per
    original pixel, interpolate both field components, apply the Lorentz impulse
    in laboratory axes, and histogram demagnified detector coordinates. Beam
    direction -1 is the bound acquisition; +1 is an explicitly wrong-sign control.
    Zero exterior field extends the finite supplied object. Rows are streamed to
    bound memory. This is an analysis helper, not the reconstruction update.
    """
    nx,ny=x.shape
    dx=x[1,0]-x[0,0];dy=y[0,1]-y[0,0]
    sx=np.linspace(x[0,0]-dx/2+dx/(2*subpixels),x[-1,0]+dx/2-dx/(2*subpixels),nx*subpixels)
    sy=np.linspace(y[0,0]-dy/2+dy/(2*subpixels),y[0,-1]+dy/2-dy/(2*subpixels),ny*subpixels)
    ex=np.linspace(x[0,0]-dx/2,x[-1,0]+dx/2,nx+1)
    ey=np.linspace(y[0,0]-dy/2,y[0,-1]+dy/2,ny+1)
    interpolators=[RegularGridInterpolator((x[:,0],y[0,:]),b,bounds_error=False,fill_value=0) for b in magnetic]
    counts=np.zeros((nx,ny),dtype=np.float64)
    for start in range(0,len(sx),128):
        xx,yy=np.meshgrid(sx[start:start+128],sy,indexing='ij')
        points=np.column_stack((xx.ravel(),yy.ravel()))
        bx,by=[f(points) for f in interpolators]
        wx=-beam_direction*ESU/(M_PROTON_G*C_CMS)*by
        wy=beam_direction*ESU/(M_PROTON_G*C_CMS)*bx
        tx=points[:,0]+RS_CM/(MAGNIFICATION*V_CMS)*wx
        ty=points[:,1]+RS_CM/(MAGNIFICATION*V_CMS)*wy
        counts+=np.histogram2d(tx,ty,bins=(ex,ey))[0]
    return counts*incident_mean/(subpixels**2)
