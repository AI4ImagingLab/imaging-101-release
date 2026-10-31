"""Local Fresnel laws and the author's mask-dependent height discretization.

Adapted from William Smith (2016), Apache-2.0; see LICENSE and provenance.
"""
import numpy as np
from scipy import sparse
from scipy.ndimage import convolve

REFRACTIVE_INDEX = 1.5
LIGHT_DIRECTION = np.array([2., 0., 7.])
SMOOTHING = .1


def diffuse_zenith(rho, refractive_index=REFRACTIVE_INDEX):
    """Invert diffuse Fresnel polarization with the original closed formula."""
    n = refractive_index
    # Complex intermediate roots reproduce MATLAB's real-part clipping rule.
    root = np.sqrt((-(rho-1)*(rho+1)).astype(complex))
    numerator = (2*rho+2*n*n*rho-2*n*n+n**4+rho*rho+4*n*n*rho*rho
                 -n**4*rho*rho-4*n**3*rho*root+1)
    denominator = n**4*rho*rho+2*n**4*rho+n**4+6*n*n*rho*rho+4*n*n*rho-2*n*n+rho*rho+2*rho+1
    return np.arccos(np.minimum(np.sqrt(numerator/denominator).real, 1.))


def specular_zenith(rho, refractive_index=REFRACTIVE_INDEX):
    """Invert the increasing specular Fresnel branch using the 0.01-rad LUT."""
    n = refractive_index
    theta = np.arange(0., np.pi/2+.0000001, .01)
    r = 2*np.sin(theta)**2*np.cos(theta)*np.sqrt(n*n-np.sin(theta)**2)
    r /= n*n-np.sin(theta)**2-n*n*np.sin(theta)**2+2*np.sin(theta)**4
    stop = int(np.argmax(r))+1
    return np.interp(rho, r[:stop], theta[:stop], left=np.nan, right=np.nan)


def fit_light(theta, phase, intensity, diffuse_mask, direction=LIGHT_DIRECTION):
    """Alternate azimuth ambiguity assignment and scalar light/albedo fitting."""
    t, p = theta[diffuse_mask], phase[diffuse_mask]
    basis = np.column_stack((np.sin(p)*np.sin(t), np.cos(p)*np.sin(t), np.cos(t)))
    observations = intensity[diffuse_mask]
    transform = np.diag([-1., -1., 1.])
    light = np.asarray(direction, dtype=float).copy()
    for iteration in range(101):
        flip = (basis@light-observations)**2 > (basis@transform@light-observations)**2
        chosen = basis.copy()
        chosen[flip] = chosen[flip]@transform
        predicted = chosen@light
        updated = light*(predicted@observations)/(predicted@predicted)
        converged = np.linalg.norm(light-updated) < 1e-9
        light = updated
        if converged:
            break
    return light, chosen, iteration+1


def angular_fields(rho, phase, specular_mask):
    """Return mixed-reflection zenith and diffuse-convention phase."""
    theta = diffuse_zenith(rho)
    theta[specular_mask] = specular_zenith(rho[specular_mask])
    adjusted = phase.copy()
    adjusted[specular_mask] = np.mod(phase[specular_mask]+np.pi/2, np.pi)
    return theta, adjusted


def assemble_height_system(theta, intensity, phase, light, mask, specular_mask, smoothing=SMOOTHING):
    """Return CSR C and d for min||Cz-d||², row-major foreground z.

    x points image-right and y image-up. The author's nine-point derivative
    applies when all eight neighbours exist, otherwise centered/one-sided
    derivatives are used. Each pixel contributes smoothing followed by its
    reflection-specific equations. The final row anchors foreground pixel 0.
    """
    shape = mask.shape
    m = np.pad(mask, 1)
    sp = np.pad(specular_mask, 1)
    th, it, ph = (np.pad(a, 1) for a in (theta, intensity, phase))
    indices = np.full(m.shape, -1, dtype=np.int64)
    count = int(mask.sum())
    indices[m] = np.arange(count)
    neighbours = convolve(m.astype(np.int8), np.array([[0,1,0],[1,0,1],[0,1,0]]), mode='constant') == 4
    rr, cc, vv = [], [], []
    rhs = []
    def append(entries, target):
        row = len(rhs)
        for coordinate, value in entries:
            rr.append(row); cc.append(indices[coordinate]); vv.append(value)
        rhs.append(target)
    half = (light+np.array([0.,0.,1.]))/np.linalg.norm(light+np.array([0.,0.,1.]))
    for r, c in np.argwhere(m):
        if neighbours[r,c] and smoothing != 0:
            append([((r,c),-4*smoothing),((r,c-1),smoothing),((r,c+1),smoothing),
                    ((r-1,c),smoothing),((r+1,c),smoothing)], 0.)
        if sp[r,c]:
            equations = [(1.,0.,-half[0]/half[2]),(0.,1.,-half[1]/half[2]),
                         (np.cos(ph[r,c]),-np.sin(ph[r,c]),0.)]
        else:
            equations = [(np.cos(ph[r,c]),-np.sin(ph[r,c]),0.),
                         (-light[0],-light[1],it[r,c]/np.cos(th[r,c])-light[2])]
        if not (m[r,c-1] or m[r,c+1]) or not (m[r-1,c] or m[r+1,c]):
            continue
        if neighbours[r,c] and m[r-1,c-1] and m[r-1,c+1] and m[r+1,c-1] and m[r+1,c+1]:
            dx = [((r,c-1),-4/12),((r,c+1),4/12),((r-1,c-1),-1/12),((r-1,c+1),1/12),((r+1,c-1),-1/12),((r+1,c+1),1/12)]
            dy = [((r-1,c),4/12),((r+1,c),-4/12),((r-1,c-1),1/12),((r-1,c+1),1/12),((r+1,c-1),-1/12),((r+1,c+1),-1/12)]
        else:
            dx = ([((r,c-1),-.5),((r,c+1),.5)] if m[r,c-1] and m[r,c+1] else
                  [((r,c-1),-1.),((r,c),1.)] if m[r,c-1] else [((r,c+1),1.),((r,c),-1.)])
            dy = ([((r-1,c),.5),((r+1,c),-.5)] if m[r-1,c] and m[r+1,c] else
                  [((r-1,c),1.),((r,c),-1.)] if m[r-1,c] else [((r+1,c),-1.),((r,c),1.)])
        for x, y, target in equations:
            append([(coordinate,value*x) for coordinate,value in dx]+[(coordinate,value*y) for coordinate,value in dy], target)
    append([tuple((tuple(np.argwhere(m)[0]),1.))], 0.)
    matrix = sparse.coo_matrix((vv,(rr,cc)), shape=(len(rhs),count)).tocsr()
    matrix.eliminate_zeros()
    return matrix, np.array(rhs)
