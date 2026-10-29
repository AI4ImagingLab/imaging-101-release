"""Explicit spatial point-response localization mathematics from PALA.

Authors: Arthur Chavignon, Baptiste Heiles, Vincent Hingot; CNRS,
Sorbonne Universite, INSERM. PALA attribution and CC BY-NC-SA 4.0 terms: test/validation.md.
"""
import numpy as np
from scipy import ndimage

def regional_maximum(image):
    """Eight-connected regional maxima of a nonnegative 2-D magnitude image."""
    a = np.asarray(image)
    maxima = a == ndimage.maximum_filter(a, size=3, mode='constant', cval=-np.inf)
    # Resolve flat connected plateaus rather than accepting a distant plateau
    # interior whose connected boundary touches a brighter value.
    equal = np.zeros(a.shape, bool)
    for dz in (-1, 0, 1):
        for dx in (-1, 0, 1):
            if dz == dx == 0: continue
            shifted = np.roll(a, (dz, dx), (0, 1))
            good = shifted == a
            if dz == 1: good[0] = False
            if dz == -1: good[-1] = False
            if dx == 1: good[:, 0] = False
            if dx == -1: good[:, -1] = False
            equal |= good
    seeds = np.argwhere(maxima & equal & (a > 0))
    visited = set()
    for z, x in seeds:
        key = (int(z), int(x))
        if key in visited: continue
        stack = [key]; component = []; peak = True; value = a[key]
        visited.add(key)
        while stack:
            zz, xx = stack.pop(); component.append((zz, xx))
            for dz in (-1, 0, 1):
                for dx in (-1, 0, 1):
                    nz, nx = zz+dz, xx+dx
                    if 0 <= nz < a.shape[0] and 0 <= nx < a.shape[1]:
                        if a[nz, nx] > value: peak = False
                        elif a[nz, nx] == value and (nz,nx) not in visited:
                            visited.add((nz,nx)); stack.append((nz,nx))
        if not peak:
            for coord in component: maxima[coord] = False
    if np.any(a > 0): maxima &= a > 0
    return maxima

def radial_centers(patches):
    """Return radial-symmetry offsets (z,x), patches (N,5,5), in pixel units."""
    a = np.asarray(patches)
    n, h, w = a.shape
    z, x = np.meshgrid(np.arange(h-1)-(h-2)/2, np.arange(w-1)-(w-2)/2, indexing='ij')
    du = a[:, :-1, 1:] - a[:, 1:, :-1]
    dv = a[:, :-1, :-1] - a[:, 1:, 1:]
    # MATLAB conv2(single,double) returns single; each patch has zero extension.
    kernel = np.ones((1,3,3))/9
    fu = ndimage.convolve(du, kernel, mode='constant', cval=0)
    fv = ndimage.convolve(dv, kernel, mode='constant', cval=0)
    g = fu*fu + fv*fv
    with np.errstate(divide='ignore', invalid='ignore'):
        m = -(fv+fu)/(fu-fv)
        fallback = (dv+du)/(du-dv)
        m = np.where(np.isnan(m), fallback, m)
        m = np.where(np.isnan(m), 0, m)
        finite = ~np.isinf(m)
        maximum = np.max(np.where(finite, m, -np.inf), axis=(1,2))
        m = np.where(np.isinf(m), 10*maximum[:,None,None], m)
        all_infinite = ~finite.any(axis=(1,2))
        m[all_infinite] = fallback[all_infinite]
        intercept = z - m*x
        total = g.sum(axis=(1,2))
        zc = (g*z).sum(axis=(1,2))/total
        xc = (g*x).sum(axis=(1,2))/total
        weight = g/np.sqrt((z-zc[:,None,None])**2+(x-xc[:,None,None])**2)
        q = weight/(m*m+1)
        sw=q.sum(axis=(1,2)); smmw=(m*m*q).sum(axis=(1,2)); smw=(m*q).sum(axis=(1,2))
        smbw=(m*intercept*q).sum(axis=(1,2)); sbw=(intercept*q).sum(axis=(1,2))
        det=smw*smw-smmw*sw
        xc=(smbw*sw-smw*sbw)/det; zc=(smbw*smw-smmw*sbw)/det
    return np.stack([zc,xc],axis=1)

def localize(magnitude, method='radial', particles=200, max_local_peaks=4):
    """Return [intensity,z,x,frame] in input real precision; z/x zero-based."""
    if method not in ('radial','wa','integer'): raise ValueError('Unknown method')
    a = np.abs(magnitude)
    h,w,t=a.shape
    reduced=np.zeros_like(a); reduced[3:-3,3:-3]=a[3:-3,3:-3]
    plane=reduced.transpose(0,2,1).reshape((h*t,w),order='F')
    mask=regional_maximum(plane).reshape((h,t,w),order='F').transpose(0,2,1)
    peaks=reduced*mask
    cutoff=np.sort(peaks.reshape((-1,t),order='F'),axis=0)[-(particles+1)]
    keep=(peaks>cutoff[None,None,:]) & mask
    z,x,frame=np.unravel_index(np.flatnonzero(keep.ravel(order='F')),a.shape,order='F')
    grid=np.arange(-2,3)
    patches=a[z[:,None,None]+grid[None,:,None],x[:,None,None]+grid[None,None,:],frame[:,None,None]]
    roi_max=patches==ndimage.maximum_filter(patches,size=(1,3,3),mode='constant',cval=-np.inf)
    count=roi_max.sum(axis=(1,2)); valid=count<=max_local_peaks
    plateau = (patches[:,1:,:] == patches[:,:-1,:]).any(axis=(1,2))
    plateau |= (patches[:,:,1:] == patches[:,:,:-1]).any(axis=(1,2))
    plateau |= (patches[:,1:,1:] == patches[:,:-1,:-1]).any(axis=(1,2))
    plateau |= (patches[:,1:,:-1] == patches[:,:-1,1:]).any(axis=(1,2))
    for index in np.flatnonzero(plateau):
        valid[index] = regional_maximum(patches[index]).sum() <= max_local_peaks
    if method=='radial': offsets=radial_centers(patches)
    elif method=='wa':
        total=patches.sum(axis=(1,2))
        offsets=np.stack([(patches*grid[None,:,None]).sum(axis=(1,2))/total,(patches*grid[None,None,:]).sum(axis=(1,2))/total],axis=1)
    else: offsets=np.zeros((len(z),2))
    # Preserve assignment in the input real precision; the selected path is double.
    positions=(offsets+np.stack([z+1,x+1],axis=1)).astype(a.dtype)
    valid &= np.isfinite(positions).all(axis=1) & (np.abs(offsets)<=1.5).all(axis=1)
    result=np.column_stack([a[z,x,frame],positions,frame+1]).astype(a.dtype)
    result[:,1:3]-=1
    return result[valid]
