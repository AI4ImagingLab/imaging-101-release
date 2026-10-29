"""Explicit PCA stellar synthesis, finite broadening, and observed continuum.

Adapted mathematics: SteParSyn/convsyn, copyright2015-2026 Hugo Tabernero,
Emilio Marfil, Jonay Gonzalez Hernandez and David Montes (BSD2Clause; LICENSE).
Only NumPy/SciPy general linear algebra, interpolation geometry and convolution
are used. Known model/calibration operators contain no observed flux, inferred
noise, stellar fit, likelihood or posterior. build_operators reconstructs them.

Selected fixed acquisition: RV=0 km/s, R=85000, limb coefficient=.6, vmac=0,
extinction=0, absolute scale exponent=0; free Teff[kK], logg[cgs dex], MH[dex],
and positive projected rotation[km/s]. Original halfpixel and finite-fill1
conventions are preserved, including the unusual source velocity-grid endpoints.
"""
from __future__ import annotations

import numpy as np
from scipy.interpolate import BSpline, splrep, splev
from scipy.linalg.lapack import dgbtrf, dgbtrs
from scipy.ndimage import convolve1d
from scipy.spatial import Delaunay

LIGHT_SPEED = 299792.458
RESOLUTION = 85000.
LIMB_DARKENING = .6
OPERATOR_KEYS = ('operator_pre_basis','operator_pre_mean','operator_velocity_xk',
 'operator_wmid','operator_post_values','operator_gaussian_offset',
 'operator_velocity_offsets','operator_observation_offsets','operator_post_offsets',
 'operator_observation_indices')


class FixedCubic:
    """Not-a-knot cubic basis with an explicitly reusable collocation LU.

    scipy's B-spline basis evaluation and banded linear solve are general
    numerical primitives; the fixed knots, boundary rule and mapping are local.
    """
    def __init__(self, coordinates):
        self.x = np.asarray(coordinates, dtype=np.float64)
        if self.x.ndim != 1 or len(self.x)<4 or np.any(np.diff(self.x)<=0):
            raise ValueError('Cubic interpolation requires four increasing coordinates')
        self.knots = np.r_[np.repeat(self.x[0],4),self.x[2:-2],np.repeat(self.x[-1],4)]
        basis = BSpline.design_matrix(self.x,self.knots,3,extrapolate=True).tocoo()
        band = np.zeros((10,len(self.x)),dtype=np.float64,order='F')
        band[6+basis.row-basis.col,basis.col] = basis.data
        self.lu,self.piv,info = dgbtrf(band,3,3)
        if info: raise ValueError('Singular cubic collocation system')

    def operator(self, query):
        """Return value-at-query matrix, with the original cubic extrapolation."""
        coefficients,info = dgbtrs(self.lu,3,3,np.eye(len(self.x),order='F'),self.piv)
        if info: raise ValueError('Cubic collocation solve failed')
        evaluation = BSpline.design_matrix(query,self.knots,3,extrapolate=True)
        return evaluation@coefficients


def velocity_grid(wavelength):
    """Original equal-log-wavelength grid and geometric mean wavelength [Angstrom].

    Preserve npix and the source's omitted central/endpoint samples exactly;
    replacing this by an ostensibly cleaner linspace is a different discretization.
    """
    x = np.asarray(wavelength)
    first = np.where(x>min(x)+.1)[0][0]+1
    last = np.where(x>max(x)-.1)[0][0]-1
    w1,w2 = x[first],x[last]
    npix = np.int64(last-first)
    mid = np.sqrt(w1*w2)
    step = (np.log10(w2)-np.log10(w1))/(npix-1.)
    lower = np.log10(mid)-step*np.arange(np.int64(npix/2.))
    upper = np.log10(mid)+step*np.arange(np.int64(npix/2.))
    lower = np.sort(10.**lower[1:len(lower)-1])
    upper = 10.**upper[1:len(upper)-1]
    return np.concatenate((lower,[mid],upper)),mid


def finite_convolution(values, kernel):
    """Odd 1D kernel, finite boundary flux1, then original second normalization."""
    values = np.asarray(values,dtype=np.float64)
    kernel = np.asarray(kernel,dtype=np.float64)
    if values.ndim!=1 or kernel.ndim!=1 or len(kernel)%2!=1:
        raise ValueError('One-dimensional values and odd kernel required')
    if not np.isfinite(values).all() or not np.isfinite(kernel).all() or kernel.sum()<=0:
        raise ValueError('Finite arrays and positive kernel sum required')
    return np.convolve(np.pad(values,len(kernel)//2,constant_values=1.),kernel,'valid')/kernel.sum()


def rotation_kernel(xk, midpoint, projected_velocity):
    """Sample the selected limb-darkened rotation profile on original halfpixels."""
    if not np.isfinite(projected_velocity) or projected_velocity<=0:
        raise ValueError('Bound projected rotation must be positive')
    width = projected_velocity/LIGHT_SPEED*midpoint
    square = 1-(xk/width)**2
    square = np.where(square<=0,0.,square)
    factor = 1/(np.pi*(1-LIMB_DARKENING/3))
    kernel = 2*(1-LIMB_DARKENING)*factor*np.sqrt(square)+.5*np.pi*LIMB_DARKENING*factor*square
    total = kernel.sum()
    return kernel/total if total>0 else None


def build_operators(raw, progress=None):
    """Rebuild pure known-model/acquisition operators from unbatched raw arrays.

    Reads only wavelength coordinates, range endpoints, synthetic model mean,
    scale, PCA basis; deliberately never reads flux, uncertainty or fitted data.
    Returns unbatched arrays keyed by OPERATOR_KEYS. Runtime continuum fitting
    remains outside these linear affine operators. ``progress(index,total)`` is
    an optional maintainer callback and does not affect numerical operations.
    """
    model_wave = np.asarray(raw['model_wavelength'])
    observed_wave = np.asarray(raw['wavelength'])
    model_basis = np.asarray(raw['pca_basis'])
    model_mean,model_scale = np.asarray(raw['pca_mean']),np.asarray(raw['pca_scale'])
    ranges = np.asarray(raw['ranges'])
    pre,means,xks,mids,post,offsets,indices = [],[],[],[],[],[],[]
    vo,oo,po = [0],[0],[0]
    for number,(low,high) in enumerate(ranges):
        center,width = .5*(low+high),.5*(high-low)
        si = np.where(abs(model_wave-center)<=width)[0]
        oi = np.where(abs(observed_wave-center)<=width)[0]
        if not len(oi): raise ValueError('The bound benchmark requires every spectral range')
        x,xo = model_wave[si],observed_wave[oi]
        v,mid = velocity_grid(x)
        n = len(v)
        forward = FixedCubic(x).operator(v)
        backward = FixedCubic(v).operator(x)
        sampling = FixedCubic(x).operator(xo)
        xk = (np.arange(n)-n/2)*(v[-1]-v[0])/(n-1)
        sigma = (LIGHT_SPEED/RESOLUTION)/(2*np.sqrt(2*np.log(2)))*mid/LIGHT_SPEED
        gaussian = np.exp(-.5*(xk/sigma)**2)/(np.sqrt(2*np.pi)*sigma)
        gaussian = gaussian/gaussian.sum()
        divisor = gaussian.sum()
        affine_offset = finite_convolution(np.zeros(n),gaussian)
        lag = n//2+np.arange(n)[:,None]-np.arange(n)[None,:]
        valid = (lag>=0)&(lag<n)
        gaussian_matrix = np.where(valid,gaussian[np.clip(lag,0,n-1)],0.)/divisor
        after = (sampling@backward)@gaussian_matrix@(forward@backward)
        constant = (sampling@backward)@affine_offset
        pre.append((model_basis[:,si]*model_scale[si])@forward.T)
        means.append(forward@model_mean[si])
        xks.append(xk);mids.append(mid);post.append(after.ravel());offsets.append(constant);indices.append(oi)
        vo.append(vo[-1]+n);oo.append(oo[-1]+len(oi));po.append(po[-1]+after.size)
        if progress is not None:progress(number+1,len(ranges))
    return dict(operator_pre_basis=np.concatenate(pre,axis=1),operator_pre_mean=np.concatenate(means),
                operator_velocity_xk=np.concatenate(xks),operator_wmid=np.asarray(mids),
                operator_post_values=np.concatenate(post),operator_gaussian_offset=np.concatenate(offsets),
                operator_velocity_offsets=np.asarray(vo,dtype=np.int64),
                operator_observation_offsets=np.asarray(oo,dtype=np.int64),
                operator_post_offsets=np.asarray(po,dtype=np.int64),
                operator_observation_indices=np.concatenate(indices).astype(np.int64))


def normalize_continuum(synthetic, observed, wavelength):
    """Runtime source order0 upper-quartile log-median continuum, no stored fit."""
    center = .5*(max(wavelength)+min(wavelength))
    half = .5*(max(wavelength)-min(wavelength))
    selected = (abs(wavelength-center)<half-.5)&(synthetic>0)&(observed>0)
    if not np.any(selected): raise ValueError('No positive continuum support')
    simulated,data = synthetic[selected],observed[selected]
    high_sim = simulated>=np.percentile(simulated,75)
    high_data = data>=np.percentile(data,75)
    continuum = np.median(np.log(data[high_data]))-np.median(np.log(simulated[high_sim]))
    return synthetic*np.exp(continuum)


def source_order_broadening(wavelength,flux,broadening,limb,kind):
    """Uncomposed finite broadening for finite-difference-sensitive fitting.

    This evaluates the identical selected physics in its original arithmetic
    order: not-a-knot FITPACK interpolation, sampled odd kernel, convolution,
    second kernel normalization, inverse interpolation. scipy.ndimage's finite
    constant-fill1 convolution has independently matched all436original C
    convolution stages bit-for-bit in the shared2.4/BLAS1 environment. No
    Astropy or other domain module is imported by this implementation.
    """
    velocity,midpoint=velocity_grid(wavelength)
    tck=splrep(wavelength,flux,k=3,s=0)
    values=splev(velocity,tck,der=0)
    n=len(velocity)
    dx=(velocity[n-1]-velocity[0])/float(n-1)
    coordinate=(np.arange(n)-n/2)*dx
    if kind=='g':
        if broadening>1000.:
            width=(LIGHT_SPEED/broadening)/(2.*np.sqrt(2.*np.log(2.)))
        else:width=broadening
        sigma=width*midpoint/LIGHT_SPEED
        z=coordinate/sigma
        amplitude=1./np.sqrt(2.*np.pi)/sigma
        kernel=amplitude*np.exp(-(z**2.)/2.)
    elif kind=='r':
        width=broadening/LIGHT_SPEED*midpoint
        factor=1./(np.pi*(1.-limb/3.))
        c1=2.*(1.-limb)*factor
        c2=.5*np.pi*limb*factor
        square=1.-(coordinate/width)**2
        square[square<=0.]=0.
        kernel=c1*np.sqrt(square)+c2*square
    else:raise ValueError('The selected source-order path has rotation/Gaussian kernels only')
    total=np.sum(kernel)
    if total>0.:
        normalized=kernel/total
        broadened=convolve1d(values,normalized,mode='constant',cval=1.)/np.sum(normalized)
    else:broadened=values
    tck=splrep(velocity,broadened,k=3,s=0)
    return splev(wavelength,tck,der=0)


class SpectralModel:
    """Read-only known-model operators plus fresh observed continuum normalization.

    raw is a dictionary with leading benchmark batch axis already removed.
    forward([Teff[kK],logg,MH,vsini[km/s]],masked=False) returns a new52570-vector.
    Invalid physical/local-grid/hull inputs raise ValueError; never substitute
    a reference. forward_batch preserves proposal order and consumes no RNG.
    """
    def __init__(self,raw):
        self.raw=raw
        self.points=np.asarray(raw['parameter_grid'])
        self.point_offset=self.points.mean(axis=0)
        self.point_scale=np.ptp(self.points,axis=0)
        if np.any(self.point_scale<=0): raise ValueError('Degenerate physical model grid')
        self.triangulation=Delaunay((self.points-self.point_offset)/self.point_scale)
        self.coefficients=np.asarray(raw['pca_weights'])
        if not all(k in raw for k in OPERATOR_KEYS):raise ValueError('Known operators missing; run build_operators(raw)')
        self.operators={key:np.asarray(raw[key]) for key in OPERATOR_KEYS}
        if not np.all(np.asarray(raw['continuum_order'])==0):raise ValueError('Selected model requires source order0 continuum')
        self.wavelength=np.asarray(raw['wavelength'])
        self.observed=np.asarray(raw['flux'])
        self.mask_indices=np.concatenate([np.where(abs(self.wavelength-center)<=width)[0] for center,width in raw['masks']])

    def interpolate_weights(self,parameters):
        """Explicit affine barycentric interpolation on the rescaled Delaunay grid."""
        p=np.asarray(parameters,dtype=np.float64)
        if p.shape!=(4,) or not np.isfinite(p).all() or p[3]<=0:raise ValueError('Invalid stellar parameters')
        selected=abs(self.points[:,0]-p[0])<=.3
        if np.unique(self.points[selected,0]).size<2:selected=abs(self.points[:,0]-p[0])<=.5
        subset=self.points[selected]
        if len(subset)<=1 or np.any(np.ptp(subset,axis=0)==0) or np.any(p[:3]<subset.min(0)) or np.any(p[:3]>subset.max(0)):
            raise ValueError('Stellar parameters outside original local-grid bounds')
        query=(p[:3]-self.point_offset)/self.point_scale
        simplex=int(self.triangulation.find_simplex(query))
        if simplex<0:raise ValueError('Stellar parameters outside interpolation hull')
        transform=self.triangulation.transform[simplex]
        delta=query-transform[3]
        bary=np.zeros(4,dtype=np.float64)
        for i in range(3):
            for j in range(3):bary[i]+=transform[i,j]*delta[j]
        bary[3]=1-bary[:3].sum()
        values=self.coefficients[self.triangulation.simplices[simplex]]
        weights=np.zeros(values.shape[1],dtype=np.float64)
        for i in range(4):weights+=bary[i]*values[i]
        return weights

    def _spectrum_from_velocity(self,velocity_flux,vsini):
        op=self.operators;parts=[]
        for i,mid in enumerate(op['operator_wmid']):
            v0,v1=op['operator_velocity_offsets'][i:i+2]
            o0,o1=op['operator_observation_offsets'][i:i+2]
            p0,p1=op['operator_post_offsets'][i:i+2]
            values=velocity_flux[v0:v1]
            kernel=rotation_kernel(op['operator_velocity_xk'][v0:v1],mid,vsini)
            rotated=values if kernel is None else finite_convolution(values,kernel)
            post=op['operator_post_values'][p0:p1].reshape(o1-o0,v1-v0)
            sampled=post@rotated+op['operator_gaussian_offset'][o0:o1]
            indices=op['operator_observation_indices'][o0:o1]
            parts.append(normalize_continuum(sampled,self.observed[indices],self.wavelength[indices]))
        result=np.concatenate(parts)
        if not np.isfinite(result).all():raise ValueError('Nonfinite synthetic spectrum')
        return result

    def forward(self,parameters,masked=False):
        p=np.asarray(parameters,dtype=np.float64)
        weights=self.interpolate_weights(p)
        op=self.operators
        velocity=op['operator_pre_mean']+weights@op['operator_pre_basis']
        result=self._spectrum_from_velocity(velocity,p[3])
        return result[self.mask_indices] if masked else result

    def forward_batch(self,parameters,masked=False):
        """Fresh ordered proposal spectra; only known PCA linear algebra is batched."""
        p=np.asarray(parameters,dtype=np.float64)
        if p.ndim!=2 or p.shape[1]!=4:raise ValueError('Expected proposal array[n,4]')
        weights=np.stack([self.interpolate_weights(row) for row in p])
        op=self.operators
        velocity=op['operator_pre_mean'][None]+weights@op['operator_pre_basis']
        result=np.stack([self._spectrum_from_velocity(v,row[3]) for v,row in zip(velocity,p)])
        return result[:,self.mask_indices] if masked else result

    def interpolate_weights_fidelity(self,parameters):
        """Original sequential fourth-barycentric subtraction, no regrouped sum.

        Kept separate from the already-validated fast arithmetic identity.
        Original Qhull interpolation weights match bit-for-bit in independent
        initial/finite-difference cases when the remaining coordinate is reduced
        successively rather than computed as1-sum(first_three).
        """
        p=np.asarray(parameters,dtype=np.float64)
        self.interpolate_weights(p)  # Same physical/local-grid/hull validation.
        query=(p[:3]-self.point_offset)/self.point_scale
        simplex=int(self.triangulation.find_simplex(query))
        transform=self.triangulation.transform[simplex]
        delta=query-transform[3]
        bary=np.zeros(4);bary[3]=1.
        for i in range(3):
            for j in range(3):bary[i]+=transform[i,j]*delta[j]
            bary[3]-=bary[i]
        values=self.coefficients[self.triangulation.simplices[simplex]]
        weights=np.zeros(values.shape[1])
        for i in range(4):weights+=bary[i]*values[i]
        return weights

    def forward_fidelity(self,parameters,masked=False):
        """Source-order forward for robust LS/noise preparation, all218ranges.

        Preserves arithmetic grouping needed for sqrtEPS finite-difference
        Jacobians and early stopping. It is freshly computed from public raw
        model tables and observations, not a lookup of fitted/reference values.
        fast forward/forward_batch remain the separate explicit affine path.
        """
        p=np.asarray(parameters,dtype=np.float64)
        weights=self.interpolate_weights_fidelity(p)
        raw=self.raw
        synthetic=raw['pca_mean']+(raw['pca_scale']*np.dot(weights,raw['pca_basis']))
        synthetic=synthetic*10.**0.
        wave=raw['model_wavelength']*(1.+0./LIGHT_SPEED)
        pieces=[]
        for lo,hi in raw['ranges']:
            center,width=.5*(lo+hi),.5*(hi-lo)
            si=np.abs(wave-center)<=width
            oi=np.abs(self.wavelength-center)<=width
            x=wave[si];xo=self.wavelength[oi]
            rotated=source_order_broadening(x,synthetic[si],p[3],LIMB_DARKENING,'r')
            broadened=source_order_broadening(x,rotated,RESOLUTION,0.,'g')
            sampled=splev(xo,splrep(x,broadened,k=3,s=0),der=0)
            pieces.append(normalize_continuum(sampled,self.observed[oi],xo))
        result=np.concatenate(pieces)
        if not np.isfinite(result).all():raise ValueError('Nonfinite source-order spectrum')
        return result[self.mask_indices] if masked else result
