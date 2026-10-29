"""Local explicit extraction of the MIT-licensed deblatting method.
Copyright (c) 2022 Denys Rozumnyi; see test/validation.md, consolidated original licenses.
"""
import numpy as np
from numpy.fft import fft2, ifft2
from scipy import sparse
import scipy.sparse.linalg

from .preprocessing import renderTraj, boundingBox

def forward(background, kernel, appearance, mask):
    """Circular exposure model in normalized BGR intensity on the crop grid."""
    shape = background.shape[:2]
    spectrum = fft2(kernel)
    object_blur = ifft2(spectrum[:, :, None] * fft2(psfshift(appearance, shape), axes=(0, 1)), axes=(0, 1)).real
    occupancy = ifft2(spectrum * fft2(psfshift(mask, shape))).real
    return object_blur + background * (1-occupancy[:, :, None])

def psffit(H, returnpars=False):
    bH = H / np.max(H) > 0.4
    x, y = np.nonzero(bH)
    res1 = np.polyfit(x, y, 3, full=True)
    res2 = np.polyfit(y, x, 3, full=True)
    if res2[1].shape[0] == 0 or (res1[1].shape[0] != 0 and res1[1][0] < res2[1][0]):
        coeffs = res1[0]
        xs = [np.min(x), np.max(x)]
        poly1d_fn = np.poly1d(coeffs)
        ys = poly1d_fn(xs)
    else:
        coeffs = res2[0]
        ys = [np.min(y), np.max(y)]
        poly1d_fn = np.poly1d(coeffs)
        xs = poly1d_fn(ys)
    pars = np.array([[xs[0], ys[0]], [xs[1] - xs[0], ys[1] - ys[0]]]).T
    Hf = renderTraj(pars, np.zeros(H.shape))
    Hf /= np.sum(Hf)
    if returnpars:
        return (Hf, pars)
    return Hf

def createDerivatives0(sz):
    N_in = sz[0] * sz[1]
    N_out = (sz[0] + 1) * (sz[1] + 1)
    idx_in = np.reshape(range(N_in), sz[:2], 'F')
    idx_out = np.reshape(range(N_out), np.array(sz[:2]) + 1, 'F')
    v1 = np.ones((sz[0], sz[1] - 1))
    v2 = np.ones((sz[0], 1))
    inds = idx_out[:-1, np.r_[:idx_out.shape[1] - 1, 1:idx_out.shape[1]]]
    index = idx_in[:, np.r_[0, :idx_in.shape[1] - 1, 1:idx_in.shape[1], idx_in.shape[1] - 1]]
    values = np.hstack((v2, -v1, v1, -v2))
    Dx = sparse.csc_matrix((values.flatten('F'), (inds.flatten('F'), index.flatten('F'))), shape=(N_out, N_in))
    v1 = np.ones((sz[0] - 1, sz[1]))
    v2 = np.ones((1, sz[1]))
    inds = idx_out[np.r_[:idx_out.shape[0] - 1, 1:idx_out.shape[0]], :-1]
    index = idx_in[np.r_[0, :idx_in.shape[0] - 1, 1:idx_in.shape[0], idx_in.shape[0] - 1], :]
    values = np.vstack((v2, -v1, v1, -v2))
    Dy = sparse.csc_matrix((values.flatten('F'), (inds.flatten('F'), index.flatten('F'))), shape=(N_out, N_in))
    return (Dx, Dy)

def createRnMatrix(img_sz, angles=None):
    if angles is None:
        angles = np.array([16, 25, 78, 152]) / 180 * np.pi
    img_sz = np.array(img_sz[:2])
    idx2, idx1 = np.meshgrid(range(img_sz[1]), range(img_sz[0]))
    offset = (img_sz + 1) / 2
    idx = np.c_[idx1.T.flatten('F') + 1, idx2.T.flatten('F') + 1] - offset
    idx_out = np.zeros((0,)).astype(int)
    idx_in = np.zeros((0,)).astype(int)
    for ki in range(len(angles)):
        ca = np.cos(angles[ki])
        sa = np.sin(angles[ki])
        R = np.array([[ca, -sa], [sa, ca]])
        res = np.round(idx @ R + offset).astype(int)
        keep = np.all(np.logical_and(res >= 1, res <= img_sz), 1)
        idx_out = np.r_[idx_out, np.nonzero(keep)[0]]
        res -= 1
        idx_in = np.r_[idx_in, res[keep, 0] * img_sz[0] + res[keep, 1]]
    values = np.repeat(1 / len(angles), idx_in.shape[0])
    R = sparse.csc_matrix((values, (idx_out, idx_in)), shape=(np.prod(img_sz), np.prod(img_sz)))
    temp = np.sum(R, 1)
    ww = np.nonzero(temp > 0)[0]
    R[ww, :] = R[ww, :] / temp[ww]
    return R

def project2pyramid_exact(m, f, eps):
    """Exact Euclidean projection for eps=1 by enumerating active color bounds.

    Corrects the upstream aliasing error in the nominal Dykstra projection.
    Conditional on opacity t, each color is clip(f,0,t). The scalar derivative
    vanishes at (m + sum(active colors))/(1 + number of active colors).
    """
    if eps != 1:
        raise ValueError('The validated exact projector requires eps=1.')
    flat = np.asarray(m).reshape(-1)
    positive = np.maximum(f, 0)
    ordered = np.sort(positive, axis=1)[:, ::-1]
    cumulative = np.c_[np.zeros(len(flat)), np.cumsum(ordered, axis=1)]
    candidates = (flat[:, None] + cumulative)/(1+np.arange(f.shape[1]+1))
    lower = np.c_[ordered, np.zeros(len(flat))]
    upper = np.c_[np.full(len(flat), np.inf), ordered]
    active = (candidates >= lower) & (candidates <= upper)
    # If the unconstrained optimum is negative, the bounded optimum is zero.
    index = np.argmax(active, axis=1)
    opacity = np.clip(candidates[np.arange(len(flat)), index], 0, 1)
    color = np.minimum(positive, opacity[:, None])
    return opacity.reshape(np.shape(m)), color


def project2pyramid(m, f, eps):
    """Upstream cyclic approximation, retained only for explicit parity runs.

    The mf_old alias is intentional here: correcting it would invalidate the
    frozen upstream reproduction. Standard reconstruction uses the exact
    projector above. This function is not an exact Euclidean projection.
    """
    maxiter = 10
    onedimm = False
    if len(m.shape) == 1:
        onedimm = True
        m = m[:, np.newaxis]
    mf = np.concatenate((m, f), 1)
    N = mf.shape[1]
    Z = np.zeros((mf.shape[0], mf.shape[1], N))
    normal = np.array([-1, eps]) / np.sqrt(1 + eps ** 2)
    for iter in range(N * maxiter + 1):
        mf_old = mf
        idx = np.mod(iter, N)
        if idx == 0:
            mf += Z[:, :, 0]
            mf[mf < 0] = 0
            mf[mf[:, 0] > 1, 0] = 1
        else:
            mf += Z[:, :, idx]
            W = mf[:, idx] * eps > mf[:, 0]
            proj = np.c_[mf[W, 0], mf[W, idx]] @ normal
            mf[W, 0] -= normal[0] * proj
            mf[W, idx] -= normal[1] * proj
        Z[:, :, idx] = mf_old + Z[:, :, idx] - mf
    if onedimm:
        m = mf[:, 0]
    else:
        m = mf[:, :1]
    f = mf[:, 1:]
    return (m, f)

def lp_proximal_mapping(val_norm, amount, p):
    shrink_factor = np.zeros(val_norm.shape)
    if p == 1:
        nz = val_norm > amount
        shrink_factor[nz] = (val_norm[nz] - amount) / val_norm[nz]
    elif p == 1 / 2:
        nz = val_norm > 3 / 2 * val_norm ** (2 / 3)
        shrink_factor[nz] = 2 / 3 * val_norm[nz] * (1 + np.cos(2 / 3 * np.acos(-3 ** (3 / 2) / 4 * amount * val_norm[nz] ** (-3 / 2)))) / val_norm[nz]
    else:
        raise Exception('not implemented!')
    return shrink_factor

def proj2simplex(Y):
    Yf = Y.flatten('F')
    X = -np.sort(-Yf)
    temp = (np.cumsum(X) - 1) / np.array(range(1, len(X) + 1))
    X = np.reshape(Yf - temp[np.nonzero(X > temp)[0][-1]], Y.shape, 'F')
    X[X < 0] = 0
    return X

def psfshift(H, usize):
    hsize = H.shape
    usize = usize[:2]
    if len(hsize) > 3:
        Hp = np.zeros((usize[0], usize[1], hsize[2], hsize[3]))
        Hp[:hsize[0], :hsize[1], :, :] = H
    elif len(hsize) > 2:
        Hp = np.zeros((usize[0], usize[1], hsize[2]))
        Hp[:hsize[0], :hsize[1], :] = H
    else:
        Hp = np.zeros((usize[0], usize[1]))
        Hp[:hsize[0], :hsize[1]] = H
    shift = tuple((-np.ceil((np.array(hsize[:2]) + 1) / 2) + 1).astype(int))
    Hr = np.roll(Hp, shift, axis=(0, 1))
    return Hr

def ipsfshift(H, hsize):
    shift = tuple((np.ceil((np.array(hsize[:2]) + 1) / 2) - 1).astype(int))
    Hr = np.roll(H, shift, axis=(0, 1))
    Hc = Hr[:hsize[0], :hsize[1], :]
    return Hc

def psfshift_idx(small, sz_large):
    if type(small) == tuple:
        temp = np.reshape(np.array(range(1, np.prod(small) + 1)), small, 'F').astype(int)
    else:
        temp = np.zeros(small.shape).astype(int)
        temp[small] = np.array(range(1, np.count_nonzero(small) + 1))
    temp = psfshift(temp, sz_large).astype(int)
    if len(temp.shape) == 2:
        idx, idy = np.nonzero(temp)
        temp_idx = temp[idx, idy]
        pos = np.zeros(temp_idx.shape).astype(int)
        pos[temp_idx - 1] = range(len(temp_idx))
        idx = idx[pos]
        idy = idy[pos]
        return (idx, idy)
    elif len(temp.shape) == 3:
        idx, idy, idz = np.nonzero(temp)
        temp_idx = temp[idx, idy, idz]
        pos = np.zeros(temp_idx.shape).astype(int)
        pos[temp_idx - 1] = range(len(temp_idx))
        idx = idx[pos]
        idy = idy[pos]
        idz = idz[pos]
        return (idx, idy, idz)
    elif len(temp.shape) == 4:
        idx, idy, idz, idf = np.nonzero(temp)
        temp_idx = temp[idx, idy, idz, idf]
        pos = np.zeros(temp_idx.shape).astype(int)
        pos[temp_idx - 1] = range(len(temp_idx))
        idx = idx[pos]
        idy = idy[pos]
        idz = idz[pos]
        idf = idf[pos]
        return (idx, idy, idz, idf)

def vec3(I):
    return np.reshape(I, (I.shape[0] * I.shape[1], -1), 'F')

def ivec3(I, ishape):
    return np.reshape(I, ishape, 'F')
