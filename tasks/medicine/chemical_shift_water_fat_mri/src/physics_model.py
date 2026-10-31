"""Extracted explicit water-fat MRI mathematics.

Copyright (C) 2016-2026 Johan Berglund. GPL-3.0-or-later.
Adapted from author revision657d7e98ee5ca1b16fc16e34ed4625bfb6643912.
The generic QPBO library solves binary energies constructed explicitly here.
"""
import numpy as np
GYRO=42.576

def predict_echoes(coefficients, fieldmap, r2star, dPar, mPar):
    """Explicit complex signal in acquisition units; coefficients are first-echo normalized."""
    t=dPar.t1+np.arange(dPar.N)*dPar.dt
    base=model_matrix(dPar,mPar,0.)
    signal=np.einsum('nm,mxyz->nxyz',base,coefficients)
    shape=(-1,1,1,1)
    signal*=np.exp(-np.asarray(r2star)[None]*(t-t[0]).reshape(shape))
    signal*=np.exp(1j*2*np.pi*GYRO*dPar.B0*np.asarray(fieldmap)[None]*(t-t[0]).reshape(shape))
    return signal

def estimate_phi(Y, D, subscripts='nv,vnk,kv->v'):
    u1Tu2 = np.einsum(subscripts, Y, D, Y)
    return np.angle(-u1Tu2) / 2 # Berglund et al. 2020, eq. 11


def estimate_rho_realvalued(Y, pinv, R2, dB0, D):
    phi = estimate_phi(Y, D[R2, dB0]) # shape: (num_voxels)
    Y *= np.exp(-1j * phi) # demodulate phi
    rho = np.einsum('vmn,nv->mv', pinv[R2, dB0], np.concat((Y, Y.conj()))) # Berglund et al. 2020, eq. 4
    phi[rho[0] < 0] += np.pi # assert phi is the phase angle of water
    return rho * np.exp(1j * phi)


def estimate_rho(Y, pinv, R2, dB0, D=None):
    if D is None: # complex-valued estimates
        return np.einsum('vmn,nv->mv', pinv[R2, dB0], Y)
    return estimate_rho_realvalued(Y, pinv, R2, dB0, D) # real-valued estimates


def _R2_residuals(Y, dB0, null_proj, nB0, D=None):
    if D is None: # complex-valued estimates
        return np.real(np.einsum('mv,rvmk,kv->rv', Y.conj(), null_proj[:, dB0 % nB0], Y))
    else: # real-valued estimates
        phi = estimate_phi(Y, D[:, dB0], 'nv,rvnk,kv->vr') # shape: (num_voxels, nR2)
        y_demod = np.einsum('nv,vr->nvr', Y, np.exp(-1j * phi)) # demodulate phi
        y = np.concat((y_demod, y_demod.conj()), axis=0) # Berglund et al. 2020, eq. 7
        return np.real(np.einsum('mvr,rvmk,kvr->rv', y.conj(), null_proj[:, dB0 % nB0], y))


# Calculate LS error as function of R2*, shape: (nR2, num_voxels)
def R2_residuals(Y, dB0, null_proj, nB0, D=None, chunk_size=10**3):
    num_voxels = Y.shape[1]
    chunks = [slice(start, min(start + chunk_size, num_voxels)) for start in range(0, num_voxels, chunk_size)]
    residuals = [_R2_residuals(Y[:, chunk], dB0[chunk], null_proj, nB0, D) for chunk in chunks]
    return np.concat(residuals, axis=1)


def _B0_residuals(Y, null_proj, iR2cand, D=None):
    if D is None: # complex-valued estimates
        residuals = np.real(np.einsum('mv,rbmk,kv->bvr', Y.conj(), null_proj[iR2cand, :], Y))
    else: # real-valued estimates
        phi = estimate_phi(Y, D[iR2cand], 'nv,rbnk,kv->vrb') # shape: (num_voxels, nR2cand, nB0)
        y_demod = np.einsum('nv,vrb->nvrb', Y, np.exp(-1j * phi)) # demodulate phi
        y = np.concat((y_demod, y_demod.conj()), axis=0) # Berglund et al. 2020, eq. 7
        residuals = np.real(np.einsum('mvrb,rbmk,kvrb->bvr', y.conj(), null_proj[iR2cand, :], y))
    return np.min(residuals, axis=2) # minimize over R2* candidates


# Calculate LS error J as function of B0, shape: (nB0, num_voxels)
def B0_residuals(Y, null_proj, iR2cand, D=None, chunk_size=10**3):
    num_voxels = Y.shape[1]
    chunks = [slice(start, min(start + chunk_size, num_voxels)) for start in range(0, num_voxels, chunk_size)]
    residuals = [_B0_residuals(Y[:, chunk], null_proj, iR2cand, D) for chunk in chunks]
    return np.concat(residuals, axis=1)


# off-resonance stepsize in ppm
def B0_stepsize(dt, nB0, B0):
    return 1.0 / abs(dt) / nB0 / B0 / GYRO


def phi_estimation_matrix(null_proj, N):
    return np.einsum('rbkn,rbkj->rbnj', null_proj[..., :N, :N], null_proj[..., N:, :N])


def modulation_vectors(nB0, N, realEstimates=False):
    B = np.zeros((nB0, N, N), dtype=np.complex64)
    b = np.arange(nB0)[:, None]
    omega = 2.0 * np.pi * b / nB0
    n = np.arange(N)[None, :]
    B[b, n, n] = np.exp(1j * n * omega)
    if not realEstimates:
        return B
    Br = np.zeros((nB0, 2*N, 2*N), dtype=np.complex64)
    Br[:, :N, :N] = B
    Br[:, N:, N:] = B.conj()
    return Br


def model_matrix(dPar, mPar, R2):
    t = np.arange(dPar.N) * dPar.dt + dPar.t1
    omega = 2.0 * np.pi * GYRO * dPar.B0 * (mPar.CS - mPar.CS[0])
    exponentials = np.exp((1j * omega[None, None, :] - R2) * t[:, None, None] + R2 * dPar.t1)
    A = np.sum(mPar.alpha[None, :, :] * exponentials, axis=2, dtype=np.complex64)
    if mPar.realEstimates:
        A = np.concat((A, A.conj()), axis=0) # Berglund et al. 2020, eq. 3
    return A


def pseudoinverse_and_projection_matrices(dPar, aPar, mPar):
    B = modulation_vectors(aPar.nB0, dPar.N, mPar.realEstimates) # shape: (nB0, N, N), 2N if real-valued estimates
    pinv = np.empty(shape=(aPar.nR2, aPar.nB0, mPar.M, B.shape[1]), dtype=np.complex64)
    null_proj = np.empty(shape=(aPar.nR2, aPar.nB0, B.shape[1], B.shape[1]), dtype=np.complex64)
    
    for r in range(aPar.nR2):
        R2 = r * aPar.R2step
        A = model_matrix(dPar, mPar, R2)
        Ap = np.linalg.pinv(A)
        proj = np.eye(A.shape[0]) - np.dot(A, Ap) # Null space projection matrix for A
        null_proj[r] = np.einsum('bni,ij,bjk->bnk', B, proj, B.conj())
        pinv[r] = np.einsum('ij,bjk->bik', Ap, B.conj())
    return pinv, null_proj


