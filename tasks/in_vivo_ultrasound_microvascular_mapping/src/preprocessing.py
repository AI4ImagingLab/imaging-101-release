"""Observation loading and the original PALA temporal clutter filter.

Adapted from PALA, Arthur Chavignon, Baptiste Heiles, Vincent Hingot;
CNRS, Sorbonne Universite, INSERM; CC BY-NC-SA 4.0.
Full attribution and license terms are retained in test/validation.md.
"""
from pathlib import Path
import numpy as np
from scipy import linalg, signal

SHAPE = (30, 83, 128, 200)
FRAME_RATE = 500.0
BLOCK_RATE = 2.0

def load_observations(path='data/raw_data.npz'):
    """Return complex64 IQ (block,z,x,time), validating public batch-first input."""
    with np.load(Path(path), allow_pickle=False) as d:
        if set(d.files) != {'iq'}: raise ValueError('Expected only iq')
        iq = d['iq']
    if iq.shape != (1,) + SHAPE or iq.dtype != np.complex64: raise ValueError('IQ shape/dtype mismatch')
    if not np.isfinite(iq).all(): raise ValueError('IQ must be finite')
    return iq[0]

def original_boundary_zero(iq):
    """Preserve MATLAB IQ(:,[1 end])=0; IQ([1 end],:)=0 exactly."""
    x = np.array(iq, dtype=np.complex128, order='F', copy=True)
    flat = x.reshape((x.shape[0], -1), order='F')
    flat[:, [0, -1]] = 0
    flat[[0, -1], :] = 0
    return x

def svd_filter(iq, cutoff=10):
    """Remove first cutoff-1 temporal Gram singular vectors in input precision."""
    if iq.ndim != 3 or not 1 <= cutoff <= iq.shape[-1]: raise ValueError('Invalid IQ/cutoff')
    x = iq.reshape((-1, iq.shape[-1]), order='F')
    gram = x.conj().T @ x
    u, _, _ = linalg.svd(gram, full_matrices=False, lapack_driver='gesvd')
    modes = u[:, cutoff-1:]
    return ((x @ modes) @ modes.conj().T).reshape(iq.shape, order='F')

def temporal_filter(iq):
    """Second-order causal Butterworth 25..249Hz; zero state per complete block."""
    b, a = signal.butter(2, [25, 249], btype='bandpass', fs=FRAME_RATE)
    real_dtype = np.float32 if iq.dtype == np.complex64 else np.float64
    result = signal.lfilter(b.astype(real_dtype), a.astype(real_dtype), iq, axis=-1)
    result[~np.isfinite(result)] = 0
    return result

def preprocess_block(iq):
    """Return tissue- and temporally-filtered full block, without joining blocks."""
    return temporal_filter(svd_filter(original_boundary_zero(iq)))
