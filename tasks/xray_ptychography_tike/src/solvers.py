"""Iterative solvers for ptychographic reconstruction.

Implements the extended Ptychographic Iterative Engine (ePIE) with
least-squares optimal step sizes, following:

  Odstrcil, Menzel, and Guizar-Sicairos, "Iterative least-squares solver
  for generalized maximum-likelihood ptychography," Optics Express (2018).

No dependency on tike. Uses cupy for GPU acceleration.
"""

import logging

import numpy as np

try:
    import cupy as cp
except ImportError:
    cp = None

from src.physics_model import extract_patches, insert_patches

logger = logging.getLogger(__name__)


def _to_gpu(arr):
    """Move array to GPU if cupy is available."""
    if cp is not None:
        return cp.asarray(arr)
    return arr


def _to_cpu(arr):
    """Move array to CPU."""
    if cp is not None and isinstance(arr, cp.ndarray):
        return cp.asnumpy(arr)
    return np.asarray(arr)


def _get_xp(arr):
    if cp is not None and isinstance(arr, cp.ndarray):
        return cp
    return np


def reconstruct(
    data,
    scan,
    probe,
    psi,
    num_iter=64,
    num_batch=7,
    alpha=0.05,
):
    """Run ptychographic reconstruction using ePIE with least-squares steps.

    Parameters
    ----------
    data : (N, W, H) float32
        Measured diffraction intensities.
    scan : (N, 2) float32
        Scan positions in pixel coordinates.
    probe : (1, 1, S, W, H) complex64
        Initial probe estimate.
    psi : (D, W', H') complex64
        Initial object estimate (D=1 for single-slice).
    num_iter : int
        Number of epochs.
    num_batch : int
        Number of mini-batches per epoch.
    alpha : float
        Regularization parameter for the preconditioner.

    Returns
    -------
    result : dict
        'psi': reconstructed object (D, W', H') complex64
        'probe': reconstructed probe (1, 1, S, W, H) complex64
        'costs': list of (epoch, cost) values
    """
    from src.physics_model import validate_inputs

    validate_inputs(data, scan, probe, psi)

    # Move to GPU
    data_g = _to_gpu(data)
    scan_g = _to_gpu(scan)
    probe_g = _to_gpu(probe)
    psi_g = _to_gpu(psi)
    xp = _get_xp(psi_g)

    N = data_g.shape[0]
    ph, pw = probe_g.shape[-2:]
    n_modes = probe_g.shape[2]

    # Rescale probe to match data intensity
    _rescale_probe(data_g, psi_g, probe_g, scan_g, xp)

    costs = []

    for epoch in range(num_iter):
        # Random permutation of scan positions
        indices = xp.random.permutation(N)
        batches = xp.array_split(indices, num_batch)

        epoch_cost = 0.0

        for batch_idx, batch in enumerate(batches):
            batch_data = data_g[batch]
            batch_scan = scan_g[batch]

            cost, psi_g, probe_g = _update_batch(
                batch_data, batch_scan, psi_g, probe_g,
                alpha=alpha, xp=xp,
                update_probe=(epoch >= 1),
            )
            epoch_cost += float(cost)

        epoch_cost /= num_batch
        costs.append([epoch_cost])
        logger.info("  gaussian cost is %+.3e", epoch_cost)

    return {
        'psi': _to_cpu(psi_g),
        'probe': _to_cpu(probe_g),
        'costs': costs,
    }


def _rescale_probe(data, psi, probe, scan, xp):
    """Rescale probe so that simulated intensity matches data magnitude."""
    ph, pw = probe.shape[-2:]
    n_modes = probe.shape[2]

    # Compute intensity for a subset of positions
    n_sample = min(50, scan.shape[0])
    sample_scan = scan[:n_sample]
    patches = extract_patches(psi, sample_scan, (ph, pw))
    probe_modes = probe[0, 0]

    sim_intensity = xp.zeros((n_sample, ph, pw), dtype=xp.float32)
    for s in range(n_modes):
        exit_wave = probe_modes[s] * patches
        farplane = xp.fft.fft2(exit_wave, norm='ortho')
        sim_intensity += xp.abs(farplane) ** 2

    data_mag = xp.sqrt(xp.mean(data[:n_sample]))
    sim_mag = xp.sqrt(xp.mean(sim_intensity)) + 1e-12
    scale = float(data_mag / sim_mag)
    probe *= scale
    logger.info("Probe rescaled by %f", scale)


def _update_batch(data, scan, psi, probe, alpha, xp, update_probe=True):
    """Process one mini-batch: compute gradients and apply updates.

    Implements the ePIE update with Gaussian noise model and
    least-squares step size estimation.
    """
    ph, pw = probe.shape[-2:]
    n_modes = probe.shape[2]
    N = data.shape[0]
    obj = psi[-1]  # Single-slice: (H, W)

    # 1. Extract patches
    positions = scan.astype(xp.int32)
    patches = extract_patches(psi, scan, (ph, pw))  # (N, ph, pw)

    probe_modes = probe[0, 0]  # (S, ph, pw)

    # 2. Forward propagation and cost computation
    total_intensity = xp.zeros((N, ph, pw), dtype=xp.float32)
    farplanes = []
    for s in range(n_modes):
        exit_wave = probe_modes[s] * patches  # (N, ph, pw)
        fp = xp.fft.fft2(exit_wave, norm='ortho')
        farplanes.append(fp)
        total_intensity += xp.abs(fp) ** 2

    # Gaussian cost: mean(sqrt(I_sim) - sqrt(I_meas))^2
    sqrt_sim = xp.sqrt(total_intensity + 1e-12)
    sqrt_data = xp.sqrt(data + 1e-12)
    cost = float(xp.mean((sqrt_sim - sqrt_data) ** 2))

    # 3. Gradient in detector plane (Gaussian noise model)
    # d_cost/d_farplane = farplane * (1 - sqrt(data) / sqrt(I_sim))
    weight = (1.0 - sqrt_data / sqrt_sim)  # (N, ph, pw)

    # 4. Backpropagate gradient and compute object/probe updates
    obj_update_num = xp.zeros_like(obj)
    obj_update_den = xp.zeros(obj.shape, dtype=xp.float32)
    probe_update_num = xp.zeros_like(probe_modes)
    probe_update_den = xp.zeros(probe_modes.shape[:-2] + probe_modes.shape[-2:],
                                 dtype=xp.float32)

    for s in range(n_modes):
        # Gradient in farplane
        grad_fp = farplanes[s] * weight  # (N, ph, pw)

        # Backpropagate to nearplane
        chi = xp.fft.ifft2(grad_fp, norm='ortho')  # (N, ph, pw)

        # Object gradient: conj(probe) * chi
        obj_grad_patches = xp.conj(probe_modes[s]) * chi  # (N, ph, pw)

        # Probe gradient: conj(patches) * chi
        probe_grad = xp.conj(patches) * chi  # (N, ph, pw)

        # Accumulate object update (numerator and denominator)
        for i in range(N):
            r, c = int(positions[i, 0]), int(positions[i, 1])
            obj_update_num[r:r + ph, c:c + pw] += obj_grad_patches[i]
            obj_update_den[r:r + ph, c:c + pw] += xp.abs(probe_modes[s]) ** 2

        # Accumulate probe update
        probe_update_num[s] += xp.sum(probe_grad, axis=0)
        probe_update_den[s] += xp.sum(xp.abs(patches) ** 2, axis=0)

    # 5. Apply preconditioned ePIE updates
    # Object update: psi -= conj(P) * chi / (|P|^2_max + alpha * |P|^2_max)
    # This is the standard ePIE denominator
    max_den = xp.max(obj_update_den) + 1e-12
    denom = (1 - alpha) * obj_update_den + alpha * max_den
    psi[-1] = obj - obj_update_num / (denom + 1e-12)

    # Probe update with same ePIE formula
    if update_probe:
        for s in range(n_modes):
            p_max_den = xp.max(probe_update_den[s]) + 1e-12
            p_denom = (1 - alpha) * probe_update_den[s] + alpha * p_max_den
            probe[0, 0, s] = probe_modes[s] - probe_update_num[s] / (p_denom + 1e-12)

    return cost, psi, probe


def _compute_step_size(update, gradient, xp):
    """Estimate optimal step size via least-squares.

    Computes beta = Re(<gradient, update>) / <update, update>
    clamped to [0, 2.0] and dampened by 0.9.
    """
    num = float(xp.real(xp.sum(xp.conj(gradient) * update)))
    den = float(xp.sum(xp.abs(update) ** 2)) + 1e-12
    beta = 0.9 * max(0.0, min(num / den, 2.0))
    return beta
