#!/usr/bin/env python
"""End-to-end ptychographic reconstruction, pure numpy.

Usage: python scripts/script_003.py <input_dir> <output_dir>
<input_dir> contains raw_data.npz (batch-dim-first).
Writes <output_dir>/case_000.npz with array 'object_phase'.
"""
import os
import sys
import time

import numpy as np


def load_raw_data(input_dir):
    archive = np.load(os.path.join(input_dir, "raw_data.npz"))
    return {
        "diffraction_patterns": archive["diffraction_patterns"][0],
        "scan_positions": archive["scan_positions"][0],
        "probe_guess": archive["probe_guess"][0],
    }


def shift_scan_positions(scan, offset=20.0):
    scan_shifted = scan.copy()
    scan_shifted -= np.amin(scan_shifted, axis=-2) - offset
    return scan_shifted


def initialize_psi(scan, probe_shape, n_slices=1, buffer=2, fill_value=0.5 + 0j):
    scan_max = np.amax(scan, axis=-2)
    probe_h, probe_w = probe_shape[-2], probe_shape[-1]
    psi_shape = (
        n_slices,
        int(np.ceil(scan_max[0])) + probe_h + buffer,
        int(np.ceil(scan_max[1])) + probe_w + buffer,
    )
    return np.full(psi_shape, dtype=np.complex64, fill_value=np.complex64(fill_value))


def extract_patches(psi, scan, probe_shape):
    ph, pw = probe_shape
    obj = psi[-1]
    n = scan.shape[0]
    patches = np.empty((n, ph, pw), dtype=psi.dtype)
    positions = scan.astype(np.int32)
    for i in range(n):
        r, c = int(positions[i, 0]), int(positions[i, 1])
        patches[i] = obj[r:r + ph, c:c + pw]
    return patches


def rescale_probe(data, psi, probe, scan):
    ph, pw = probe.shape[-2:]
    n_modes = probe.shape[2]
    n_sample = min(50, scan.shape[0])
    sample_scan = scan[:n_sample]
    patches = extract_patches(psi, sample_scan, (ph, pw))
    probe_modes = probe[0, 0]
    sim_intensity = np.zeros((n_sample, ph, pw), dtype=np.float32)
    for s in range(n_modes):
        exit_wave = probe_modes[s] * patches
        farplane = np.fft.fft2(exit_wave, norm="ortho")
        sim_intensity += np.abs(farplane) ** 2
    data_mag = np.sqrt(np.mean(data[:n_sample]))
    sim_mag = np.sqrt(np.mean(sim_intensity)) + 1e-12
    probe *= float(data_mag / sim_mag)


def update_batch(data, scan, psi, probe, alpha=0.05, update_probe=True):
    ph, pw = probe.shape[-2:]
    n_modes = probe.shape[2]
    n = data.shape[0]
    obj = psi[-1]
    positions = scan.astype(np.int32)
    patches = extract_patches(psi, scan, (ph, pw))
    probe_modes = probe[0, 0]

    total_intensity = np.zeros((n, ph, pw), dtype=np.float32)
    farplanes = []
    for s in range(n_modes):
        exit_wave = probe_modes[s] * patches
        fp = np.fft.fft2(exit_wave, norm="ortho")
        farplanes.append(fp)
        total_intensity += np.abs(fp) ** 2

    sqrt_sim = np.sqrt(total_intensity + 1e-12)
    sqrt_data = np.sqrt(data + 1e-12)
    cost = float(np.mean((sqrt_sim - sqrt_data) ** 2))
    weight = 1.0 - sqrt_data / sqrt_sim

    obj_num = np.zeros_like(obj)
    obj_den = np.zeros(obj.shape, dtype=np.float32)
    probe_num = np.zeros_like(probe_modes)
    probe_den = np.zeros(probe_modes.shape[:-2] + probe_modes.shape[-2:], dtype=np.float32)

    for s in range(n_modes):
        grad_fp = farplanes[s] * weight
        chi = np.fft.ifft2(grad_fp, norm="ortho")
        obj_grad_patches = np.conj(probe_modes[s]) * chi
        probe_grad = np.conj(patches) * chi
        for i in range(n):
            r, c = int(positions[i, 0]), int(positions[i, 1])
            obj_num[r:r + ph, c:c + pw] += obj_grad_patches[i]
            obj_den[r:r + ph, c:c + pw] += np.abs(probe_modes[s]) ** 2
        probe_num[s] += np.sum(probe_grad, axis=0)
        probe_den[s] += np.sum(np.abs(patches) ** 2, axis=0)

    max_den = np.max(obj_den) + 1e-12
    denom = (1 - alpha) * obj_den + alpha * max_den
    psi[-1] = obj - obj_num / (denom + 1e-12)

    if update_probe:
        for s in range(n_modes):
            p_max_den = np.max(probe_den[s]) + 1e-12
            p_denom = (1 - alpha) * probe_den[s] + alpha * p_max_den
            probe[0, 0, s] = probe_modes[s] - probe_num[s] / (p_denom + 1e-12)

    return cost, psi, probe


def reconstruct(data, scan, probe, psi, num_iter=64, num_batch=7, alpha=0.05):
    rescale_probe(data, psi, probe, scan)
    n = data.shape[0]
    costs = []
    for epoch in range(num_iter):
        indices = np.random.permutation(n)
        batches = np.array_split(indices, num_batch)
        epoch_cost = 0.0
        for batch in batches:
            cost, psi, probe = update_batch(
                data[batch], scan[batch], psi, probe,
                alpha=alpha, update_probe=(epoch >= 1),
            )
            epoch_cost += cost
        epoch_cost /= num_batch
        costs.append([epoch_cost])
    return psi, probe, costs


def main():
    if len(sys.argv) != 3:
        print(__doc__)
        sys.exit(1)
    input_dir, output_dir = sys.argv[1], sys.argv[2]
    raw = load_raw_data(input_dir)
    data = raw["diffraction_patterns"]
    scan = raw["scan_positions"]
    probe = raw["probe_guess"]
    scan = shift_scan_positions(scan, offset=20.0)
    psi = initialize_psi(scan, probe.shape)
    print("object frame:", psi.shape)
    t0 = time.time()
    psi, probe, costs = reconstruct(data, scan, probe, psi, num_iter=64, num_batch=7)
    print("done in %.1fs; first/last cost %.4f / %.4f"
          % (time.time() - t0, costs[0][0], costs[-1][0]))
    phase = np.angle(psi[0])
    assert np.isfinite(phase).all(), "non-finite phase values"
    assert phase.shape == (482, 479), "unexpected object frame %s" % (phase.shape,)
    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, "case_000.npz")
    np.savez(out_path, object_phase=phase.astype(np.float32))
    print("saved", out_path)


if __name__ == "__main__":
    main()
