# Batch update reference (pure numpy, no GPU required)

def update_batch(data, scan, psi, probe, alpha=0.05, update_probe=True):
    ph, pw = probe.shape[-2:]
    n_modes = probe.shape[2]
    n = data.shape[0]
    obj = psi[-1]
    positions = scan.astype(np.int32)
    patches = extract_patches(psi, scan, (ph, pw))          # (N, ph, pw)
    probe_modes = probe[0, 0]                                # (S, ph, pw)

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
    weight = 1.0 - sqrt_data / sqrt_sim                     # (N, ph, pw)

    obj_num = np.zeros_like(obj)
    obj_den = np.zeros(obj.shape, dtype=np.float32)
    probe_num = np.zeros_like(probe_modes)
    probe_den = np.zeros(probe_modes.shape[:-2] + probe_modes.shape[-2:], dtype=np.float32)

    for s in range(n_modes):
        grad_fp = farplanes[s] * weight
        chi = np.fft.ifft2(grad_fp, norm="ortho")          # (N, ph, pw)
        obj_grad_patches = np.conj(probe_modes[s]) * chi
        probe_grad = np.conj(patches) * chi
        for i in range(n):
            r, c = int(positions[i, 0]), int(positions[i, 1])
            obj_num[r:r+ph, c:c+pw] += obj_grad_patches[i]
            obj_den[r:r+ph, c:c+pw] += np.abs(probe_modes[s]) ** 2
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

Notes:
- All arithmetic in complex64/float32 as inherited from the arrays.
- The scatter-add loops are the exact adjoint of patch extraction; the object
  numerator/denominator share the same scatter pattern, which is what makes
  the preconditioner the correct per-pixel Hessian estimate.
- alpha=0.05 blends the per-pixel denominator with its global max ("lstsq"
  step-size stabilization); both 1e-12 guards prevent division by zero.
