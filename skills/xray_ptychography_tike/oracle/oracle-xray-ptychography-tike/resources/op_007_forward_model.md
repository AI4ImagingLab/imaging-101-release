# Forward model pseudocode and FFT convention

Measurement model per scan position j:
    I_j = | F[ P . O_j ] |^2

Implementation (numpy):

    patches  = extract_patches(psi, scan, (ph, pw))   # (N, ph, pw) complex64
    probe_s  = probe[0, 0]                            # (S, ph, pw)
    I        = np.zeros((N, ph, pw), np.float32)
    farplanes = []
    for s in range(S):
        exit_wave = probe_s[s] * patches
        fp = np.fft.fft2(exit_wave, norm="ortho")
        farplanes.append(fp)
        I += np.abs(fp) ** 2

Conventions:
- np.fft.fft2 with norm="ortho": the forward and inverse transforms are
  unitary, so the gradient back-propagation (ifft2 of the detector gradient)
  needs no extra normalization factors.
- Grid is the raw (128, 128) patch; no zero-padding before the FFT.
- Mode summation is incoherent: intensities add, phases do not.
- Keep the per-mode farplanes for the gradient computation; the summed
  intensity alone is not enough to back-propagate the update.
