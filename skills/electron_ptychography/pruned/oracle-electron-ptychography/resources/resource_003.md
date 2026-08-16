# Ptychography update reference

Key constants: max_iter=10, step_size=0.5, batch_fraction=4 (batch_size=J//4~576), eps=1e-16.

Amplitude-projection (Fourier magnitude) update per pattern:
1. exit = P * O[r:r+Qx, c:c+Qy]
2. ff = fft2(exit)
3. ff_mod = measured_amp * exp(1j*angle(ff))   # keep phase, replace magnitude with measured
4. delta = ifft2(ff_mod) - exit
5. obj gradient  += conj(P)*delta ; obj weight  += |P|^2
6. probe gradient += conj(patch)*delta ; probe weight += |patch|^2

Batch update (note weight is squared inside the root):
  O += step_size * obj_grad   / sqrt(eps + obj_weight**2)
  P += step_size * probe_grad / sqrt(eps + probe_weight**2)

Error:
  nmse = sum_over_batch((measured_amp - |ff|)^2) accumulated over the epoch,
         divided by (mean_intensity * J * Qx * Qy)

Probe intensity normalization before iterating:
  probe *= sqrt(mean_intensity / mean(|fft2(probe)|^2))

The probe is refined jointly with the object every batch; do not freeze it.