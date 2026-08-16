# Line-shape and resampling helpers

gaussian_line(w,w0,sigma) = (2/sigma)*sqrt(ln2/pi)*exp(-4*ln2*((w-w0)/sigma)^2); returns zeros if sigma==0.

lorentz_line(w,w0,sigma) = (1/pi)*(sigma/2)/((w-w0)^2 + sigma^2/4).

asym_Gaussian: split at w0; low side uses width (sigma-a_sigma) and exponent (k-a_k); high side (sigma+a_sigma),(k+a_k); add offset, raise to power_factor (=1), normalize by max, nan_to_num.

Slit defaults: [sigma=slit_width(0.5), k=slit_shape_factor(2.0), a_sigma=0, a_k=0].

downsample local_mean: hw = max(1, int(dstep/fstep/2)); index via searchsorted(clip); average spec_fine over [idx-hw, idx+hw].

Pump convolution: I_as = 0.5*(I_as + dnu*|conv(chi_total, gaussian_line)|^2), dnu = nu_s[1]-nu_s[0], mode 'same'. Applied only when pump_lw>0. Fine-grid step = 0.05 cm^-1, padded +-10 cm^-1 beyond experimental range.
