# Band separation

## Phase matrix (calibration form, phase-normalised)
total_spjg = sum(spjg) = 11
cum = [0, 4/11, 8/11]
pm[j] = [1, exp(+1j*2pi*cum[j]), exp(-1j*2pi*cum[j])]
inv_pm = inv(pm)

## Frame averaging per phase slot i
indices = range(i, nframes*beishu, nframes); fd[:,:,i] = mean(raw[indices], axis=0)

## Spectrum + support mask
pad fd[:,:,i] centred into (n,n); DIbar = fftshift(fft2(ifftshift(padded)))
DIbar *= H1  (binary OTF support broadcast over frames)

## Unmix (per angle block of 3)
sp[:,:, itheta*3 + j] = sum_k inv_pm[j,k] * DIbar[:,:, itheta*3 + k]

## Calibration-only amplitude normalisation
sp = sp / (|sp| + 1e-12)   # keeps phase, drops magnitude
