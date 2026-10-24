# U Gem Velocity Emissivity

> Reconstruct a two-dimensional velocity-plane emission field from phase-resolved optical spectra whose finite exposures and incomplete angular coverage blur the emitting structures.
> Domain: astronomy | Keywords: spectral tomography, maximum entropy | Difficulty: Hard

## Background

Gas orbiting an accreting binary emits spectral lines whose Doppler shifts encode its line-of-sight velocity. As the binary rotates, each emitting region traces a different radial-velocity curve. Combining these projections reveals where emission lies in velocity space, helping distinguish a rotating disk, gas stream and emission associated with the companion.

The recovered plane describes velocity rather than sky position. Several physical locations can share a velocity, and finite exposures mix nearby orbital phases. Uneven phase coverage, continuum subtraction and changes in line brightness make the inverse problem incomplete and sensitive to regularization.

## Problem Description

The observations are 28 native spectra of the H-alpha region of U Gem. Recover the complete signed, background-subtracted velocity-plane emission density. For orbital angle theta and velocity coordinates vx and vy, the instantaneous projected radial velocity is:

```math
v_{\mathrm{rad}}=v_x\sin\theta-v_y\cos\theta.
```

Here theta is the orbital phase multiplied by the angular conversion stated below. A finite-exposure projection distributes an emitting velocity pixel over the interval between its two exposure-endpoint radial velocities, widened by half a reconstruction pixel at each end. Its contribution to a spectral bin is the integral of a unit-area triangular profile over that bin. The spectrum is the sum of these contributions over the velocity plane. The triangular cumulative distribution is zero below zero, 2t² on [0,1/2], 1−2(1−t)² on [1/2,1], and one above one; t is the position normalized to the widened interval. Normalize each emitting pixel's weights across the entire observation set to sum to one.

Save `output/reconstruction.npz` with key `reconstruction`, real float32 or float64, shape `(1, 144, 144)`, axis order `(batch, vy, vx)`. Both velocity axes increase from −va to va, where va is the maximum absolute endpoint of the calibrated spectral velocity grid. Their spacing is dv=2va/143. Return zero outside vx²+vy² < (va+dv/10)². The field includes signed continuum/background-subtracted values; do not take its magnitude or clip negative values. Its unit is U Hz/(cm/s)², where U denotes the author's unspecified spectral flux unit. This is a conditional emission estimate, not an independently calibrated physical emissivity.

Evaluation uses cosine similarity without mean subtraction, RMS field error divided by the comparison field's full dynamic range, and complete forward spectral RMS error divided by the positive spectra's dynamic range. All output samples and all 28 spectra enter evaluation, without translation, rotation, sign, flux or amplitude alignment. Agreement with a standard reconstruction does not establish astrophysical truth.

## Data Description

| Key | Shape | Dtype | Description |
|---|---|---|---|
| wavelength | (1, 28, 2048) | float64 | Native per-spectrum wavelengths, Angstrom |
| spectrum | (1, 28, 2048) | float64 | Native measured spectral flux, author unit U |
| orbital_phase | (1, 28) | float64 | Author orbital phases, cycles |

### Acquisition metadata

| Parameter | Value | Unit | Meaning |
|---|---|---|---|
| Data origin | San Pedro Martir, January 1999 | — | Real phase-resolved echelle observations |
| Rest wavelength | 6562.8 | Angstrom | H-alpha line center |
| Systemic velocity | 36 | km/s | Source's systemic correction |
| Systemic conversion speed | 299800 | km/s | Legacy wavelength factor (1+b)sqrt(1−b²), b=36/299800 |
| Velocity conversion speed | 299700 | km/s | Native velocity convention v=c(r²−1)/(r²+1), r=wavelength/rest wavelength |
| Angular conversion | 6.2831854820251465 | rad/cycle | Source's 8 atan(float32(1)) constant |
| Common exposure width | 0.03930 | cycle | Author phase-file common-width approximation; individual 300/600-second assignments are not traceable for these 28 spectra |
| Continuum intervals | (6500,6537), (6591,6620) | Angstrom | Cubic continuum fitted after systemic correction |
| Line interval | (6527.8,6597.8) | Angstrom | Open-interval selection; 477 native selected samples |
| Wavelength grid | First spectrum's grid | Angstrom | Linearly interpolate other spectra to this grid before continuum fitting |
| Spectral velocity grid | 477 equally spaced endpoints | cm/s | Linear interpolation of continuum-subtracted spectra onto velocity grid |
| Native serialization | Flux 5 decimal places; velocity 6 significant figures; phase 6 decimal places after float32 conversion | U; cm/s; cycle | Observation precision used by the selected physical discretization |
| Velocity-plane size | 144 by 144 | pixel | Known circular support defined above |
| Eclipse interval | (0.95,1.05) modulo one | cycle | Predetermined zero likelihood weight; retain these observations for full forward consistency |
| Absolute flux calibration | Unavailable | U | Do not interpret U as mJy |

The legacy conversion constants and common exposure width are acquisition approximations. They should not be interpreted as exact relativistic calibration or separately known exposure durations.

## Method Hints

A positive latent emission field with an observation-derived flat background permits a Poisson-like likelihood. Add a flat-circle projected baseline whose central response is 0.1 times the mean continuum-subtracted data; replace values below one third of this scalar level by 10⁻⁵ times that level. Normalize each non-eclipse spectrum's integrated positive flux to the non-eclipse mean. Keep the eclipse spectra unchanged. Set likelihood weights to zero in the eclipse interval and where the flat response vanishes, normalizing their sum to 28.

Maximum entropy with a floating spatial default suppresses unsupported velocity structure. A useful objective is:

```math
Q(x)=\sum_j w_j\bigl[y_j\log(Px)_j-(Px)_j\bigr]
-\alpha\sum_i\bigl[x_i\log(x_i/q_i)-x_i+q_i\bigr],\qquad q=Gx.
```

Here x is the positive latent pixel flux, P is the finite-exposure projection, y is the positive normalized spectrum, w is its likelihood weight, and alpha is the entropy strength. The selected conditional estimate uses alpha=0.002 and G with separable normalized weights exp(−j²/49), j=−10,...,10, half-sample reflected boundaries after zero extension of the circular support, then restriction to that support. A flux-conserving accelerated Lucy iteration combines the current scaled gradient with the previous direction using a two-direction Newton calculation. Subtract the known flat background and convert latent pixel flux to density by multiplying by the spectral frequency step and dividing by dv². The smoothing and regularization choose a conditional reconstruction; weak structures remain uncertain.

## References

- H. C. Spruit, *Fast maximum entropy Doppler mapping*, [arXiv:astro-ph/9806141v1](https://arxiv.org/abs/astro-ph/9806141v1).
- J. Echevarria et al., *High-resolution optical spectroscopy of U Gem*, [Astronomical Journal 134, 262 (2007)](https://doi.org/10.1086/518562), [author manuscript](https://arxiv.org/abs/0704.1641).
