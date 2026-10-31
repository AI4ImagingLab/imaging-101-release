# Implemented phase-only conductivity estimate

## Physical binding and scope

The unknown is apparent electric conductivity, not a separately identified complex permittivity. Twenty-five real transceive-phase maps share the native grid `(z,y,x)=(96,96,80)`, spacing 0.002 m and radiofrequency 128 MHz. The phase-only Helmholtz approximation is

```math
\sigma=\frac{\Delta\phi}{2\mu_0\omega},\qquad \omega=2\pi f,\qquad \mu_0=4\pi10^{-7}\ \mathrm{H/m}.
```

Phase phi is in radians and conductivity sigma in S/m. The factor two assumes transmit phase equals half of transceive phase. It omits transmit-magnitude gradient terms; agreement with this estimate does not establish physical accuracy. No independently measured conductivity field is available. The nominal bulk 0.56 S/m was explicitly not experimentally characterized with uncertainty and is not scored as truth.

## Averaging and local derivative

Cast each native float32 repeat to float64, divide it by 25, then accumulate repeats in acquisition order. This division-before-summation preserves the author dataset program's arithmetic. Retain native NaNs outside the fixed source observation support. Do not unwrap, resize or subtract phase.

The primary derivative uses a quadratic least-squares fit on the full cuboid of integer offsets -5,...,5 along every axis. In the symmetric stencil, even terms decouple from odd linear and mixed quadratic terms. Let F have one row `[1,x*x,y*y,z*z]` for each active offset. The dimensionless second-derivative weights for axis d are twice row d+1 of `(F.T F)^(-1) F.T`. Multiply each stencil sum by the reciprocal of that axis spacing twice, then add x, y and z contributions in that order. No regularizer, iterative objective, line search, initialization or stopping criterion is involved: this is a closed-form linear derivative followed by a physical scale factor.

Only centers with every active stencil sample measured are valid. Erode the source mask by the actual stencil, treating points beyond the array as absent. The primary 11-cuboid leaves 859 centers from 11513 observed voxels. No fitted value affects these masks. Correlation uses zero-filled missing data only as an arithmetic implementation detail; invalid derivative centers remain NaN internally. The released array writes exactly zero outside the valid support and preserves signed conductivity inside it.

## Active scientific comparisons

The notebook reconstructs every individual repeat to estimate empirical repeatability and reconstructs means using cross, sphere and cube stencils of radii 1 through 5. Cross contains the three coordinate axes, sphere uses the original sequential floating-point normalized-distance test, and cube contains every offset. Common-support statistics use the fixed primary support; support counts are also shown so different stencils are not compared under silent selection.

The magnitude-assisted comparison constructs a complex transmit-field approximation `U = magnitude * exp(0.5j*phase)` from the separately averaged supplied observations. It computes complex permittivity as `Delta(U)/(-mu0*omega**2*U)`, conductivity as minus its imaginary part times omega, and relative permittivity as its real part divided by `epsilon0=1/mu0/c0/c0`, with c0=299792458 m/s. This preserves the original comparison; its output is neither truth nor the selected required estimate. A spatially uniform common magnitude scale cancels, but spatial magnitude bias does not.

Repeat standard deviation uses the sample convention ddof=1; standard error divides by sqrt(25). These are empirical repeat diagnostics, not the original paper's full James-Stein covariance propagation. A separate diagonal-noise propagation applies the squared physical derivative coefficients to the measured phase variance of the mean. Its disagreement with empirical conductivity variation exposes the effect of neglected spatial correlation; it is not a calibrated uncertainty guarantee.

## Evaluation and independent checks

The hidden reference is the unmodified EPTlib 0.1.1 cuboid-radius-5 result on these exact observations. On all 859 fixed support voxels, NCC is cosine similarity without centering; NRMSE is RMS error divided by reference range. The frozen bounds are NCC >= 0.999999 and NRMSE <= 1e-6. A measured float32 one-ULP envelope was 8.136333964421884e-7 dynamic-range NRMSE before local implementation. This justifies a nonzero numerical tolerance independently of self-comparison. Invalid arrays, nonfinite values or nonzero exterior padding are contract errors.

Frozen upstream tests check all fifteen derivative coefficient sets, averaging order, representative full phase-only and magnitude-assisted fields and the primary output. Independent polynomial moments, analytic constant-conductivity phase, complex plane waves, frequency/spacing conventions and common magnitude-scale invariance test the physical factors independently of the original implementation. No continuum experiment is substituted for the real input.
