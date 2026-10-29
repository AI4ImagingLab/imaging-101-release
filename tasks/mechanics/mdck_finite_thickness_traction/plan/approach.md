# Implemented finite-layer traction inversion

## Bound physical problem

Measured in-plane displacements of an MDCK epithelial cell colony are inferred from the complete experimental fluorescent-bead image pair (1989 rows ×2034 columns). Public observations contain the derived PIV field, not the images or any traction estimate. Target is the two signed traction components on its 48×49 observation grid, at Gaussian spatial resolution sigma=6 µm. The elastic medium is an isotropic linear layer, Young's modulus E=49000 Pa, Poisson ratio ν=0.49, thickness h=300 µm, rigidly bonded at the lower face. No cell-sheet FEM stress inverse problem is part of this task.

The PIV observation is produced privately by the original pyTFM calculate_deformation(after,before,window_size=100,overlap=60): Gaussian subpixel correlation, peak2peak threshold1.05, norm threshold mean+20*std, local-mean NaN replacement max_iter10 kernel_size2; output v is positive upward. Since original image rows increase downward, reverse the field's row order to make physical y increase with array row. Preserve signed components. The exact measurement pitch is (100-60)*0.201=8.040000000000001 µm, rather than approximate image-size/field-size scaling used by the upstream tutorial. This is an explicitly calibrated application of the same elastic inverse method, not an unchanged tutorial/paper-Fig5 reproduction. A private original-convention capture is retained separately.

## Preparation and discretization

Subtract the arithmetic mean of each displacement component across all 2352 points; this removes uninformative rigid translation. Multiply the supplied original-image pixel displacements by pixel_size=0.201 µm/pixel. Pad to N=50, the smallest even integer no smaller than max(48,49), placing the measured field in the final48 rows and final49 columns. Remaining values are zero. After inversion crop those exact same rows/columns. The published finite-layer inverse uses this support; the half-space code pads at the upper left, which is a periodic shift and produces identical cropped translation-invariant results.

Use scipy.fft.fft2/ifft2; forward transform unnormalized, inverse normalized by N². Angular frequencies are kx,ky=2π*fftfreq(N,pitch), rad/µm; k=sqrt(kx²+ky²), n=(kx,ky)/k. At nonzero frequency the elastic stiffness is

K = αperp(I - n n^T) + αparallel n n^T,

αperp=E*k/[2(1+ν)tanh(kh)], αparallel=E*k*γ/[2(1-ν²)],

γ=[(3-4ν)+(1-2ν)²sech²r+r²sech²r]/[(3-4ν)tanh(r)+r sech²r], r=kh.

Here I is2×2 identity and all matrix entries have units Pa/µm. The compliance G=K^-1 maps traction to displacement. Explicit Kxx=αperp*ny²+αparallel*nx², Kyy=αperp*nx²+αparallel*ny², Kxy=(αparallel-αperp)*nx*ny. Compute tanh and sech² through e=exp(-2r), tanh=(1-e)/(1+e), sech²=4e/(1+e)², avoiding the upstream cosh overflow for large r without altering the formula. Handle r=0 analytically by setting DC traction to zero rather than evaluating an undefined ratio.

For a real input, upstream computes the real part of the inverse FFT. Its off-diagonal coefficient is not Hermitian on single Nyquist lines. The effective real operator sets Kxy to zero on those lines but retains the double-Nyquist corner: that corner maps to itself and its coefficient is real. The half-space implementation explicitly sets Kxy=0 on both Nyquist lines including that corner. Preserve both executable conventions. This subtlety was independently investigated after parity identified a0.12Pa discrepancy; no threshold was changed. All nonzero-frequency effective matrices are positive definite, and stiffness/compliance roundtrip and positive work are tested independently.

## Closed-form inverse and regularization

For Fourier-transformed prepared displacements U,V, compute Tx=Kxx*U+Kxy*V, Ty=Kxy*U+Kyy*V. Inverse FFT, retain the real part, crop to48×49. The inverse is closed form with no optimization initialization, iterations, line search or stopping rule. There is no Tikhonov objective, neural training or hidden domain solver.

Apply a2-D Gaussian to each traction component after cropping, scipy.ndimage.gaussian_filter with sigma=6.0/pitch pixels, default truncate4, reflective boundary. Components are not mixed; this is an explicit spatial regularization/resolution convention. Smoothing_um=0 retains an optional unfiltered diagnostic. Output is float64 Pa, shape(1,2,48,49), component order x,y. Infinite mode sets γ=1 and coth=1, with the preserved Nyquist convention above; it is a separately computed notebook comparison, not substituted for finite-layer output.

## Evaluation and limitations

Truth is unavailable; hidden baseline_reference.npz contains the unmodified upstream core algorithm run on the same canonicalized full displacement field and exact pitch. Frozen tests compare finite, unfiltered, and infinite results at rtol1e-10/atol1e-8Pa. Judge cosine NCC flattens all signed components without centring; NRMSE is RMS error divided by the full reference maximum minus minimum. Boundaries0.99999999 and1e-7 are nonzero numerical reproducibility tolerances, fixed in private acceptance before local implementation and retained after coordinate calibration. No crop, scale fit, magnitude conversion or alignment. Zero or constant hidden references are evaluator errors. Independent physical tests are necessary because reference agreement is not physical accuracy.

The paper's broader monolayer stress and cell-boundary analysis is outside the declared unknown. The maintained notebook covers the task-relevant displacement/traction fields, signed vectors, finite-versus-half-space comparison, upstream/fresh comparison, residual, and numeric metrics. No archived traction is presented as fresh. Displacement noise is not calibrated, so noise-weighted chi-square is not asserted.
