# Implemented reconstruction mathematics

The selected method is the author's automated PerovskiteOxide110 API, applied to the complete calibrated tutorial ADF/ABF pair with the tutorial's initial separation12pixels. It is not the distinct interactive tutorial sequence or a claim of reproducing its manual plane choices. All1009 A,958 B and922 oxygen columns are reconstructed. No field crop, saved positions, synthetic observation or trained checkpoint is used.

## Image model, units and parameter conventions

Internal coordinates x,y and Gaussian widths sx,sy are in pixels. Internal parameters are `[cx,cy,sx,sy,theta,A]`, with integrated amplitude A in ADF a.u.×pixel² or reciprocal-ABF a.u.⁻¹×pixel². For coordinate offsets dx=x−cx and dy=y−cy the image profile is

```math
g(x,y)=\frac{A}{2\pi s_xs_y}\exp[-(a\,dx^2+2b\,dx\,dy+c\,dy^2)],
```

```math
a=\frac{\cos^2\theta}{2s_x^2}+\frac{\sin^2\theta}{2s_y^2},\quad
b=-\frac{\sin(2\theta)}{4s_x^2}+\frac{\sin(2\theta)}{4s_y^2},\quad
c=\frac{\sin^2\theta}{2s_x^2}+\frac{\cos^2\theta}{2s_y^2}.
```

The normalized source Gaussian uses integrated A; the paper's peak-amplitude notation is converted explicitly. Its rotation convention has the opposite cross-term sign to some published parameterizations. Compare covariance, not raw angles. With delta=0.012443924564929288nm/pixel, output centres are delta×internal centres, output covariance is delta² times the pixel covariance, and peak is A/(2πsxsy), retaining the source intensity scale.

```math
C_{xx}=\delta^2(s_x^2\cos^2\theta+s_y^2\sin^2\theta),\quad
C_{xy}=\delta^2(s_y^2-s_x^2)\sin\theta\cos\theta,\quad
C_{yy}=\delta^2(s_x^2\sin^2\theta+s_y^2\cos^2\theta).
```

This is a projected contrast model, not multislice electron-wave inversion, absolute oxygen occupancy or isolated-atom radius estimation. No actual spatial truth is available.

## Initialization and decomposition

Subtract a nearest-extension Gaussian background with standard deviation30pixels from ADF and shift the minimum to zero. Apply a randomized rank22 decomposition to the transposed image with32 trial vectors,5 LU-normalized power iterations and an economic QR/SVD. The fixed initialization seed20261022 is an algorithm seed, not a hidden observation generator. Project through the retained right-singular subspace, shift a negative minimum if present and divide by the maximum. This filtering is used only to initialize strong peaks.

Detect local maxima with square25×25 neighborhood, relative threshold.02,12pixel excluded border and descending intensity order. Apply minimum infinity-norm spacing12. Initial Gaussian widths are1.2pixels, rotation.01radians and provisional amplitude1. Quantitative fitting always uses the unfiltered ADF or decomposed reciprocal-ABF images.

Refine A using two nominal Gaussian refinement iterations. Each iteration invokes two source fitting passes. The source's nominal rotation-disabled pass still frees rotation internally; the actual selected source behavior is preserved and disclosed. Construct A crystallographic directions and initialize B at adjacent-column midpoints along named100, the second direction. Subtract refitted A profiles from original ADF with fitting radius.35 times same-sublattice nearest-neighbor distance; refine B with one centre-of-mass pass and one nominal two-pass Gaussian iteration, radius fraction.25.

Initialize oxygen at B midpoints along named110, the first direction. Use reciprocal ABF exactly, rather than max-minus inversion. Refit and subtract A in reciprocal ABF at fraction.35, then B at fraction.30. Refine oxygen with one centre-of-mass pass and one nominal two-pass Gaussian iteration at fraction.25. Auxiliary subtraction fits start with unfitted-amplitude initialization, as in the author's thin sublattice copies. They do not replace the reported A/B profiles from ADF.

## Local nonlinear least squares

Neighbor identities are frozen when each sublattice refinement begins, using8 neighbors from a9-neighbor query including self. Distances use their current positions as sequential fits update columns. A's later zone construction leaves14 neighbor identities available to its auxiliary subtraction. B's auxiliary subtraction retains identities from its pre-refinement positions.

For each atom, crop round(cx−radius):round(cx+radius)+1 and similarly in y, clipped at the available image boundary. Use a circular mask centred on the current noninteger location; zero samples outside it. Subtract the median of the lowest3percent of in-mask intensities and clip negative residual samples to zero. This current-source convention differs from the paper's minimum-background description and is recorded rather than silently equated.

Initialize integrated A as ten times the median of the highest3percent of all crop pixels unless that column was already successfully fitted. Minimize the unweighted sum of squared residuals over the entire rectangular crop, including its zeroed exterior. There is no regularizer, loss rescaling, positivity projection inside LM or analytic-gradient claim. SciPy `leastsq` implements the selected general Levenberg–Marquardt optimizer, with its unchanged default tolerances/finite-difference step and maximum evaluations. Free parameter order is `[A,cx,cy,theta,sx,sy]`, matching the upstream alphabetic parameter order; model parameter order differs. The result is accepted when its centre lies strictly inside the image crop and mask, A is nonnegative, and absolute width ratio is at most4. Store absolute widths and angle moduloπ.

If rejected, use the source's at-most10 retries. Preserve its existing crop and mask-radius convention: the first retry reduces radius by.95, later retries reuse that assigned radius; each pass masks/subtracts the existing crop again. After final failure, use centre of mass and amplitude zero as the source does; this would fail the declared physical-profile gate. The bound complete original and fresh local runs contain no such zero-amplitude fallback.

Centre-of-mass uses a zero-padded square around rounded coordinates, circular support at the selected radius, normalized image moments and the original pixel origin. It does not subtract a separately estimated background. Its purpose is initialization of weaker columns, not a replacement inverse output.

## Lattice direction and midpoint construction

Compute all ordered pair displacements in float32, omitting self and retaining norms below84pixels. This preserves the reference's intentional precision conversion. Normalize by their global standard deviation; apply explicitly implemented density-connected clustering with radius.1 and minimum10samples. Cluster means use float32 arithmetic, are rescaled and rounded to.01pixel. There are fewer than150000 eligible displacements in this binding, so the source's stochastic fingerprint subsampling is inactive.

Sort means by length, remove vectors within8pixels, then remove integer multiples and antiparallel copies using multipliers−4..4, preferring the positive principal component. Retain six named directions in the selected source order. For each required direction, walk the14 neighbor identities forward/backward, choosing the closest candidate within6pixels of the expected translated point. Group columns into planes, order along each plane from its negative-direction endpoint, and sort planes by distance to the reference's remote orthogonal point. Exclude a direction if more than60percent of its planes have only2atoms. Adjacent within-plane midpoints initialize the next sublattice. There is no outer-edge extension.

## Subtraction and evaluation

Render each auxiliary Gaussian on a box of5times its larger width, with floor/ceil limits clipped to the image, accumulating in source column order. No periodic wrap or fitted scalar offset is introduced. Preserve the source normalization, all pixels and reciprocal-intensity units.

The independent Judge matches identities within each lattice using minimum total centre distance; it fits no geometric transformation. All matched centres, covariance entries and peaks are scored. Before any local code, independent complete source seeds20261022/20261023 gave maximum centre variation.076579925pixel, RMS.002410928pixel, covariance relative L2.000900950 and peak relative L2.0000243183. Rounded-up threefold gates are respectively.23pixel,.0073pixel,.0028 and.000075. Each gate applies to each lattice. Invalid profile count must be zero. These nonzero tolerances derive from independent reruns rather than self-comparison; no postfit mask is used. Analytic rotated-Gaussian controls independently establish formula/sign/unit consistency and known-centre recovery; they do not supply specimen truth.
