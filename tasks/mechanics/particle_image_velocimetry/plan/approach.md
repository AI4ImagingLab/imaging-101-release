# Implemented mathematics

## Scope and observations

The selected standard method is OpenPIV's noniterative extended-area correlation
estimator, `pyprocess.extended_search_area_piv`, explicitly using its SciPy/Python
path. The bound case comprises two 369-row by 511-column uint8 monochrome images.
All internal arithmetic is float64. The image origin is top left; x increases
right and y down. Pixel values are digital counts. No calibrated camera length
or elapsed seconds are available, so the interval is Delta t = 1 frame interval
and reconstructed velocity is pixel per interval. No ground truth is available.

This task selects the published standard correlation stage, not the upstream
tutorial's subsequent outlier interpolation, physical scaling or axis reversal.
That tutorial uses an undocumented scale factor and does not establish its
physical timing. Those operations would invent unsupported calibration here.
The task requires every raw estimated vector and its confidence, with fixed axes.

## Discretization and normalization

For search size S=64 and overlap O=16, windows start at (48j,48i) for
i=0,...,6 and j=0,...,9. Their centers are (48j+32,48i+32). The image is not
recentered and its unused trailing margins are discarded. Each window pair is
stacked in row-major grid order, giving 70 by 64 by 64 arrays.

Define N(W)=(W-mean(W))/std(W), with population standard deviation (denominator
S squared), all calculated separately for each search window. A constant window
is replaced with zero. Let M be one on rows and columns 16 through 47 inclusive
and zero elsewhere. The first preparation stage is A0=M N(A), B0=N(B).
The correlation stage again normalizes both, producing A1=N(A0), B1=N(B0).
This second normalization is essential upstream behavior; after masking, the
first window is no longer mean-zero or unit-variance.

For displacement indices p,q in 0,...,63,

```math
C[p,q]=\frac{1}{S^2}\sum_{r=0}^{S-1}\sum_{c=0}^{S-1}
A1[r,c]\,B1[(r+p-S/2)\bmod S,(c+q-S/2)\bmod S].
```

The implementation evaluates this expression with SciPy real FFTs:
`fftshift(irfft2(conj(rfft2(A1))*rfft2(B1)))/4096`. SciPy's default forward FFT
is unnormalized and inverse is normalized by the number of pixels. The final
division therefore makes C a mean normalized product. Periodicity applies only
inside each window. The original zero-support mask is applied before the second
normalization and is not applied again afterward.

## Inverse peak estimate

Let (r0,c0)=argmax C, using NumPy's row-major first maximum. A peak on any map
boundary is invalid. For an interior peak, mutate the whole map by adding
epsilon=1e-7. This upstream numerical convention precedes both the subpixel fit
and confidence calculation; copying the map and measuring confidence on an
unmodified version would not reproduce the method.

Let z=C[r0,c0], l=C[r0-1,c0], h=C[r0+1,c0], d=C[r0,c0-1], and t=C[r0,c0+1]
after adding epsilon. For nonnegative z,l,h,d,t, the row correction is

```math
e_r=\frac{\log l-\log h}{2\log l-4\log z+2\log h}.
```

The column correction uses d,t in place of l,h. If a Gaussian denominator is
exactly zero, its correction is zero. If any of the five intensities is negative,
both corrections instead use the parabolic form
`(l-h)/(2*l-4*z+2*h)` and its column counterpart. These conventions preserve
upstream behavior; for the bound observations all outputs are finite.

Velocity is `u=(c0+e_c-32)/Delta t`, `v=(r0+e_r-32)/Delta t`.
It is stored as real float64 with shape (1,7,10,2), component order (u,v).
There are no optimization objectives, losses, regularizers, adjoints, gradient
updates, initialization, line searches, iterative stopping criteria or training
weights: the inverse is a finite direct spectral correlation and peak fit.

## Confidence

On the epsilon-shifted map, p is the largest intensity and q the largest outside
the square from r0-2 to r0+2 and c0-2 to c0+2 (clipped to the map). Confidence
is p/q unless: p<0.001; the first maximum lies on an edge; q=0; or the second
maximum is on an edge and q>p/2. These cases return zero. This is a dimensionless
matching diagnostic, not a physical unknown or calibrated uncertainty.
Its full (1,7,10) array is required and independently evaluated.

## Evaluation conventions

The hidden reference is an actual unmodified original-method run on these exact
images, not simulated fluid truth. Velocity NCC is flattened dot product divided
by vector norms without centering; a zero estimate has NCC zero. NRMSE is RMS
error over both components divided by max(reference)-min(reference). Confidence
uses relative L2 norm. There is no fitted alignment, scale, crop or masking.
The Judge rejects constant/zero reference definitions rather than reporting NaN.
The fixed pre-extraction bounds are NCC>=0.99999999, NRMSE<=1e-8 and confidence
relative L2<=1e-8. Two isolated upstream reruns were bitwise identical; the
nonzero allowance accommodates numerical FFT differences, not physical error.
Frozen stage tolerances are absolute 1e-10 and full-output tolerances 1e-8.
Direct-sum correlation, an exact periodic translation and analytic Gaussian /
parabolic peaks independently check signs, units, normalization and subpixel fit.

## Derivation of the direct inverse and its limits

For a fixed integer shift s, let a be the normalized interrogation search array
and b_s the circularly shifted second normalized search array. The local squared
matching distance is E(s)=sum(a-b_s)^2. Circular translation preserves ||b_s||,
so E(s)=||a||^2+||b||^2-2 S^2 C(s): minimizing this unweighted distance is exactly
maximizing the shipped normalized correlation. The apparent sign of translation
is fixed by conj(FFT(a))*FFT(b); a second image translated right/down produces
positive column/row displacement. The extra 1/S^2 affects peak amplitude and
confidence's low-peak rejection, even though it does not move an integer argmax.
This derivation concerns prepared arrays, not an assertion that separately
normalized/masked windows obey unmodified raw-image brightness conservation.

For an interior positive Gaussian peak, write log C(k)=a*k^2+b*k+c around integer
center k=0. Neighbor values give a=(log C(-1)-2log C(0)+log C(1))/2 and
b=(log C(1)-log C(-1))/2. Its stationary location -b/(2a) yields the documented
logarithmic offset. The fallback fits the same quadratic directly to C instead
of log C. The code does not regularize a zero parabolic denominator; nonfinite
velocities are rejected by solve rather than synthesized. A zero Gaussian
curvature returns offset zero. The five-value sign decision applies jointly to
both directions; separate per-direction decisions would change the method.

The peak is fitted in row/column order, then reversed to public x/y order. The
integer zero-shift index is32 on this even64-point grid. The output grid is7x10;
its last search window ends at row352 and column496 (exclusive), leaving17 rows
and15 columns unused. At no point is the correlation spatially upsampled or a
vector interpolated. A second-peak exclusion mask covers5x5 pixels by default;
confidence has no Gaussian uncertainty interpretation and no independent truth.

## Bound parameter and execution inventory

| Setting | Value | Numerical consequence |
|---|---|---|
| Interrogation/window |32 pixels | Central first-window support16:48 |
| Search |64 pixels | Circular FFT map64x64; zero shift32 |
| Overlap |16 pixels | Stride48;70 row-major samples |
| Delta t |1 frame interval | Report pixels per frame interval |
| Normalization |Population ddof0, twice | No count/velocity rescaling |
| Peak epsilon |1e-7 | In-place map change before confidence |
| Confidence width |2 pixels | Clip5x5 exclusion at borders |
| Weak-peak rejection |p<1e-3 | Applied after epsilon |
| Secondary edge rejection |q>0.5p | Strict inequality, edge only |
| Arithmetic |float64 | Real FFT; no stochastic state |

The finite algorithm terminates after70 maps have each been evaluated once.
There is no convergence tolerance or training state to tune. No subsequent
median validation, interpolation, physical scaling or y-axis reversal is shipped.
The notebook compares both images, vector overlay, paired vertical fields,
confidence distribution, component errors and paired NCC/NRMSE values. Its
archived upstream result is an independent software reference; the active local
solve is the only source of the newly plotted reconstruction.
