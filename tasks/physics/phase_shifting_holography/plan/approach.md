# Four-frame synthetic-reference quadrature: implemented approach

## Bound inverse problem, notation and physical scale

Recover a camera-plane complex interferometric cross-term C from four released
experimental Fresnel-lens intensity exports. Rows y=0,...,M-1 and columns
x=0,...,N-1 have M=N=1024. I_j[y,x] is the j-th image in DN for j=0,1,2,3;
the nominal reference phase increment is pi/2. O and R denote complex object
and reference amplitudes in sqrt(DN), not calibrated electric-field units.
The ideal measured model and its expansion are

```math
I_j=|O+R e^{ij\pi/2}|^2+\epsilon_j
=|O|^2+|R|^2+O\overline R e^{-ij\pi/2}
+\overline O R e^{ij\pi/2}+\epsilon_j.
```

Epsilon includes noise, departures from nominal phase steps and lossy export
error. The files are real experimentally acquired, already cropped/scaled
grayscale images; their misleading .bmp suffix contains JPEG RGB upstream.
No independent height profile, optical power calibration, magnification or
object true wave is provided. The task is this quadrature/SOSR camera-plane
stage, not numerical Fresnel propagation, refractive-index inversion or lens
height estimation. All transformations below act on the fixed full grid.

## Observation preparation and derivation of the inverse combination

Load uint8 (1,4,1024,1024), remove only the leading batch axis, and promote
to float64. Spatial frame means are removed separately:

```math
H_j=I_j-\langle I_j\rangle,\qquad
Q=H_0+iH_1-H_2-iH_3.
```

The brackets mean an arithmetic mean over all MN pixels. Summing i^j times
the ideal expansion cancels the phase-independent terms because sum i^j=0
and the conjugate term because sum (-1)^j=0, while the desired term has
sum 1=4. Consequently

```math
Q=4\left(O\overline R-\langle O\overline R\rangle\right)
+\sum_{j=0}^3 i^j(\epsilon_j-\langle\epsilon_j\rangle).
```

Thus the factor four and removed cross-term mean are physical/numerical output
conventions, not arbitrary display normalizations. The implementation retains
the upstream ordered accumulation R0*H0+(iR0)*H1+(-R0)*H2+(-iR0)*H3;
algebraically factoring it is not needed for reproduction. Q and C have DN
units. They are not divided by four, projected to magnitude, unwrapped or
converted to height. Independent ideal-interference tests prove the sign and
factor rather than taking another implementation as the only evidence.

## Discrete spectral carrier initialization

For a 2D array A, define the negative-exponential unnormalized DFT
F[A][p,q]=sum_y sum_x A[y,x] exp(-2*pi*i*(p*y/M+q*x/N)). The centered
spectrum used here is fftshift(F[fftshift(H0)]), including the pre-transform
shift from the original function. The finite window is not padded, tapered
or resampled; its spectral interpretation has periodic DFT boundaries.
The unnormalized Parseval identity is sum|F(A)|^2=MN sum|A|^2.

Compute power as the squared modulus and select the largest value in the
literal original ROI. For upper=True, rows use Python slice
`1:int(M/2)-int(M-M*.1)` and columns `1:N`. At M=1024 the stop is -409,
so rows 1 through 614 are included; this is wider than the comment's upper
half-plane. The local code preserves the actual slice. An independent strict
upper-half search returns the same peak (column 514,row 468) on the bound
observations. Using power instead of log(power) preserves the maximum and
avoids log(0); this is not a silently corrected sideband selection. The optional
upper=False helper instead uses rows 1:M and the analogous column slice;
the standard reconstruction always calls the default upper=True.

## Unit synthetic reference and exhaustive inverse update

Let f_x,f_y denote candidate centered spectral coordinates in DFT bins.
Lambda=632.8e-9 m and d_x=d_y=6.9e-6 m/pixel are explicit acquisition constants.
With X=x-N/2 and Y=y-M/2 in pixels, not (N-1)/2 centering, the code evaluates

```math
\theta_x=\arcsin\frac{(N/2-f_x)\lambda}{Nd_x},\quad
\theta_y=\arcsin\frac{(M/2-f_y)\lambda}{Md_y},\quad
R_d=\exp\left[\frac{2\pi i}{\lambda}
  (\sin\theta_x Xd_x+\sin\theta_y Yd_y)\right],\qquad C=R_d Q.
```

The inverse-sine argument must be in [-1,1]. Algebraically lambda and pitch
cancel, giving exp(2*pi*i*((N/2-fx)*X/N+(M/2-fy)*Y/M)); the implemented
asin/sin order is retained for float64 parity. |R_d|=1 preserves amplitude.
The phase origin, carrier sign and frame order are fixed by this formula;
there is no global phase or additional fitted-plane freedom at evaluation.

Initialize at the integer peak and enumerate x in linspace(fx-1,fx+1,4),
then y in linspace(fy-1,fy+1,4), including endpoints. These are 16 discrete
candidates, not four updates of a continuous optimizer. At each candidate,
compute phi=angle(C) in the principal [-pi,pi] convention, a=min(phi),
b=max(phi), and score

```math
S(f_x,f_y)=\#\{(y,x):(\phi[y,x]-a)/(b-a)<0.1\}.
```

Initialize best=-1; replace best/carrier only when S>best. Hence the first
candidate wins ties in this precise x-outer/y-inner order. Stop after all
16 candidates and recompute C at the chosen carrier. The bound selected
carrier is (513.6666666666666,467.6666666666667), score 156192. With zero
wrapped-phase span the local helper returns zero; all actual candidate spans
are nonzero, and this guard does not affect the selected upstream path.
There is no likelihood, regularizer, gradient, proximal update, learning,
line search, residual stop or arbitrary extra iteration in this finite search.
Nondefault s,steps are validated by steps>=s>=0 and steps>=1.

## Output, review comparisons and independent evaluation

Pack real(C),imag(C) as float64 (1,2,1024,1024), axis order batch,component,
row,column, in DN. The public contract also permits float32. The CLI writes
only reconstruction.npz and prints carrier metadata. The notebook actively
reconstructs, evaluates and retains the spectrum/refined-carrier heatmap,
measured frame, uncompensated phase, selected fresh/upstream phase, amplitude,
absolute complex error and paired metric/runtime table. All full scientific
arrays remain untouched by display-percentile contrast or phase color limits.

The reference is an independent unchanged pyDHM run on exactly these pixels,
not actual truth. Flatten the two real channels to a,b. NCC=(a dot b)/
(||a|| ||b||) without centering; NRMSE=sqrt(mean((a-b)^2))/(max(b)-min(b)).
No phase, scale, crop, tilt or translation fitting occurs. Both metrics must
pass NCC>=0.9999999999 and NRMSE<=6.6583485927655845e-8. The latter is
10 times a distinct original float32-input discrepancy, fixed before local
extraction; repeated original float64 calls were identical. This is numerical
method agreement, not proof of lens metrology accuracy.

Independent checks include analytic four-step DC/twin cancellation, plane-wave
carrier slope and SI cancellation, half-array coordinates on odd toy sizes,
Parseval scaling, |C|=hypot(H0-H2,H1-H3), zero centered frame means and exact
carrier/score/full-pipeline parity. Frozen upstream arrays verify preprocessing,
spectrum, peak and selected field. Judge/plot/notebook metrics share separately
known small analytic cases, while the standalone Judge imports no solver code.
Missing/invalid files, keys, dtypes, shapes, NaN/Inf, poor output, unavailable
assets and report-write failures are distinct tested errors or failures.


## Current benchmark scoring (2026-10-29)

The reference implementation still runs its historical synthetic-reference
search. Benchmark acceptance instead independently verifies the four-frame
quadrature up to a unit linear-phase gauge with fixed half-array origin.
Unknown reference tilt is not independently calibrated. Finite search grid
frequency, lower-tail score, and optimizer trajectory no longer determine
acceptance. See README and evaluation/metrics.json for exact tolerance.
