# Calibrated Fourier electron-wave reconstruction: implemented approach

## Physical scope, data and unknown

Two real 512-by-512 count images are the object hologram of an Fe/YOx needle
and an independently recorded vacuum hologram. They are supplied already
binned/cropped by the original dataset, converted losslessly from int64 HDF5
samples to batch-first NPZ. Computation promotes them to float64. No synthetic
observations, extra crop, intensity normalization or fitted background enters
the pipeline. The output is q, a finite-bandwidth dimensionless object/vacuum
complex transmission. It is not independently known true wave, charge density,
mean inner potential, thickness or the full associated charge-density article.

Let y,x be row,column indices 0..511, d=0.9197516441345215 nm/pixel, and
N_y=N_x=512. Physical sample coordinates are x_nm=d*x and
y_nm=183.9503288269043+d*y. This offset fixes physical labels; the DFT uses
array indices. Carrier k=(k_y,k_x) is measured in unshifted Fourier bins.
Object/reference electron amplitudes O,R have sqrt(count) units. For noise
eta in counts, the acquisition and interference terms are

```math
I_o=|O+R e^{i\psi}|^2+\eta_o,
\quad \psi=2\pi(k_y y/N_y+k_x x/N_x),
\quad I_o=|O|^2+|R|^2+O\overline R e^{-i\psi}
+\overline O R e^{i\psi}+\eta_o.
```

The vacuum exposure replaces O by the vacuum object-path wave V. The desired
negative carrier band therefore contains O*conj(R), and its vacuum counterpart
V*conj(R). Their ratio cancels shared instrumental/reference distortions to the
extent that the reference is stable. Selecting the opposite sideband conjugates
the phase and violates the declared sign. Filtering followed by division does
not generally equal filtering the physical ratio; the observable is precisely
the ratio of the two filtered cross terms below.

## Discrete DFT, angular calibration and aperture

Define F with the unnormalized negative-exponential 2D DFT and F_inverse with
the reciprocal N_y*N_x inverse factor. The original convention explicitly uses
S=F(I)/(N_y*N_x) and multiplies its inverse transform by N_y*N_x. It is
algebraically equivalent to inverseF(roll(F(I),k)*W); retaining these factors
prevents confusing count cross-term amplitudes with sqrt(count) electron waves.
There is no fftshift, output crop, interpolation or Fourier-size change. The
full finite observation is interpreted with periodic DFT boundaries.

Unshifted physical frequency coordinates are fftfreq(N_y,d), fftfreq(N_x,d)
in inverse nm; radial frequency r is their Euclidean norm. The smooth aperture
for positive radius a and smoothness b in inverse nm is

```math
A(r;a,b)=\tfrac12\left[1-\tanh\frac{|r|-a}{0.5b}\right].
```

It is approximately one at the center, one-half at the radius and approaches
zero outside; b controls the transition. Keeping the exact tanh convention
matters for parity and bandwidth. A hard cutoff or Gaussian is a different
method. Pixel spacing calibrates both grid and aperture, so final count-wave
normalization remains unchanged.

## Vacuum-only carrier estimate and bandwidth selection

First form S_v=F(I_v)/(N_y*N_x). Suppress the origin by multiplying S_v by
1-A(r;0.05*max(r),1e-6 inverse nm). Search |S_v times stop| over rows 0..255
and all columns, take NumPy's first row-major argmax. This preserves the
upstream half-plane naming, which need not match a displayed upper/lower image.
The result is k=(122,454), equivalent to (122,-58), estimated only from the
legitimate measured vacuum calibration.

Find the closest of four Fourier corner origins (0,0),(0,N_x),(N_y,0),
(N_y,N_x). Set rho to half the Euclidean distance from k to that corner,
rho=67.54257916307313 Fourier pixels. Let delta_f be the mean of
1/(N_y*d) and 1/(N_x*d), in inverse nm per Fourier pixel. For the selected
strong-phase-object reconstruction choose a=rho*delta_f and
b=0.05*rho*delta_f. This half-distance aperture limits overlap with zero/twin
orders while retaining spatial bandwidth. Original smoothing is five percent
of the radius; no new regularization weight is inferred from target truth.

## Closed-form reconstruction and division stability

For j=o,v apply the same roll, aperture and inverse:

```math
w_j=N_yN_x F^{-1}\!\left[
 A(r;\rho\Delta f,0.05\rho\Delta f)\,
 \operatorname{roll}\left(F(I_j)/(N_yN_x),k\right)\right],
\qquad q=w_o/w_v.
```

Roll is positive on row and column axes: it moves the negative-carrier order
to the origin under the declared DFT sign. w_o,w_v are count-valued complex
cross terms; q is dimensionless and has principal phase arg(q) in radians.
Before dividing, require every |w_v|>float64_epsilon*max|w_v|. Otherwise raise
ValueError; adding a hidden epsilon would change the inverse and is not done.
There are no numerical zeros on this dataset, although small modulation can
amplify experimental errors (observed vacuum magnitude approximately 3.118 to
370.552 counts). No mask is fitted to suppress difficult specimen regions.

This inverse is closed form. It has no iteration, initialization guess, loss,
gradient, proximal map, optimizer, training, step-size or convergence schedule.
The finite aperture is its stabilization, and processing finishes after two
filtered inverse transforms and one pointwise division. No phase unwrapping,
global phase fit, intensity scaling, translations or material inference occurs.

## Active scientific comparison paths

The standard CLI and notebook selected result use the above rho. The notebook
also actively evaluates the raw uncorrected object cross term w_o and a weak-
object bandwidth alternative with radius 2*rho/3 and its own five-percent edge,
then divides its object/vacuum waves. The original HoloSpy documentation
discusses these strong/weak aperture choices; neither comparison replaces the
evaluated selected result. Keep measured object/vacuum intensity and spectra,
selected amplitude/phase, original-code error, uncorrected phase, smaller-radius
phase and selected-minus-smaller wrapped phase. The metric table compares all
three labeled original, fresh-selected and fresh-smaller measurement groups.
Fresh runtime records belong in the private review archive; packaged reference
snapshots remain byte-frozen across notebook executions.

## Evaluation and independent physical evidence

For complex arrays a,b define relative_L2(a,b)=||a-b||2/||b||2, where complex
Euclidean norms include real and imaginary components. Both reference agreement
relative_L2(q,q_standard) and observation-sideband consistency
||q*w_v-w_o||2/||w_o||2 must be <=1e-5. The independent Judge recomputes
the latter from public data with the fixed contract carrier/aperture and never
imports the solver. It does not fit phase, mask, scale, shift or bandwidth.
A nonzero reference norm is required. No NCC or magnitude-only NRMSE obscures
phase errors; a global phase change or conjugation is penalized.

The standard result is an independently executed original HoloSpy Fourier
stage, not actual truth. Original double reruns agreed; original float32/double
relative L2 differed by 9.561154009425683e-7, fixing the nonzero 1e-5 bounds
before extraction. Fresh selected wave agreement is approximately 8.70e-16 and
sideband consistency 3.87e-16; these validate numerical reproduction, not material
accuracy. The smaller aperture substantially changes bandwidth (relative L2
about 0.3279); it is a real scientific comparison rather than a free alignment.

Frozen original fixtures verify radial grid, carrier, radius and both complex
waves before full-ratio parity. Independent constant known complex O/R
interference verifies phase sign and normalization, a DC intensity is rejected,
vacuum/vacuum returns one, and analytic aperture limits verify its radial scale.
Malformed predictions, nonfinite values, missing assets, poor valid waves,
complex64 compatibility, foreign-cwd CLI and report-write errors are separately
tested. No conclusion about charge density follows from these gates.
