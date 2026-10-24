# Implemented RF-to-vector mathematics

## Observation binding, physical target and axes

The observation is the complete existing author `PWI_disk.mat`: measured signed
int16 RF(334 fast-time samples,128 elements,32 pulses) and known instrument
calibration. The released RF is reordered to batch/ensemble/element/time,
shape(1,32,128,334); loading restores source time/element/ensemble order as
float64. Coordinates x,z and the author's exact known geometric disk mask are
public. Grid rows are depth z, columns lateral x, both in metres. The mask has
31404 pixels on the complete251 by251 grid, centred(0,.023)m with radius.01m.
At floating-point points on the circle, preserve the supplied boolean mask;
an independent geometry check permits only rounding within64epsilon times
radius. This is not a mask inferred from velocities.

The unknown is both physical in-plane velocity components, vx laterally right
and vz away from the transducer, in m/s. There is no independent rotation-rate
or velocity truth. The target is the selected author recipe's finite-protocol
conditional estimate, rather than a claimed fully converged continuous robust
optimum or a calibrated true mechanical speed. All support pixels of both
components are evaluated; outside-support output values must be zero.

The exact acquisition constants are c=1480m/s, fc=5e6Hz,
fs=6666666.666666667samples/s, PRF=10000Hz, t0=9.95e-6s,
pitch=.000298m, width=.000262m,128 zero-depth elements, all transmit delays0,
fractional bandwidth15%, normal plane-wave transmission and receive angles
plus/minus pi/12. The grid is the exact source floating-point x/z arrays,
nominally x=-.0125:.0001:.0125 and z=.01:.0001:.035m. The RF file is already
retrospectively downsampled. Its disk diameter/ensemble differ from the paper's
historical experiment; no unsaved high-rate RF is invented.

## Complex demodulation and narrow-band undersampling

For fast-time index n, t[n]=t0+n/fs. Multiply each independent element/pulse
trace by exp(-i2pi fc t[n]), where i is the imaginary unit. Apply a fifth-order
Butterworth low-pass, with normalized Nyquist cutoff Wn=(.15fc)/fs=.1125,
forward and backward with15 samples of odd-reflected padding and steady
initial conditions. Multiply by2 to preserve the analytic envelope amplitude.
The physical baseband half-width is375kHz. A general `scipy.signal.butter` and
`filtfilt` implement these filter primitives; no domain demodulator is called.

Padding15 is three filter orders, matching the successful official author
path in Octave. SciPy's default18 would differ; an independent general filter
case validates the selected convention. The source operates on independent
time columns; no filtering is performed across elements or pulses. The
carrier used for phase conversion remains5MHz, even though its sampled alias
lies near1.6667MHz. At the saved rate the two real passband aliases are
disjoint; reducing carrier frequency to the alias would change physical scale.

## Receive directivity and explicit discrete travel-time operator

The element centres are xe[j]=(j-63.5)pitch, j=0,...,127, all ze=0.
For receive angle beta, define RX=abs(beta) and shortest band wavelength
lambda=c/[fc(1+.15/2)]. Minimize

```math
g(\alpha)=\left|\cos(\alpha+\mathrm{RX})
\operatorname{sinc}\left(\frac{w}{\lambda}\sin(\alpha+\mathrm{RX})\right)-0.71\right|,
\quad 0\leq\alpha\leq\frac\pi2-\mathrm{RX},
\quad F=\frac{1}{2\tan\alpha}.
```

Here w is element width and sinc(a)=sin(pi a)/(pi a). General bounded scalar
Brent minimization with x tolerance pi/100 matches the source's8 evaluations,
alpha=.14746717627109673 and F=3.365971459145058. This is an instrument
directivity calculation independent of any target velocity, not fitting hidden
calibration. Both angle signs give the same F. Keep element j at point(x,z)
when abs(x-xe[j]-z tan(beta))<=z/[2F cos(beta)]. There is no amplitude
aperture-size normalization in this source recipe.

The source's virtual array for zero transmit delays is the actual linear
element array. Its parabolic TX-distance reader is retained explicitly. For
the element j nearest x, let a[k]=abs(x-xe[k]) and
b[k]=hypot(x-xe[k],z). Set

```math
T=\frac{b_{j-1}+b_{j+1}}{2}
-\frac{(b_{j+1}-b_{j-1})(a_{j+1}-a_{j-1})}
{2a_{j-1}-4a_j+2a_{j+1}},
\qquad \tau_k=\frac{T+b_k}{c}.
```

The earliest element wins a nearest-distance tie. Every bound point has an
interior nearest element; the function also retains the source endpoint
distance if that is not so. Evaluate only the selected parabola: unused
far-away normal-distance second differences can be zero and are not required
to determine the selected T. This removes unused division warnings without
altering any selected source matrix coefficient. Do not replace T by ideal z.

For each element, source one-based fast-time coordinate is
u=(tau-t0)fs+1. Keep1<=u<=nfast-1. Let k=floor(u), f=u-k. Its two zero-based
matrix columns are k-1+j*nfast and k+j*nfast, with coefficients
(1-f)exp(i2pi fc tau) and f exp(i2pi fc tau). Grid rows are Fortran-flattened
z/x order; input columns have fast time first then element. Local COO-to-CSR
assembly and general sparse multiplication evaluate the full acquisition
operator. Point blocks2048 control memory, with no scientific crop. Both
full251² by32 complex focused ensembles are freshly computed.

## Ensemble phase, finite-packet variance and vector inversion

For focused complex echo I[y,x,q], lag l and packet length Q=32, use

```math
A_l=\sum_{q=0}^{Q-l-1}I_q\overline{I_{q+l}},
\qquad P=\sum_{q=0}^{Q-1}|I_q|^2,
\qquad V_N=\frac{c\,\mathrm{PRF}}{4f_c l},
\qquad d=-\frac{V_N}{\pi}\arg A_l.
```

The overline denotes complex conjugation. Default lag1 uses all31 adjacent
pairs. Before the phase and variance, convolve A and P with the outer product
of symmetric Hamming11 windows, using replicated boundaries. The unnormalized
window scale cancels in phase and A/P. Keep the source variance estimator
2(VN/pi)^2[1-abs(A)/P] in(m/s)^2. Because P uses32 terms but A31, even a
constant noiseless packet has a nonzero finite-packet estimate; it is never
called calibrated noise variance or physical uncertainty.

At lag1 VN=.74m/s. The complex phase convention gives positive d for motion
toward the effective bistatic view. For receive plus/minus theta=pi/12,

```math
d_\pm=-\tfrac12[\pm\sin\theta\,v_x+(1+\cos\theta)\,v_z],
\quad v_x=\frac{d_--d_+}{\sin\theta},
\quad v_z=-\frac{d_-+d_+}{1+\cos\theta}.
```

No angle correction based only on a monostatic cosine is substituted. The
lateral difference is more sensitive to noise than the depth sum. The principal
directional interval is plus/minus VN; wrapping is explicit. Longer-lag
notebook comparisons use VN/l and are diagnostics, not new benchmark tasks.

## Explicit weighted robust DCT penalty

The selected author display/inference recipe applies fixed smoothing s=1e4 to
the two raw vector components jointly. For fixed scalar weights W on grid
pixels and either component y, the conditional penalized quadratic is

```math
J(u)=\tfrac12\|W^{1/2}(u-y)\|_F^2+\tfrac{s}{2}\|Lu\|_F^2.
```

L is the positive grid Laplacian with repeating, not periodic and not
reflect101, boundaries: its1D diagonals are(1,2,...,2,1), neighbours-1.
Its2D orthonormal DCT-II eigenvalues at zero-based(k,l) are
Lambda=4-2cos(pi k/H)-2cos(pi l/W), H=W=251. Pixel spacing is one in this
regularizer; s is not silently reinterpreted as a metre-scaled physical modulus.
The inverse filter is Gamma=1/(1+s Lambda²). A small explicit matrix test
checks this penalty and its boundary factors independently of source parity.

Outside the public mask W=0. Source toolbox-free initialization fills missing
raw velocities with the corresponding component's measured-support mean,
takes a DCT, keeps the first ceil(H/10)=26 and ceil(W/10)=26 frequencies per
axis and inverse-transforms. This preserves the selected original fallback
path; it is not nearest-neighbour interpolation or a zero initial field.

With relaxation r=1.75, each weighted update is

```math
u^{k+1}=r\,\mathcal C^{-1}
\bigl[\Gamma\,\mathcal C\{W(y-u^k)+u^k\}\bigr]+(1-r)u^k.
```

C is orthonormal2D DCT-II; inverse uses the corresponding inverse DCT.
Weights and Gamma multiply pointwise. Stop each inner pass when the joint
two-component Frobenius change divided by new-field Frobenius norm is at most
1e-3 or100 updates are reached. The source uses three robust passes and a
shared vector residual scale. Default actual inner counts are100,43,12; changes
are.0022206293909106094,.0009977235547358643,.0009915145713361665.
The original returned convergence flag refers only to the final pass. The first
pass hits its prescribed budget; no full-quadratic or all-pass convergence is
claimed. This finite protocol is the selected conditional reference definition.

After each pass, let r_i be its2-component residual at pixel i and m its vector
of marginal medians on the mask. Set MAD=median(norm(r_i-m)) and

```math
h=\left[\frac{\sqrt{1+\sqrt{1+16s}}}{\sqrt2\sqrt{1+16s}}\right]^2,
\quad a_i=\frac{\|r_i\|_2}{1.4826\,\mathrm{MAD}\sqrt{1-h}},
\quad W_i=\begin{cases}(1-(a_i/4.685)^2)^2,&a_i<4.685,\\0,&a_i\geq4.685.\end{cases}
```

Multiply W by the public mask. The uncentred residual norm is used in the
studentization, while the median-centred norm estimates scale, exactly as in
the source vector implementation. The final recomputed weights are returned
separately from weights actually used in the final solve; no fourth solve is
invented. An undefined zero robust scale raises explicitly instead of emitting
NaN; it does not occur in this acquisition. Full extrapolated fields are kept
for regularization review; only the declared output sets exterior values0.

The2010 article's corrigendum proves positive definiteness when the weight
mask contains at least one positive value; its original nonsingularity claim
for L is not repeated. Robust reweighting remains a chosen finite procedure,
not a proof of true velocity accuracy or a single globally convex fit.

## Evaluator and experiment conventions

The reference is the complete independent official-author MATLAB result,
obtained on exact bound RF through documented general Octave API adapters.
Each component's NCC is uncentred cosine similarity. Each NRMSE is RMS error
divided by that reference component's dynamic range on all31404 support pixels.
There is no fitted gain, translation, rotation, sign or vector alignment.
Independent unchanged repeat was bitwise equal; source complex64-intermediate
and float32-serialization measurements froze nonzero limits before extraction:
vx NRMSE<=3.1279966911092165e-8, vz<=3.449385143851913e-8; each
NCC>=.9999999999999956. These are reproduction tolerances, not accuracy bounds.

An independent evaluator re-derives both focused complex correlations directly
from public RF and grid; no solver or hidden observational cache is imported.
Given predicted directional d from the submitted field and observed A, measure
sqrt(sum(abs(A)*abs(exp(-i pi d/VN)-A/abs(A))²)/sum(abs(A))) over both beams and
all support pixels. The original value is.05201278063077122; frozen maximum is
.057214058693848345. Zero and sign-reversed fields yield1.21664 and1.56248,
respectively. This tests observation consistency while allowing the selected
spatial prior's measured bias. It is not full RF waveform fitting: unknown
scatterer amplitudes/speckle are not reconstructed. Exterior max magnitude
must be at most1e-12m/s.

Notebook comparisons actively reuse the fresh RF-derived complex beams to
compute raw versus nonrobust/robust fields, smoothing-scale sensitivity,
spatial averaging and temporal-lag/alias sensitivity. All comparisons are
labeled conditional-method differences; none is substituted for a fresh run
or relabeled as true speed error. The paper's different acquisitions and
historical high-rate comparisons are identified as unavailable for this exact
file; the notebook does not invent unsaved observations.
