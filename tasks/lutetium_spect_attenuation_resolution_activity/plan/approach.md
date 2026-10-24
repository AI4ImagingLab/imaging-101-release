# Implemented mathematics

The selected baseline is the author's complete real lutetium-177 NEMA-phantom
example: all 96 camera angles, 128 by 128 detector pixels, 128 cubed object,
CT attenuation, depth-dependent Gaussian collimator response, additive
triple-energy-window (TEW) scatter, four iterations of eight-subset OSEM.
This is a modern source port, not a claim to execute the paper-era release.
The complete primary paper and Appendix A.2 were read; Figure A.6 concerns
this acquisition. The historical Siemens reconstruction is an external
comparison, not truth or a fitted calibration. A MIM reconstruction is not
supplied with the selected public data and is not fabricated.

## Observations, coordinates and calibration

The storage interface is batch first. `energy_windows` is uint16
(1,3,96,128,128); the three windows are original peak, lower, upper counts.
Every stored integer is identical to the corresponding original DICOM count.
Original energy-window and detector-head frames were selected and sorted by
the native reader. Original detector rows were reversed and radial/axial axes
transposed, exactly as the author's reader. The supplied angles already have
the reader's `(angle+180) modulo 360` convention and increasing sort. No
cropping, downsampling, fitted registration or output-dependent mask is used.
The input attenuation float32 (1,128,128,128) is independent CT calibration,
not activity derived from the object being reconstructed. Radii retain
float64; camera angles retain float32.

The source independently converts HU to 208-keV attenuation using the scanner
table row for 130-kVp Symbia T2. For HU below zero the slope is 0.00014376;
above zero it is 0.00008787; both intercepts are 0.1352 cm inverse. Negative
converted attenuation is clipped at zero before original affine registration
onto the NM lattice. This full source stage is completed privately, and its
registered result is a public legitimate calibration array. Runtime does not
need a DICOM reader or perform another interpolation of this calibration.

For voxel indices $(i,r,k)$ the axes are x, y, z. Spacing in all axes is
$d=0.47951998710631993$ cm. The native millimetre affine is diagonal with
entries 4.7951998710632 and translation
(-304.49519181251316,-462.5951918125131,110.4048081874937).
The detector azimuth is $\phi=270-\beta$ degrees, with $\beta$ the camera
angle. This subtraction is evaluated in float32 before conversion to a host
float, preserving the native arithmetic. Rotation uses negative phi in
forward projection and positive phi in backprojection.

## Rotate-and-sum forward acquisition

For nonnegative activity $a$, attenuation $\mu$, photopeak counts $g$ and scatter
$s$, the physical discrete model is

```math
g_{jrk}\sim\mathrm{Poisson}(\lambda_{jrk}),\quad
\lambda=Ha+s,\quad (Ha)_{jrk}=\sum_i[K_j(p_j\,R_j a)]_{irk}.
```

The sum has no extra voxel-size factor; the activity is in the author's native
count-contribution units. This is not Bq/ml and no absolute sensitivity or
partial-volume truth is invented. The spatial domain is the full 128 cubed
volume. Before each rotation, zero padding is applied to x/y with

```math
P=\left\lceil\frac{(\sqrt{2}-1)128}{2}\right\rceil=27.
```

The padded lattice is 182 by 182 by 128; z is unpadded. The centre of the
rotation plane is (90.5,90.5). A homogeneous translation $T$ moves the centre
to the origin, a rotation $Q$ maps coordinates, and the inverse translation
restores them. Pixel-space transform is the ordered product

```math
M=T(c)\,Q(\theta)\,I\,T(-c),\qquad
Q(\theta)=\begin{pmatrix}\cos\theta&\sin\theta&0\\-\sin\theta&\cos\theta&0\\0&0&1\end{pmatrix}.
```

Here $\theta$ is signed degrees times the source's float32 pi divided by 180.
The explicit identity factor retains source multiplication order. Pixel
normalisation is

```math
N=\begin{pmatrix}2/(W-1)&0&-1\\0&2/(H-1)&-1\\0&0&1\end{pmatrix},\quad
u_{\mathrm{source}}=[N(MN^{-1})]^{-1}u_{\mathrm{destination}}.
```

The normalisation matrix is formed in float32 and then cast, including in
float64 validation cases. `affine_grid` uses corner alignment and
`grid_sample` evaluates bilinear interpolation with zero extension. This
matches the generic geometric primitives of the original Kornia path without
importing Kornia. Grids are cached by signed angle; no rotated attenuation
probability cache changes the calculation.

The detector lies at increasing x. Path integration uses a reverse cumulative
sum of the rotated attenuation, subtracting half the emitting voxel:

```math
p_{jirk}=\exp[-\operatorname{reversecumsum}(dR_j\mu)_{irk}
+(dR_j\mu)_{irk}/2].
```

This is a centre-to-face path convention. An analytic constant slab verifies
the half voxel and centimetre conversion independently.

For camera radius $r_j$ in cm, distance of padded depth plane $i$ is

```math
D_{ji}=\max(r_j+(90.5-i)d,0),\quad
\sigma_{ji}=\frac{\sqrt{(A D_{ji}+B)^2+C^2}}{d}.
```

Here sigma is in voxel units. Exact coefficients are
A=0.03161982752665455, B=0.1248503046423388 cm,
C=0.16137114205472364 cm. For each view,
$L_j=2\lceil3\max_i\sigma_{ji}\rceil+1$ is the odd support. At integer
offset $t$ from minus floor(L/2) through plus floor(L/2), the kernel is

```math
k_{ji}(t)=\frac{\exp[-t^2/(2\sigma_{ji}^2+\delta)]}
{\sum_h\exp[-h^2/(2\sigma_{ji}^2+\delta)]},\quad\delta=10^{-11}.
```

Two grouped one-dimensional convolutions apply these depth-specific kernels
along y then z with zero padding. Every depth kernel is discretely normalised;
boundary flux can leave the finite domain. Sum along x, then remove 27 pixels
from each radial side; retain all 128 z pixels.

## Scatter and inverse update

TEW uses three independent windows without smoothing or a fit to the activity.
Use exactly archived window endpoints, not rounded nominal widths. If their
widths are $w_p,w_l,w_u$ then

```math
s=(g_l/w_l\cdot0.5+g_u/w_u\cdot0.5)w_p.
```

All terms here are float32 Torch tensors on the selected device. Counts are
converted losslessly from uint16, then divisions and sums preserve native
operation order. No subtraction or clipping of the measured photopeak is used:
scatter remains additive in the Poisson prediction.

Let $B_m$ be the native approximate continuous backprojection for subset $m$.
First pad projections radially by zero. Replicate each detector plane along
all 182 x planes, with a boundary box equal to one in the original radial
interval and zero in radial padding. Apply the symmetric Gaussian blur, then
multiply by the same photon survival map. Rotate back by positive phi, sum
views in native order, then unpad x/y. The inverse rotation is **not** the exact
transpose of bilinear interpolation. Independent weak-integrated refinement
supports its continuous-adjoint interpretation, while pointwise discrete
transpose differences persist. It must not be described as an exact gradient
or silently replaced by automatic differentiation.

The objective motivating the inverse is the unnormalised Poisson negative
log likelihood, omitting the count factorial term:

```math
\mathcal L(a)=\sum_{jrk}\{(Ha+s)_{jrk}-g_{jrk}\log[(Ha+s)_{jrk}+\delta]\}.
```

No prior, spatial penalty, objective division or line search is present. Source
OSEM uses eight angle subsets $m,m+8,\ldots$ for $m=0,\ldots,7$, four complete
iterations (32 updates) and initial $a=1$ at every voxel. Cached sensitivities
are $n_m=B_m\mathbf1$. For the current subset calculate

```math
q_m=\frac{g_m}{H_m a+s_m+\delta},\qquad
v_m=B_m q_m-n_m,\qquad
a\leftarrow a+[a(1/(n_m+\delta))]v_m.
```

After each update, set values less than or equal to zero to zero, as the
published estimator does; no voxel is removed from the output/evaluation
support. The reciprocal multiplication is retained rather than reordering
as a division, preserving float32 results. Fixed iteration count is the
stopping rule; there is no convergence tolerance. CPU threads are one. The
default device is CUDA when available and CPU otherwise. No domain package
is imported by cleaned runtime or tests.

## Evaluation and actual comparisons

The hidden reference is the independently executed original author's result,
not physical truth. The nominal 10:1 phantom filling ratio cannot be used as a
voxelwise truth map. NCC is cosine similarity without centring. NRMSE is RMS
error divided by full reference range. Maximum error is divided by the same
range. The full forward L2 error is computed in float64 reductions relative
to the measured photopeak norm, with all detector bins included. Independent
CPU/GPU source runs justify pre-extraction numerical limits; bitwise GPU
repeat alone is not their rationale. All four limits are fixed in metrics.json.
Truth takes precedence whenever the actual truth file exists, including when
invalid: no fallback is allowed for invalid truth.

The notebook actively computes the full standard reconstruction, all subset
states, physical forward residuals, actual removal-of-physics ablations,
iteration and subset comparisons. Ablations change explicitly labelled
acquisition assumptions, not the benchmark method. The Siemens result is
historical and uses the original author's stated division by 96; there is no
least-squares fitted scaling or claim it is fresh. Observations, calibration,
method-reference agreement and physical accuracy are kept distinct. Remaining
limitations include approximate continuous backprojection, fixed TEW scatter,
Gaussian PSF approximation, finite iterations and unknown voxelwise activity.

The physical output additionally requires zero negative voxels (negative_count<=0); this is independent of reference-agreement tolerances. Invalid metric domains and nonfinite derived metrics are evaluator errors with empty metrics, not quality passes or nonstandard JSON numbers.
