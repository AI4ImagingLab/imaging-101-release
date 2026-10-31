# Implemented mathematics: Hansen-Law emission reconstruction

## Scope and physical unknown

This is a modern, independently checked extraction of the PyAbel v 0.9.1
Hansen-Law transform, using its first-order hold option. It reproduces that
implementation on a new synthetic emission acquisition, not the original
1985 numerical experiment or all eight methods of the 2019 comparison paper.
The paper's Gaussian accuracy comparison motivates the analytic check, while
the benchmark object includes a hollow radial structure and an axial core.
Zero-order hold is an actively computed notebook comparison.

The unknown f (r,z), in relative radiance/mm, is stationary and cylindrically
symmetric. With x along the line of sight and y the impact parameter (all
coordinates in mm), r=sqrt (x^2+y^2). The calibrated point-sampled projection is

```math
P(y,z)=\int_{-\infty}^{\infty} f(\sqrt{x^2+y^2},z)\,dx
      =2\int_y^\infty \frac{r f(r,z)}{\sqrt{r^2-y^2}}\,dr.
```

The measured p=P+e has independent normal noise e of standard deviation
0.002 radiance units. Batch dimension is 1. Rows i=0,...,64 are z=(i-32)*0.1
mm; columns j=0,...,256 are y=r=j*0.025 mm. No top/bottom averaging,
centering, smoothing, clipping, rebinning or intensity normalization occurs.
Computations and stored observations use float64. Radial tails are negligible
at 6.4 mm; no periodic extension is used.

## Continuous inverse and discrete driving function

```math
f(r,z)=-\frac{1}{\pi}\int_r^\infty
\frac{\partial P(y,z)/\partial y}{\sqrt{y^2-r^2}}\,dy.
```

The state-space coefficients below incorporate the negative sign and 1/pi
factor through their kernel approximation. Do not add another 1/pi factor.
Let d=0.025 mm and C be the number of columns. For first-order hold, the
inverse drive u is the default NumPy gradient of p along the last axis:
u (j)=(p (j+1)-p (j-1))/(2 d) in the interior, with one-sided first-order
differences at j=0 and C-1. The zero-order variant uses forward differences
u (j)=(p (j+1)-p (j))/d for j<C-1, and u (C-1)=0.
The inverse drive has radiance/mm units. The forward drive is u (j)=-2*pi*d*f (j),
with radiance units. No derivative in z is taken.

## Nine-state kernel and recurrence

State index k=0,...,8 uses the fixed coefficient vectors

```math
h=(0.318,0.19,0.35,0.82,1.8,3.9,8.3,19.6,48.3),
```
```math
\lambda=(0,-2.1,-6.2,-22.4,-92.5,-414.5,-1889.4,-8990.9,-47391.1).
```

For descending integer n=C-1,...,2, let q=n/(n-1). With a=0 for inverse
and a=1 for forward, define the signed interval integral and transition

```math
I_k(n,a)=\begin{cases}
-\log q,&a=0,\ k=0,\\
(n-1)^a\dfrac{1-q^{\lambda_k+a}}{\lambda_k+a},&\mathrm{otherwise},
\end{cases}
\qquad \Phi_k(n)=q^{\lambda_k}.
```

Let Gamma0(k,n)=h (k)*I(k,n,a). For first-order hold additionally set
Gamma1(k,n)=h (k)*I(k,n,a+1). The coefficients are

```math
B_{0,k}(n)=\Gamma_{1,k}(n)-(n-1)\Gamma_{0,k}(n),\qquad
B_{1,k}(n)=n\Gamma_{0,k}(n)-\Gamma_{1,k}(n).
```

For zero-order hold, B0=0 and B1=Gamma0. Each row has nine states s (k),
initialized to zero outside the computed sweep. For j=n-1, update all rows:

```math
s_k\leftarrow\Phi_k(n)s_k+B_{0,k}(n)u(j+1)+B_{1,k}(n)u(j),
\qquad v(j)=\sum_{k=0}^{8}s_k.
```

The sum v is emission for the inverse direction and projection for the
forward direction. The sweep computes j=C-2,...,1 once, then sets v (0)=v (1)
and v (C-1)=v (C-2), exactly preserving the selected release's endpoint
convention. These copies are numerical endpoint approximations, not claims
that the physical derivative vanishes at the detector boundary. The known
nonzero but tiny analytic tail remains part of the Gaussian accuracy check.
The random signed probe tests ensure the convention also holds for nonzero
outer boundary data, rather than testing only negligible-tail images.

## Objective, initialization, stopping and parameters

This is direct linear approximate inversion of an integral equation, not
optimization. There is no loss normalization, regularizer, learned prior,
learning rate, line search or convergence threshold. States start at zero;
termination is the end of the single inward radial sweep. The default hold
order is 1, an explicit choice rather than upstream's default of 0. Both
choices are compared to upstream with the same option. No iterations are
added artificially. Work is O(9*65*257), with nine state values per row and
coefficient arrays sized (255,9). Negative noise fluctuations are retained to
preserve linearity.

## Generation, evaluation and validation

`src.generate_data.synthetic_fields` constructs analytic Gaussian components
and their exact infinite-line integrals, independently of this recurrence.
The precise object and acquisition seed are hidden, documented in
`test/validation.md`, and do not enter the solver. Numerical quadrature along
x checks the analytic projection. A Gaussian pair checks absolute scale and
coordinate conventions; spacing, linearity and endpoint tests add independent
invariants. No adjoint or optimization gradient is required by this direct
method; the projection derivative and state update are checked explicitly.

NCC is flattened cosine similarity without mean subtraction. NRMSE is RMS
error divided by the reference's dynamic range. All (1,65,257) samples are
evaluated against actual synthetic emission truth, with no crop or alignment.
A zero estimate has NCC=0; constant references are evaluator errors. Upstream
baseline metrics determine boundaries 0.9*baseline_NCC and
1.1*baseline_NRMSE before local implementation. Numerical parity uses
rtol=atol=1e-12 on float64 arrays, fixed before local implementation.
Gaussian relative L2 errors must be below 0.015; independent line integrals
must agree within 1e-6 radiance units. These are separate checks, not Judge
threshold adjustments.


## Complete acquisition and reconstruction sequence

The maintenance generator and the inverse have distinct caller roles. For Gaussian width w and axial amplitude A(z), the analytic line integral of A(z) exp (-(r/w)^2) is sqrt (pi) w A(z) exp (-(y/w)^2). Superposition therefore generates projection independently of the Hansen-Law approximation, including a negative depletion component whose sum remains nonnegative. Additive noise is drawn only when creating a separate acquisition; the inverse neither regenerates nor looks up the hidden target.

For the fixed benchmark, load the exact float64 array, remove its leading batch axis, form a first-order centered radial derivative, construct transitions for n=256 down to 2, zero all 9×65 state entries, and process 255 interior radial columns from outside inward. At each column update all states for every row before summing the nine states. B0 multiplies the next outer derivative and B1 the current derivative. Copy the adjacent computed output at the two endpoints, restore the batch axis and save the signed density. This explicit sequencing preserves which drive sample accompanies which transition; reversing B0/B1 or adding a second minus/pi changes the method.

The zero-order comparison uses a different driving difference and interval hold together: forward derivative with last drive zero, B0 zero and B1 Gamma0. It is freshly recomputed, not constructed by smoothing the first-order output. The upstream release defaults to zero order; both reference calls explicitly choose their order so the selected first-order benchmark is not a default mismatch.

## Error mechanisms and independent measurements

The continuous inverse is a first-kind ill-conditioned integral inverse with a differentiated noisy drive. First-order interpolation improves smooth-data accuracy but is not a noise regularizer. The nine fixed kernel terms approximate the Abel singularity; finite spacing, a truncated radial domain and endpoint copies contribute distinct approximation errors. Nonnegative true emission does not imply nonnegative noisy direct inverse estimates. No prior is silently used to suppress these fluctuations.

The release comparison keeps four quality groups: independently reproduced upstream first order, fresh first order, independently reproduced upstream zero order and fresh zero order. All use identical bound observations and hidden analytic truth. Separate Gaussian tests isolate absolute normalization/sign/origin from complicated source morphology. Signed probes make the nonzero outer-boundary convention observable. Scaling dr by 2 must multiply forward radiance by 2 and inverse density by 1/2; this dimensional identity detects the common omission of radial pitch. Independent cylindrical flux compares 2∫P(y) dy with 2 pi∫r f (r) dr rowwise, rather than comparing only two implementations of the same recurrence.

Fresh first-order NCC is about 0.99982661 and dynamic-range NRMSE 0.00411980; zero-order NRMSE is about 0.00966408. The notebook reports full precision and produces the Gaussian/residual panels, six physical image panels, z=0 radial profiles and paired NCC/NRMSE bars. These comparisons are retained alongside the fresh standalone Judge. The evidence supports the pinned direct transform on known-axis synthetic optically thin sources. It does not validate unknown-center estimation, non-axisymmetric reconstruction, experimental noise models or radiative-transfer corrections.
