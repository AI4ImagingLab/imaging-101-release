# Implemented mathematics

## Binding and qualification

The target is the signed, conditional single-shell vertical-TEC field for the
complete author-provided 2017-01-02 test day: all313 station directories and all
504358 retained current/reference phase-difference rays. This is real processed
measurement data, without independent electron-content truth. The source is
MosGIM2 fc42e31f3559c2bb57e0cef636c378c3d9bb25cc. Two independent full qualified
source runs completed before local extraction; the unmodified full source run
is retained as a distinct, physically failed continuous-interpolation control.

Four source factors incorrectly assigned the later-node time fraction to the
earlier node and its complement to the later node. The private qualified oracle
exchanges only those current/reference factors. A constant field, a linear
hourly ramp, every node's one-sided continuity, partition of unity, and actual
source-captured rows establish the correction independently. No shell geometry,
observations, weights, harmonic order, regularization or LCP update was changed.

The selected paper parameters are degree/order15, one450000m shell, MODIP
coordinates, 24 hourly intervals and continuous interpolation, with sigma0=.02
TECu and sigmaRel=.005TECu. The supplied default software configuration uses
other layers/weights; this is explicitly the article's single-layer parameter
binding, not an untouched default example. The full test package has313
GPS/GLONASS stations; the article's real illustration uses286 GPS stations.
The executable preprocess decimates at600s and does not implement its Savitzky–
Golay TODO. We preserve the supplied processed measurements, not a fabricated
reproduction of the article's smoothing step.

## Representation, coordinates and forward operator

Earth radius R=6371000m and shell height h=450000m define the mapping factor

```math
s(e)=[1-(R\cos e/(R+h))^2]^{-1/2}.
```

Elevation e, local-time longitude lambda and MODIP colatitude theta are supplied
in radians. Local-time longitude is a periodic coordinate; MODIP is not
geographic latitude. The calibrated ray coordinates are independent of the
unknown field. One TECu means10^16 electrons/m². No amplitude normalization is
performed.

For n=0,...,15 and m=-n,...,n, the real basis H uses normalized complex spherical
harmonics with the Condon–Shortley convention. Negative orders use
sqrt(2)(-1)^m Im(Y(n,abs(m))), positive orders use
sqrt(2)(-1)^m Re(Y(n,m)), and zero orders use Re(Y(n,0)). Columns are ordered by
degree n, then signed order m. There are256 columns. The local implementation
evaluates the associated Legendre polynomials explicitly:

```math
P_m^m=(-1)^m(2m-1)!!\sin^m\theta,\quad
P_{m+1}^m=(2m+1)\cos\theta P_m^m,
```
```math
(n-m)P_n^m=(2n-1)\cos\theta P_{n-1}^m-(n+m-1)P_{n-2}^m.
```

Multiply by sqrt((2n+1)(n-m)!/(4pi(n+m)!)) and the sine/cosine of m*lambda.
The recurrence is mathematically the selected normalized harmonic definition,
not a replacement physical model. Frozen legacy scipy sph_harm values and
independent analytical degree-zero/one cases validate phase and normalization.
The shared SciPy's newer spherical-harmonic primitive was initially tested and
passed but was slow on this large batch; its interrupted local run is retained.

Let c(j,k) be the TECu coefficient of mode k at hourly node j=0,...,24. For time
t in seconds, j=floor(t/3600), u=t/3600-j. The observation row is the sum of four
256-mode blocks: +(1-u)sH at node j, +u sH at node j+1, and the analogous two
negative reference-ray blocks. All retained observation times are below86400s.
The CSR constructor sums repeated column entries. Thus an identical current and
reference ray cancels exactly. Coefficients flatten in hour-major order, giving
6400 unknowns. A has504358rows and6400columns. A*c predicts observed slant-TEC
differences in TECu, not absolute phase. No per-ray bias is fitted.

## Objective and normal equations

With y the measured differences, diagonal W has

```math
W_{ii}=\sigma_0^{-2}\frac{\sin^2e_i\sin^2e_i^{r}}
{\sin^2e_i+\sin^2e_i^{r}},\qquad \sigma_0=0.02\ \mathrm{TECu}.
```

D differences consecutive coefficient blocks without identifying the two
midnight endpoints. Its6144 rows are dimensionless first differences. The
selected objective, without a half factor or observation-count normalization, is

```math
J(c)=(Ac-y)^T W(Ac-y)+\sigma_{rel}^{-2}\|Dc\|_2^2,
\quad\sigma_{rel}=0.005\ \mathrm{TECu}.
```

This diagonal weighting ignores covariance caused by shared reference rays.
The scales are conditional assumptions; actual residuals greatly exceed .02TECu.
All outliers remain in the objective. Form N=A^TWA+sigmaRel^-2 D^TD,
b=A^TWy and rPr=y^TWy. The unconstrained solution is c0=N^-1 b. The source's
dense inverse is preserved. No optimizer, line search, random initialization or
iterative normal-equation stopping rule is introduced.

For numerical source parity, partition observations using the source memory
rule: nchunks=int(504358/(30000*.1))=168. Start with3002rows per chunk and merge
the final small remainder into the last3024row chunk. Sum the chunks in input
order. Add the four signed precision entries per adjacent-hour coefficient:
+w on both diagonal entries and -w on both symmetric off-diagonals, with
w=sigmaRel^-2. The source scalar dispersion diagnostic is
(rPr-c0^T b)/DOF, DOF=-6400+504358+6144=504102. It uses the unconstrained fit and
includes the prior; it is not an empirical, independent measurement uncertainty.

## Author's one-pass finite-grid LCP

The constraint grid has colatitudes2.5:2.5:177.5degrees and longitudes0:5:355
degrees, evaluated at all25 time nodes:127800 samples. Let G be its block-diagonal
harmonic sampling matrix. Select I solely where G*c0<0, before correction. There
are2620 selected nodes for the independent source run. With Gs=G[I,:], define

```math
q=G_s c_0,\quad M=G_sN^{-1}G_s^T,\quad
z\ge0,\quad Mz+q\ge0,\quad z^T(Mz+q)=0,
```
```math
c=c_0+N^{-1}G_s^Tz.
```

Here z is the nonnegative multiplier vector in the source scaling. Lemke's
explicit tableau starts as [I,-M,-1,q], with labels for w,z,artificial-driver and
right-hand-side columns. Initialize by eliminating the most negative q row and
pivoting its complementary variable into the driver column. Subsequent rows
use the minimum nonnegative ratio q(row)/driver(row) among strictly positive
driver entries. Strict '<' preserves the source's first-row tie rule. Normalize
the selected row, clear the driver column from all other rows, swap complementary
columns and update their positions. Stop when the artificial variable leaves
the basis; return the basic z values. The1000000step maximum, secondary-ray and
budget-failure conditions match the selected source. The cleaned implementation
raises on failure instead of silently returning an uncorrected result.

This is one pass on initially negative nodes. It is not a strict all-grid
nonnegative solution: the original result contains one newly negative other
node at approximately-.011068TECu. Preserve it. No clipping, re-selection,
postfit support mask or repeated positivity solve is added. Selected-node KKT
stationarity and complementarity are independently checked, and the full-grid
limitation is displayed in the notebook.

## Output, comparisons and evaluation

Evaluate all coefficients on the full declared grid to obtain shape
(1,25,71,72), float64, TECu, with axes batch/hour/north-to-south colatitude/local
longitude. No additional coefficient file is required from a solver. The hidden
reference is an independently executed qualified standard method, not truth.
Every hour receives its own cosine NCC and own-reference-range NRMSE. Frozen
limits are NCC>=.999999999999 and NRMSE<=2.9645042444036815e-8. Independent full
source reruns are bitwise equal; the nonzero error boundary is1.5times the
measured maximum per-hour map error from float32 serialization of original
coefficients. It was fixed before local scoring. Zero, uniform, shuffled,
time-reversed and sign-wrong fields must fail.

The independent Judge first tests every map. For passing fields it fits their
degree15 harmonic representation using every grid sample, evaluates every ray
pair and requires weighted RMS<=1.5272988162761219TECu. The unscaled elevation
weight is sin²(e)sin²(er)/(sin²(e)+sin²(er)); sigma0 cancels in this normalized
RMS. This uses public ray calibration. Reference and forward consistency have
different meanings; neither proves unique physical accuracy.

The notebook actively reconstructs the full day, then computes fresh no-LCP
and temporal-weight comparisons using the same observations.
Independent original arrays are identified separately. The failed source-time
convention is historical comparison evidence, never a fresh physical method.
Every figure has numeric source arrays and units. New synthetic perturbations,
if used, must go through the documented generator; the main real binding is
never changed. Scientific changes from the article (multi-GNSS package,
executable decimation and unavailable author synthetic NeQuick truth) are
disclosed rather than silently dropped or labeled as paper replication.
