# Implemented mathematics

## Binding and scientific limits

The bound inverse uses the exact author `tutoInversionCreep` observations and
native geometry: three LOS tracks with1341/770/1365pixels and39GNSS stations,
1292main triangular patches,5deep rectangles, and one eastern and one western
rectangle. All inputs are existing real curated velocity products. The target
is the tutorial's last conditional tied-deep NNLS estimate, not geological
truth or the article's posterior mean. The published article instead uses
node-interpolated slip, layered elasticity, additional nuisance terms and an
AlTar posterior; none of those unimplemented paths is claimed here.

We retain all three actual author comparison groups: zero-prior unconstrained
GLS, shifted-prior NNLS, and shifted-prior NNLS with a shared deep rate.
The article's time-series/slow-event analysis is a different observation and
unknown; it is not part of this velocity-only tutorial binding.

## Observations and coordinates

All geometry coordinates are km with axes east,north,depth positive down.
The data already contain a known WGS84 transverse-Mercator projection centered
at33°E,40.8°N, scale1. No projection library is required at reconstruction.
Elastic response components are east,north,up. Every receiver is on depth0.
The data vector is all three LOS blocks followed by39east and39north GNSS
velocities. Velocity units are mm/year. GNSS vertical components are excluded
by the selected original pipeline, not masked after reconstruction.

## Surface elastic response

Let a triangle have vertices $p_1,p_2,p_3$ in positive-down coordinates.
Its unit normal is

$$n=\frac{(p_2-p_1)\times(p_3-p_1)}{\|(p_2-p_1)\times(p_3-p_1)\|}.$$

If its depth component is negative, swap vertices2and3 and reverse the normal.
With $\gamma=\operatorname{atan2}(n_y,n_x)$, the strike and dip directions are
$e_s=(-\sin\gamma,\cos\gamma,0)$ and $e_d=n\times e_s$.
For source-convention strike,dip,opening slips $(s_s,s_d,s_o)$, the global
Burgers vector is $v=-[e_s,e_d,n](s_s,s_d,s_o)^T$. The source's negative signs
and final upward-component sign conversion are preserved. The selected
unit response has $s_s=1,s_d=s_o=0$; the tutorial then negates this column to
make the reconstructed rate right-lateral positive.

For a directed triangle edge, write its horizontal azimuth as $\alpha$ and
its dip as $\delta$. Horizontal rotation is

$$Q(\alpha)=\begin{pmatrix}\cos\alpha&-\sin\alpha\\\sin\alpha&\cos\alpha\end{pmatrix}.$$

The source `RotateXyVec` evaluates this rotation and adds one double-precision
unit spacing to each returned component; that numerical convention remains.
Coordinates are translated to each endpoint and rotated by $Q(-\alpha)$.
The angular-dislocation inclination is $\beta=\pi/2-\delta$ for nonnegative
dip and $\beta=-\pi/2-\delta$ otherwise, with the source's range adjustments.
An edge with inclination within1e-6radians of0or pi contributes zero under
the original branch convention. Edge-parallel, horizontal-normal and vertical
unit directions are $l_s=(\cos\alpha,\sin\alpha,0)$,
$l_t=(-\sin\alpha,\cos\alpha,0)$ and $l_d=l_s\times l_t$.
The angular slip components are $(B_1,B_2,B_3)=(v\cdot l_s,v\cdot l_t,v\cdot l_d)$.

`adv` explicitly evaluates the corrected angular-dislocation half-space
formula. To make its numerical decomposition precise, its coordinates are
$y_1,y_2,y_3$, endpoint depth $a$, inclination $\beta$, and Poisson ratio
$\nu=0.25$. It defines

$$z_1=y_1\cos\beta-y_3\sin\beta,\quad z_3=y_1\sin\beta+y_3\cos\beta,
\quad R=\sqrt{y_1^2+y_2^2+y_3^2},$$

$$\bar y_3=y_3+2a,\quad \bar z_1=y_1\cos\beta+\bar y_3\sin\beta,
\quad \bar z_3=-y_1\sin\beta+\bar y_3\cos\beta,
\quad\bar R=\sqrt{y_1^2+y_2^2+\bar y_3^2}.$$

The angular terms are

$$F=-\operatorname{atan2}(y_2,y_1)+\operatorname{atan2}(y_2,z_1)
+\operatorname{atan2}(y_2R\sin\beta,y_1z_1+y_2^2\cos\beta),$$

with $\bar F$ replacing $R,z_1$ by their barred versions. For each of the
three Burgers-vector components, the executable local formula adds the
infinite-medium/image contribution scaled by $[8\pi(1-\nu)]^{-1}$ to the
traction-free surface correction scaled by $[4\pi(1-\nu)]^{-1}$. Its expanded
terms consist only of these variables, logarithms of $R-y_3$, $R-z_3$,
$\bar R+\bar y_3$, $\bar R+\bar z_3$, and the associated rational factors.
The three component responses are multiplied by $B_1,B_2,B_3$ and summed.
These complete expanded algebraic terms are executable locally in `adv`,
not an imported Green-function implementation or numerical lookup table.

Subtract the two endpoint angular responses, rotate the horizontal response
back with $Q(\alpha)$ and sum over the three directed edges. This cancels the
common vertical legs of the dislocation loop and yields the triangle response.
Negate the returned depth displacement to obtain up. The source's internal
prism translation applies only below a buried triangle; with the selected
free-surface receivers and nonnegative vertex depths it is identically
inactive. The local surface-only operator therefore has no polygon library
or interior-observation branch.

Each known rectangle is split along vertices0–2 into triangles(0,1,2) and
(0,2,3), whose contributions add. This is the exact common-edge cancellation
identity, not a mesh approximation. Before local extraction, all seven
rectangles at all3515receiver positions were compared using the independent
original Okada and original triangular formulas; maximum absolute difference
was7.8830193e-13per unit slip. Additional local checks cover vertex reversal,
translation invariance and slip linearity at floating-point precision.

## Nuisance basis and dimensional closure

The fixed basin center is(-30.343562481406668,8.946455159060937)km.
Translate coordinates to this center and convert them to metres, denoting
the differences by $X,Y$. Set $h=3000$m,$r=6000$m,$\mu=30\times10^9$Pa.
The formal deflation basis is

$$b_{ENU}=-10^6\frac{(1-\nu)r^3(X,Y,h)^T}{\mu(X^2+Y^2+h^2)^{3/2}}.$$

It is in mm/kPa, and its coefficient is a formal kPa/year rate. The original
tutorial's `-1e6` column has exactly this numerical interpretation:
$1000$mm/m times$1000$Pa/kPa. The tutorial print label MPa is dimensionally
inconsistent with fitting mm/year values; no matrix or result is rescaled.
An independent on-axis formula and mirrored radial directions test the
factor and deflation sign. Since radius exceeds depth, the small-cavity
physical approximation fails. This is an empirical subsidence shape with
fixed normalization; its coefficient is not validated physical pressure.

Project each elastic/basin response onto the supplied LOS vector at a radar
pixel; select E and N at GNSS stations. Add three track-indicator columns and
two GNSS translation-indicator columns. The full matrix $A$ is3554by1305:
1299slip columns,one basin column,five datum columns.

## Noise and conditional prior

The data covariance $C_d$ is block diagonal. Within each radar track its
off-diagonal entries are $\sigma^2\exp(-d/\ell)$, where $d$ is horizontal
separation in km. Its diagonal is replaced by the supplied total variance,
not added to the exponential variance. The three $(\sigma,\ell,\mathrm{std})$
triples are explicit constants in preprocessing and README. GNSS E/N
variances are the squared supplied uncertainties, with component-first order.

Let $q_i$ be main-patch arithmetic centers. Define $\ell_0$ as the norm of
three coordinate spacings, each coordinate's span of distinct center values
divided by its number of distinct values. For horizontal and vertical center
separations $H_{ij},V_{ij}$, the selected executable prior is

$$C_m(i,j)=\frac{(20\ell_0)^2}{10\times1}
\exp[-H_{ij}/(2\times10)]\exp[-V_{ij}/(2\times1)].$$

All lengths here are km;20has units mm/year. This preserves the actual source
half-exponents and product amplitude. The upstream docstring instead writes
other powers/exponents; the executable tutorial is the selected definition.
Other original prior variances are100for each deep slip,.01for each remote
slip,1for basin,1for each datum,in their respective squared units. They are
variances, not standard deviations to be squared again.

## Inverse groups and exact equality

For a prior mean $m_0$, minimize the strictly convex quadratic

$$J(m)=\tfrac12(Am-d)^TC_d^{-1}(Am-d)
+\tfrac12(m-m_0)^TC_m^{-1}(m-m_0).$$

Its gradient is $A^TC_d^{-1}(Am-d)+C_m^{-1}(m-m_0)$.
The GLS comparison has $m_0=0$ and uses the original inverse-normal-matrix
formula $(A^TC_d^{-1}A+C_m^{-1})^{-1}A^TC_d^{-1}d$.
For both NNLS groups, set the five datum prior entries to−100and all others
to zero. Thus their prior shift and nonnegative constraints both differ
from GLS; comparisons are not attributed solely to nonnegativity.

If $L L^T=C_d^{-1}$ and $M M^T=C_m^{-1}$ are lower Cholesky factorizations,
set $F=[L^TA;M^T]$ and $b=[L^T(d-Am_0);0]$. SciPy's general-purpose NNLS
solves $\min_{z\ge0}\|Fz-b\|_2$, and $m=z+m_0$. The source's default NNLS
iteration budget is retained through the generic API default; there is no
custom reconstruction update, training, early stopping or hidden initializer.

For the final selected group, remove the five deep columns and append their
sum, producing1301variables. Delete the associated covariance rows/columns
and append variance100for the one shared deep rate. A1305by1301matrix $E$
maps the reduced solution to all original parameters. Its five deep rows
select the same final reduced variable; other rows are identity selections.
It is incorrect to compute the new covariance as $E^TC_mE$, which would sum
the five old variances. The original explicit100variance is preserved.
The restored1299slips and all six nuisance coefficients are output.

KKT diagnostics use the projected-gradient infinity norm
$\|z-\max(z-\nabla J,0)\|_\infty$, checking both active and free coordinates.
An independent small quadratic central-difference case validates the
objective/gradient, and a known mapping case validates exact equality.

## Experiments and evaluation

The notebook actively rebuilds the full operator and all three inverse groups.
It compares each fresh group against separately labeled original arrays,
plots all three LOS data/residual groups and GNSS vectors, and studies
diagonal-noise, changed prior strength and a seed20261022correlated-noise
perturbation with covariance0.15²times the known observation covariance.
The diagnostic generator changes only copied velocity observations. Its
result is a sensitivity experiment, not additional benchmark data or truth.

The hidden standard-method reference is not ground truth. Original native
and independent original reruns are bitwise equal. A before-local upstream
float32-serialization run establishes nonzero numerical error. Its main-slip
dynamic-range NRMSE is5.8648899380666354e-9; the frozen boundary is1.5times
that value. Each constant/deep/remote/basin/datum quantity has its own
absolute boundary from the same independent measurement. No pooled range
conceals weak or scalar quantities. All field groups and noise-weighted
observation RMS must pass. The forward criterion is
$\sqrt{(d-Am)^TC_d^{-1}(d-Am)/3554}\le1.2399866256976022$.
Zero,mean,reversed,sign-wrong and wrong-unit controls fail. Earlier GLS or
untied estimates also fail this particular conditional target definition.
No spatial alignment,rescaling,cropping or sign ambiguity is applied.
