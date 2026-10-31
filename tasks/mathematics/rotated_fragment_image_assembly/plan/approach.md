# Implemented mathematics

## Observation and pose

There are $N=432$ original uint16 RGB tiles, each $P=28$ pixels square.
The complete image has 18 by 24 cells. Tile index is public observation order.
For image $X$, block extraction $B_p$, and counterclockwise rotation $R_s$,
```math
Y_i=R_{s_i}B_{p_i}X.
```
The returned orientation $q_i=-s_i\bmod4$ acts directly on the observed tile.
Output positions are zero-based row/column pairs. Internal assembly grids instead
contain one-based tile IDs and zero holes. All rotations preserve pixels exactly.
No radiance conversion, noise likelihood, interpolation or learned prior is used.

## Edge statistics and compatibility

For every tile, top/right/bottom/left boundary strips are stored in consistent
clockwise traversal. Their reversed strips reverse pixel order within each RGB
channel. Inward-to-outward one-pixel differences are formed in float32 from native
code values; their 28-sample RGB means are also evaluated in float32. The arrays
are subsequently stored in float64, as in the source's preallocated arrays.
Nine rows are appended to the 28 gradient rows:
```math
(0,0,0),(1,1,1),(-1,-1,-1),(0,0,1),(0,1,0),(1,0,0),
(-1,0,0),(0,-1,0),(0,0,-1).
```
The covariance is the centered sample covariance of these 37 float64 rows,
divided by 36. The augmented rows make otherwise uniform-color edges usable.
Covariances have squared code-value units; RGB means have code-value units.

For a candidate seam, let $x_t$ be the RGB boundary difference at pixel $t$.
Let $\mu_i,C_i$ and $\mu_j,C_j$ be the two corresponding edge statistics.
The selected Mahalanobis gradient compatibility is
```math
d(i,j)=\sum_{t=1}^{P}
\left[\sqrt{(x_t-\mu_j)^T C_j^{-1}(x_t-\mu_j)}
+\sqrt{(-x_t-\mu_i)^T C_i^{-1}(-x_t-\mu_i)}\right].
```
It is dimensionless. Explicit Cholesky whitening evaluates these quadratic forms; no
explicit covariance inverse or domain solver is used. Square roots occur before
the sum over pixels. Final scores are float32. Self-pairs are positive infinity.
Four seam directions times four relative rotations give 16 channels, ordered
direction + 4 times relative turn, with direction top/right/bottom/left.
Reverse-pair channels use the exact source mirror permutation.

The shipped numerical primitive is explicit-spd-v1.2. For a real symmetric
positive-definite covariance $C=(c_{ab})$, form lower-triangular $L$:
```math
\ell_{00}=\sqrt{c_{00}},\quad \ell_{10}=c_{10}/\ell_{00},\quad
\ell_{20}=c_{20}/\ell_{00},\quad
\ell_{11}=\sqrt{c_{11}-\ell_{10}^2},
```
```math
\ell_{21}=(c_{21}-\ell_{20}\ell_{10})/\ell_{11},\qquad
\ell_{22}=\sqrt{c_{22}-\ell_{20}^2-\ell_{21}^2}.
```
For residual $u=(u_0,u_1,u_2)$, explicit forward substitution gives
```math
z_0=u_0/\ell_{00},\quad
z_1=(u_1-\ell_{10}z_0)/\ell_{11},\quad
z_2=(u_2-\ell_{20}z_0-\ell_{21}z_1)/\ell_{22},\qquad
M(u,C)=z_0^2+z_1^2+z_2^2=u^TC^{-1}u.
```
All arithmetic is float64. Covariance and residual must be finite and real;
covariance symmetry and positive finite pivots are checked. $L$ has DN units,
while $z$ and $M$ are dimensionless. Zero residual returns exactly zero.
The original negative-quadratic-form guard remains. This is a squared norm,
not clipping or adding jitter to a negative value. Full frozen float32 MGC is
bitwise identical. The earlier owned-input-v1.1 experiment failed an additional
CLI and was rejected; both native failure records and finite stability results
remain archived, without an asserted native root cause.

## Initial connection graph and synchronization

The initial graph combines best-pair orientation proposals and four directional
best proposals. Ambiguous nearest edges are weakened; conflicting proposals in
the same direction are pruned. A two-hop common-neighbor check contributes 0.8
and the original tested graph contributes 0.2. If disconnected, minimum-distance
bridges of weight 0.001 are added. Ties may use a random choice; statistical graph
properties are tested rather than random trajectory equality. The bound graph is
already connected before this optional bridge operation.

Specifically, a directional score's best/second-distinct-best ratio above 0.95
replaces its connection by $(1.1z_1+0.9z_2)/4$, for the two complex C4
proposals $z_1,z_2$. A unilateral neighbor weight becomes 0.08 after
symmetrization. The four-cycle routine visits ordered pairs in source order:
inconsistent two-path orientations halve each involved edge weight immediately.
All weights are then divided by 1.5; edges on retained consistent squares are
restored to unit weight. The synchronization weight below is specifically the
current neighbor matrix plus half the diagonal-neighbor matrix.

Four-edge cycles add agreement evidence and diagonal-neighbor weights. Write
the resulting nonnegative weights as $W$, the complex quarter-turn connection
as $T$, and the degree matrix as
```math
D_{ii}=\sum_j |W_{ij}T_{ij}|,\qquad
L=D-W\odot T,\qquad H=D^{-1/2}LD^{-1/2}.
```
The minimum **left** singular vector of $H$, multiplied by $D^{-1/2}$, supplies
complex orientation estimates. The author requests five smallest singular
triplets in descending order and selects the fifth returned column: this is the
smallest mode, not the fifth-smallest mode. General full SVD is the validated
numerical backend. Phases are rounded to the nearest quarter-turn, retaining
the source's strict midpoint comparisons. If the rounded source number is
$r_i$, the actual tile action is $R_{-r_i}$; cumulative output turns therefore
update as $q_i\leftarrow q_i-r_i\bmod4$.

## Position relaxation and discrete assembly

Already oriented tiles use four score layers. For each direction, normalized
compatibility divides the score by the smaller competing-best score for either
endpoint. The selected candidate is the minimum normalized score. Its weight
is $\max(\exp(-5d^2),10^{-15})$; floor-weight candidates are removed.
The source threshold 0.8 belongs to an unused weighting mode and is not a cutoff
in this selected probabilistic mode.

The normalization adds 1e-15 to both numerator and denominator. The original
infinite self-pair arithmetic produces NaN diagonal entries, which remain
explicitly excluded by the selected candidate ordering rather than being
silently replaced by zeros.

For a proposed directed neighbor pair $i,j$ with unit displacement
$\delta_{ij}$, the relaxation minimizes
```math
\sum_{(i,j)} w_{ij}
\left(|x_j-x_i-\delta^r_{ij}|+
|y_j-y_i-\delta^c_{ij}|\right).
```
Here $x,y$ are row and column coordinates in grid cells. Both coordinates are
free real variables; nonnegative positive/negative residual variables express
the absolute values. Explicit anchor equalities set one row and column to
-100000, removing translation. They do not identify any true tile location.
The sparse equality/inequality matrices are implemented locally; SciPy's general
HiGHS dual-simplex primitive solves the LP, with one thread and primal/dual
tolerances $10^{-8}$. Termination success and finite outputs are required;
maximum equality or inequality violation above $2\times10^{-6}$ is rejected.
LP nonuniqueness is assessed by feasibility and objective, never by requiring
an arbitrary basis solution to equal another solver's coordinates.

Integer-consistent relative positions define connected blocks. The source
candidate-update loop has at most five updates after its initial LP and stops
early when no candidates remain or component labels cease changing. Greedy
block joins preserve the selected source tie/order conventions; trimming uses
separate row and column occupancy marginals. Missing cells are filled by local
compatibility. Selected runtime uses no replicated pieces and no unknown
rotation inside this position stage. Historical source details, including the
one final greedy join after its continuation flag is cleared, are preserved.

## Two orientations and five refinements

Each stage runs the position method twice: current synchronized tiles and all
tiles additionally turned counterclockwise once. The score-channel transformation
is explicitly permuted. A source compatibility cost selects the second candidate
only if it is strictly smaller and its grid area is no smaller. The historical
cost counts left/up seams but omits the last forward row/column terms; this
documented source criterion is not relabeled as the paper's full-neighbor sum.
An accepted global turn increments every cumulative $q_i$.

After the initial assembly, all five selected refinements execute. They detect
high-cost tile neighborhoods using 1.5 times source median costs, retain the
largest connected block, update connection weights (0.6 previous plus 0.4
placement evidence), add hole-based evidence at the source 0.6/0.3 levels, and
repeat four-cycle synchronization and both position candidates. A history
contains the initial stage and five refinements. No true positions, rotations,
source photograph, historical inverse result or fixture enters this path.

## Evaluation and evidence

A single global C4 action is chosen jointly for all metrics. Position and
orientation must be simultaneously correct for joint-piece accuracy. Oriented
neighbor accuracy counts each true grid seam once and requires its directed
displacement and both tile orientations to be correct. Both rates must be at
least 0.95. Position, orientation, ordinary neighbor accuracy, and dynamic-range
image NRMSE are diagnostics. The independent source method attains both required
rates 1; thresholds were frozen before local implementation, rather than using
an arbitrary exact-reference tolerance.

Parity covers edge attributes, all 2,979,072 finite MGC entries and 6,912 infinite
diagonal entries, initial and refined graph operators, minimum-mode subspaces,
LP matrices/objectives/feasibility, components and assembly. Source compatibility
adaptations and failed native attempts remain privately archived. The shipped
local implementation explicitly evaluates all mathematics without invoking
Octave, the original repository, or a domain reconstruction library.
