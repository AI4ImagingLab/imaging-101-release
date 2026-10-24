# Implemented mathematics

The task reconstructs the consensus coordinates of the complete author-distributed five-chromosome sample. The350 bins retain their native identities;343 have positive incident counts. All350 coordinates are present during initialization and optimization, because removing unobserved bins before drawing the random initialization changes the nonconvex problem's selected solution. Only the seven input-unobserved coordinates are replaced with zero in the output. There is no nuclear-envelope constraint, chain regularizer, distance clipping, spatial interpolation or hidden geometry calibration.

## Observations and visibility correction

Let $C$ be the350-by350 strictly upper-triangular matrix of nonnegative integer counts, and let $D$ contain its53,883 positive pairs. The diagonal and lower triangle are storage zeros. Each distinct observed contact is counted once; the total count is1,902,039. Genomic bounds are one-based inclusive base-pair intervals, unrelated to Cartesian model-distance units.

Sparse ICE estimates dimensionless visibility factors from these observations alone. Initialize the working upper triangle $W=C$ and bias vector $b=1$. At each iteration, form each locus's incident count $r_i=\sum_j(W_{ij}+W_{ji})$, divide nonzero $r$ values by their nonzero mean to obtain $q$, and replace zero entries of $q$ by1. Apply the operations in this order:

```math
b_i\leftarrow b_iq_i,\qquad
W_{ij}\leftarrow\frac{W_{ij}}{q_i},\qquad
W_{ij}\leftarrow\frac{W_{ij}}{q_j}.
```

For original total count $T=\sum_{i<j}C_{ij}$ and current working total $T'=\sum_{i<j}W_{ij}$, rescale $b\leftarrow b\sqrt{T'/T}$ and then $W\leftarrow WT/T'$. Stop after the summed absolute bias increment is below $10^{-4}$, or at3000 iterations. The source order of division, summation and rescaling is retained. Unobserved loci receive neutral bias1 in the cleaned interface; the upstream routine returns NaN there, which has no likelihood contribution because those indices occur in no pair. This is representation of the known input support, not a fitted quality mask.

## Selected PM1 model and initial stress

For Cartesian coordinates $x_i\in\mathbb R^3$, define $d_{ij}=\lVert x_i-x_j\rVert_2$ and predicted count $\lambda_{ij}=b_ib_j\beta d_{ij}^{\alpha}$. Fix $\alpha=-3$ and $\beta=1$. The latter defines arbitrary model-distance units; no conversion to nanometres is available. The coordinates represent a population consensus, not a measured single-cell configuration.

Use desired distances $\delta_{ij}=[C_{ij}/(b_ib_j\beta)]^{1/\alpha}$ and initialize the coordinates as $1-2u$, where the1050 entries of $u$ are consecutive MT19937 uniform samples from seed0. The first stage minimizes the unnormalized weighted MDS2 sum

```math
S(X)=\sum_{(i,j)\in D}\frac{(d_{ij}-\delta_{ij})^2}{\delta_{ij}^2}.
```

One pair contributes $2(d_{ij}-\delta_{ij})(x_i-x_j)/(d_{ij}\delta_{ij}^2)$ to the gradient at $i$ and its negative at $j$. There are no zero-count or diagonal terms and no extra factor from mirroring the matrix.

## Poisson objective and numerical update

From the MDS2 coordinates, minimize the source's positive-pair Poisson objective

```math
L(X)=\sum_{(i,j)\in D}\left[\lambda_{ij}-C_{ij}\log\lambda_{ij}\right].
```

The factorial constant is omitted. This is the source convention of omitting zero-count pairs; it is not the formally zero-truncated Poisson likelihood. The paper introduces an observed-pair set but also writes a sum over all pairs, so zero handling is explicitly bound to the implemented positive-pair estimator. The notebook separately examines predictions for omitted pairs and the resulting model limitations. No inference that zero contact means infinite distance is made.

The contribution to the coordinate gradient is

```math
\frac{\partial L}{\partial x_i}\bigg|_{ij}
=\frac{\alpha(\lambda_{ij}-C_{ij})}{d_{ij}^2}(x_i-x_j).
```

The code evaluates this as the author's sequence of elementary products and divisions, and accumulates per-locus positive and negative pair sums in source order. This matters because the optimizer is nonconvex. All task-specific distances, intensities, objective and derivatives are explicit local mathematics. SciPy supplies only the same general L-BFGS-B optimizer used by the validated original Python implementation, rather than a genomic reconstruction library.

Both stages use maximum5000 iterations. MDS2 uses `factr=1e12`; PM1 uses the optimizer default `factr=1e7`. In both cases the relative-function criterion is $f_k-f_{k+1}\leq\mathrm{factr}\,\epsilon\max(|f_k|,|f_{k+1}|,1)$, where $\epsilon$ is float64 machine precision. Other retained general-optimizer defaults are memory10, projected-gradient tolerance $10^{-5}$, maximum15000 function evaluations and maximum20 line-search steps. The built-in L-BFGS-B limited-memory curvature updates and safeguarded line search are retained, rather than replacing them with a different inverse update. No coordinate bounds are imposed. Solver histories store actual objective values at completed optimizer iterations, and recorded stopping messages distinguish convergence from an iteration/evaluation limit.

After fitting, subtract the mean coordinate over the343 supported loci and set unsupported positions to zero. Centering is a gauge choice and changes no distance. No scaling or reflection is applied to the submitted result.

## Active scientific comparisons

The notebook actively runs MDS2, PM1, PM2 and NMDS on the same observations. MDS2 and PM1 use their exact source formulas. The original NMDS wrapper fails under current NumPy because an ndarray initialization is compared to the string `random` as a boolean. A separately identified private oracle repairs only this type guard; it is not labeled an unchanged original NMDS run. Local NMDS retains the numerical algorithm and its five outer steps: MDS2 initialization, increasing isotonic regression of distance versus inverse count, then MDS2 with those precomputed desired distances. Isotonic regression is implemented locally by equal-abscissa aggregation and weighted pool-adjacent-violators. As in the source, visibility is used in its first MDS2 stage but is not reapplied to later precomputed isotonic distances.

PM2 alternates five scalar-exponent and coordinate stages. For fixed coordinates and exponent,

```math
\widehat\beta(\alpha)=\frac{\sum_{D}C_{ij}}
{\sum_{D}b_ib_jd_{ij}^{\alpha}}.
```

Substitute this expression into the Poisson objective and fit the scalar $\alpha$ by L-BFGS-B, retaining the source's bounds $[-100,0.01]$, maximum1000 iterations and remaining optimizer defaults. Its profiled gradient is $-\sum_D C_{ij}\log d_{ij}+\widehat\beta\sum_D b_ib_jd_{ij}^{\alpha}\log d_{ij}$. The implementation retains the source's expanded log-likelihood arithmetic. The source permits a slightly positive exponent at its upper bound although the physical interpretation expects a negative exponent; actual fitted exponents are reported without silently clipping them. With the updated exponent and scale, optimize coordinates using the PM1 coordinate update. These auxiliary PM2 parameters are comparison diagnostics; the scored inverse fixes PM1's exponent and scale.

Read-coverage diagnostics thin the actual integer observations with identified binomial random seeds; they are simulated retention experiments, not new measured replicates or enzyme replicates. Resolution diagnostics aggregate consecutive bins separately within each chromosome and discard only contacts within a new bin. Every original bin is assigned to one coarse bin. Transfer-exponent and visibility comparisons are identified conditional-estimator sensitivity experiments. Cross-resolution or differently scaled method plots may use explicitly labeled similarity alignment for review; this never enters the Judge. Historical biological replicates, mouse experiments and ChromSDE results from the publication are not relabeled as experiments on this five-chromosome sample.

## Evaluation conventions

There is no independent three-dimensional truth. The hidden reference is the validated original PM1 result on precisely these observations, using the same NumPy2.4.6/SciPy1.17.1 backend, then the same support-centering and zero-padding convention. Earlier NumPy1.26.4/SciPy1.13.1 runs demonstrate a different nonconvex trajectory and remain separately archived.

The Judge compares every one of the58,653 supported-locus pair distances. Cosine NCC has no mean subtraction; distance NRMSE is RMS distance error divided by the comparison distance range. Pair distances remove global translation and all orthogonal transformations, including reflection, without any fitted alignment. They retain absolute scale within the declared model-distance convention, correspondence, chromosome placement and all supported loci. There is no per-chromosome alignment or arbitrary geometry rescaling.

Every required metric must pass: distance NCC at least0.999999, distance NRMSE at most $10^{-6}$, and relative positive-pair Poisson excess at most $10^{-5}$. The last quantity is $(L(\widehat X)-L(X_{\mathrm{ref}}))/\max(1,|L(X_{\mathrm{ref}})|)$. Source-only float32-coordinate quantization measured distance NRMSE $1.9671533816699332\times10^{-9}$ before local implementation; the nonzero tolerance permits numerical representation differences. A good match is agreement with a conditional standard estimator, not proof of biological coordinate accuracy.

The evaluator selects ground_truth.npz when present, otherwise baseline_reference.npz. Invalid existing actual truth is an evaluator error, without fallback; each reference contains coordinates and the independently input-derived visibility needed for count consistency. No actual geometry truth is present in this task.

Maintainer-only sensitivity observations are generated by src.generate_data.retain_contacts(counts,probability,seed=0): a fresh integer square matrix from independent binomial retention of every observed contact, with RandomState0 restarted per0.5/0.2 case. Standard CLI retains original counts unchanged. The active notebook calls this helper, preserves original-support scoring, and labels perturbations as simulated rather than real replicates. No hidden object enters generation.
