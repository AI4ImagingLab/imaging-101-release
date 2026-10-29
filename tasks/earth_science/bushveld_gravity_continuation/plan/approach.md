# Implemented approach

## Scope and physical interpretation

The task estimates an exterior gravity-disturbance image from real Bushveld observations. The selected published method is the standard generalized equivalent-source formulation in Soler and Uieda (2021), Sections 2.1-2.2, equations (1)-(10). Gradient boosting and block aggregation are distinct methods discussed later in that paper and are not claimed here. There was no supplied upstream notebook to preserve. This task does not infer unique rock density, point masses, or interior geology.

The observations d have length N=306, in mGal. Coordinates P contain easting, northing, upward position in metres, shape (N,3). Projection is WGS84 UTM35S; horizontal offsets of 600000 m and 7200000 m were subtracted privately. Vertical coordinates retain WGS84 geometric heights. Input disturbances have normal gravity removed upstream, but not Bouguer correction. Their observational uncertainties are unspecified; weights W=I impose equal weighting, not a claim of identical calibrated uncertainties.

## Forward model, geometry and units

Use M=N equivalent mathematical sources, one directly under each station. Source j has coordinates Q[j]=P[j]-(0,0,h), with h=6000 m. The unbounded Cartesian Green matrix is A[i,j]=1/||P[i]-Q[j]||, in inverse metres. No periodic extension, image boundary stencil or FFT discretization is introduced. Discretization is a collocation expansion at actual station coordinates. Field prediction at any point p is sum_j c[j]/||p-Q[j]||. The coefficient c has units mGal m; it is not physical mass. The unknown exterior disturbance is approximately harmonic above all sources, consistent with potential-field continuation. Cartesian curvature and unresolved terrain contribute approximation error.

The target points are G, shape (48,64,3), supplied publicly. Their upward height is 3000 m; each horizontal axis spans the minimum station coordinate plus 10000 m to the maximum minus 10000 m, endpoint-inclusive. Axis order is northing/easting, both increasing. There is no arbitrary output registration or amplitude normalization.

## Scaled objective and solution

For each column j compute its population standard deviation s[j]=sqrt(mean_i((A[i,j]-mean_i A[i,j])^2)). Means are used only for computing s; the kernel is never centered. S is diag(s). Let B=A S^{-1}, and m=S c. B and lambda are dimensionless, while m and d have units mGal. Minimize

    Phi(m) = ||B m - d||_2^2 + lambda ||m||_2^2,
    lambda = 0.1.

The data term is a sum, not a mean. There is no hidden N factor, intercept, trend removal or data standardization. Scaling regularizes S c, rather than c directly; replacing this with lambda||c||^2 changes the method. The equivalent unscaled penalty is lambda||S c||^2.

The gradient is 2 B^T(Bm-d)+2 lambda m. The stationary system is (B^T B+lambda I)m=B^T d, followed by c=S^{-1}m. The local implementation forms the matrix and calls scipy.linalg.solve(assume_a='pos'), a permitted general linear-system primitive. This reproduces the original scikit-learn Ridge Cholesky solution with no intercept and equal weights. There is no initialization, iteration, line search, update step size or stopping tolerance: the selected method is a direct positive-definite solve. Nonpositive damping and degenerate column scaling are rejected. Float64 is used for all local numerical steps.

## Parameters and comparisons

The reference reconstruction uses h=6000 m and lambda=0.1. These are selected model hyperparameters, not acquisition facts, and are fixed across the oracle and cleaned run. The notebook additionally computes a weaker damping lambda=0.0001 using the same local model to illustrate the tradeoff between station residual and coefficient norm; it is explicitly labeled and is not substituted for the standard result. No parameter is tuned against hidden geological truth, which is unavailable.

The forward image is A(G,Q)c in mGal, shape (48,64), saved under reconstruction with batch shape (1,48,64). Intermediate coefficients and station predictions remain maintainer quantities and are not additional required solver outputs.

## Validation and evaluation

Frozen independent upstream fixtures contain source locations, the unscaled Green matrix, coefficients, station predictions, and the complete target image. Tolerances were committed privately before local extraction: kernel rtol3e-15; coefficient relative L2<1e-8; station/image rtol1e-9 and atol1e-8 mGal. Independent checks use a 3-4-5 inverse-distance case, coordinate scaling, an adjoint identity, finite-difference harmonic Laplacian and an augmented least-squares solution distinct from normal-equation solving. All validate units, signs and objective scaling in addition to upstream parity.

Evaluation compares submitted real arrays at their exact declared shape and absolute scale. NCC is flattened cosine without mean subtraction. NRMSE is RMS error divided by reference dynamic range. There is no crop, alignment, normalization or hidden mask. With no true field measurement on the plane, the hidden reference is explicitly a standard-method result. Double-precision independent reruns matched exactly, whereas an independent float32-input oracle run gave NRMSE 2.6443191016598888e-5. The nonzero NRMSE bound is ten times that value, 0.0002644319101659889, fixed before extraction; NCC must be at least 0.9999999999. This numerical agreement criterion is separate from physical accuracy. Standard station residual RMS is about 0.636177 mGal, checked independently; it is not an image truth error.

## Operator transpose, discretization and normal-system derivation

Let G(X,Q) be the dense collocation matrix for any set of evaluation points X:
G_ij=[(x_i-qx_j)^2+(y_i-qy_j)^2+(z_i-qz_j)^2]^(-1/2).
It is not a finite-difference Poisson solve or Newtonian vertical-acceleration
kernel; the selected generalized harmonic basis is used for disturbance data.
For a residual r at evaluation points, the Euclidean transpose is
(G^T r)_j=sum_i r_i/||X_i-Q_j||. Consequently
< Gc,r >=< c,G^T r > under ordinary unweighted sums. The source plane follows
station heights rather than being forced flat; every source is6000m below its
own station. The target plane is above the source set. No interpolation onto an
intermediate image grid is used in the fit.

For s_j=std_i(A_ij)>0 and m_j=s_j c_j, the linear perturbation of the objective
in direction v is2 v^T[B^T(Bm-d)+lambda m]. Its second derivative is
2(B^TB+lambda I), positive definite for lambda>0 even if columns are dependent.
This establishes the normal-system sign and damping factor without relying on
a solver wrapper. An equivalent independent least-squares system is
[B; sqrt(lambda) I]m=[d;0]. The validation compares this augmented solution to
the direct positive-definite solve. The factor2 cancels; using mean squared error
instead would require rescaling lambda by observation count and is not shipped.

Population standard deviation subtracts column means to measure scale, but the
forward operator still includes its constant component. Thus setting an intercept,
subtracting data mean or centering the matrix changes the continuation. Coordinates
are stored after a common horizontal translation, which leaves all distances and
fields invariant; no upward coordinate offset is applied. Endpoint-inclusive grid
spacing is determined by public coordinates, and not by an implied pixel size.

## Numerical choices, diagnostics and scientific limits

| Choice | Standard value and convention |
|---|---|
| Source count |306; one source per station in source order |
| Source vertical offset |6000m downward |
| Kernel |Positive1/r in inverse metres; unbounded Cartesian space |
| Arithmetic |float64 arrays and dense solve |
| Scaling |Population std, ddof0; matrix itself uncentered |
| Damping |Dimensionless0.1 on scaled coefficients |
| Direct solve |SciPy positive-definite normal system; no iterative tolerance |
| Standard output |All48x64 target values, mGal, northing/easting row/column |
| Additional notebook comparison |Fresh lambda0.0001, otherwise identical |

No initial coefficient vector or step size exists. The mathematical stationary
system, augmented independent solve and station residual assess numerical
correctness; no invented convergence history is shown. Numerical failure in the
general linear algebra primitive propagates. Explicit validation rejects zero
column scale, nonpositive damping and source/evaluation coincidence. It does not
claim a general depth-sign validator or calibrated observational-noise model.

The standard station RMSE is0.636177032347mGal versus0.178264810674mGal for the
weak-damping computation. Their coefficient L2 norms are approximately
2116633.8643 and10107171.89896mGal*m respectively. The smaller residual alone
therefore does not establish a better continuation image: field truth at height
3000m is unavailable and coefficient instability can amplify noise. The notebook
retains station/image/reference panels, a fresh-minus-upstream map and the
standard/weak/difference comparison. The numerical baseline thresholds evaluate
agreement with the selected method and never convert station fit into truth.
