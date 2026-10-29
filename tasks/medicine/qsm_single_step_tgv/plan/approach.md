# Implemented mathematical approach

## Scope, acquisition and fixed scientific question

The benchmark solves the single-step second-order TGV QSM problem of Langkammer et al. (NeuroImage 111, 2015, 622-630), using the finite-difference/primal-dual implementation in QSM.m revision f22dc40d81495b37438fbf66f86c9281f8326813. It binds the complete real 2016 challenge single-orientation wrapped phase, not the paper's original EPI cohort. The paper is the model/method source; independently executed code is the numerical discretization oracle. This distinction matters for acquisition, metric and performance claims.

Public data contain measured phase phi (float64, rad), magnitude (float32, acquisition arbitrary units) and mask M (bool), each with batch-first shape (1,160,160,160). Computation removes only that batch axis. Magnitude is an acquisition-time observation displayed in the notebook; it is not used to generate a morphology prior in this selected method. No synthetic observations, target-derived calibration or hidden tensor susceptibility enter reconstruction.

The grid has pitches h=(1.0625,1.0625,1.0714285714285714) mm along x,y,z, B0 direction e_z, B0=2.89362 T, TE=0.025 s, and gamma=42.58 MHz/T. Consequently s=2*pi*gamma*B0*TE=19.35383488668318 rad/ppm. The exact field is tied to the raw data by independent complex-image/protocol identity evidence, rather than fitted to a susceptibility reconstruction. The original image phases match every bound phase voxel within 1.2e-7 rad. The author-hosted official ZIP and fixed-version repository files are byte-identical for all five observation/reference files checked.

The full volume has 4,096,000 voxels; M contains 933,481 active brain voxels. The six-neighbor interior Mi contains 887,715 rows. The original wrapper's bounding-box-plus-two crop is [18:143,7:157,11:137] in zero-based Python notation, shape (125,150,126). This removes only exterior computational padding, not any brain observations. Outputs are expanded back to the full original grid. There is no center slice approximation or downsampling.

## Physical model, sign and unavoidable ambiguities

For signed scalar susceptibility chi in ppm, the ideal phase is s*D chi, where the continuous dipole Fourier multiplier is

```math
D(k)=1/3-k_z^2/(k_x^2+k_y^2+k_z^2),\quad k\ne0.
```

The measured phase is phi=wrap(s*D chi+b+epsilon), where b is a background phase and epsilon describes measurement/model discrepancy in radians. The background is harmonic only under the external-source assumption. Chemical exchange, anisotropic susceptibility, nonharmonic transmit phase and the source reconstruction do not obey that scalar model perfectly. This task does not identify b or an unwrapped phase image as separate outputs.

Let L=dxx+dyy+dzz in mm^-2 and W=dzz-L/3. Then L*D=-W, not +W. A transverse cosine mode independently checks that sign. With x=s*chi and a discrepancy potential eta in radians, the implemented PDE is

```math
Mi(L eta-W x-lphi)=0,\qquad lphi=Lwrap(phi).
```

The paper's discrepancy variable has the opposite sign under its displayed constraint; replacing psi by -psi leaves its squared norm unchanged. The shipped eta follows the source's positive-eta constraint exactly. For harmonic b, Lb=0 on the interior, explaining the integrated background-removal mechanism. A quadratic harmonic polynomial added to phase is used as an independent check, separately from parity with source formulas.

Dipole inversion cannot determine a global susceptibility constant. The implementation starts all fields at zero and retains the source's zero brain-mean gauge, without a fitted translation or rescaling. Both chi and eta are identically zero outside M. Measured COSMOS and STI arrays are compared at their supplied ppm reference; their very small original offsets are reported, not optimized away. A successful numerical reconstruction is not exact biological truth.

## Spatial discretization and boundary conventions

For axis a with pitch h[a], the forward difference G_a u=(u[i+e_a]-u[i])/h[a]. The divergence component B_a p=(p[i]-p[i-e_a])/h[a] is its negative transpose on the zero-extended masked grids. L is the sum of central second differences, and W uses coefficients (-1/3,-1/3,2/3) for those second differences. The z pitch must not be replaced by the transverse pitch.

Lwrap(phi) is computed by taking the forward and backward first phase differences separately, wrapping differences outside [-pi,pi], subtracting them and dividing by h[a]^2. Only interior array sites are written; outer faces remain zero. The source explicitly preserves differences already equal to either endpoint. On the bound raw phase the possible differences lie within [-2*pi,2*pi]. This is the source's wrapped first-difference discretization, not a replacement by the Laplacian of the already processed tissue field. The continuous paper expression Im(L exp(i*phi)*exp(-i*phi)) motivates the phase operator, but at finite pitch it is not numerically identical to this source stencil. That paper/code discretization difference is recorded rather than silently called exact equation parity.

Mi is true when all six axial neighbors of a voxel belong to M. The upstream function does not additionally test the center voxel. That convention is preserved and explicitly tested with an artificial one-voxel hole; the actual acquisition has zero Mi sites outside M. All primal updates are multiplied by M, with outermost array faces forced inactive. Dual updates are restricted to Mi. The source crop leaves two zero-margin cells, so the roll operations used as array primitives never create a physical periodic connection between active brain sites. Standalone Judge uses independent interior slices rather than roll operations.

For a vector v=(vx,vy,vz), Ev=(Gv+Gv^T)/2 has six stored components (xx,xy,xz,yy,yz,zz). Its Frobenius norm is sqrt(qxx^2+qyy^2+qzz^2+2*(qxy^2+qxz^2+qyz^2)). The corresponding Div has components Bx qxx+By qxy+Bz qxz; Bx qxy+By qyy+Bz qyz; Bx qxz+By qyz+Bz qzz. Thus E and -Div are adjoints under the Frobenius-weighted tensor inner product. Tests verify both this identity and the derivative of half the squared E norm.

## Selected objective and regularization-unit derivation

The actual source-normalized objective is

```math
min_{x,eta,v} 0.5||eta||_2^2
 +lambda sum_{Mi} ||Gx-v||_2
 +r*lambda sum_{Mi} ||Ev||_F,
 subject to Mi(L eta-Wx-lphi)=0.
```

The eta norm uses the complete primal mask M. Vector and tensor penalties are isotropic Euclidean and Frobenius norms, respectively, not componentwise anisotropic L1 norms. No magnitude weighting or additional ridge penalty is added. Multiplying the entire discrete objective by the common voxel volume would not change its minimizer; the source therefore uses unweighted voxel sums with mm-based derivatives.

The paper prints a full squared discrepancy term, not a half-squared term, and gives physical regularization (alpha1,alpha0)=(0.0015,0.0005). Its definitions associate alpha1 with Gchi-w and alpha0 with Ew. The source instead implements the half-squared discrepancy through eta's prox factor 1/(1+tau). Under x=s*chi, eta=s*psi, v=s*w, the squared term scales by s^2 and each TGV term by s. Divide the source objective by s^2, then compare it with half the published objective. This gives lambda=s*alpha1/2=0.014515376165012384 and r=alpha0/alpha1=1/3. These constants are DEFAULT_LAMBDA and DEFAULT_RATIO.

The original QSM.m default uses lambda=1e-4 and r=3; it is a different parameter setting, not the paper's printed pair. Both default and canonical original-source outputs remain identified in notebook references. The canonical physical pair was frozen before inspecting its measured quality. The private diagnostic sweep (lambda .0003,.001,.003,.01,.03,.1; r=3), and stricter 5000-step runs, are retained with data/method/metric provenance. They are not relabeled paper defaults and are not silently used to select the most favorable test-reference score.

The TV comparison retains the same lambda, phase PDE and eta penalty, removes v and q entirely, and uses lambda*sum||Gx||_2. It is the source's single-step TV branch, not a separate already-background-corrected deconvolution. Fresh TV is required in the notebook because the selected paper makes scientifically relevant comparisons between first- and second-order regularization.

## Primal-dual algorithm and implemented stopping

Dual fields are nu (scalar PDE dual), p (three-component first-order dual), q (six-component symmetric dual). All primal and dual arrays are zero-initialized float64; no hidden target or cached reconstruction initializes them. A bar denotes theta=1 extrapolation. Define g=4*sum(1/h[a]^2), a=2*(1/hx^2+1/hy^2+2/hz^2), and tau=sigma=1/sqrt(g^2+a^2), using the source's conservative bound. This is a fixed-step algorithm; there is no line search, automatic optimizer or learned network.

In each iteration, update the PDE dual first: nu += sigma*Mi*(L eta_bar-W x_bar-lphi). Update p by projecting p+sigma*Mi*(G x_bar-v_bar) pointwise onto the radius-lambda Euclidean ball. Update q by projecting q+sigma*Mi*E v_bar onto the radius-r*lambda Frobenius ball. A projection divides each vector/tensor by max(1,norm/radius); zero norms are therefore harmless.

Using these freshly updated dual variables, set eta_new=M*(eta-tau*L nu)/(1+tau), x_new=M*(x+tau*(div p+W nu)), and v_new=M*(v+tau*(p+Div q)). Do not swap the dual and primal order. Extrapolate each primal field as bar_u=2*u_new-u_old, then retain the new primal variables. The TV branch skips q/E/Div/v updates while retaining all other steps.

Every tenth iteration, compare each updated scalar primal field with the field from the immediately preceding SINGLE update, not with the field ten iterations ago. Stop when ||x_new-x_old||/||x_new|| < 3e-4 and ||eta_new-eta_old||/||eta_new|| < 6e-4. Maximum iterations are 5000. A zero denominator produces an internal infinite stopping ratio and does not declare convergence. History is a numeric (K,3) array of checked iteration, x ratio and eta ratio; internal Inf is never emitted as a JSON metric. Complete canonical C runs stop at 2480 TGV and 2040 TV iterations. The notebook follows these actual stopping conditions; no fixed small smoke run is substituted.

## Investigated correction and validation boundaries

The original tensor-divergence C code uses hx for every derivative in the vx update, hy for every derivative in vy, and hz for every derivative in vz. This is not the Frobenius adjoint of its anisotropic symmetric gradient. A deterministic independent random check using the actual spacing gives relative discrepancy 0.019643358890651864. Replacing each factor by its derivative-axis pitch gives 9.041385288914794e-16. The six minimal edits are the three component formulas in both float and double branches; the model, stencil, masks, regularizers, initialization and stopping are not replaced.

The unmodified original source is archived privately and independently executed on the full bound observations. The minimally corrected C oracle is archived separately with its exact patch. Capture instrumentation only exposes existing state arrays; it does not compute replacement outputs. Real-patch states (chi,eta,v,nu,p,q,Mi) at 1,2,10,50 updates are frozen before extraction. Both local NumPy and PyTorch-CPU paths agree with all those independent states to a maximum absolute error below 5e-16, compared with a predeclared 5e-11 ceiling. Wrapped Laplacian error is below 2e-15 against a 2e-13 ceiling. GPU validation independently covers the same states before full volume reconstruction.

Complete upstream canonical runs, the minimally corrected repeat that exposes eta, full local GPU reconstruction, standard CLI, standalone Judge and active notebook are independently identified in validation. CPU small-array execution and a ten-update full-volume timing are validated; the very slow complete 2480-update NumPy CPU benchmark is not claimed executed. The standard published benchmark path is GPU float64, with the tested device/runtime recorded separately.

## Evaluation, physical consistency and scientific limitations

The hidden baseline is the independently validated minimally corrected C standard-method result, containing signed chi (ppm) and eta (rad). It is not called biological truth. COSMOS and STI chi33 remain independent measured comparisons in the notebook, not substitute calibrations or targets used by the solver.

Array agreement uses cosine NCC without mean subtraction and RMS error normalized by each reference's dynamic range, on M. Zero estimates have NCC=0 rather than NaN. The standalone evaluator validates exact shape, float64 real dtype, required keys, finiteness, zero exterior and zero mean gauge. It checks both quantities, not chi alone. Numerical agreement boundaries are fixed before the fresh GPU run; no crop, magnitude operation or alignment is fitted.

The independent observation metric is ||Mi*(L eta-W(s*chi)-Lwrap phi)||_2 / ||Mi*Lwrap phi||_2. It is evaluated directly on original public arrays using separate code. The independent C canonical value is 0.004286750793256695. Eta RMS on M is 0.08087621554939445 rad. Their fixed upper boundaries are 1.1 times those independently measured values and must pass fresh results. Zero chi/eta produces a PDE ratio of 1 and fails; zero chi with eta set to the masked measured phase exceeds the eta-energy boundary and fails. Reference agreement and energy checks prevent fitting all data with an arbitrary nuisance phase potential.

The paper challenge reported historical L2-percent errors of 69.0-140.9%, median 83.9%; GRAZ TGV was approximately 77.5%. Our canonical TGV measurement is 108.1079% against STI chi33, with COSMOS NCC 0.7171844 and relative L2 error 88.5155%. TV is better on these chosen global metrics (STI 95.22095%, COSMOS NCC 0.7312303). These are candidly retained results from a different implementation and parameter convention, not reproduced historical leaderboard values. Original-source defaults are worse; tightening stopping alone does not fix the model discrepancy. The raw Wave-CAIPI dataset and the LBV-plus-polynomial processing of the multi-orientation reference introduce known differences. The task is a validated implementation benchmark, not a claim of superiority, an exact tissue measurement, or a proof that TGV beats TV.


## Public-contract revision 2026-10-29

Version2.0-joint-tgv-primal-witness adds actual source-computed auxiliary_w, physical ppm/mm, shape(1,3,160,160,160). Source primal updates unchanged; reconstruct requests existing return_state and zero-pads/scales its real w. Historical output/baseline/source/interface archived under benchmark_readiness_20261029/qsm_v1_archive. Fresh CUDA main2480updates31.2764s and full actual TGV+TV nbconvert exit0/no cell errors completed with .24process GPU memory cap coordinated with ADJUST. The new independently implemented physical objective is14.571308579816776; bound14.571308725529862 fixed with relative1e-8 numerical allowance, original PDE/eta thresholds unchanged. No actual tissue truth exists. Independent conditional witness descent lowers objective to14.570909040355454 and passes, without changing chi/eta; zero auxiliary and all-zero outputs fail. This control is not a fresh blind full susceptibility solve.
