# Implemented offline tomography

The selected method is the published offline LIUQE-informed MAP comparison, not the realtime region estimator. All 190 native times,113 good channels, and120×41 image cells are reconstructed. The companion realtime code has a separately documented reduced-grid derivative defect; its results are not relabeled as a validated local method.

## Physical model and preprocessing

For time t, detector c and image pixel p, P(t,c)=sum G(c,p)x(t,p)+error. Emissivity x is W/m³, calibrated power P is W and G has effective collection-volume units m³. The instrument's stored b is converted to P=4pi E b with independent étendue E. We use s=1/max(G), A=sG restricted to one-based good-channel identifiers after subtracting one, and y=sP. G and P use the identical normalization; x retains physical scale. Each frame is divided by q=max(y), and its solution is multiplied by q afterward. No sign clipping, temporal smoothing or resampling of bolometer values occurs.

Grid cell centers use R=0.624+(j+0.5)dR and Z=0.75−(i+0.5)dZ, with dR=(1.1376−0.624)/41 and dZ=1.5/120 metres. Native magnetic flux has axes R,Z,time. Transpose and flip its Z axis, then sample with order2 spline interpolation, prefilter enabled and nearest exterior extension, preserving the upstream map_coordinates convention. Select the closest magnetic timestamp, with first-index tie breaking. The default uses measured LIUQE flux; the comparison helper can use planned FBT flux.

## Diffusion prior

The selected upstream convention uses h=(dR+dZ)/2 for both differentiation axes, rather than their separate physical spacings. This slight anisotropic-grid approximation is retained. Central finite differences with symmetric padding give v=(dpsi/drow,dpsi/dcolumn), with endpoint half differences. The structure tensor S=vvᵀ has its dominant singular vector normal to flux surfaces. NumPy's Hermitian SVD supplies U; W=U diag(alpha,1) Uᵀ with alpha=0.01. The tangent direction is smoothed more strongly. Exact zero gradients use the original SVD basis convention.

The image gradient D stacks all vertical forward differences, then all horizontal forward differences. The final row/column derivatives vanish by symmetric exterior boundary conditions. Its block weight matrix contains the two diagonal components of W and corresponding off-diagonal entries. L=DᵀWD is symmetric positive semidefinite. Vessel support is enforced by projection after each update. Gradients are formed over the complete rectangular image before clipping, so interfaces between active and zero outside pixels retain a penalty; this differs from a no-flux boundary on the vessel-shaped mask itself.

## Objective and iterations

For normalized data z=y/q, minimize J(u)=||Au−z||²/(2sigma²)+lambda uᵀLu/2 over u zero outside support. The chosen sigma=0.025, lambda=10, alpha=0.01 are the paper's fixed standard settings, not instrument metadata. Gradient is Aᵀ(Au−z)/sigma²+lambda Lu. Initialization u0=Aᵀz is the original unprojected backprojection.

Chambolle-Dossal acceleration uses a(k)=k/(k+4), k starting at0. Form v=u+a(k)(u−u_previous), then u_new=mask*(v−tau grad J(v)). The previous iterate initially equals u0. The fixed step is tau=1/(||A||²/sigma²+lambda16/h²), preserving the original conservative Pyxu derivative bound. The largest eigenvalue of AAᵀ provides ||A||² using a general symmetric eigensolver; the original uses its equivalent sparse spectral norm. No line search is used. Stop when ||u_new−u||₂≤1e-5||u||₂ or after10000 updates. A zero denominator uses the floating-point tiny bound. The first-frame3-update fixture checks acceleration and initialization independently.

The optional positivity comparison additionally clips every iterate below zero; it is not the default estimator. The isotropic ablation sets alpha=1. The planned-field comparison changes only auxiliary flux data. Neither comparison substitutes for complete default reconstruction.

## Evaluation

There is no independent physical truth. The hidden comparison is the independently executed unmodified author's offline solver on the same complete observations. NCC is uncentered cosine similarity over every image element. NRMSE divides global RMS error by the global reference dynamic range. All 190 frames also satisfy the fixed maximum normalized observation RMS bound0.035, measured over all 113 valid channels. Outside-support amplitude must be exactly zero. No alignment, magnitude transform, clipping or rescaling is applied by the Judge.

For toroidally symmetric interpretation, integrate P_rad(t)=sum x(t,i,j)2pi R(j)dR dZ. Units are W. Differences among plausible priors reveal conditional reconstruction uncertainty, not independently validated spatial accuracy.
