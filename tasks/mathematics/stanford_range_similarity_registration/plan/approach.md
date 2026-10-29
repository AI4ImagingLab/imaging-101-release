# Implemented similarity CPD

This task fits two actual raw scanner point sets. The synthetic translated-duplicate author demonstration is not used. There is no exact pose truth. The selected standard is the published similarity CPD, with scale optimized; it is not the upstream scale-disabled variant. The latter's variance shortcut does not apply to a constrained fixed scale and is deliberately outside scope.

## Observations and conditioning

The full target X0 has shape (40256,3), source Y0 (40097,3), float64 metres. Fitting selects indices 0,40,80,… from each original table, producing N=1007 target and M=1003 source points. The target selected mean o and scalar a=sqrt(mean((X0_selected-o)^2)) define X=(X0_selected-o)/a, Y=(Y0_selected-o)/a. No scanner-pose information or target truth is used. These are numerical conditioning parameters computed from public observations, not instrument calibration. Every source point is transformed in the final result.

## Gaussian mixture and E step

Dimension D=3. For transformed source Z=s Y R+t, the posterior P has shape (M,N), with P[j,i]=exp(-||X[i]-Z[j]||²/(2v))/(sum_k exp(-||X[i]-Z[k]||²/(2v))+c). Here v>0 is normalized isotropic variance and c=(2πv)^(D/2) w/(1-w) M/N; fixed w=0.1. This is the original uniform-outlier convention. Terms smaller than machine range underflow to zero, with the source denominator safeguard retained. P1=sum_i P, Pt1=sum_j P, PX=P X and mass=sum(P1). There is no mean subtraction in any evaluator metric.

## M step and objective

Weighted means are mx=sum(PX)/mass and my=Yᵀ P1/mass. Let Xc=X-mx, Yc=Y-my, A=Xcᵀ Pᵀ Yc, YPY=sum_j P1[j]||Yc[j]||², and XPX=sum_i Pt1[i]||Xc[i]||². For A=U S Vᵀ, C=diag(1,1,det(U Vᵀ)), R=(U C Vᵀ)ᵀ. The row-vector rotation preserves orientation. Then s=tr(AᵀRᵀ)/YPY and t=mx-s my R. Source samples become Z=s Y R+t. The normalized variance update v=(XPX-s tr(A R))/(D mass) uses the optimal free scale; if nonpositive, source behavior replaces it with tolerance/10.

The recorded source EM surrogate Q=(XPX-2s tr(A R)+s²YPY)/(2v_old)+(D mass/2)log(v_old) is evaluated with previous variance and current weights. It is not claimed to be a monotonically decreasing observed negative log likelihood. Initialize R=I, s=1, t=0, Q=∞ and v=sum_ij||X[i]-Y[j]||²/(D M N). Stop when absolute Q change <=1e-6 or after 400 iterations. No line search or additional regularization is used. The bound run stops after 189 iterations.

## Physical output and evaluation

Convert translation to metres by t_phys=a t+o-s o R; output every original point s Y0 R+t_phys. Return R,s,t_phys alongside the full point table. The evaluator compares all four quantities to the separately run original method using relative L2 <=1e-8 (nonzero numerical budget, not self-comparison error). It additionally requires all-source nearest-target RMS <=0.0024525798338538444m, transform RMS <=1e-10m, rotation orthogonality Frobenius error <=1e-8 and determinant error <=1e-8. No implicit registration or scaling is permitted. Nearest-target residual is not independent pose accuracy because sampling and overlap differ.

## Fresh review experiments

The active notebook repeats w in {0,.05,.1,.2,.3} and fit stride in {20,40,80,160}; each nondefault is an actual local reconstruction. Every metric uses the full native output, not just selected samples. A separately implemented rigid nearest-neighbor ICP comparison uses stride40, identity initialization, at most100 SVD updates, mean squared coordinate change stopping1e-12. ICP fixes scale=1 and is labeled accordingly. Figures preserve before/after projections, full residual CDF/spatial distribution/RMS bars, upstream-vs-fresh EM traces and both sensitivity curves. Observational mismatch and unsuccessful variants are retained.


## Public-contract revision 2026-10-29

Convergence audit resolves the provisional finite-protocol isolation. On exactly the public observations, unchanged source CPD with tolerance1e-12 reaches475updates and passes original v1Judge: relative output/rotation/translation/scale differences1.5808e-9/1.7428e-9/3.8983e-9/6.7225e-11, all below1e-8; full native nearest RMS.00222961804m. This is a source convergence probe, not an independent blind reconstruction. README now specifies fixed40-stride fit support, public-data coordinate normalization, w=.1 and observed mixture NLL/stable nondegenerate stationary solution convention. No iteration recipe is exposed, no true pose or noise floor invented, no claim of unrestricted global maximum because mixture boundary singularity remains. Judge/source/arrays unchanged.
