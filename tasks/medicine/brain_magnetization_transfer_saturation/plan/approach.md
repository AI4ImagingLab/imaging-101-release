# Implemented mathematics

This task implements qMRLab's no-B1-map branch of MTSAT_exec, including the 2010 correction to the two-point formulas. No microscopic two-pool fit, B1 correction, spatial smoothing, or iterative regularizer is added. The three measured images are preserved exactly; integer scanner values are represented losslessly in float64. Array axes are x,y,z and voxel spacing1.5mm. Nominal angles are converted by multiplication with pi/180; repetition times are .028,.018,.028 seconds in PD,T1,MT order.

Let p,t,m denote the three observed signals, a,b,c their angles, and u,v,w their repetition times. The small-angle short-TR FLASH approximation is S=A alpha TR R/(TR R+alpha²/2+delta), with delta zero for PD and T1 and d for MT. A is a common arbitrary signal amplitude, R is R1 in s^-1, and d is fractional saturation per repetition. Common echo attenuation is absorbed in A. This model ignores higher-order angle/TR terms and does not estimate transmitter inhomogeneity.

The explicit inverse is

```math
R=\frac{1}{2}\frac{(b/v)t-(a/u)p}{p/a-t/b},\qquad
A=\left(\frac{ub}{a}-\frac{va}{b}\right)\frac{pt}{ubt-vap},
```

```math
100d=100\left[w\left(cA/m-1\right)R-c^2/2\right].
```

Every symbol is defined above. Multiplication/division grouping follows the original code. There is no optimization objective, objective normalization, gradient, initialization, step size, line search, iteration limit, or stopping rule: this is a direct algebraic inverse. Artificial iteration would change the implementation without benefit. Signed estimates are retained.

The task domain is fixed before extraction: intersect the index-space ellipsoid ((x-63)/38)^2+((y-64)/53)^2+((z-48)/30)^2<1 with p>100,t>100,m>50 and abs(p/a-t/b)/(p/a+t/b+1e-100)>.2. These conditions use only visible observations and coordinates. Of253021 ellipsoid voxels219351 pass. This is a numerical analysis region, not a segmentation or target-derived calibration. Outside it, assign exact zeros. Original MTSAT_exec computes the whole volume and multiplies MTsat by the mask; 0*NaN can remain NaN outside the domain. The oracle capture explicitly assigns zero outside for both outputs, matching the declared boundary convention, without changing any evaluated voxel. The local method evaluates only the domain.

Predicted observations infer nuisance A from observed p and submitted R: A=p(uR+a²/2)/(auR). Substituting A,R,d into all three signal equations yields a relative L2 observation residual. It is an algebraic consistency diagnostic, not independent physical truth. Exact no-MT Bloch signal A sin(alpha)(1-exp(-TR R))/(1-cos(alpha) exp(-TR R)) provides an independent approximation check in tests and notebook; it is not substituted for the published inverse.

The notebook compares the fresh maps with independent upstream outputs, displays MTR=100(p-m)/p as a separately labeled contrast with different units/interpretation, examines a wrong-TR ablation by rebuilding the same closed-form formulas with T1 TR=.028, and plots perturbation sensitivity after adding explicitly labeled1% independent synthetic signal noise. Perturbations are supplementary experiments; the benchmark input remains real observations. Perturbations preserve the original fixed mask when measuring common-domain differences; reconstruct may change its own validity and comparisons use the intersection, reported explicitly.

Evaluation uses mask-specific absolute RMSE (MT percent and R1 s^-1), uncentered cosine, zero outside, and three-image relative L2 residual. Thresholds were frozen in mts_pre_extraction_contract.json before extraction: MT RMSE1e-5percent,R1 RMSE1e-6s^-1,NCC.999999,outside1e-12,signal residual1e-6. Two independent original reruns matched exactly; these are nonzero numerical reproducibility bounds, not clinical error allowances. No truth exists: evaluation uses the original standard-method result.
