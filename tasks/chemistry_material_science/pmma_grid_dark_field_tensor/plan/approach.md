# Implemented mathematics

This is the BSD-licensed DiDaFi local correlation method on an existing measured ROI, with no new benchmark simulation. Raw float32 grid and sample images are divided by the same float32 flat. No dark frame is supplied; zero dark and unit exposure normalization are explicit assumptions. Period P=12.68550106609808 detector pixels was calibrated from the full reference grid alone.

K=7. At every one of 455 by 455 positions, take a 14 by 14 image window and the reference-grid 7 by 7 patch beginning at offset round(7/2)=4. Compute valid cross-correlation, giving 8 by 8 correlation samples; use grid/grid for the reference channel and sample/grid for the sample channel. Correlation coordinates are x=0,...,7 and y=7,...,0. Fit

```math
c_0+c_1\cos(2\pi x/P+\phi_1)+c_2\cos(2\pi y/P+\phi_2)
+c_3\cos(2\pi(x-y)/P+\phi_1-\phi_2)
+c_4\cos(2\pi(x+y)/P+\phi_1+\phi_2).
```

All seven parameters start at one. scipy.optimize.curve_fit uses its default unconstrained Levenberg-Marquardt implementation, default stopping tolerances and evaluation limit, and unweighted squared correlation residuals. There is no custom regularizer, line search, or outer iterative solver. Preserve the source expression's arithmetic order, correlation float32 inputs, and float64 fitted parameters. Take absolute values of the first five fitted amplitudes; failed fits become NaN, as upstream. Average each amplitude over nonoverlapping 13 by 13 blocks with nanmean, yielding 35 by 35 maps. Do not average ratios before forming the block amplitude means.

Let R_i be averaged sample coefficient i divided by averaged grid coefficient i, i=0,...,4, and T=R_0. Define a_1=P^2 log(R_1/T)/pi^2, a_2=P^2 log(R_2/T)/pi^2, a_3=P^2 log(R_3/T)/(2 pi^2), a_4=P^2 log(R_4/T)/(2 pi^2). The unweighted least-squares directional inversion is

```math
C_{xx}=(-3a_1+a_2-a_3-a_4)/8,\quad
C_{yy}=(a_1-3a_2-a_3-a_4)/8,\quad
C_{xy}=(a_3-a_4)/4.
```

C is a signed second-moment estimate in detector-pixel squared units. Do not clip eigenvalues or discard weak-scattering locations. The predicted mode ratios are T for DC and T exp[-2 pi^2 v/P^2], with v=(Cxx,Cyy,Cxx+Cyy-2Cxy,Cxx+Cyy+2Cxy). The normalized mode residual is the L2 norm of predicted minus identified ratios divided by the L2 norm of identified ratios. This residual diagnoses the stationary Gaussian approximation, rather than proving a physical positive semidefinite tensor.

The notebook also propagates the conditional curve-fit covariance through the amplitude absolute-value map, block mean, logarithms, linear inversion, and smallest eigenvalue. Overlapping correlation windows are dependent: use the mean of individual standard deviations as a Cauchy bound for a block mean, then add grid and sample standard-deviation bounds. These are conditional fit diagnostics, not calibrated camera confidence intervals. Directional closure mismatch is reported separately. A known periodic Gaussian blur verifies all tensor signs and factors independently.

Map centers are original rows 913+13*i and columns 1013+13*j. This coordinate definition follows the reference patch center: its local first pixel is round(7/2)=4 and its center is 4+3=7. The first 13-position averaging block has mean start offset 6, so its center is ROI start+7+6=ROI start+13. Consequently the first center is row 900+13 and column 1000+13. This is the reference-patch center convention, rather than the 14-window geometric center at offset 6.5. The physical component basis is x right/y up; stored map rows increase down. CLI and notebook use the same bound constants and solve all 455 by 455 correlation fits for both channels. The notebook actively reconstructs once and derives actual transmission-only and isotropic comparisons from that fresh result; no cached reconstruction substitutes for this computation.

The hidden standard-method reference is the original implementation on identical observations, not actual material truth. Evaluate every covariance component and transmission separately with cosine NCC and dynamic-range NRMSE. Independent original reruns are bitwise equal. Float64 linear algebra and optimizer parity tolerances are tested before judging; the mode-residual boundary is 1.01 times the measured source residual. No field alignment or postprocessing is allowed.
