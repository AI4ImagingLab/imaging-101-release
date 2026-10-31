# Implemented approach

The selected method is the author's BaSiCPy 2.0 approximate Torch implementation, translated into explicit NumPy/SciPy mathematics. It is related to the 2017 BaSiC model but is not claimed to reproduce the original MATLAB implementation unchanged. Native uint8 observations remain intact; all 100 frames are resampled to float32 128² model values using the normalized antialiased triangle matrices defined in README. Sparse separable matrix products avoid duplicating the full native stack.

Let I be the T×128×128 model observations in DN. Factor the background as B_t F_p, where B is temporarily dimensionless and F is in DN. Let R be a DN residual, Y the dimensionless Lagrange multiplier, W adaptive dimensionless residual weights, μ the inverse-DN penalty, and H=C(F) the orthonormal DCT-II coefficients. The mathematical inner update, in this order, is:

```math
\begin{aligned}
H^+&=\operatorname{shrink}\left(H+C\left(\operatorname{mean}_t[I-R-BF+Y/\mu]\right),\lambda/\mu\right),\\
F^+&=C^{-1}(H^+),\\
R^+&=\operatorname{shrink}\left(I-BF^++Y/\mu,W/\mu\right),\\
B_t^+&=\max\left(0,\frac{\operatorname{mean}_p(I-R^+)_{tp}}{\operatorname{mean}_{t,p}(I-R^+)_{tp}}\right),\\
D&=I-B^+F^+-R^+,\qquad Y^+=Y+\mu D .
\end{aligned}
```

Shrink(x,a)=sign(x)max(|x|−a,0), applied elementwise. Set μ to min(1.5μ, μmax) after the update. The initial μ is 12.5 divided by the largest singular value of the flattened I matrix, and μmax=10^7μinitial. Each inner loop starts with F=1, B=frame mean/global mean, R=Y=0, and stops at ||D||₂/||I||₂≤0.001 or 500 updates. A converged field has S=mean_t(BF)/mean_t,p(BF), and b=B mean_t,p(BF), restoring the physical units and gauge.

Initially W=1. At an outer reweighting, compute W̃=1/(|R/(mean_p(BF)+10^-6 DN)|+0.1) and normalize W̃ to unit global mean. Restart the inner state, preserving these weights. Stop when sum|S−Sprevious|/sum|Sprevious|≤0.01 or after 10 outer iterations. This bound dataset uses two outer iterations at λ=5. The objective is an adaptive weighted residual plus a DCT penalty; its final weighted objective is distinct from the fixed unweighted public J.

The complete original automatic parameter selection is retained. Select every second frame (50 frames); if H×W exceeds 1024², first antialias-resize to floor(H/r)×floor(W/r), r=HW/1024², then use the 128² model basis. Fit an initial field at the mean of the author's 17-element λ pool. Upsample fields with non-antialiased aligned-corner bilinear interpolation, and refit the temporal baseline against their resampled values. This baseline-only refit uses a per-frame lowest-10-percent mask, initially weight 1 inside and 0.1 outside, five outer reweightings, and the same μ budget and 500-update/0.001 inner stopping. Each outer baseline initialization is the masked frame mean, R=Y=0. The baseline step updates the residual by shrinkage, then b=max(mean_p(I−R),0); the constraint residual uses the pre-update b as in the selected original code. Reweight with 1/(|R/b|+0.1), downweight outside by 0.1, normalize globally, then multiply by the binary mask. No darkfield is estimated.

Automatic selection scores corrected preliminary images I/S−b using the author's 1000-bin differential histogram entropy plus 30 times the fraction of nontrivial cosine coefficients outside index-radius 10 exceeding 0.1. The histogram lower edge is 0.6 times the first percentile; the common histogram width is 1.5 times (initial 99th percentile−0.6 initial first percentile). Quantile sampling strides are ceil(number of preliminary pixels/2^24). The Fourier penalty uses unnormalized DCT-II, excludes either zero index, and counts i²+j²>100.

The first search evaluates λ=0.01,0.1,0.5,2,8,10. Its best neighboring bracket selects the corresponding entries from [0.01,0.1,0.5,0.75,1,1.25,1.5,1.75,2,2.5,3,4,5,6,7,8,10]; already evaluated endpoints are reused. Here seven additional candidates are 2.5,3,4,5,6,7,8 and the chosen λ is 5. All 13 original groups are preserved in notebook comparison. A final complete 100-frame fit uses this λ and the same full stopping rules.

Evaluation scores both physical outputs jointly with the README's unweighted DN objective J and MAE A, strict positivity and mean-one gauge. Bounds were frozen from a complete original run and negative controls before local extraction. They do not claim unique truth or exact minimization by the adaptive algorithm. There is no measured true flatfield for this acquisition. Historical original corrected image examples are method outputs, never ground truth.
