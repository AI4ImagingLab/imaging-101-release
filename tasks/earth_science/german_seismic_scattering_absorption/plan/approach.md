# Implemented transport inversion

The task uses 24 measured three-component records for five earthquakes, each sampled at 20 Hz for 4401 points. All five frequency bands and every valid record enter preprocessing. One low-frequency station/event has no usable coda after the measured SNR rule; it remains a visible observation but is excluded from that fit. The last event has four acquired stations. All 25 event-frequency inversions execute.

## Preprocessing and units

For each component, remove a least-squares straight line, apply a second-order Butterworth bandpass in SOS form forward and backward with zero initial state, and compute the periodic FFT Hilbert transform. If $v_k$ denotes filtered component k and $h_k$ its Hilbert transform, form

```math
E=\frac{2700}{2\Delta f\cdot4}\sum_{k=1}^3(v_k^2+h_k^2),\qquad
\Delta f=\frac{20}{1024}\sum_{j=0}^{511}|H_j|^4.
```

$H$ is single-pass digital filter frequency response on the default 512-point grid; $\Delta f$ is in Hz and $E$ in J m$^{-3}$ Hz$^{-1}$. No FFT amplitude normalization is introduced beyond the Hilbert implementation. Noise is the minimum energy mean over [-10,-5] and [-5,0] seconds. Subtract it, floor at noise/100, then Bartlett-smooth 20 samples. Sample boundaries use floor((time-start)*20+0.5). The recorded S onset has microsecond timestamp precision and is preserved verbatim; theoretical Green times use continuous distance/3400.

Bulk energy is the unsmoothed mean from S-2 to S+20 inclusive; its representative time is the energy-weighted mean on a continuous theoretical-arrival grid. Coda starts S+20 and ends at S+150 or the first smoothed crossing below three times noise, whichever is earlier. The minimum accepted duration is 0.05 s. Coda time starts distance/3400+20. Arrays retain inclusive endpoints.

## Transport and discrete forward model

Let r be distance in m, t time in s, c=3400 m/s, g scattering in m^-1, b intrinsic absorption in s^-1, W source in J/Hz, and R dimensionless site response. The prediction is WR G(r,t;g) exp(-bt). Define u=cgt, v=gr, a=1-v²/u². For ct>r the scattered density is

```math
G_c=\frac{g^3a^{1/8}}{(4\pi u/3)^{3/2}}\exp[u(a^{3/4}-1)]\sqrt{1+2.026/(ua^{3/4})}.
```

The direct term has integrated temporal weight exp(-gr)/(4 pi c r²). `green` represents it as one bin divided by time spacing. It is zero before arrival; following upstream, the coda begins one index after the first nonnegative ct-r index. This omission is preserved and explicitly tested. Coda Green density extends 9 points left/10 right before Bartlett convolution; bulk Green is the mean over 1000 points including endpoints. Absorption for bulk uses the energy-weighted representative time rather than integrating the absorption exponential inside the bulk kernel, as upstream specifies.

## Weighted variable projection

For fixed g, stack all station coda samples followed by one bulk observation per station. Set target y=log(E)-log(G). The matrix A has one station indicator column per station and final column -t; its unknown vector contains station log amplitudes and b. Coda weights are 1; bulk weight is number of original bulk samples. Solve min ||sqrt(w)(A x-y)||² by SVD of sqrt(w)A. Singular values below 1e-15 times largest are discarded, preserving upstream pseudoinverse arithmetic. Degrees of freedom use matrix_rank(diag(singular_values)), matching the original weighted fit. Report error sqrt(weightedSSR/(N-rank)). This is a diagnostic with correlated coda samples, not an independent-noise likelihood.

```math
\log W=\frac1S\sum_{s=1}^S x_s,\qquad R_s=\exp(x_s)/W,\qquad \prod_s R_s^{1/S}=1.
```

S is number of usable stations; N is total stacked observations. The common scale is not identifiable without this gauge. Search g using SciPy golden minimization, initial bracket (1e-8,1e-4), relative tolerance 1e-3, default maximum iterations 5000. The objective is infinite outside that interval and whenever fitted b is outside (0.001,10). No additional regularizer, random initialization or gradient is used. Failure to converge raises an error. All bound observations have interior feasible solutions. The solver's scalar profile repeats explicit SVD at each trial; there is no domain inversion library.

## Evaluation and scientific limitations

For g/b jointly, W and valid R separately, maximum absolute log estimate/reference ratio must be <=0.005. Invalid slots must be zero and valid site geometric-mean log must be <=1e-5 in magnitude. The independent original repeated exactly; a float32 submission is checked separately. Each of the 25 forward errors must be <=1.05 times its independently captured native value +0.001. These are method agreement and forward consistency, not accuracy to unavailable crustal truth. The source/site split is gauge dependent; frequency/event spread reflects regional path heterogeneity as well as estimation uncertainty. Notebook comparisons include fixed scattering, full fitted transport, source/site differences and residual behavior; perturbation experiments are labeled supplementary and never replace the actual observations.
