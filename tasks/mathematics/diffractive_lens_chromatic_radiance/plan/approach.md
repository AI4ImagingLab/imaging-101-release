# Implemented alternating inverse

Input is decoded RGB float64 on the original digital0--1 scale. Green is measured guidance, not truth. Independently restore red and blue without altering green. Primary parameters: PSF patch500pixels, step100; transfer patch60pixels, step12; alpha=beta=.3; three iterations. Patch starts advance by the stride, with the final patch forced flush to the bottom/right edge. Order is column outer, row inner. Vectorization within a patch is MATLAB column-major. A separable symmetric Hamming window is 0.54-0.46cos(2pi*n/(L-1)); accumulated weighted values are divided by accumulated weights at every pixel.

For each transfer patch initialize sharp color as mean(observed color)/mean(green) times green. The bundled nonzero green patch means make this defined; no epsilon or clipping is silently inserted. Assemble overlapping estimates with Hamming weights. Compute forward green differences, with zero last-column x difference and zero last-row y difference (replicated boundary).

For patch estimates x and observations y, their two-dimensional Fourier transforms X and Y give the regularized kernel H:

```math
H=\frac{\overline X Y}{\overline X X+\beta+\alpha(|D_x|^2+|D_y|^2)}.
```

Dx,Dy are FFTs of [1,-1] horizontal/vertical filters zero-padded to500by500. This solves squared convolution residual plus beta times squared kernel norm and alpha times squared kernel gradient norm, using the same unnormalized FFT convention as the original. No positivity, kernel-sum-one constraint, recentering or magnitude operation is introduced. Apply H to the green patch's FFT, inverse transform and retain the real component; aggregate with the500pixel Hamming window.

Let g be this blurred green guide. On each60pixel patch construct Z with columns one, g, its forward x difference and its forward y difference. Let y be the observed color patch vector. Solve Z-transpose Z times theta = Z-transpose y. Theta consists of offset, color scale and two derivative coefficients. Predict a sharp color patch using the same theta with columns one, original green and its derivatives. Aggregate with the60pixel Hamming window. Repeat kernel fitting/transfer exactly three times; no convergence check or extra optimizer is inserted.

The local implementation streams patches instead of materializing all four-dimensional arrays. Normal equations are solved independently rather than as one block-diagonal sparse system; numerical parity is measured against the original native MEX/sparse-solve run. This changes storage, not the objective or discretization. After the last iteration, inverse-transform each full Fourier kernel and expose its centered49by49 subset only as a diagnostic. Full500by500 kernels, not the displayed crop, perform convolution.

Notebook comparisons retain input, mean-ratio initialization, each of the three fresh iterations, independently reproduced original result, regularization factors0.5 and2 and transfer-window sensitivity. All sharp display images are clipped solely by plotting; stored arrays and metrics retain overshoot. No independent sharp GT exists. Original reference agreement uses cosine NCC and full-image range-NRMSE, frozen NCC>=.999999,NRMSE<=1e-5; green max error<=1e-7. The green criterion is acquisition consistency, not proof of red/blue physical fidelity.
