# Implemented calibrated AFM inverse

The physical unknown is a set of 30 molecular specimen topographies, each sampled on a 30 by 30 grid at lateral pitch 15 Angstrom. Heights and reflected probe ordinates retain the author's stored Angstrom unit. The independent double-apex reflected probe is a 14 by 14 float64 array with its active symmetry origin at index (6,6), and with maximum zero. The substrate is zero. The public observations are the exact first 30 stored noisy images, without normalization, cropping, denoising or downcast. Observation storage and main computation are float64; the explicitly labeled blind comparison uses float32 to match its independent original run.

Let s(n,r,c) denote specimen height; y(n,r,c) denote the observed contact height; p(i,j) denote the nonpositive reflected probe cap. Here n=0,...,29, r,c=0,...,29, i,j=0,...,13. Define offset u=(i-6,j-6). The implemented forward model is

```math
D_p(s)_{nrc}=\max_u[s_{n,r+u_y,c+u_x}+p_u],\qquad y=D_p(s)+\epsilon.
```

The entries of epsilon are independent Gaussian height noise with standard deviation 3 Angstrom. The forward maximum excludes out-of-field specimen points; there is no periodic wraparound. Row and column coordinates in the displayed local frame are 15*r and 15*c Angstrom relative to the first stored grid point. This display translation does not move any data or alter the inverse. Image and sample axes are identical. No tip-shape information is derived from hidden surfaces.

The original morphological envelope inverse is an erosion, not a least-squares optimization or linear deconvolution. For the declared forward-offset convention its order adjoint is

```math
E_p(y)_{nrc}=\min_u[y_{n,r-u_y,c-u_x}-p_u].
```

Missing exterior image samples are plus infinity, so they impose no upper-bound constraint. The independent original modern implementation evaluates same-offset erosion; on the bound data its active symmetric probe support makes the values exactly equal to the adjoint expression above. This equality was measured, not assumed generically. For an asymmetric probe the local code follows the mathematically correct reversed-offset definition and is tested by order-adjunction and re-dilation identities. The inactive skirt of this particular probe is below -223 Angstrom; it cannot be selected by erosion for the bounded observations. Original and local routines therefore agree exactly at every bound pixel.

The shipped estimator is the declared constrained variant

```math
\hat{s}=\max(0,E_p(y)).
```

This projection uses the independently known substrate height. It is a visible modeling choice beyond raw upstream erosion and is reported separately from that erosion. It does not claim exact inversion of noisy contact data. No iterative initialization, step size, convergence tolerance or stopping rule applies to this closed-form main solver. There is no smoothing regularizer or fitted noise parameter in the shipped estimator. Every finite-grid offset is evaluated once. The maximum/minimum reductions preserve the physical height scale and signs.

The notebook additionally performs a fresh modern differentiable blind-tip reconstruction as an explicitly different comparison. It uses 30 float32 frames, a 10 by 10 zero-initialized tip, 200 epochs, and sequential frame updates. Each update minimizes the mean squared opening error `mean((dilation(erosion(frame,tip),tip)-frame)**2)`. AdamW uses learning rate 0.1 and weight decay 0.0011, with PyTorch defaults betas=(0.9,0.999) and epsilon=1e-8. The tip is clamped nonpositive after every optimizer step and recentered by the weighted center of its heights above the current minimum; weights below 1e-8 are zeroed, with a uniform fallback when all weights are below 1e-10. The origin used by that recenter routine is Python round((10-1)/2)=4, while the unfold padding is (4,5). Loss history is the sum of the 30 post-update frame mean squared errors per epoch. This comparison intentionally retains the original same-offset morphology convention, unlike the general adjoint main operator. The fixed 200-epoch stop, probe size and regularization are modern notebook defaults, not acquisition parameters. Neither its estimated probe nor its recovered surfaces are used as ground truth.

Actual author surfaces, not reconstructed envelopes, provide evaluation truth. All 27,000 specimen pixels are evaluated without masks, alignment, offset subtraction, rescaling or magnitude conversion. NCC is flattened cosine similarity with no mean subtraction. NRMSE is root mean square height error divided by the full reference maximum minus minimum. A zero prediction has NCC=0. The nonconstant truth makes NRMSE well-defined. Forward observation consistency is `sqrt(mean((D_p(estimate)-y)**2))` in Angstrom, using the public probe and observations. All three boundaries must pass. The standard result has NCC 0.9514423828199022, NRMSE 0.03055631860849182 and observation RMSE 3.018926125977721 Angstrom. Boundaries were fixed before local scoring at 0.9 times baseline NCC and 1.1 times each error. This is an actual-truth comparison, not self-comparison to a reconstruction.

Sensitivity analyses perturb only the known probe-height scale by 0.7,0.85,1,1.15,1.3 and separately prefilter the observations with Gaussian sigma 0,0.35,0.7,1 pixel. These are additional measured ablations, not alternative acquisition metadata or shipped hyperparameters. They do not change the main baseline thresholds. Five exported figure groups retain raw erosion, the declared substrate projection and the fresh blind comparison. Limitations include inaccessible valleys, extrema bias under additive noise, a geometric nondeforming contact approximation, known-probe dependence, and use of author-existing simulations rather than real AFM measurements.
