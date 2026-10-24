# Implemented mathematics and estimator

## Bound acquisition and unknowns

The input is the full native CFIN human19-slice diffusion acquisition: int16 signal `(1,96,96,19,496)`, one b0 and33 directional observations at each of15 b shells200:200:3000s/mm². All native voxels/volumes remain public; an input-derived pre-fit foreground mask fixes44427 evaluation voxels. No fitted-field mask or spatial interpolation is used. Native voxel spacing is(2.5,2.5,2.500000476837158)mm. Axes remain batch,x,y,slice,acquisition. The original file slope1/intercept0 means scanner intensities are preserved exactly.

The selected published stage is multi-compartment spherical-mean microstructure estimation, corresponding to the author's `fitmcmicro` executable. We do not claim to reconstruct the paper's separate downstream fiber-orientation distribution or entropy fields. At each voxel the free unknowns are intra-neurite signal fraction v and intrinsic longitudinal diffusivity D. Derived extra-neurite transverse diffusivity is(1-v)D and extra-neurite mean diffusivity is(1-2v/3)D. All four fields are submitted/scored. S0 is the arithmetic mean of observed b0 intensities and is fixed, not a fitted target.

The two-compartment signal assumes zero intra-neurite transverse diffusion, equal longitudinal diffusivity across pools, tortuosity-constrained extra-neurite transverse diffusion, negligible inter-compartment exchange during encoding and equal response weighting aside from their signal fraction. Because this acquisition uses inversion-recovery suppression and finite TE, v is a conditional signal-weighted fraction. No histological calibration or absolute cellular volume is inferred.

## Explicit forward response and objective

For b in s/mm² and microscopic longitudinal/transverse diffusivities p,t in mm²/s, the uniform orientation integral is

$$A(b,p,t)=\int_0^1 e^{-b[t+(p-t)u^2]}du
=e^{-bt}\frac{\sqrt\pi\,\operatorname{erf}\sqrt{b(p-t)}}{2\sqrt{b(p-t)}}.$$

The ratio has value1 when b(p-t)=0. The implementation reproduces the source convention of swapping p,t when p<t; the selected compartment model always supplies p>=t. At b=0 the response is1. `spherical_mean` broadcasts scalar/array inputs and evaluates the formula using NumPy and the general SciPy error function. No domain reconstruction function is delegated.

The expected shell-mean signal is

$$s(b;v,D)=S_0[vA(b,D,0)+(1-v)A(b,D,(1-v)D)].$$

`signal_objective` implements the unnormalized original intensity-domain SSE

$$F(z)=\sum_{j:b_j>0}[Y_j-s(b_j;v(z),D(z))]^2.$$

Every positive-b directional observation is included. The b0 observation does not enter the fit residual. There is no noise-variance division, logarithmic transform, Rician correction, regularizer, spatial prior or additional compartment. Repeated b values share the explicit forward evaluation but predictions are gathered back to every native directional residual before summation. This changes neither observation weights nor the mathematical objective. With33 observations per shell, F equals33 times shell-mean SSE plus the parameter-independent within-shell sum of squares. Directional residuals include orientation variation; powder-mean residuals quantify the model fit to its intended observable.

The uniform-sphere integral is exact for the theoretical model. Finite33-direction acquisition averaging is approximate, particularly because this acquisition combines a24-point spherical design and nine additional directions. Gradient vectors are supplied as native observations/calibration. The original loader normalizes their nonzero lengths without changing nominal b values; no gradient-deviation map is supplied. The spherical-mean response depends on b and the orientation integral, not on a fitted orientation field.

## Bounded parameterization and initialization

Unconstrained logits z=(z1,z2) map to v=exp(z1)/(1+exp(z1)) and D=Dmax exp(z2)/(1+exp(z2)), with Dmax=3.05e-3mm²/s. Preserve the original arithmetic and its overflow branch: above log(max_float64), return the upper bound; for finite smaller values use the exp(z) fraction. Do not substitute a different logistic arithmetic in tightly compared iteration fixtures.

The starting vertex is(0,0), corresponding to v=.5,D=.001525mm²/s. The other stored vertices are(1,0) and(0,1). There is no RNG or estimated starting parameter. There are exactly three stored vertices, with a separate rank index. Storage identities remain unchanged during rank sorting.

## Explicit original simplex updates

At each update rank vertices by objective and average the best and second-best vertices to form centroid c. For worst vertex w, reflection is r=2c-w. If its objective lies between best and second-best, replace worst with r. If it improves best, evaluate expansion e=3c-2w and choose e only when it improves r. Otherwise, if the reflected objective lies between second-best and worst, evaluate outside contraction1.5c-.5w and accept it if no worse than reflection. In the remaining branch evaluate inside contraction.5c+.5w and accept only if it improves worst. Failed contraction shrinks each non-best stored vertex halfway toward the best. Updated worst vertices use strict-less rank insertion. These are the source coefficients1,2,.5,.5; no generic optimizer substitutes another update rule.

The original initial/shrink sort supplies `<=` to `std::stable_sort`, formally violating strict weak ordering. This is not silently copied as a generic comparator. In the fixed original libstdc++ three-element runtime, small-array insertion sorting places an equal newly inserted key before preceding equal keys. `_sort_three` explicitly implements bounded insertion with that tie ordering, preserving the actual captured sequence without undefined generic-sort behavior. `_sort_last` preserves the source strict-less update insertion. Independent quadratic and all-tied constant objectives, source first-ten-update fixtures and full native-field parity investigate this discrepancy. The constant objective has arbitrary equally optimal coordinates; no uniqueness claim is made.

## Stopping and batching

After every update, objective spread is abs(Fbest-Fworst). The source's coordinate test uses **storage slots0 and2**, rather than the full simplex span or ranked extrema. Preserve that selected convention:

$$|F_{best}-F_{worst}|\le\max(10\epsilon,1000\epsilon|F_{worst}|),$$
$$\|Z^{(0)}-Z^{(2)}\|_\infty\le\max(10\epsilon,1000\epsilon\|Z^{(2)}\|_\infty).$$

Here epsilon is float64 machine precision. Stop a voxel when both hold or at10000 updates. Actual convergence flags and iteration counts are returned, not inferred from output existence. A fitted bound or a difficult voxel is retained; it is not removed from the mask.

Independent voxels are processed in deterministic batches of1024 by default, CPU one thread. No spatial coupling exists, and batch partitioning does not alter the model or drop observations. Inactive converged voxel states remain fixed while other batch members continue. Optional history records the actual best stored logits/objective after each executed update. All final scientific fields use float64 calculations and exact zero padding outside the fixed input mask. Source original output is float32 storage; its storage envelope is measured independently.

## Evaluation, references and review experiments

The hidden reference is the independently executed original `fitmcmicro` result on these exact observations and this same input-only mask, not truth. Every one of four quantities is compared over all44427 voxels without rescaling, axis changes, registration or outcome-dependent masking. NCC is uncentered cosine similarity; NRMSE is RMS error divided by reference max-minus-min. For all fields, numerical boundaries NCC>=.999999 and NRMSE<=1e-5 were fixed before local inverse extraction from original float64/float32 and one-ULP envelopes. No boundary is tuned to local results.

Original source directional SSE, independent sphere quadrature, captured first ten simplex updates, full float64 sample fits and full native float32 output provide complementary evidence. Agreement with a method reference does not validate biological specificity. Boundary saturation, finite angular quadrature, imperfect input foreground, inversion-recovery weighting and model mismatch remain disclosed limitations.

The active notebook reconstructs full raw data, compare fresh and independently original fields, show all native slices and four quantities, evaluate powder-mean/directional residuals separately, inspect optimization histories and physical boundaries, and perform explicitly labeled shell-coverage/noise perturbations. The initial complete execution exited0 in653.8276844 seconds; the final visual revision and its execution are identified in validation. No cached comparison substitutes for fresh reconstruction.

The directional squared residual matrix is explicitly made C contiguous before summation. This fixes the same acquisition-order reduction for single-row and multi-row batches, rather than allowing NumPy advanced-index memory layout to change a sensitive optimizer branch. An independent batch-size test and original first-ten-step/full-field fixtures validate the convention.

A notebook-only null model sets v=0 and fits the isotropic response S0 exp(-bD) on [0,0.00305]mm?/s using explicit golden-section interval updates (tolerance1e-12, maximum100). This is a freshly fitted diagnostic comparison, not a reproduced competing paper algorithm. Shell-coverage experiments fit full support with five low-b shells or seven/eight alternating shells. Added-noise experiments add Gaussian1%/3% of observed b0 to positive-b measurements with seed20261022 and retain fixed b0/support without clipping.

The added-noise observation generation is implemented in src.generate_data.add_gaussian_noise and invoked by the notebook. It preserves the exact previous RandomState seed, draw ordering, scaling and all-b0/no-clipping conventions. Judge reference selection gives actual truth precedence when supplied and treats malformed existing truth as an evaluator error, without baseline fallback.
