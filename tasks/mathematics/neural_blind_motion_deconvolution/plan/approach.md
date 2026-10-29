# Implemented mathematical approach

## 1. Bound scientific problem and selected method

This task jointly estimates a full-resolution sharp luminance raster and a motion
kernel from the actual RealBlur-J scene026 exposure `blur_3.png`. The observation
has 753 rows and 668 columns. It is not resized, cropped, synthetically blurred or
noised during packaging. The independent short-exposure image is an evaluation
measurement, not an input to either generator or its initialization. Both images
were distributed after the dataset's registration and intensity processing;
neither is a linear raw sensor image. The uniform pixel-domain blur approximation
does not explain every JPEG, noise or depth-dependent effect in this real scene.

The selected method is the **joint dual-deep-prior reference-release variant** of
SelfDeblur, specifically the numerical path in `selfdeblur_ycbcr.py` at the
revision recorded in validation. The published paper motivates image and kernel
generators and joint optimization. The selected release additionally contains
three nonlocal image blocks, removes the paper's TV term, and switches from MSE
to SSIM after 500 updates. These are actual upstream choices, not changes made to
simplify cleaning. This task does not claim to reproduce the paper's historical
Levin/Lai result tables or their different acquisition conditions. No pretrained
inverse model is loaded; all 5,170,122 trainable parameters are fit to this one
observation. The complete five-scale network and all 2500 updates are retained.

## 2. Discrete acquisition, units and boundary domains

Let H=753, W=668, K=51 and q=(K-1)/2=25. Let y in R^(H x W) be the visible
luminance observation, u in R^((H+2q) x (W+2q)) the padded sharp latent image,
and k in R^(K x K) the unknown correlation kernel. Row increases downward and
column rightward. Intensities are dimensionless fractions of 8-bit full scale;
kernel entries are dimensionless weights. All model computations use float32.

```math
(A_k u)_{r,c}=\sum_{a=0}^{K-1}\sum_{b=0}^{K-1}k_{a,b}u_{r+a,c+b},\qquad
y=A_k u+\epsilon,\qquad k\geq0,\quad\sum_{a,b}k_{a,b}=1.
```

The unknown epsilon includes shot/read noise, quantization, preprocessing and
nonuniform-blur mismatch; no Gaussian variance calibration is invented. This is
PyTorch's cross-correlation convention. If a physical convolution kernel is
written with negative displacement, it is the spatial reversal of k. There is no
FFT-periodic blur in the optimization. Reflection padding belongs only to the
image generator's feature convolutions; it is not an acquisition boundary model.

The actual dimensions and crops are distinct:

| Quantity | Shape excluding batch/channel | Meaning |
|---|---|---|
| y | 753 x 668 | Complete visible observation |
| u | 803 x 718 | Generated latent; includes unobserved 25-pixel border |
| A_k u | 753 x 668 | Complete predicted exposure, used in every training loss |
| x = u[25:778,25:693] | 753 x 668 | Submitted sharp reconstruction |
| valid correlation of x with k | 703 x 618 | Independently evaluable forward region |
| y[25:728,25:643] | 703 x 618 | Matching observation region for that check |

The last two rows follow directly by substituting x(r,c)=u(r+q,c+q):
A_k x(r,c)=A_k u(r+q,c+q). The Judge does not try to reconstruct the missing
latent border. It does not pad x with zeros, wrap it, or borrow a target boundary.
Training always uses the complete y; the smaller diagnostic domain is a
consequence of submitting the centered image rather than the padded latent.

Before packaging, the integer YCrCb luminance is obtained from RGB values R,G,B
using `(4899 R + 9617 G + 1868 B + 8192) >> 14`, then cast to float32 and divided
by 255. This reproduces the selected upstream OpenCV channel conversion on every
bound pixel. The runtime simply loads the already-converted NPZ; it does not
depend on an image reader, OpenCV or hidden source PNG files.

## 3. Parameterization of the two unknowns

Write u=G_theta(z+eta) and k=H_phi(v). Theta denotes all image-generator
parameters and phi all kernel-generator parameters. The fixed image input z has
shape (1,8,803,718); its entries are independent U[0,0.1). The fixed kernel input
v is a 200-vector from the same uniform law. A fresh eta with entries
0.001*N(0,1) perturbs the image input on every iteration. There is no kernel-input
perturbation. These are optimization inputs, not seeds or objects used to create
the measured scene. Seed 42 fixes initialization and the perturbation sequence.

### Image generator G_theta

There are five levels, each with 128 main channels and 16 skip channels. At a
level receiving C channels, the skip branch is 1x1 convolution C->16, batch
normalization and LeakyReLU with negative slope 0.2. The deeper branch is a 3x3
stride-2 convolution C->128, batch normalization and LeakyReLU, an optional
nonlocal block at levels 2,3,4, then 3x3 convolution 128->128, normalization and
activation. The next level processes this deeper branch. The decoder doubles
resolution using bilinear interpolation with align_corners=False, centers both
branches to their common minimum shape, and concatenates skip before deeper
features. A 144-channel normalization is followed by 3x3 convolution 144->128,
normalization/activation and 1x1 convolution 128->128 with normalization/activation.
The final 1x1 map 128->1 and sigmoid enforce 0<u<1.

Every convolution has a bias. Feature padding is reflection with radius floor
(filter_size/2); 1x1 convolutions have zero-radius reflection. Normalization stays
in training mode, including when producing the final reconstruction: the affine
map uses each current feature channel's spatial mean and biased variance with
epsilon 1e-5. Running buffers use PyTorch's default momentum 0.1 and are updated,
but are never substituted for current-batch moments in this reconstruction path.

The downsampled spatial sizes are (402,359), (201,180), (101,90), (51,45),
(26,23). Center-cropping after each bilinear upsample handles odd sizes and returns
exactly (803,718). Crops use integer floor division; odd surplus pixels therefore
remain on the bottom/right before exclusion. This convention is parity-tested.

For a nonlocal block, let F be its 128-channel input on N spatial locations.
Three biased 1x1 maps produce query Q, key T and value V with 64 channels. Keys
and values undergo 2x2 max pooling, giving M=floor(height/2)*floor(width/2) sites.
Using matrices Q in R^(N x 64), T in R^(64 x M), V in R^(M x 64), compute

```math
Z=F+\mathrm{BN}\!\left(W\,\mathrm{reshape}\left((QT/M)V\right)\right).
```

W is a biased 64->128 pointwise convolution. This dot-product interaction is
divided by M, not softmax-normalized. The output BN scale and bias start at zero,
so each such block is initially an identity but can learn a nonlocal correction.
Queries/keys/values are task-local executable operations, not an imported imaging
or attention reconstruction framework.

### Kernel generator H_phi

For fixed v in R^200, compute h=ReLU6(W1 v+b1), where W1 has shape (1000,200).
Compute l=W2 h+b2 in R^2601 and k_j=exp(l_j)/sum_i exp(l_i), reshaped row-major
to (1,1,51,51). ReLU6 clamps between 0 and 6. The softmax gives strict positivity
and unit sum without projection. There is no assumed target trajectory or
target-derived kernel initialization. The kernel is not recentered after fitting.

## 4. Objective and explicit mathematical stages

For zero-based update index t, compute yhat_t=A_(k_t) u_t. Define N=H*W=503004.

```math
\mathcal L_t=
\begin{cases}
N^{-1}\sum_{r,c}(\widehat y_{r,c}-y_{r,c})^2,&0\leq t<500,\\
1-\mathrm{SSIM}(\widehat y,y),&500\leq t<2500.
\end{cases}
```

The MSE is a mean, not a sum or half-mean; its prediction gradient is
2*(yhat-y)/N. There is no TV penalty, kernel sparsity term, weight decay or other
regularizer in this selected release. Regularization comes from the generator
parameterization, fixed noise inputs, input perturbations and finite iteration
budget. Switching the discrepancy does not reset either Adam state.

For SSIM, let g_j be the normalized 11-point Gaussian with sigma=1.5 and center
index 5, and let w_ab=g_a*g_b. The window is built on CPU in float32 and moved
without changing its coefficients. For two images a and b, zero-padded window
convolution produces mu_a=w*a, mu_b=w*b, variances
v_a=w*(a*a)-mu_a^2, v_b=w*(b*b)-mu_b^2 and covariance
c_ab=w*(a*b)-mu_a*mu_b. No unbiased-sample correction or variance clipping occurs.

```math
\mathrm{SSIM}(a,b)=\frac1N\sum_{r,c}
\frac{(2\mu_a\mu_b+C_1)(2c_{ab}+C_2)}
{(\mu_a^2+\mu_b^2+C_1)(v_a+v_b+C_2)},\quad
C_1=0.01^2,\quad C_2=0.03^2.
```

Zero padding here defines the **loss window moments**, not the optical boundary.
The local implementation writes every window moment and quotient explicitly;
autograd differentiates this graph and the generators. No ready-made SSIM or
blind-deconvolution function is called.

For an arbitrary prediction-space residual derivative R, the forward adjoints
have direct finite-support meanings:

```math
(A_k^*R)_{i,j}=\sum_{r,c:\ 0\leq i-r<K,\ 0\leq j-c<K}
R_{r,c}k_{i-r,j-c},\qquad
\frac{\partial\mathcal L}{\partial k_{a,b}}=
\sum_{r,c}R_{r,c}u_{r+a,c+b}.
```

These identities are checked against an independent discrete correlator and
finite differences; agreement with an upstream copy alone would not establish
the acquisition sign or boundary. SSIM input derivatives are likewise checked by
central finite differences and frozen upstream gradients.

## 5. Initialization, optimizer and exact update order

The implementation fixes Torch seed 42; draws z on CPU and transfers it; builds
the image network on CPU and transfers it; draws v on CPU and transfers it; then
builds/transfers the kernel network. Standard PyTorch convolution/linear
initialization and default BN initialization are preserved, with the special
zero output-BN initialization in the nonlocal blocks described above. Layer
allocation remains top-down to preserve the reference's random draw order,
although the independently written class organization differs.

Both parameter groups use Adam with beta1=0.9, beta2=0.999, epsilon=1e-8,
weight_decay=0, amsgrad=False. For a parameter p and its gradient g_t,
m_t=0.9*m_(t-1)+0.1*g_t, v_t=0.999*v_(t-1)+0.001*g_t^2, both initialized at zero.
Bias-corrected moments divide by 1-beta^(t+1). The update subtracts
alpha_t*mhat_t/(sqrt(vhat_t)+1e-8). Torch's general optimizer evaluates this
standard update; no task-specific inverse update is hidden inside it.

| Zero-based iteration | Image learning rate | Kernel learning rate |
|---|---:|---:|
| 0 through 1599 | 0.01 | 0.0001 |
| 1600 through 1899 | 0.005 | 0.00005 |
| 1900 through 2199 | 0.0025 | 0.000025 |
| 2200 through 2499 | 0.00125 | 0.0000125 |

The reference calls its epoch scheduler before the optimizer and passes the
explicit epoch index. The cleaned code evaluates the same closed-form table,
avoiding modern scheduler-order warnings without changing any learning rate.
For each iteration: draw image perturbation; set both rates; clear gradients;
evaluate both generators and the complete valid exposure; evaluate the current
loss; reject a nonfinite objective; backpropagate; apply one joint Adam update;
record index, objective, and both rates. There is no alternation, early stopping,
line search, learned training set, best-GT checkpoint selection, or extra nonblind
postprocessing. All 2500 iterations execute.

The reported image and kernel are the tensors computed during iteration 2499,
immediately before its optimizer step, matching the reference save convention.
Reevaluating the updated networks on unperturbed noise would be a different
output. The final arrays are float32 and have batch axes added only on saving.
CUDA synchronization makes reported elapsed time include completed device work.

## 6. Translation gauge and scientifically fixed evaluation

With cross-correlation, shifting both array functions as u_s(r)=u(r-s) and
k_s(a)=k(a-s) preserves A_k u on a domain where neither support wraps or truncates.
Changing variable b=a-s in the sum proves this identity. In the physical
convolution PSF convention, the kernel reversal turns this into opposite shifts
of image and PSF. An independent compact-support test verifies the sign and
identity. This is a genuine identifiability limitation, not free image warping.

Image-only evaluation therefore selects one integer shift (dy,dx) in [-25,25]^2
by maximizing full-image cross-correlation after subtracting each image's mean.
The FFT computes correlation; it does not imply periodic optical blur. Its
search box and the row-major tie rule are fixed. After selecting the shift,
compare x[50-dy:H-50-dy,50-dx:W-50-dx] with reference[50:H-50,50:W-50].
The 50-pixel margin is kernel radius 25 plus maximum gauge shift 25. It leaves
653*568=370904 pixels, about 73.7% of the entire exposure, including the information
sign, fine lettering and background lights. It is not an adaptively chosen easy
mask. No scale, offset, subpixel warp or per-region alignment is fitted.

NCC is the cosine of flattened aligned image and reference, without mean
subtraction in the score itself. NRMSE is RMS difference divided by the reference
interior's maximum-minus-minimum. Nonfinite or degenerate evaluator references
are errors. A zero estimate has NCC zero. The unaligned forward NRMSE compares
valid correlation of the submitted x and k with y[25:-25,25:-25], dividing by that
observed region's range. Kernel mass error is abs(sum(k)-1); negative mass is
sum(max(-k,0)). Image extrema must remain within [0,1]. Every boundary must pass.

Thresholds were frozen from the independent upstream run before the cleaned full
run: NCC >=0.9*0.9918410391059359; NRMSE <=1.1*0.029215276490004403;
forward NRMSE <=1.1*0.008502817858528754; kernel mass error <=1e-5 and negative
mass <=1e-8. The nonzero error tolerances use independent real truth and measured
observation discrepancy, not self-comparison. The blurred identity has NRMSE
0.041142400423377426, so it is rejected even though a delta kernel explains its
own observation. Historical paper metrics are not relabeled as task results.

## 7. Evidence, execution limits and interpretation

The original five-scale release runs privately on the exact bound luminance.
Frozen fixtures capture image/kernel outputs, complete predicted exposure, first
MSE and both first-layer gradients, first Adam updates, and SSIM plus its input
gradient. Local comparisons test these before the full pipeline. The full
integration test invokes the complete CLI in a fresh temporary directory and
compares both arrays to the independently produced upstream result; it cannot
pass by reading a stale output. Judge contract tests include bad image quality,
incorrect kernels, malformed archives and missing evaluator assets.

CPU tests validate explicit forward mathematics, adjoints, SSIM derivatives and
metrics. The complete 2500-step reconstruction is validated on RTX 4080 SUPER,
not claimed as a practical full CPU benchmark. Numerical conventions disable
TF32 and cuDNN benchmarking and enable deterministic cuDNN selection. The
validated Torch/CUDA versions and measured runtimes are in validation. Full
deterministic backward operations cost substantially more than the historical
six-minute cuDNN-only run: approximately 0.76 seconds/update on this machine.
The complete notebook/test timeout is 5400 seconds, allowing the roughly
32-minute deterministic reconstruction plus review work and shared-GPU load.
Library imports do not change Matplotlib's backend.

The reconstruction improves the fixed real-reference interior error over the
blurred exposure, but residual ringing, stripe texture and noise remain. Its
kernel is an effective model fit, not an independently measured true camera
trajectory. Forward consistency and upstream parity do not prove complete
physical accuracy. Independent short-exposure agreement, the physical checks
and an explicitly scoped approximation are all needed to assess the result.
