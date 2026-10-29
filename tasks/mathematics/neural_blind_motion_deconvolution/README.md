# Neural Blind Motion Deconvolution

> Recover a sharp full-resolution luminance image and an unknown motion kernel from a real handheld-camera exposure using a strongly nonconvex blind imaging model.
> Domain: Mathematics | Keywords: blind inverse problems, image restoration, deep priors | Difficulty: Hard

## Background

During an exposure, camera motion distributes the light from a scene point across several detector samples. A spatially invariant point-spread function approximates this integration when motion and scene depth produce similar local trajectories across the field. The resulting image loses fine texture and develops directional streaks. JPEG processing and read noise add discrepancies beyond this idealized optical model.

Blind restoration must infer both the scene and the point-spread function. A sharp image with a broad kernel can resemble a smoother image with a narrow kernel, and shifting the scene while adjusting the kernel origin creates an additional ambiguity. This makes blind reconstruction fundamentally different from inversion with a measured, fixed blur kernel. Applications include handheld photography, document recovery and motion-degraded scientific imaging.

## Problem Description

The observation is a real, processed camera exposure of an indoor information sign and background lights. Reconstruct its normalized luminance and the unknown nonnegative motion kernel. The selected model is valid cross-correlation:

```math
y_{r,c}=\sum_{a=0}^{50}\sum_{b=0}^{50}k_{a,b}u_{r+a,c+b}+\epsilon_{r,c},
\qquad k_{a,b}\geq0,\qquad\sum_{a,b}k_{a,b}=1.
```

Here $y$ is measured luminance, $u$ is the sharp latent image including an unobserved boundary, $k$ is the cross-correlation kernel, and $\epsilon$ includes noise and model mismatch. Indices $r,c$ denote image row and column; $a,b$ denote kernel row and column. All intensities are dimensionless in the interval $[0,1]$. The corresponding physical convolution point-spread function is the spatially reversed kernel. The reported image $x$ is the center of $u$, with 25 pixels removed on every side. Neither $u$ outside this center nor a calibrated noise distribution is supplied.

Save **`output/reconstruction.npz`**, relative to the task workspace, with exactly these arrays:

| Output key | Batch-first shape | Numeric dtype | Units and convention |
|---|---|---|---|
| `reconstruction` | `(1, 753, 668)` | `float32` | Dimensionless normalized luminance in `[0,1]`; axes are batch, row, column; no color or complex component |
| `kernel` | `(1, 51, 51)` | `float32` | Dimensionless cross-correlation weights; axes are batch, positive row offset, positive column offset; nonnegative and unit sum |

Image accuracy uses cosine NCC without mean subtraction and RMS error normalized by the reference dynamic range. Blind imaging has an unavoidable translation gauge. Before these two image metrics only, one integer translation is selected by the maximum of demeaned full-image cross-correlation within **plus or minus 25 pixels** on either axis. Ties use row-major order. The fixed reference region is rows `50:703` and columns `50:618`, a `(653,568)` interior; the image is sampled at the corresponding translated coordinates. The 50-pixel exclusion combines the 25-pixel kernel radius and 25-pixel allowed gauge displacement. There is no photometric rescaling, rotation, subpixel interpolation or independent local registration.

The kernel is also evaluated: nonnegativity, unit sum and its coupled ability to explain the observation are required. Forward consistency uses the **original, unshifted submitted image and kernel**, applying valid cross-correlation to obtain a `(703,618)` array and comparing it with `observation[0,25:728,25:643]`. Its RMS error is normalized by the dynamic range of that observed region. No boundary is padded or inferred from privileged data. Every required metric and physical constraint must pass; a sharp image paired with an unrelated kernel does not satisfy the contract.

## Data Description

The bound observation is the complete `scene026/blur/blur_3.png` camera exposure from the RealBlur-J dataset, converted to the integer luminance channel of the full-range YCrCb convention, then divided by 255. It has not been resized, cropped, synthetically blurred or corrupted with additional noise. RealBlur-J is processed JPEG-domain data distributed in PNG containers. Dataset registration and intensity processing predate this task; this is not raw sensor radiance.

| Key | Shape | Dtype | Description |
|---|---|---|---|
| `observation` | `(1, 753, 668)` | `float32` | Real processed exposure luminance, dimensionless relative full scale; axes batch, row, column |

**Acquisition metadata**

| Parameter | Value | Unit | Meaning |
|---|---|---|---|
| Source modality | Sony A7R3 real camera exposure | none | Measured handheld motion blur, not a numerical blur simulation |
| Spatial sampling | 753 rows by 668 columns | pixel | Complete distributed image raster; row increases downward and column rightward |
| Photometric convention | Integer full-range YCrCb luminance divided by 255 | relative full scale | Quantized processed intensity; no linear-radiance claim |
| Luminance conversion | `(4899 R + 9617 G + 1868 B + 8192) >> 14` | integer DN | RGB channels have 8-bit range; divide the integer result by 255 |
| Quantization step | `1/255` | relative full scale | Precision of the distributed luminance observation |
| Assumed kernel support | 51 by 51 | pixel | Specified finite-support blind forward-model assumption, not a measured calibration |
| Latent model extent | 803 rows by 718 columns | pixel | Adds a 25-pixel latent boundary on each side to model the complete measurement |
| Forward boundary convention | Valid cross-correlation | none | No circular convolution; no observed boundary padding |
| Kernel origin convention | Index `(25,25)` corresponds to the reported image origin | pixel | Translation remains ambiguous within the specified gauge |
| Exposure trajectory and noise variance | Unknown | none | Must not be assumed calibrated; structured residuals may remain |
| Pixel pitch and exposure time | Not supplied with this sample | none | Neither is needed for the pixel-domain model; no metric displacement or radiometric inference is requested |

## Method Hints

Joint deep priors parameterize the image and the kernel using two untrained generators. An image encoder-decoder can encode spatial structure, while a small fully connected kernel generator can impose positivity and unit sum through a softmax. Fitting both generators to this single observation regularizes a severely underdetermined problem without requiring paired training data.

Let $G_{\theta}$ and $H_{\phi}$ denote the image and kernel generators with weights $\theta$ and $\phi$, and let $z$ and $v$ denote fixed random inputs. A suitable model uses $u=G_{\theta}(z)$ and $k=H_{\phi}(v)$. Joint optimization fits the predicted blur to the observed luminance. A pixelwise discrepancy can establish the initial fit, followed by a structural discrepancy that emphasizes local contrast and texture. A deep prior does not remove translation ambiguity or guarantee a unique solution; retain the declared kernel convention and inspect physical observation consistency.

The validated runtime uses the shared research environment with PyTorch 2.9.1 and CUDA 12.8. Reproducible GPU optimization requires `CUBLAS_WORKSPACE_CONFIG=:4096:8` before CUDA work, disabled TF32, and `torch.use_deterministic_algorithms(True)`; selecting deterministic cuDNN alone does not control every backward operation. Full reconstruction requires a CUDA GPU with sufficient memory; CPU validation covers the mathematical operators and metrics.

## References

- Dongwei Ren, Kai Zhang, Qilong Wang, Qinghua Hu and Wangmeng Zuo. *Neural Blind Deconvolution Using Deep Priors*. CVPR 2020. [Published paper](https://openaccess.thecvf.com/content_CVPR_2020/html/Ren_Neural_Blind_Deconvolution_Using_Deep_Priors_CVPR_2020_paper.html).
- Jaesung Rim, Haeyun Lee, Jucheol Won and Sunghyun Cho. *Real-World Blur Dataset for Learning and Benchmarking Deblurring Algorithms*. ECCV 2020. [Dataset](https://huggingface.co/datasets/rimchang/RealBlur). RealBlur data are provided under CC BY 4.0; this task converts one complete distributed observation to normalized luminance and retains the attribution.
