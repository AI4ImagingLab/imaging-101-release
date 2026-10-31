## Problem Statement

Reconstruct a high-resolution image of a planetary/lunar surface from a short-exposure video sequence degraded by atmospheric turbulence, using a select-align-stack (lucky imaging) pipeline with local adaptive stacking.

## Mathematical Formulation

### Atmospheric degradation model

Each observed frame $y_k$ is a spatially-varying degradation of the underlying sharp scene $x$:

$$y_k(\mathbf{r}) = h_k(\mathbf{r}) * x(\mathbf{r}) + n_k(\mathbf{r}), \quad k = 1, \ldots, K$$

where $h_k(\mathbf{r})$ is the instantaneous atmospheric point spread function (PSF), $*$ denotes spatial convolution, and $n_k$ is additive noise. The PSF varies both across frames (temporal turbulence) and across the field (anisoplanatism).

### Frame quality metric

The Laplacian variance measures high-frequency content as a proxy for instantaneous seeing quality:

$$Q_k = \mathrm{Std}\bigl[\nabla^2 \tilde{y}_k\bigr]$$

where $\tilde{y}_k$ is the Gaussian-blurred monochrome version of frame $k$, and $\nabla^2$ is the discrete Laplacian. Higher $Q_k$ indicates less atmospheric blurring.

### Global alignment

Frame-to-frame tip-tilt is estimated via normalised cross-correlation in a reference patch $\mathcal{P}$:

$$(\Delta y_k, \Delta x_k) = \arg\max_{\boldsymbol{\delta}} \frac{\sum_{\mathbf{r} \in \mathcal{P}} R(\mathbf{r}) \cdot y_k(\mathbf{r} + \boldsymbol{\delta})}{\|R\|_{\mathcal{P}} \cdot \|y_k\|_{\mathcal{P}+\boldsymbol{\delta}}}$$

where $R$ is the reference (mean of top-ranked frames). Sub-pixel refinement fits a 2D paraboloid $f(\delta_y, \delta_x) = a \delta_y^2 + b \delta_x^2 + c \delta_y \delta_x + d \delta_y + e \delta_x + g$ to the $3 \times 3$ correlation peak.

### Local stacking

At each alignment point $j$ on a staggered grid, the stacked patch is:

$$\hat{x}_j = \frac{\sum_{k \in S_j} w_j \cdot y_k\bigl[\mathbf{r} + \boldsymbol{\Delta}_k + \boldsymbol{\delta}_{k,j}\bigr]}{\sum_{k \in S_j} w_j}$$

where $S_j$ is the per-AP selected frame subset, $\boldsymbol{\Delta}_k$ is the global shift, $\boldsymbol{\delta}_{k,j}$ is the local warp correction, and $w_j(\mathbf{r})$ is a triangular (tent) blending weight that ramps linearly from 0 at the patch edge to 1 at the AP centre.

### Final blending

The full image is assembled by merging AP patches:

$$\hat{x}(\mathbf{r}) = \frac{\sum_j w_j(\mathbf{r}) \cdot \hat{x}_j(\mathbf{r})}{\sum_j w_j(\mathbf{r})}$$

with background fill from the globally-aligned mean in regions not covered by any AP.

### Unsharp masking

The raw stack suppresses the highest spatial frequencies because stacking averages over residual sub-pixel misalignments. Unsharp masking recovers this detail:

$$\hat{x}_\text{sharp}(\mathbf{r}) = \hat{x}(\mathbf{r}) + \alpha \bigl[\hat{x}(\mathbf{r}) - (G_\sigma * \hat{x})(\mathbf{r})\bigr]$$

where $G_\sigma * \hat{x}$ is the Gaussian-blurred stack with blur radius $\sigma$, and $\alpha \geq 0$ is the sharpening strength. The bracket term isolates high-frequency residuals. Default parameters: $\sigma = 2.0$ pixels, $\alpha = 1.5$.

## Solution Strategy

1. **Preprocessing** — extract frames from video, convert to monochrome (luminance), apply Gaussian blur (kernel size 7), compute normalised brightness.

2. **Frame ranking** — compute Laplacian-variance quality metric for each frame at stride 2, normalise by brightness, sort in descending order.

3. **Global alignment** — select the best frame as the initial reference. Using a patch of size `(H/3, W/3)` centred on the highest-structure region, align each frame to the reference via two-phase multi-level normalised cross-correlation:
   - Phase 1: stride-2 coarsened images, search width = `(34 - 4) / 2 = 15` pixels.
   - Phase 2: full resolution, ±4 pixel refinement around phase-1 result.
   - Sub-pixel: paraboloid fit on 3×3 peak neighbourhood.

4. **Mean reference** — average the top 5% of globally-aligned frames to form a cleaner reference for local operations.

5. **Alignment point grid** — create a staggered grid with step size derived from `half_patch_width = round(24 * 1.5) = 36`, step = `round(36 * 1.5) = 54`. Filter APs by structure threshold (0.04) and brightness threshold (10).

6. **Local ranking and selection** — at each AP, rank all frames by local sharpness within the AP box. Select the top 10% (≈10 frames from 101).

7. **Local shift computation** — for each selected frame at each AP, compute the local warp via multi-level cross-correlation within a ±14 pixel search window.

8. **Stacking** — accumulate shifted patches into per-AP float32 buffers. Apply triangular blending weights. Merge all AP buffers, normalise by weight sum, fill background gaps, trim borders. Convert to uint16 output.

9. **Unsharp masking** — apply post-processing sharpening to the uint16 stack: compute `sharpened = stack + alpha * (stack - GaussianBlur(stack, sigma))`, clip to `[0, 65535]`, return uint16. Default `sigma=2.0`, `alpha=1.5`.

## Expected Results

The stacked output (after unsharp masking) should show:
- Significantly sharper features (craters, ridges) compared to any single frame — target ≥3× Laplacian-variance ratio vs best frame
- Reduced noise (SNR improvement ≈ $\sqrt{N_{\text{stacked}}}$ per AP)
- No visible seams between alignment point regions
- 16-bit dynamic range output

Note: the raw stack (before unsharp masking) may appear softer than the single best frame. This is expected — stacking averages over residual sub-pixel misalignments, and the sharpness recovery requires the post-processing step.

Quality will be assessed by visual comparison to the single best frame and to the simple mean of all frames.

## Default Parameters

These values are hard-coded in `main.py` and were removed from `meta_data.json` (which is reserved for imaging parameters only):

| Parameter | Default | Description |
|-----------|---------|-------------|
| `gauss_width` | 7 | Gaussian blur kernel size for frame preparation |
| `ranking_method` | "Laplace" | Sharpness metric used for frame ranking |
| `ranking_stride` | 2 | Pixel stride when computing sharpness |
| `normalize_brightness` | True | Normalize per-frame brightness before ranking |
| `alignment_method` | "MultiLevelCorrelation" | Global and local alignment method |
| `alignment_search_width` | 34 | Global alignment search radius (pixels) |
| `average_frame_percent` | 5 | Percentage of top frames averaged to build reference |
| `ap_half_box_width` | 24 | Half-width of each alignment point box (pixels) |
| `ap_search_width` | 14 | Local search radius at each AP (pixels) |
| `ap_structure_threshold` | 0.04 | Minimum normalised structure to keep an AP |
| `ap_brightness_threshold` | 10 | Minimum brightness to keep an AP |
| `ap_frame_percent` | 10 | Percentage of best frames stacked per AP |
| `drizzle_factor` | 1 | Super-resolution factor (1 = off) |
| `usm_sigma` | 2.0 | Gaussian sigma for unsharp masking |
| `usm_alpha` | 1.5 | Amplification factor for unsharp masking |
