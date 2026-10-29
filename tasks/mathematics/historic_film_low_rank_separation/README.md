# Historic Film Low-Rank and Sparse Separation

> Separate persistent image structure and sparse temporal changes from a complete real historical film despite camera jitter, film texture and overlapping motion.
> Domain: mathematics | Keywords: low-rank imaging, sparse decomposition, temporal imaging | Difficulty: Hard

## Background

A sequence of intensity images combines persistent scene appearance with changes caused by moving objects, illumination and recording defects. When successive frames share much of their structure, a matrix whose columns are images may have a compact low-rank component. Local changes can instead occupy a small subset of the space-time entries.

This decomposition is useful for video analysis, archival restoration and separating structured signals from impulsive interference. It describes a statistical image formation model rather than a unique partition into physical objects. Camera motion, shadows, repeated movement and textured film can violate its assumptions, so a sparse component need not be a true foreground mask.

## Problem Description

Analyze all84 native digital frames of Traffic Crossing Leeds Bridge, recorded in1888 and preserved as a public-domain digital film. Let T=84, H=240, W=320. Flatten each image in row-major (y,x) order to form D in real pixels-by-time coordinates, with measured uint8 intensity divided by255. The additive observation model is

```math
D=L+S+E.
```

Here D has HW rows and T columns, L is a low-rank image sequence, S is a signed sparse change sequence, and E is the residual in normalized digital intensity units. The components overlap in support and are not uniquely identifiable without structural assumptions. The complete native field includes film grain, borders, scene motion and camera jitter; no supplied mask isolates moving objects. No independent component truth or calibrated noise distribution is available.

Save `output/reconstruction.npz` relative to the task workspace, with both required arrays:

| Key | Shape | Numeric dtype | Units and convention |
|---|---|---|---|
| reconstruction | (1, 84, 240, 320) | float64 | L as batch,time,downward y,rightward x; normalized digital intensity (ADU/255), real and signed, without clipping. |
| sparse_component | (1, 84, 240, 320) | float64 | S in identical axis order and units; signed changes, not absolute differences or a binary mask. |

All values must be finite. Each complete component is evaluated by relative L2 error, norm(estimate-reference)/norm(reference), against a validated standard decomposition. Additive consistency is measured as norm(D-L-S)/norm(D). Norm denotes the Euclidean norm over every entry. No translation, intensity scaling, sign alignment, cropping or magnitude conversion is performed. Both components and additive consistency must pass. Component agreement does not establish foreground or restoration accuracy.

## Data Description

`data/raw_data.npz` contains the entire decoded grayscale movie. Conversion from the source's slightly tinted color encoding used the standard BT.601 digital grayscale convention; no crop, resize, added corruption or temporal averaging was applied.

| Key | Shape | Dtype | Description |
|---|---|---|---|
| frames | (1, 84, 240, 320) | uint8 | Digital grayscale intensity in ADU; axes batch,time,y,x. |

**Acquisition metadata**

| Parameter | Value | Unit | Meaning |
|---|---|---|---|
| Historical scene | Traffic Crossing Leeds Bridge,1888 | identifier | Original film by Louis Le Prince. |
| Native digital frame count | 84 | frames | Every decoded frame is retained in order. |
| Native digital image dimensions | 240 by320 | pixels | Downward y and rightward x. |
| Grayscale convention | BT.601 digital luminance | ADU | Conversion from decoded color samples; intensity is not calibrated radiance. |
| Numeric intensity scale | D=frames/255 | normalized ADU | Exact normalization used by the observation model. |
| Pixel flattening | C-order (y,x), images as columns | convention | Pixel index y*320+x; matrix shape (76800,84). |
| Digital playback rate | 29.97002997002997 | frames/s | Container playback timing only; the1888 camera rate is unverified. |
| Physical pixel size, exposure and camera response | unavailable | uncalibrated | No physical spatial or radiometric scale is inferred. |
| Boundary treatment | no spatial or temporal extension | convention | The finite native image matrix is decomposed directly. |
| Noise model | uncalibrated recording/film/digitization variability | normalized ADU | No known Gaussian variance or photon count calibration is imposed. |

The original film is public domain; the source's rights statement identifies Public Domain Mark1.0 and applicable expired copyrights. Digital sampling does not create a calibrated physical acquisition clock.

## Method Hints

For the evaluated decomposition, form D with one frame per column and row-major flattened pixels in its rows, using observed digital codes divided by 255 without centering. Fix the objective coefficient to lambda=1/sqrt(max(D.shape))=1/sqrt(76800). The constraint and the two norms below use all entries with no sample-count normalization. This coefficient defines the requested decomposition and is a modeling choice rather than an acquisition calibration.

Principal component pursuit uses nuclear-norm regularization for the persistent component and an entrywise L1 penalty for sparse changes:

```math
\min_{L,S}\ \lVert L\rVert_*+\lambda\lVert S\rVert_1
\quad\mathrm{subject\ to}\quad D=L+S.
```

The nuclear norm is the sum of singular values, the L1 norm is the sum of absolute entries, and lambda>0 balances the two structural penalties. An augmented-Lagrangian method alternates singular-value shrinkage, entrywise soft thresholding and a dual update. Preserve signed values and assess the full additive residual. Ordinary truncated principal component analysis offers a useful comparison but need not recover the same sparse changes.

## References

- Candes, Li, Ma and Wright, Robust Principal Component Analysis?, Journal of the ACM58(3),2011, DOI: https://doi.org/10.1145/1970392.1970395 ; preprint arXiv:0912.3599v1.
- Louis Le Prince, Traffic Crossing Leeds Bridge (1888), full digital film and public-domain rights statement: https://commons.wikimedia.org/wiki/File:Traffic_Crossing_Leeds_Bridge_film.ogv .
- Locally extracted fixed-penalty principal-component-pursuit implementation; MIT source license is preserved in LICENSE.
