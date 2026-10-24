# Natural Image Opacity Reconstruction

> Recover the complete spatial foreground-opacity field from a real RGB photograph and an independent coarse hand-drawn trimap despite ambiguous foreground-background mixing and fine hair boundaries.
> Domain: mathematics | Keywords: inverse compositing, variational reconstruction, image matting | Difficulty: Medium

## Background

An image pixel can receive light from a foreground object and the scene behind it. Partial coverage, fine hair, translucency and optical averaging create fractional foreground contributions. Representing these contributions as a continuous opacity field enables accurate boundary representation and subsequent image compositing.

A single color observation does not separately determine foreground color, background color and opacity. Coarse manual foreground/background constraints help resolve this ambiguity, while local color regularity propagates the information into mixed regions. Such reconstruction is useful for image editing and visual effects, but an inferred opacity is still sensitive to these modeling and annotation assumptions.

## Problem Description

The observation is the full native lemur photograph, accompanied by the source's independent manual trimap. For each row-major pixel i, the nominal compositing model is

```math
I_i=\alpha_i F_i+(1-\alpha_i)B_i.
```

Here I is the observed three-channel RGB vector, F and B are unknown foreground and background RGB vectors in the same dimensionless encoded color scale, and alpha is foreground opacity in [0,1]. The requested unknown is the complete spatial opacity field; F and B are nuisance fields, not supplied observations. The usual linear mixture is an approximation in encoded photographic RGB, so no absolute radiometric opacity calibration is implied.

The independent trimap supplies exact opacity 0 at background labels and 1 at foreground labels. Its gray value 0.4 identifies an unknown pixel and is **not** an opacity observation. Most ambiguous pixels lie near the object's hair and boundary. The inverse is underdetermined because one observed RGB vector permits many combinations of alpha, F and B. Spatial/color assumptions and the stated constraints are necessary to obtain a useful field.

Save `output/reconstruction.npz` relative to your task workspace, with exactly one key `reconstruction`: a finite real float64 array of shape `(1, 440, 680)`. Axes are batch, native image row, native image column. Values are dimensionless foreground opacity in [0,1]; rows increase down, columns right. All native pixels must be reconstructed without resizing or cropping. Known trimap labels must remain exact within numerical precision. Evaluation uses flattened cosine NCC without mean subtraction, RMS error normalized by the reference dynamic range, and maximum error at known opacity labels. No fitted alignment, rescaling, thresholding or foreground-background exchange is applied.

## Data Description

`data/raw_data.npz` contains the real photograph and its independently authored trimap, converted losslessly from the source's 8-bit images to the declared normalized float64 values. Neither a target opacity field nor foreground/background estimates are visible inputs.

| Key | Shape | Dtype | Description |
|---|---|---|---|
| image | (1, 440, 680, 3) | float64 | Observed encoded RGB photograph, dimensionless values equal to original 8-bit channel codes divided by 255; channel order red, green, blue |
| trimap | (1, 440, 680) | float64 | Independent manual labels, dimensionless: 0 background, 1 foreground, 0.4 unknown |

**Acquisition metadata**

| Parameter | Value | Unit | Meaning |
|---|---|---|---|
| Observation origin | Real photograph | dimensionless | No new synthetic observation or noise realization is generated |
| Native rows | 440 | pixel | Complete released image height |
| Native columns | 680 | pixel | Complete released image width |
| Color order | RGB | dimensionless | Last array axis is red, green, blue |
| Color encoding | Original 8-bit photographic RGB codes divided by 255 | dimensionless | Preserve source encoding; no linear-light or absolute flux conversion |
| Known opacity labels | 0 and 1 | dimensionless | Independent background and foreground annotations |
| Unknown annotation | 0.4 | dimensionless label | No opacity value is specified at these pixels |
| Unconstrained pixels | 38666 | pixel | Number of unknown trimap labels |
| Coordinates | Row down, column right, top-left origin | pixel | Original array coordinates, no physical length calibration |
| Image boundary | Finite image, no periodic extension | dimensionless | No observations beyond the supplied image are assumed |
| Sensor noise | Unspecified photographic acquisition noise | encoded RGB unit | No calibrated variance or added noise is supplied |

The photograph is by Mathias Appel, released under CC0 1.0. The manual trimap is supplied with the associated research implementation. This is a real-image reconstruction with manual constraints, not a simulated opacity benchmark.

## Method Hints

Closed-form color-line matting eliminates local affine foreground/background nuisance coefficients to obtain a positive-semidefinite quadratic energy in opacity. For overlapping local windows, color covariances define a matting Laplacian L. Partitioning pixels into known K and unknown U gives the constrained linear system

```math
L_{UU}\alpha_U=-L_{UK}\alpha_K.
```

The subscripts denote matrix blocks and corresponding opacity entries. A local covariance regularizer stabilizes nearly uniform color windows. Preconditioned conjugate gradients can solve the sparse system while preserving independent manual labels. A fractional field retains boundary structure that a hard binary segmentation discards; it does not uniquely identify separate foreground colors.

## References

- Levin, A., Lischinski, D., and Weiss, Y. *A Closed Form Solution to Natural Image Matting*. CVPR (2006). [Author-hosted paper](https://www.ee.technion.ac.il/people/anat.levin/papers/Matting-Levin-Lischinski-Weiss-CVPR06.pdf).
- Germer, T., Uelwer, T., Conrad, S., and Harmeling, S. *PyMatting: A Python Library for Alpha Matting*. JOSS 5(54), 2481 (2020). [Published paper](https://doi.org/10.21105/joss.02481).
- Appel, M. *Lemur photograph*, CC0 1.0. [Original photograph](https://www.flickr.com/photos/mathiasappel/25419442300/).
