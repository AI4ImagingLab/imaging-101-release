# Edge-Welded Beam Contour Normal Stress

> Reconstruct a complete cut-face normal-stress map from measured surface contours despite incomplete reliable profiling, elastic coupling and amplification of surface noise.
> Domain: mechanics | Keywords: residual stress imaging, elastic inversion | Difficulty: Hard

## Background

A cut through a stressed solid releases internal tractions and changes the shape of its exposed faces. Surface profilometry records these small changes over a much larger cross-section than a single strain gauge can sample. Linear elastic superposition connects the measured contour with the stress that acted across the cut.

Contour measurements are used to examine welded components, manufactured metal parts and structures where residual loading affects fatigue and fracture. The measurement is destructive. Reliable interpretation depends on coordinate conventions, material properties and a sufficiently elastic cutting process; surface artefacts and differentiation of noisy profiles limit spatial detail.

## Problem Description

The observations are the author's registered and averaged real coordinate-measuring-machine profiles from an edge-welded steel beam. This benchmark starts with those existing averaged observations. It does not require repeating registration or averaging of the original two face scans. Infer normal stress on all 3,166 specified cut-face nodes of the complete supplied half-body geometry.

Let the displacement vector be u, the infinitesimal strain tensor be epsilon and the Cauchy stress tensor be sigma. In the fixed elastic restoration model:

```math
\nabla\cdot\boldsymbol\sigma=0,\qquad
\boldsymbol\epsilon=\tfrac12\left(\nabla\mathbf u+(\nabla\mathbf u)^T\right),\qquad
\boldsymbol\sigma=\lambda\operatorname{tr}(\boldsymbol\epsilon)I+2\mu\boldsymbol\epsilon.
```

Here I is the identity tensor, and lambda and mu are the Lame moduli determined by Young's modulus E and Poisson's ratio nu:

```math
\lambda=\frac{E\nu}{(1+\nu)(1-2\nu)},\qquad
\mu=\frac{E}{2(1+\nu)}.
```

The unknown is q=sigma(zz) on the cut face. Its outward normal is minus the model z unit vector. Its corresponding normal traction is therefore minus q times that unit vector. Other exterior surfaces are traction-free, and tangential traction on the cut face is zero. The acquisition maps this unknown traction through elastic compliance to sampled normal displacements. The supplied signed heights define the restoration displacement u(z)=h in the author's convention. Recovering q reverses that compliance map; incomplete reliable sampling and fine-scale noise make the inverse unstable. Rigid translations and rotation are fixed by the stated anchor convention.

This is a conditional restoration-stress estimate. Positive q means tensile normal stress in this restoration model. The original measurement-axis-to-release-sign calibration is not independently available, so externally calibrated pre-cut tensile/compressive sign accuracy is not claimed. Interior relaxation stresses are not the requested original residual-stress tensor.

Save `output/reconstruction.npz` with exactly key `reconstruction`, shape `(1, 3166)`, real `float64`, in MPa. Axis order is `(batch, cut-face node)` following `surface_node_indices` exactly. Retain signed values and every declared node, including regions where the surface fit extrapolates. No magnitude conversion, clipping, registration, gain fitting or reference-dependent alignment is applied. Evaluation uses cosine similarity without mean subtraction, RMS error divided by the comparison map's full dynamic range, and maximum absolute error in MPa; every criterion must pass.

## Data Description

| Key | Shape | Dtype | Description |
|---|---|---|---|
| surface_samples | (1, 2010, 3) | float64 | Existing averaged CMM observations: author x coordinate, author y coordinate, signed restoration height; all three columns in mm |
| fit_mask | (1, 2010) | bool | Original input-only profile quality selection, dimensionless; true identifies points eligible for fitting |
| outline | (1, 128, 3) | float64 | Author cross-section outline coordinates, mm; x/y bound the profile fit |
| mesh_nodes | (1, 34826, 3) | float64 | Known pre-deck half-body geometry in model x/y/z coordinates, mm |
| mesh_elements | (1, 30370, 8) | int64 | Zero-based eight-node brick connectivity, dimensionless node index |
| surface_node_indices | (1, 3166) | int64 | Ordered zero-based cut-face nodes to reconstruct, dimensionless node index |
| rigid_anchor_indices | (1, 2) | int64 | Known nodes fixing the in-plane rigid-motion gauge, dimensionless node index |

### Acquisition metadata

| Parameter | Value | Unit | Meaning |
|---|---|---|---|
| Material | Edge-welded AISI316H steel | — | Existing author specimen; no simulated target is substituted |
| Young's modulus | 195600 | MPa | Author's uniform elastic model parameter |
| Poisson's ratio | 0.294 | — | Author's uniform isotropic model parameter |
| Displacement and geometry scale | 1 | mm | CMM coordinates and all displacements use millimetres |
| Model versus author profile coordinates | Add1 to x/y/z geometry coordinates | mm | Fixed author's FE-deck geometry translation; height remains a displacement and is not translated |
| Cut plane | z=1 | mm | Minimum model z; outward normal points in negative z |
| Half-body extrusion | 31 | mm | Complete supplied body; remaining exterior traction is zero |
| Height convention | u(z)=signed height | mm | Fixed restoration-model displacement convention |
| Reliable profile selection | Supplied original fit_mask | — | Retains1,917points; excludes identified cut-completion artefacts before inference |
| Unobserved average-grid positions | Excluded before packaging | — | Only the2,010finite recorded averages are observations |
| Boundary displacement representation | float32, then six decimal places | mm | Author's point-container and text-deck precision, after evaluating the fitted profile |
| FE geometry representation | Six decimal places | mm | Preserve supplied full-precision geometry for profile evaluation; round geometry for elastic assembly using the original numerical protocol |
| Rigid gauge | First anchor u(x)=u(y)=0; second anchor u(y)=0 | mm | Removes the remaining in-plane rigid modes without changing stress |
| Stress convention | Positive normal tension in restoration model | MPa | Fixed model sign, without independent pre-cut sign calibration |
| Noise model | Measurement covariance not provided | — | Do not assume a calibrated Gaussian noise level |

## Method Hints

A contour-method reconstruction can fit a smooth bivariate spline to the reliable observations, use that fit as normal displacement on the full cut face, and solve the isotropic elastic restoration problem. Knot spacing controls spatial smoothness and limits noise amplification. Preserve the model's geometry and boundary precision conventions.

An explicit finite-element stiffness matrix, displacement constraints and constitutive stress evaluation are sufficient. Fully integrated bricks provide stresses at eight quadrature points; recovering nodal values requires inverse shape interpolation in the correct quadrature order, followed by averaging contributions from neighboring elements. A forward shape matrix is not the Gauss-to-node extrapolator. The reconstructed cut-face normal stress is invariant to an admissible rigid-motion gauge.

## References

- Roy, Stoyanov, Moat and Withers (2020), *pyCM: An open-source computational framework for residual stress analysis employing the Contour Method*, [SoftwareX11,100458](https://doi.org/10.1016/j.softx.2020.100458).
- Roy (2019), *EC5thFramework ENPOWER austenitic edge welded beam contour cut metrology for assessing residual stress*, [author dataset](https://doi.org/10.5281/zenodo.3373916). This task reuses the paper repository's higher-precision registered/averaged processing record; the separate dataset describes the original face scans.
