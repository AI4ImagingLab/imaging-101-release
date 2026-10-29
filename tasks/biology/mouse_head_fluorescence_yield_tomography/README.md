# Mouse Head Fluorescence Yield Tomography

> Reconstruct a three-dimensional fluorescence-yield field inside a mouse head from boundary excitation and fluorescence measurements, despite strong tissue scattering and sparse angular sampling.
> Domain: biology | Keywords: fluorescence tomography, diffuse light, regularized inversion | Difficulty: Hard

## Background

Fluorescent molecular probes emit light after optical excitation. Binding a probe to a biological target can reveal molecular activity. In thick tissue, repeated scattering prevents direct assignment of detected light to its origin.

Fluorescence tomography combines several illumination and detection positions with a light-propagation model. Anatomical imaging can supply the tissue boundary and optical-fiber positions. Applications include preclinical studies of targeted probes, tumor biology and treatment response. Uncertain optical properties and limited measurement coverage constrain recoverable spatial detail.

## Problem Description

The author-supplied observations are spectrally processed real measurements of an EGFR-targeted probe in a mouse-head acquisition. MRI-derived geometry is provided; internal anatomical regions are not reconstruction priors. The unknown yield is the product of quantum yield and fluorophore absorption.

For excitation fluence $u$, fluorescence fluence $v$, unknown yield $q$, and known point-source distribution $s$, the continuous-wave model is

```math
-\nabla\cdot(\kappa_x\nabla u)+\mu_xu=s,\qquad
-\nabla\cdot(\kappa_m\nabla v)+\mu_mv=qu.
```

Here $\mu$ denotes background absorption, $\kappa$ denotes diffusion coefficient, and subscripts $x,m$ denote excitation and emission. For either fluence $w$, every exterior face has outward-normal condition

```math
\kappa\partial_nw+\xi w=0,\qquad
\xi=\frac{1}{2A},\quad
A=\frac{2/(1-R_0)-1+c^3}{1-c^2},\quad
R_0=\left(\frac{n_r-1}{n_r+1}\right)^2,\quad
c=\left|\cos(\arcsin(1/n_r))\right|.
```

The refractive index relative to air is $n_r$. Linear tetrahedral basis functions $N_i$ give mass entries $M_{ij}=\int N_iN_j\,dV$. The discrete emission load is $M(q\odot u)$, where $\odot$ denotes nodal multiplication. Exterior faces follow from tetrahedral connectivity.

Detectors interpolate boundary fluence. A calibrated datum is $y_a=F_aX^{\mathrm{model}}_a/X_a$, where $F_a,X_a$ are observed fluorescence and excitation in channel $a$, and $X^{\mathrm{model}}_a$ is predicted excitation. Only 56 channels constrain thousands of spatial unknowns. Give a nonnegative field that explains all 56 observations under this model. Such a feasible field is not unique; this task does not certify a true tumor distribution or independently measured molecular concentration.

Save `output/reconstruction.npz`, relative to the task workspace, with exactly this key:

| Key | Shape | Dtype | Description |
|---|---|---|---|
| reconstruction | (1, 5707) | float64 | Nonnegative real fluorescence yield in mm^-1, in the exact node order of `nodes`. |

All nodes contribute to an independently assembled P1 finite-element forward calculation. Let predicted channel values be h(q), with the fixed calibration y above. Mean squared log residual, mean over all 56 channels of [log(h(q))-log(y)] squared, may exceed an attainable conditional fit by at most 1e-6. This predeclared dimensionless objective tolerance is not a measured noise variance. Full relative L2 error ||h(q)-y||/||y|| must also be at most 0.191. Zero channel predictions fail. No masking, permutation, alignment or fitted scale is applied. Nonnegative fields with equal or better supported fits are accepted; no unique spatial truth is assumed.

## Data Description

Every active author measurement is retained. XYZ coordinates are the author's Cartesian mesh coordinates in millimetres. Indices are zero-based.

| Key | Shape | Dtype | Description |
|---|---|---|---|
| fluorescence | (1, 56) | float64 | Observed processed probe fluorescence, instrument-relative intensity units. |
| excitation | (1, 56) | float64 | Observed excitation for the same channels, instrument-relative intensity units. |
| links | (1, 56, 2) | int64 | Source and detector indices per channel, dimensionless. |
| nodes | (1, 5707, 3) | float64 | Tissue FEM coordinates, XYZ in mm. |
| tetrahedra | (1, 29448, 4) | int64 | Tissue-node indices per tetrahedron, dimensionless. |
| boundary_vertices | (1, 5707) | bool | Known boundary-vertex indicator, dimensionless; exterior faces follow from connectivity. |
| source_positions | (1, 8, 3) | float64 | Model point sources displaced inward by the author's optical-transport convention, XYZ in mm. |
| detector_positions | (1, 8, 3) | float64 | Surface detector coordinates, XYZ in mm. |
| coarse_nodes | (1, 3323, 3) | float64 | Fixed geometry-only reconstruction-basis coordinates, XYZ in mm. |
| coarse_tetrahedra | (1, 17350, 4) | int64 | Reconstruction-basis connectivity, dimensionless indices. |

**Acquisition metadata**

| Parameter | Value | Unit | Meaning |
|---|---|---|---|
| Geometry | 8 sources x 7 other detectors | dimensionless | All 56 ordered pairs. |
| Modulation frequency | 0 | MHz | Continuous-wave data; no lifetime recovery. |
| Excitation absorption | 0.009 | mm^-1 | Fixed background coefficient. |
| Emission absorption | 0.006 | mm^-1 | Fixed background coefficient. |
| Excitation diffusion | 0.251953 | mm | Exact supplied author mesh value. |
| Emission diffusion | 0.260620 | mm | Exact supplied author mesh value. |
| Refractive index | 1.33 | dimensionless | Tissue relative to air. |
| Point-source amplitude | cos(0.15) | dimensionless | Real CW part of the author's source; angle is in radians. |
| Source footprint | Barycentric point load | mm | No Gaussian broadening. |
| Detector convention | Boundary-only P1 interpolation | dimensionless | Remove interior-vertex weights and renormalize in local order. |
| Reconstruction geometry | Author-generated 20 x 20 x 20 grid clipped to tissue | dimensionless | Fixed tetrahedralization avoids triangulation ties; no fitted values supplied. |
| Fine interpolation | P1, with nearest inside fine-node fallback | dimensionless | Outside the coarse mesh, copy interpolation from the nearest fine node that lies inside. |
| Noise covariance | Not supplied | intensity units^2 | No independent covariance estimate available. |

## Method Hints

Use nonnegative diffuse-light finite-element inversion with excitation-normalized measurements. A matched forward sensitivity or adjoint supports constrained optimization of the calibrated log-observation residual.

A coarse geometric basis can reduce parameter count; nonnegativity restricts the weakly determined yield. Use the transpose or derivative of the same discretized diffusion/measurement model in inverse sensitivities. Initialization, numerical damping, line search and stopping do not supply independent physical information or a static spatial prior. Observation residuals and sensitivity to unsupported spatial structure should be assessed together.

## References

- Davis et al., *Comparing implementations of magnetic-resonance-guided fluorescence molecular tomography for diagnostic classification of brain tumors*, Journal of Biomedical Optics 15, 051602 (2010). [Original method family](https://doi.org/10.1117/1.3483902).
- NIRFAST, *Tutorial: Imaging a mouse brain tumor in an MRI-coupled fluorescence molecular tomography system*. Author tutorial and sample data. The selected configuration is the three-dimensional unguided reconstruction, with explicitly corrected exterior-face assembly.
