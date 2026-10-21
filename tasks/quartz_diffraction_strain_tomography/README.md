# Quartz Diffraction Strain Tomography

> Reconstruct all six spatial elastic-strain components inside a known quartz grain from noisy directional diffraction line averages with uneven angular sensitivity.
> Domain: chemistry & material science | Keywords: diffraction imaging, tensor tomography, material characterization | Difficulty: Hard

## Background

Elastic strain changes lattice spacings and therefore the lengths and directions of reciprocal-lattice diffraction vectors. Scanning diffraction records these changes along rays through crystalline material. Unlike ordinary attenuation imaging, each diffraction event probes a direction-dependent contraction of a tensor field.

Spatial strain maps help characterize residual stress, mechanical response, and local deformation inside grains without cutting the specimen. Recovering a complete tensor is difficult because different rays and diffraction directions constrain its components unevenly. Noise and sparse angular information amplify this imbalance.

## Problem Description

This task uses an **existing author-supplied simulated quartz diffraction dataset**, rather than a real instrument measurement. The unknown is a symmetric strain tensor E at each material voxel. For event a and detector bin d, the measured scalar strain b is

```math
b_{ad}=k_{ad}^{\mathsf T}\left(\frac{1}{L_{ad}}\int_{\ell_{ad}\cap\Omega}E(u,v)\,ds\right)k_{ad}+\eta_{ad}.
```

Here Omega is the supplied known grain support, ell is a parallel ray, L is its path length in the support, k is the measured unit diffraction direction, and eta is measurement noise. Obtain k from the supplied diffraction vector g as g/(norm(g)+1e-12), preserving the numerical regularization. The contraction uses E=(Exx,Eyy,Ezz,Exy,Exz,Eyz), with factors two multiplying the three off-diagonal terms. All components and measured strains are dimensionless, with strain 0.001 corresponding to 1000 microstrain. Tensor axes x,y,z are the supplied diffraction-vector component basis.

Image coordinates u and v denote centered row and column positions in voxel units, both increasing with the stored index. A detector ray satisfies `v*cos(omega)+u*sin(omega)=d-(192-1)/2`, with omega the supplied projection angle in radians. The image shape is 106 rows by 96 columns; a voxel center has u=r-(106-1)/2 and v=c-(96-1)/2. Integrate using dominant-axis Joseph sampling, with linear interpolation perpendicular to the ray. The bound numerical acquisition uses float32 CUDA texture interpolation with 1/256 fractional weights, 32-slice accumulation blocks, zero extension outside the image, and division by the identically projected support. A zero-support ray carries zero predicted signal. These details matter because the inverse is sensitive to very small operator perturbations.

Save `output/reconstruction.npz` with key `reconstruction`, a finite real float32 or float64 array of shape `(1, 6, 106, 96)`. Axes are batch, tensor component, original row, original column. Components are ordered `(xx, yy, zz, xy, xz, yz)` in dimensionless strain units. Strain is zero outside the given support. The evaluation compares every component on the entire declared array using cosine similarity without mean subtraction and RMS error normalized by that component's reference dynamic range. No rotation, scaling, masking, cropping, or alignment is fitted during evaluation.

## Data Description

| Key | Shape | Dtype | Description |
|---|---|---|---|
| strain_measurements | (1, 500, 192) | float64 | Existing noisy directional line-average strain, dimensionless |
| diffraction_vectors | (1, 3, 500, 192) | float64 | Measured reciprocal-space vectors in the fixed x,y,z basis, inverse angstrom |
| noise_std | (1, 500, 192) | float64 | Propagated per-ray strain standard deviation, dimensionless |
| angles | (1, 500) | float64 | Parallel-ray projection angles, degrees |
| support | (1, 106, 96) | bool | Known grain morphology, dimensionless Boolean prior |

### Acquisition metadata

| Parameter | Value | Unit | Meaning |
|---|---|---|---|
| Observation origin | Existing author simulation | - | No new observations generated for this task |
| Projection events | 500 | event | Each has its own diffraction direction |
| Detector bins | 192 | bin | Center index 95.5 |
| Detector spacing | 1 | voxel | Image and detector coordinate scale |
| Voxel spacing | 1 | voxel | Both image axes; absolute length unnecessary for normalized averages |
| Image origin | row 52.5, column 47.5 | voxel index | Zero of centered coordinates |
| Tensor convention | xx,yy,zz,xy,xz,yz | - | Off-diagonal entries are tensor strains, not doubled engineering shear |
| Reciprocal basis | x,y,z supplied vector order | - | Used consistently for every tensor contraction |
| Outside support | zero strain | dimensionless | Known void boundary condition |
| Outside-support ray | zero predicted signal | dimensionless | Avoid division by zero path length |
| Empty-ray noise entry | 1 | dimensionless strain | Author sentinel for rays not intersecting material; not actual camera noise |
| Simulation wavelength | 0.2845704100778472 | angstrom | Author acquisition wavelength |
| Unstrained unit cell | 4.926,4.926,5.4189;90,90,120 | angstrom; degree | Author quartz reference cell |
| Noise weighting | reciprocal supplied standard deviation | inverse strain | Zero standard deviation receives zero weight |

## Method Hints

A compact Gaussian radial-basis expansion within the known support regularizes the spatial tensor field. Combine normalized ray integration, directional tensor contraction, and per-observation precision weights into a least-squares objective. An ordinary gradient optimizer is suitable, but its gradient must use the transpose of the same forward Jacobian. A generic image backprojection need not be that transpose. Both the explicit discretization and the floating-point accumulation order influence convergence.

The validated numerical runtime uses an NVIDIA Ada GPU, CUDA toolkit 12.2 with nvcc, MSVC 14.29, and Windows SDK 10.0.19041.0; the general numerical Python dependencies are listed in requirements.txt. An explicit implementation may use general GPU computation facilities. The acquisition mathematics above defines the problem independently of any solver layout.

## References

Henningsson and Hall. *An efficient system matrix factorization method for scanning diffraction based strain tensor tomography*. Acta Crystallographica A79, 542-549 (2023). [Published article](https://doi.org/10.1107/S2053273323008136).
