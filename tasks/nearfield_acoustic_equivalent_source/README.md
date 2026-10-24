# Near-field Acoustic Equivalent Sources

> Reconstruct a spatial complex equivalent-source distribution from a sparse plane of measured acoustic pressure, despite severe underdetermination and evanescent-wave attenuation.
> Domain: physics | Keywords: acoustic imaging, inverse wave propagation, spatial regularization | Difficulty: Hard

## Background

Near-field acoustic measurements contain information about vibrating objects and sound-radiating structures. A nearby microphone plane captures propagating sound together with rapidly decaying spatial components. Recovering their source distribution supports noise-source localization, loudspeaker design and vibration diagnosis.

The measurement plane usually contains far fewer samples than the desired source image. High spatial frequencies decay rapidly with distance, and pressure measurements contain noise and interference. An equivalent-source image therefore depends on spatial regularity and a specified source-plane model; it need not equal the true mechanical surface velocity.

## Problem Description

The bound experiment measures a composition of four loudspeakers at a single frequency. Recover complex equivalent monopole strengths on a rectangular source plane from the complex pressure measured on a parallel plane. With the time convention exp(+i omega t), the model is


```math
p_m=\sum_{j=1}^{2000}G_{mj}q_j+e_m,\qquad G_{mj}=\frac{i\omega\rho}{4\pi r_{mj}}\exp(-ikr_{mj}),\qquad k=\omega/c.
```


Here p is the measured pressure, q is the equivalent monopole-strength image, e is measurement/model error, i is the imaginary unit, omega is angular frequency, rho is air density, c is sound speed and r is the Euclidean receiver-to-source distance. Each discrete q is a point-source strength; there is no pixel-area multiplier. There are 108 complex measurements and 2000 complex unknowns. Free-field propagation and a finite rectangular equivalent-source support are assumed.

Save `output/reconstruction.npz` with key `reconstruction`, numeric dtype **complex64**, shape **(1,40,50)**, axes **(batch,y,x)**. Real and imaginary components use the supplied phase convention. Pressure is in author-processed units U, so q has units **U m³/(Pa s)**, a calibration-relative volume velocity. Absolute pressure or absolute volume velocity cannot be inferred from this processed dataset. Preserve the supplied amplitude and phase without fitting a complex gain, translation, or phase rotation.

Evaluation uses relative complex L2 error, magnitude cosine similarity and relative complex pressure reprojection error. For vectors a and b, relative L2 is norm(a-b)/norm(b), and magnitude cosine is dot(abs(a),abs(b))/(norm(abs(a))*norm(abs(b))). The reprojection error is norm(Gq-p)/norm(p). No implicit crop, alignment or normalization is applied. Agreement with a regularized standard method measures reproducibility, not independently measured source accuracy.

## Data Description

These are existing, author-processed **real acoustic measurements**, not simulated observations. Separate loudspeaker impulse responses were Fourier transformed, each speaker's maximum pressure amplitude was equalized, its maximum-pressure phase shifted to zero, and the resulting fields summed. Only this composed field is supplied; the independent individual-speaker signals are unavailable. U denotes the retained author numerical pressure scale, not calibrated pascals.

| Key | Shape | Dtype | Description |
|---|---|---|---|
| pressure | (1,9,12) | complex64 | Composed pressure in U; axes batch,y,x; real+i imaginary under exp(+i omega t). |

**Acquisition metadata**

| Parameter | Value | Unit | Meaning |
|---|---|---|---|
| Frequency | 1150 | Hz | Single Fourier frequency; omega=2 pi frequency. |
| Sound speed | 343 | m/s | Uniform free-field medium. |
| Air density | 1.2 | kg/m³ | Factor in pressure-to-volume-velocity propagation. |
| Measurement plane height | 0.025 | m | Receiver z relative to source z=0. |
| Rectangle width | 0.825 | m | Both grids span x=0 to this inclusive endpoint. |
| Rectangle height | 0.600 | m | Both grids span y=0 to this inclusive endpoint. |
| Receiver grid | 9 rows, 12 columns | samples | Uniform inclusive y,x grids; row-major flattening. |
| Source grid | 40 rows, 50 columns | samples | Uniform inclusive y,x grids; row-major flattening. |
| Receiver spacing | 0.075 | m | Both x and y directions. |
| Source spacing x | 0.825/49 | m | Inclusive endpoints. |
| Source spacing y | 0.600/39 | m | Inclusive endpoints. |
| Coordinate axes | x columns, y rows, z toward receivers | convention | Right-handed spatial model. |
| Temporal convention | exp(+i omega t) | convention | Outgoing Green function exp(-ikr)/(4 pi r). |
| Boundary | free field, finite source rectangle | convention | No reflections or periodic extension; regularization may impose edge assumptions. |
| Noise | unknown | U | No independently calibrated noise variance is supplied. |

## Method Hints

Complex Tikhonov inversion provides a useful reference for the underdetermined linear system. Spatial derivative penalties can concentrate the equivalent sources while controlling unstable spatial oscillations. A primal-dual splitting method separates the smooth pressure-fit term from a nonsmooth isotropic derivative penalty. First- and second-order finite differences impose different spatial assumptions; their boundary conditions and adjoints matter.

## References

Pagavino, M. and Zotter, F. (2025). Data-driven and physics-constrained acoustic holography based on optimizer unrolling. *Acta Acustica*, 9, 46. [Published article](https://doi.org/10.1051/aacus/2025030). The task selects the article's classical equivalent-source baselines and existing processed real experiment.
